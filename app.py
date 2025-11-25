import os
import re
import time
import logging
import requests
import openai
from anthropic import Anthropic
from flask import Flask, render_template, request, session, jsonify, abort, send_file
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from io import BytesIO
import random

# === Load and Validate Environment Variables ===
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY   = os.getenv("CLAUDE_API_KEY")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
CLAUDE_MODEL     = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not CLAUDE_API_KEY:
    raise RuntimeError("Missing CLAUDE_API_KEY in environment")
if not MISTRAL_API_KEY:
    raise RuntimeError("Missing MISTRAL_API_KEY in environment")

openai.api_key = OPENAI_API_KEY
anthropic_client = Anthropic(api_key=CLAUDE_API_KEY)

# === Base System Prompt Template (editable via UI) ===
# This is what appears in the System Prompt popup and can be changed there.
BASE_SYSTEM_PROMPT_TEMPLATE = (
"""You are AIAgent, a rigorously analytical, adversarial-but-constructive participant in a multi-agent focus group. Your task is to advance the discussion, not to smooth it.

Your role:
- Stay tightly on topic and respond directly to the most recent claim or argument.
- Build on others’ points only if doing so introduces a new mechanism, contradiction, or deeper layer of analysis. Never paraphrase, restate, or praise what has already been said.
- Do NOT use token-wasting preambles such as “X raises a valid point…” or “As Y mentioned…”. Skip directly to your critique, extension, or counterargument.
- Challenge other agents by uncovering mechanisms (how something actually works), demanding evidence, examples, or logic, and identifying trade-offs, constraints, and perverse incentives.
- If another agent makes a vague or abstract claim, force specificity: ask how, why, compared to what, or under what conditions.
- If an argument contradicts itself, expose the contradiction and press on it until resolved.
- If another agent avoids uncomfortable implications, name them explicitly and analyze their consequences.
- Be direct, rational, unsentimental, and concise. No flattery, no softening, no conflict avoidance.

Output quality rules:
- No redundancy. Each message must introduce a new insight, mechanism, example, or sharper framing.
- Avoid meta-commentary and transitions; start immediately with substance.
- Use concrete cases, historical precedents, analogies, or counterexamples whenever they clarify the argument.
- Do not converge prematurely. If an argument is weak, say so; if consensus is unwarranted, disrupt it.
- Escalate the analysis: move from surface claims → mechanisms → structural causes → strategic implications → actionable consequences.
- Maintain a respectful but sharp tone: critique ideas, not personas.
- Do not mention AI, system prompts, or meta-instructions.
"""
)

# === Backend-only constraints (NOT editable via UI) ===
# These are always prepended by build_system_prompt().
def build_backend_constraints(max_chars: int) -> str:
    return (
        "You respond concisely but with substance, aiming for at most about "
        f"{max_chars} characters per turn.\n"
        "Do not use markdown formatting such as **bold**, bullet lists, or numbered lists. "
        "Write in plain text prose only."
    )

# Additional role-specific constraints that we keep regardless of user edits
CLAUDE_EXTRA = (
    "You may speak in the first person as Claude and adopt a recognizable voice, "
    "but do not describe physical actions, sounds, or emotions such as clears throat, "
    "smiles, laughs, pauses, or similar. Focus on informational content and reasoning only."
)

MISTRAL_EXTRA = (
    "Use coherent, complete sentences in 2–5 sentences "
    "and always end your answer with a full sentence and proper punctuation."
)

def get_base_system_prompt_template() -> str:
    """
    Return the current base system prompt template.
    If the user customized it, read from session; otherwise use default constant.
    """
    return session.get("custom_system_prompt_template") or BASE_SYSTEM_PROMPT_TEMPLATE

def build_system_prompt(agent_name: str, mtype: str, max_chars: int) -> str:
    """
    Build the concrete system prompt for a given agent, based on:
    - backend-only constraints (max_chars, no markdown)
    - (possibly customized) base template
    - role-specific extras (Claude/Mistral)
    """
    backend_constraints = build_backend_constraints(max_chars)
    template = get_base_system_prompt_template()

    # Allow user templates optionally to use {max_chars}, but don't require it.
    try:
        user_part = template.format(max_chars=max_chars)
    except Exception:
        user_part = template

    # Replace AIAgent placeholder by the agent name (if present).
    user_part = user_part.replace("AIAgent", agent_name)

    extra = ""
    if mtype == "claude":
        extra = CLAUDE_EXTRA
    elif mtype == "mistral":
        extra = MISTRAL_EXTRA

    parts = [backend_constraints, user_part]
    if extra:
        parts.append(extra)

    return "\n\n".join(parts)

# === Agent Definitions ===
agents = [
    ("ChatGPT", "gpt"),
    ("Claude",      "claude"),
    ("Mistral",     "mistral"),
]

# === Flask App Setup ===
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# === Rate Limiter ===
limiter = Limiter(key_func=get_remote_address, default_limits=["10 per minute"])
limiter.init_app(app)

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === Utility Functions ===

def get_context(conv, max_turns=10):
    return "\n".join(f"{e['speaker']}: {e['text']}" for e in conv[-max_turns:])

def truncate_to_complete_sentence(text: str) -> str:
    """
    Truncate text to the last full sentence that ends with . ! or ?.
    If none found, return stripped text.
    """
    if not text:
        return ""
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    if sentences:
        return ''.join(sentences).strip()
    return text.strip()

def is_repetitive(name, resp, conv, last_n=6):
    if not resp:
        return True
    recent = [e['text'] for e in conv[-last_n:] if e['speaker'] == name]
    return any(resp.strip() == r.strip() for r in recent)

def is_trivial(resp):
    if not resp:
        return True
    trivial_patterns = [
        r"^I agree\b",
        r"^I concur\b",
        r"^Yes, I agree\b",
        r"^Good point\b",
        r"^I support that\b",
        r"^That makes sense\b",
    ]
    return any(re.match(p, resp.strip(), re.IGNORECASE) for p in trivial_patterns)

def mentions_other_agents(resp, real_agents=None):
    if not resp:
        return False
    if real_agents is None:
        real_agents = agents
    text = resp.lower()
    for name, _ in real_agents:
        n0 = name.split()[0].lower()
        if n0 in text:
            return True
    return False

def is_shallow_agreement(resp):
    return bool(re.match(r"^I (agree|concur|support)( with .+)?[.!]?$", resp.strip(), re.IGNORECASE))

def get_next_agent(conv, agents_list):
    """
    Randomizes the speaking order per round:
    - Each round, each agent speaks exactly once.
    - Order within a round is random.
    """
    if not conv:
        return random.choice(agents_list)

    agent_names = [a[0] for a in agents_list]
    total_agent_msgs = sum(1 for e in conv if e['speaker'] in agent_names)
    pos_in_round = total_agent_msgs % len(agent_names)

    already_spoken = []
    if pos_in_round > 0:
        for e in reversed(conv):
            speaker = e['speaker']
            if speaker in agent_names and speaker not in already_spoken:
                already_spoken.append(speaker)
                if len(already_spoken) == pos_in_round:
                    break

    remaining = [a for a in agents_list if a[0] not in already_spoken]
    if not remaining:
        remaining = agents_list

    return random.choice(remaining)

def clean_agent_response(text: str) -> str:
    """
    Normalizes all agent outputs to plain text:
    - remove **bold** markers but keep inner text
    - remove any *...* segments entirely (stage directions etc.)
    - remove leading bullets (-, *, •) at line start
    - collapse to a single paragraph
    """
    if not text:
        return ""

    # Remove bold markdown but keep content
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Remove any *...* segments entirely (roleplay like *clears throat*)
    text = re.sub(r'\*[^*]+\*', '', text)

    # Split into lines and clean bullets
    lines = text.splitlines()
    cleaned_lines = []
    for ln in lines:
        ln = re.sub(r'^\s*[-*•]+\s*', '', ln)  # bullets at line start
        if ln.strip():
            cleaned_lines.append(ln.strip())

    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# === Smart continuation helper ===

def generate_full_sentence_response(call_fn, prompt, sys_prompt, max_tokens_initial, max_chars):
    """
    Calls an LLM repeatedly until:
    - we have a full ending sentence (., !, ?)
    - OR we reach approximately max_chars
    - OR we hit a small max_rounds limit
    """
    final_text = ""
    max_rounds = 3   # keep continuation limited
    max_tokens = max_tokens_initial

    for _ in range(max_rounds):
        chunk = call_fn(prompt, sys_prompt, max_tokens)
        if not isinstance(chunk, str):
            break

        final_text = (final_text + " " + chunk.strip()).strip()

        # Stop if we already end with . ! ?
        if final_text and final_text[-1] in ".!?":
            break

        # Stop if we reached our char limit (soft cap)
        if len(final_text) >= max_chars:
            break

        # Otherwise, allow the model to continue with slightly more tokens
        max_tokens = int(max_tokens * 1.2)

    # Clean and enforce full sentence ending
    final_text = clean_agent_response(final_text)
    final_text = truncate_to_complete_sentence(final_text)
    if final_text and final_text[-1] not in ".!?":
        final_text += "."

    return final_text

# === API Call Wrappers ===

def call_gpt_turbo(message, system_prompt, max_tokens):
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": message},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.exception("Error calling GPT: %s", e)
        return f"[Error from GPT]: {e}"

def call_claude(message, system_prompt, max_tokens):
    try:
        resp = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        )
        parts = []
        for block in resp.content:
            # Anthropics SDK: either objects with .text or dicts with type="text"
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    except Exception as e:
        logger.exception("Error calling Claude: %s", e)
        return f"[Error from Claude]: {e}"

def call_mistral(message, system_prompt, max_tokens):
    """
    Call Mistral with basic retry & rate-limit handling.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": message},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=40,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429 and attempt < 2:
                wait = 2 ** attempt
                logger.warning("Mistral 429 rate limit, retrying in %s seconds (attempt %s)...", wait, attempt + 1)
                time.sleep(wait)
                continue
            logger.exception("HTTP error calling Mistral: %s", e)
            if status == 429:
                return "[Error from Mistral]: rate limit reached, please try again later."
            return f"[Error from Mistral]: HTTP error {status}"
        except Exception as e:
            logger.exception("Error calling Mistral: %s", e)
            return f"[Error from Mistral]: {e}"

# === Routes ===

@app.route("/", methods=["GET"])
def index():
    session.clear()
    return render_template("index.html")

@app.route("/start", methods=["POST"])
@limiter.limit("5 per minute")
def start():
    data = request.get_json() or {}
    msg = data.get("message") or abort(400)
    session.update({
        "conversation":   [{"speaker": "Facilitator", "text": msg}],
        "max_characters": int(data.get("max_characters", 250)),  # default now 250
        "strict_limit":   bool(data.get("strict_limit", True)),  # kept for compatibility, not used anymore
        "rounds_left":    int(data.get("initial_rounds", 1)),
    })
    return jsonify({"facilitator": msg})

@app.route("/next_response", methods=["POST"])
@limiter.limit("10 per minute")
def next_response():
    conv   = session.get("conversation", [])
    rounds = session.get("rounds_left", 0)
    if rounds <= 0:
        return jsonify({"speaker": None, "text": "", "done": True})

    name, mtype = get_next_agent(conv, agents)
    context = get_context(conv)
    first   = not any(e['speaker'] in [a[0] for a in agents] for e in conv)

    if first:
        facil = conv[0]['text']
        prompt_text = (
            f"Facilitator asked: \"{facil}\"\n\n"
            f"As {name}, please provide your opening statement in response."
        )
    else:
        prompt_text = (
            f"Based on the following context:\n{context}\n\n"
            "Please provide your next statement."
        )

    max_chars  = session.get("max_characters", 250)
    # Rough token estimate
    max_tokens = round((max_chars / 4.2) * (0.9 if mtype == "gpt" else 1.1))

    # Build system prompt dynamically (using possibly customized base)
    sys_prompt = build_system_prompt(name, mtype, max_chars)

    # Choose appropriate call function for the model
    if mtype == "gpt":
        call_fn = lambda msg, sys, tok: call_gpt_turbo(msg, sys, tok)
    elif mtype == "claude":
        call_fn = lambda msg, sys, tok: call_claude(msg, sys, tok)
    else:
        call_fn = lambda msg, sys, tok: call_mistral(msg, sys, tok)

    last_raw = None
    response = None

    for attempt in range(3):
        # Generate a full-sentence response with continuation if needed
        raw = generate_full_sentence_response(
            call_fn=call_fn,
            prompt=prompt_text,
            sys_prompt=sys_prompt,
            max_tokens_initial=max_tokens,
            max_chars=max_chars
        )

        last_raw = raw.strip() if isinstance(raw, str) else None
        resp = last_raw or ""

        # Remove any leading "Name: " if the model prints it
        resp = re.sub(rf"^{re.escape(name)}[:\s-]*", "", resp, flags=re.IGNORECASE).strip()

        if not first:
            if is_repetitive(name, resp, conv):
                logger.info("Filtered (repetitive) from %s, attempt %d", name, attempt + 1)
                continue
            if is_trivial(resp) or is_shallow_agreement(resp):
                logger.info("Filtered (trivial/shallow) from %s, attempt %d", name, attempt + 1)
                continue
            if mentions_other_agents(resp, agents):
                logger.info("Filtered (mentions other agents) from %s, attempt %d", name, attempt + 1)
                continue

        response = resp
        break

    if response is None:
        response = last_raw or "[No valid response generated.]"

    conv.append({"speaker": name, "text": response})
    session["conversation"] = conv

    # Full round completed?
    if sum(1 for e in conv if e['speaker'] != "Facilitator") % len(agents) == 0:
        session["rounds_left"] = rounds - 1

    return jsonify({
        "speaker": name,
        "text":    response,
        "done":    session.get("rounds_left", 0) <= 0,
    })

@app.route("/facilitator_input", methods=["POST"])
@limiter.limit("5 per minute")
def facilitator_input():
    data = request.get_json() or {}
    msg  = data.get("message") or abort(400)
    conv = session.get("conversation", [])
    conv.append({"speaker": "Facilitator", "text": msg})
    session.update({
        "conversation":   conv,
        "max_characters": int(data.get("max_characters", 250)),  # default now 250 here as well
        "rounds_left":    int(data.get("rounds", 1)),
    })
    return jsonify({"status": "added"})

@app.route("/intervene", methods=["POST"])
@limiter.limit("5 per minute")
def intervene():
    data = request.get_json() or {}
    msg  = data.get("message", "").strip()
    conv = session.get("conversation", [])
    if msg:
        conv.append({"speaker": "Facilitator", "text": msg})
    session["conversation"] = conv
    return jsonify({"status": "ok"})

@app.route("/system_prompt", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def system_prompt():
    """
    GET: return current base system prompt template (default or customized).
    POST:
      - if {"reset": true}: reset to default, test, and return default prompt + status.
      - else: set new template, test, return prompt + status.
    """
    if request.method == "GET":
        return jsonify({
            "system_prompt": get_base_system_prompt_template()
        })

    data = request.get_json() or {}

    # RESET CASE
    if data.get("reset"):
        session.pop("custom_system_prompt_template", None)
        # test default prompt
        try:
            test_sys_prompt = build_system_prompt("GPT (Chat)", "gpt", 300)
            test_resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": test_sys_prompt},
                    {"role": "user",   "content": "Reply with a short confirmation sentence that you received the system prompt."},
                ],
                max_tokens=20,
                temperature=0,
            )
            _ = test_resp.choices[0].message.content
            ok = True
            message = "System prompt reset to default and test request succeeded."
        except Exception as e:
            logger.exception("Error testing reset system prompt: %s", e)
            ok = False
            message = f"System prompt reset to default, but test request failed: {e}"

        return jsonify({
            "ok": ok,
            "message": message,
            "system_prompt": get_base_system_prompt_template()
        })

    # UPDATE CASE
    new_prompt = data.get("system_prompt")
    if not new_prompt or not isinstance(new_prompt, str):
        abort(400, description="system_prompt is required and must be a string")

    session["custom_system_prompt_template"] = new_prompt

    try:
        test_sys_prompt = build_system_prompt("GPT (Chat)", "gpt", 300)
        test_resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": test_sys_prompt},
                {"role": "user",   "content": "Reply with a short confirmation sentence that you received the system prompt."},
            ],
            max_tokens=20,
            temperature=0,
        )
        _ = test_resp.choices[0].message.content
        ok = True
        message = "System prompt updated and test request succeeded."
    except Exception as e:
        logger.exception("Error testing new system prompt: %s", e)
        ok = False
        message = f"System prompt saved, but test request failed: {e}"

    return jsonify({
        "ok": ok,
        "message": message,
        "system_prompt": get_base_system_prompt_template()
    })

@app.route("/export", methods=["GET"])
def export_conversation():
    conv = session.get("conversation", [])
    lines = []
    for e in conv:
        lines.append(f"{e['speaker']}: {e['text']}")
    content = "\n" + "\n\n".join(lines)

    buf = BytesIO()
    buf.write(content.encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="conversation.txt",
        mimetype="text/plain",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5001)
