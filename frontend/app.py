from __future__ import annotations

import html as html_lib
import json
import os
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PRISM Assessment",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# API URLS
# ============================================================

# Requests made by Streamlit itself.
# Streamlit runs INSIDE the Docker frontend container,
# therefore it must use the Docker service name "api".
#API_URL = os.getenv("PRISM_API_URL", "http://api:8000").rstrip("/")

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Requests made by JavaScript running in the candidate's browser.
# The browser runs on the host machine, therefore localhost is correct.
BROWSER_API_URL = os.getenv(
    "PRISM_BROWSER_API_URL",
    "http://localhost:8000",
).rstrip("/")

# ============================================================
# QUESTION BANK
# ============================================================

QUESTIONS = [
    {
        "topic": "Work From Home",
        "question": (
            "What does the company's work-from-home policy say about "
            "employee eligibility, and what conditions must be satisfied "
            "before working remotely?"
        ),
    },
    {
        "topic": "Work From Home + Security",
        "question": (
            "If an employee works from home using a personal laptop, "
            "what security and data-protection requirements should they "
            "follow according to company policy?"
        ),
    },
    {
        "topic": "Leave",
        "question": (
            "What types of leave are available to employees, and what "
            "does the company policy specify about applying for and "
            "managing leave?"
        ),
    },
    {
        "topic": "Performance Reviews",
        "question": (
            "How does the company conduct performance reviews, and what "
            "factors are considered during employee evaluation?"
        ),
    },
    {
        "topic": "Travel and Expenses",
        "question": (
            "If an employee travels for business, what expenses can be "
            "claimed and what requirements apply when submitting those "
            "expenses?"
        ),
    },
    {
        "topic": "Code of Conduct",
        "question": (
            "What does the company's code of conduct say about conflicts "
            "of interest and the professional responsibilities of employees?"
        ),
    },
    {
        "topic": "IT and Data Security",
        "question": (
            "What rules does the company provide for protecting "
            "confidential company information and using company IT resources?"
        ),
    },
    {
        "topic": "Sexual Harassment Policy",
        "question": (
            "What does the company's policy say about reporting and "
            "handling workplace sexual-harassment concerns?"
        ),
    },
    {
        "topic": "Onboarding and Separation",
        "question": (
            "What procedures does the company define for employee "
            "onboarding and separation from the organization?"
        ),
    },
    {
        "topic": "Compensation and Benefits",
        "question": (
            "What benefits and compensation-related provisions are "
            "described in the company's policy documents?"
        ),
    },
]


# ============================================================
# SMALL HELPERS
# ============================================================

def esc(value) -> str:
    return html_lib.escape(str(value))


def h(markup: str) -> str:
    """Collapse HTML to one line so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #06101d;
    --surface: #0d1a2c;
    --surface-2: #122238;
    --border: #22354d;
    --border-strong: #33465e;
    --text: #eef4ff;
    --muted: #8fa3bd;
    --accent: #63e6be;
    --accent-2: #8ab4ff;
    --warn: #f5c451;
    --danger: #ff7b8b;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
textarea:focus, textarea:focus-visible { outline: none !important; box-shadow: none !important; }
[data-baseweb="textarea"] { overflow: hidden; }
[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,230,190,.18) !important;
}
.stApp {
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(99, 230, 190, 0.10), transparent 60%),
        radial-gradient(900px 500px at 95% -5%, rgba(138, 180, 255, 0.10), transparent 60%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden;
    height: 0;
}
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4, h5, p, label, li, span, .stMarkdown { color: var(--text); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* ---------- Header ---------- */
.prism-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 22px;
    background: linear-gradient(135deg, rgba(18,34,56,.95), rgba(13,26,44,.95));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 10px 40px rgba(0,0,0,.35);
}
.brand { display: flex; align-items: center; gap: 14px; }
.logo {
    width: 46px; height: 46px; border-radius: 13px;
    display: grid; place-items: center; font-size: 26px; color: #061019;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 6px 22px rgba(99,230,190,.35);
}
.wordmark { font-size: 24px; font-weight: 800; letter-spacing: .14em; color: var(--text); line-height: 1; }
.tagline { font-size: 12.5px; color: var(--muted); margin-top: 5px; }
.header-right { display: flex; gap: 10px; align-items: center; }
.badge {
    font-size: 12px; font-weight: 600; color: var(--muted);
    padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,.04); border: 1px solid var(--border);
    display: inline-flex; align-items: center; gap: 8px;
}
.badge.live { color: var(--accent); border-color: rgba(99,230,190,.35); background: rgba(99,230,190,.08); }
.pulse {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 0 0 rgba(99,230,190,.7); animation: pulse 1.8s infinite;
}
@keyframes pulse {
    70% { box-shadow: 0 0 0 9px rgba(99,230,190,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,230,190,0); }
}

/* ---------- Progress steps ---------- */
.steps { display: flex; gap: 6px; margin: 16px 2px 6px; }
.step { flex: 1; height: 6px; border-radius: 99px; background: #1a2b42; }
.step.done { background: var(--accent); }
.step.current { background: linear-gradient(90deg, var(--accent), var(--accent-2)); box-shadow: 0 0 12px rgba(99,230,190,.5); }

/* ---------- Section titles ---------- */
.section-title {
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--muted); margin: 22px 0 10px;
}
.section-title::after { content: ""; flex: 1; height: 1px; background: var(--border); }

/* ---------- Question card ---------- */
.q-card {
    padding: 22px 24px;
    background: linear-gradient(160deg, var(--surface-2), var(--surface));
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 16px;
}
.q-meta { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.chip {
    font-size: 11.5px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 999px; color: var(--accent);
    background: rgba(99,230,190,.10); border: 1px solid rgba(99,230,190,.25);
}
.chip.topic { color: var(--accent-2); background: rgba(138,180,255,.10); border-color: rgba(138,180,255,.25); }
.q-text { font-size: 20px; line-height: 1.5; font-weight: 600; color: var(--text); }

.notice {
    margin-top: 12px; padding: 11px 14px; border-radius: 12px; font-size: 13.5px; line-height: 1.5;
    color: #cfe0f7; background: rgba(138,180,255,.08); border: 1px solid rgba(138,180,255,.22);
}

/* ---------- Native containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(13,26,44,.85);
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

/* ---------- Inputs ---------- */
textarea, input[type="text"], [data-baseweb="input"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
}
[data-baseweb="textarea"], [data-baseweb="input"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
}
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,230,190,.15) !important;
}
textarea::placeholder, input::placeholder { color: #6c8099 !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 46px; border-radius: 12px !important; font-weight: 700 !important;
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border-strong) !important; transition: all .15s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #4fd1c5) !important;
    color: #061019 !important; border: none !important;
    box-shadow: 0 8px 24px rgba(99,230,190,.25);
}
.stButton > button[kind="primary"]:hover { color: #061019 !important; filter: brightness(1.08); }

/* ---------- Status tiles ---------- */
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 6px 0 12px; }
.stat {
    padding: 12px 14px; border-radius: 14px; background: var(--surface);
    border: 1px solid var(--border);
}
.stat .k { font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.stat .v { font-size: 17px; font-weight: 700; margin-top: 4px; color: var(--text); }
.stat.ok { border-color: rgba(99,230,190,.35); }
.stat.ok .v { color: var(--accent); }
.stat.warn { border-color: rgba(245,196,81,.4); background: rgba(245,196,81,.06); }
.stat.warn .v { color: var(--warn); }
.stat.bad { border-color: rgba(255,123,139,.4); background: rgba(255,123,139,.06); }
.stat.bad .v { color: var(--danger); }
.stat.idle .v { color: var(--muted); }

/* ---------- Agent card ---------- */
.agent {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 12px 16px; border-radius: 14px; background: var(--surface);
    border: 1px solid var(--border); margin-bottom: 12px;
}
.agent .name { font-size: 11px; font-weight: 700; letter-spacing: .1em; color: var(--muted); text-transform: uppercase; }
.agent .flow { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
.tag { font-size: 12px; font-weight: 700; padding: 5px 11px; border-radius: 999px; }
.tag.ok { color: var(--accent); background: rgba(99,230,190,.12); }
.tag.warn { color: var(--warn); background: rgba(245,196,81,.12); }
.tag.info { color: var(--accent-2); background: rgba(138,180,255,.12); }

/* ---------- Timeline ---------- */
.timeline { display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
.ev {
    display: flex; gap: 10px; align-items: center; padding: 8px 12px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border); font-size: 13px;
}
.ev .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2); flex: none; }
.ev.warning .dot { background: var(--warn); }
.ev .t { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
.ev .n { font-weight: 600; color: var(--text); }
.ev .d { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- Assistant ---------- */
.empty {
    text-align: center; padding: 34px 20px; border: 1px dashed var(--border-strong);
    border-radius: 16px; background: rgba(13,26,44,.5);
}
.empty .ico { font-size: 34px; margin-bottom: 8px; }
.empty .t { font-weight: 700; font-size: 16px; }
.empty .s { color: var(--muted); font-size: 13.5px; margin-top: 6px; line-height: 1.5; }
.answer-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.answer-query { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.answer-query b { color: var(--text); }

[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stExpander"] { border-color: var(--border) !important; border-radius: 12px !important; }
hr { border-color: var(--border) !important; }

/* ---------- Complete screen ---------- */
.done-card {
    text-align: center; padding: 60px 30px; margin-top: 24px; border-radius: 20px;
    background: linear-gradient(160deg, var(--surface-2), var(--surface));
    border: 1px solid rgba(99,230,190,.35);
}
.done-card .big { font-size: 54px; }
.done-card h2 { margin: 10px 0 6px; }
/* ---------- Tiles: 4-up ---------- */
.stats.four { grid-template-columns: repeat(4, 1fr); }
.stats.four .v { font-size: 14px; }

/* ---------- Timeline tweaks ---------- */
.ev { flex-wrap: wrap; }
.ev .d { white-space: normal; }
.ev.warning { border-color: rgba(245,196,81,.35); }

/* ---------- IP card ---------- */
.kv {
    display: grid; grid-template-columns: auto 1fr; gap: 6px 18px;
    padding: 12px 16px; border-radius: 14px; margin-bottom: 12px;
    background: var(--surface); border: 1px solid var(--border); font-size: 13.5px;
}
.kv .k { color: var(--muted); }
.kv .v { color: var(--text); font-weight: 600; word-break: break-all; }
.kv .v.warn { color: var(--warn); }
.kv .v.ok { color: var(--accent); }

/* ---------- Expanders (fixes white header / invisible text) ---------- */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] details { background: var(--surface) !important; border: none !important; }
[data-testid="stExpander"] summary { background: var(--surface-2) !important; color: var(--text) !important; }
[data-testid="stExpander"] summary:hover { background: #1a2e49 !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: var(--text) !important; font-weight: 600; }
[data-testid="stExpander"] summary svg { color: var(--muted) !important; fill: var(--muted) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "session_id": str(uuid.uuid4()),
    "question_index": 0,
    "submitted_answers": [],
    "clear_after": "",       # ignore PRISM answers older than this timestamp
    "ask_notice": None,
    "flash": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

session_id: str = st.session_state.session_id
question_index: int = st.session_state.question_index
total_questions = len(QUESTIONS)
is_complete = question_index >= total_questions

if not is_complete:
    current_question = QUESTIONS[question_index]
    topic = current_question["topic"]
    question = current_question["question"]


# ============================================================
# BACKEND HELPERS
# ============================================================

def send_monitor_event(event_type: str, severity: str = "info", detail: str = "") -> None:
    try:
        requests.post(
            f"{API_URL}/monitoring/event",
            json={
                "session_id": session_id,
                "event_type": event_type,
                "severity": severity,
                "detail": detail,
            },
            timeout=2,
        )
    except requests.RequestException:
        pass


def ask_prism(query: str) -> None:
    query = query.strip()
    if not query:
        st.warning("Type a policy question for PRISM first.")
        return

    send_monitor_event(
        "authorized_rag_used", "info", "Candidate requested authorized RAG assistance"
    )

    try:
        with st.spinner("PRISM is retrieving and grounding an answer…"):
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "session_id": session_id,
                    "text": query,
                    "is_final": True,
                    "sequence_id": int(datetime.now().timestamp()),
                },
                timeout=90,
            )
            response.raise_for_status()
            data = response.json()
    except requests.RequestException as exc:
        st.session_state.ask_notice = f"PRISM API request failed: {exc}"
        return

    if data.get("answer"):
        st.session_state.ask_notice = None
    else:
        st.session_state.ask_notice = (
            f"No passages were retrieved (action: {data.get('action', '—')}). "
            f"{data.get('reason', '')}".strip()
        )
    st.rerun()


def submit_answer(answer: str) -> None:
    answer = answer.strip()
    if not answer:
        st.warning("Please type or speak an answer before submitting.")
        return

    try:
        response = requests.post(
            f"{API_URL}/assessment/submit",
            json={"session_id": session_id, "question": question, "answer": answer},
            timeout=10,
        )
        if not response.ok:
            st.session_state.flash.append(
                "The answer was captured, but the backend could not persist it."
            )
    except requests.RequestException:
        st.session_state.flash.append(
            "The answer was captured locally; assessment persistence is temporarily unavailable."
        )

    st.session_state.submitted_answers.append(
        {
            "number": question_index + 1,
            "topic": topic,
            "question": question,
            "answer": answer,
        }
    )

    send_monitor_event(
        "answer_submitted", "info", f"Candidate submitted answer for question {question_index + 1}"
    )

    st.session_state.question_index += 1
    st.session_state.clear_after = now_iso()   # hides the previous question's PRISM answer
    st.session_state.ask_notice = None
    st.rerun()


def fetch_latest_answer() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/chat/latest/{session_id}", timeout=3)
        if response.ok:
            data = response.json()
            stamp = data.get("updated_at") or ""
            if data.get("answer") and stamp > st.session_state.clear_after:
                return data
    except (requests.RequestException, ValueError):
        pass
    return None


# ============================================================
# HEADER
# ============================================================

def render_header() -> None:
    steps = []
    for i in range(total_questions):
        cls = "done" if i < question_index else ("current" if i == question_index else "")
        steps.append(f'<div class="step {cls}"></div>')

    st.markdown(
        h(
            f"""
<div class="prism-header">
  <div class="brand">
    <div class="logo">◈</div>
    <div>
      <div class="wordmark">PRISM</div>
      <div class="tagline">Voice-enabled assessment · Streaming live RAG · Integrity monitoring</div>
    </div>
  </div>
  <div class="header-right">
    <span class="badge live"><span class="pulse"></span>LIVE</span>
    <span class="badge">Session {esc(session_id[:8])}</span>
  </div>
</div>
<div class="steps">{''.join(steps)}</div>
"""
        ),
        unsafe_allow_html=True,
    )


render_header()

for message in st.session_state.flash:
    st.warning(message)
st.session_state.flash = []


# ============================================================
# COMPLETION SCREEN
# ============================================================

if is_complete:
    st.markdown(
        h(
            f"""
<div class="done-card">
  <div class="big">✅</div>
  <h2>Assessment complete</h2>
  <p style="color:#8fa3bd">You answered {len(st.session_state.submitted_answers)} of {total_questions} questions.
  Your responses have been submitted.</p>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Review submitted answers", expanded=False):
        for item in st.session_state.submitted_answers:
            st.markdown(f"**Q{item['number']} · {item['topic']}**")
            st.caption(item["question"])
            st.write(item["answer"])
            st.divider()

    if st.button("Start a new attempt", type="primary"):
        st.session_state.question_index = 0
        st.session_state.submitted_answers = []
        st.session_state.clear_after = now_iso()
        st.rerun()

    st.stop()


# ============================================================
# IFRAME WIDGETS (plain templates — no f-string brace escaping)
# ============================================================

IFRAME_CSS = r"""
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; background: transparent; color: #eef4ff;
               font-family: Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }
  .card { border: 1px solid #22354d; border-radius: 14px; padding: 12px;
          background: linear-gradient(160deg, #122238, #0d1a2c); }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button { border: 0; border-radius: 10px; padding: 9px 14px; font-weight: 700; font-size: 13px;
           cursor: pointer; transition: all .15s ease; font-family: inherit; }
  .primary { background: linear-gradient(135deg, #63e6be, #4fd1c5); color: #061019; }
  .primary:hover { filter: brightness(1.08); }
  .ghost { background: #17263b; color: #eef4ff; border: 1px solid #33465e; }
  .ghost:hover { border-color: #f5c451; color: #f5c451; }
  .stop { background: #f5c451; color: #061019; }
  .muted { color: #8fa3bd; font-size: 12px; }
  .pill { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: #8fa3bd;
          padding: 5px 10px; border-radius: 999px; background: rgba(255,255,255,.04);
          border: 1px solid #22354d; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #566a85; }
  .dot.ok { background: #63e6be; box-shadow: 0 0 8px #63e6be; }
  .dot.warn { background: #f5c451; }
  .dot.bad { background: #ff7b8b; }
</style>
"""

CAMERA_HTML = IFRAME_CSS + r"""
<div class="card">
  <div class="row" style="align-items:stretch">
    <div style="width:132px;height:99px;border-radius:10px;overflow:hidden;background:#07111f;
         border:1px solid #22354d;display:grid;place-items:center;flex:none">
      <video id="video" autoplay muted playsinline style="width:100%;height:100%;object-fit:cover;display:none"></video>
      <span id="placeholder" class="muted">No preview</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;justify-content:center;flex:1;min-width:180px">
      <div class="row">
        <span class="pill"><span id="camdot" class="dot"></span><span id="status">Camera not started</span></span>
        <span class="pill"><span id="focusdot" class="dot ok"></span><span id="focus">Page focus: active</span></span>
      </div>
      <div class="row">
        <button id="start" class="primary">Enable camera</button>
        <button id="stop" class="ghost">Stop camera</button>
      </div>
    </div>
  </div>
</div>

<script>
const API = __API__;
const SESSION = __SESSION__;
const $ = (id) => document.getElementById(id);
let stream = null;
let cameraBeat = null;

async function sendEvent(type, severity = "info", detail = "") {
  try {
    await fetch(API + "/monitoring/event", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-PRISM-Browser": "1" },
      body: JSON.stringify({ session_id: SESSION, event_type: type, severity: severity, detail: detail })
    });
  } catch (e) {}
}

/* ---------- Session link: always on, independent of the camera ---------- */
sendEvent("session_opened", "info", "Assessment page opened");
setInterval(() => sendEvent("page_heartbeat"), 5000);

/* ---------- Camera ---------- */
function setCam(text, kind) { $("status").textContent = text; $("camdot").className = "dot " + kind; }
function showVideo(on) {
  $("video").style.display = on ? "block" : "none";
  $("placeholder").style.display = on ? "none" : "block";
}

async function stopCamera(reason, severity) {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  clearInterval(cameraBeat);
  $("video").srcObject = null;
  showVideo(false);
  setCam("Camera stopped", "warn");
  await sendEvent("camera_stopped", severity, reason);
}

$("start").onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 }, audio: false });
    $("video").srcObject = stream;
    showVideo(true);
    setCam("Camera active", "ok");
    stream.getVideoTracks().forEach((t) => {
      t.onended = () => { if (stream) stopCamera("Camera track ended unexpectedly", "warning"); };
    });
    await sendEvent("camera_started", "info", "Candidate enabled camera");
    clearInterval(cameraBeat);
    cameraBeat = setInterval(() => sendEvent("camera_heartbeat"), 10000);
  } catch (e) {
    setCam("Camera unavailable", "bad");
    await sendEvent("camera_unavailable", "warning", String(e));
  }
};
$("stop").onclick = () => { if (stream) stopCamera("Candidate stopped camera", "warning"); };

/* ---------- Focus tracking: tab switch AND window/app switch ---------- */
let host;
try { host = window.parent; void host.document; } catch (e) { host = window; }
let lost = false;
let lostSince = 0;

function awayReason() {
  if (host.document.visibilityState !== "visible")
    return "Switched to another browser tab or minimised the window";
  if (!host.document.hasFocus())
    return "Moved to another window or application";
  return "";
}

function checkFocus() {
  const why = awayReason();
  if (why && !lost) {
    lost = true;
    lostSince = Date.now();
    $("focus").textContent = "Page focus: LOST";
    $("focusdot").className = "dot bad";
    sendEvent("page_focus_lost", "warning", why);
  } else if (!why && lost) {
    const secs = Math.round((Date.now() - lostSince) / 1000);
    lost = false;
    $("focus").textContent = "Page focus: active";
    $("focusdot").className = "dot ok";
    sendEvent("page_focus_returned", "info", "Returned after " + secs + "s away");
  }
}

["blur", "focus"].forEach((name) => host.addEventListener(name, () => setTimeout(checkFocus, 250)));
host.document.addEventListener("visibilitychange", checkFocus);
setInterval(checkFocus, 1000);   // safety net
</script>
"""

SPEECH_HTML = IFRAME_CSS + r"""
<!-- question __QIDX__ -->
<style>
  .bars { display: none; gap: 3px; align-items: center; height: 22px; }
  .bars.on { display: inline-flex; }
  .bars i { width: 3px; height: 6px; border-radius: 2px; background: #63e6be; animation: bar 0.9s ease-in-out infinite; }
  .bars i:nth-child(2) { animation-delay: .12s; } .bars i:nth-child(3) { animation-delay: .24s; }
  .bars i:nth-child(4) { animation-delay: .36s; } .bars i:nth-child(5) { animation-delay: .48s; }
  @keyframes bar { 0%,100% { height: 5px; } 50% { height: 22px; } }
  #transcript { margin-top: 10px; padding: 10px 12px; min-height: 42px; max-height: 76px; overflow-y: auto;
                border-radius: 10px; background: #07111f; border: 1px solid #22354d;
                font-size: 13px; line-height: 1.5; color: #dbe7f5; }
  #events { margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; max-height: 40px; overflow-y: auto; }
  .chip { font-size: 11px; padding: 3px 8px; border-radius: 999px; background: rgba(255,255,255,.05);
          border: 1px solid #22354d; color: #8fa3bd; }
  .chip.go { color: #63e6be; border-color: rgba(99,230,190,.35); }
  .chip.err { color: #ff7b8b; border-color: rgba(255,123,139,.35); }
</style>

<div class="card">
  <div class="row">
    <button id="mic" class="primary">🎙 Start speaking</button>
    <span class="bars" id="bars"><i></i><i></i><i></i><i></i><i></i></span>
    <span class="muted" id="status">Speech is typed into your answer box</span>
    <span class="pill" style="margin-left:auto"><span id="apidot" class="dot"></span><span id="apitext">Checking API…</span></span>
  </div>
  <div id="transcript" class="muted">Live transcript will appear here.</div>
  <div id="events"></div>
</div>

<script>
const API = __API__;
const SESSION = __SESSION__;

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const $ = (id) => document.getElementById(id);
const mic = $("mic"), statusEl = $("status"), transcriptEl = $("transcript"), eventsEl = $("events"), bars = $("bars");

let recognition = null, listening = false, shouldListen = false;
let base = "", fullTranscript = "", sentWords = 0, sequence = 0, timer = null, prefix = "";
let inflight = false, pendingFinal = null;
let errChip = null;

/* ---------- API health ---------- */
async function checkApi() {
  try {
    const r = await fetch(API + "/health", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const j = await r.json();
    $("apidot").className = "dot " + (j.pipeline_ready === false ? "warn" : "ok");
    $("apitext").textContent = j.pipeline_ready === false ? "API up · loading models" : "API connected";
    return true;
  } catch (e) {
    $("apidot").className = "dot bad";
    $("apitext").textContent = "API unreachable";
    $("apitext").title = "Cannot reach " + API;
    return false;
  }
}
checkApi();
setInterval(checkApi, 5000);

/* ---------- chips ---------- */
function chip(text, cls) {
  const el = document.createElement("span");
  el.className = "chip " + (cls || "");
  el.textContent = text;
  eventsEl.appendChild(el);
  eventsEl.scrollTop = eventsEl.scrollHeight;
  return el;
}
function showError(text) {
  if (!errChip) errChip = chip("", "err");
  errChip.textContent = text;
}
function clearError() { if (errChip) { errChip.remove(); errChip = null; } }

/* ---------- answer box ---------- */
function answerBox() {
  try { return window.parent.document.querySelector("textarea"); } catch (e) { return null; }
}
function pushToAnswerBox(text) {
  const box = answerBox();
  if (!box) return;
  const value = (prefix ? prefix.trimEnd() + " " : "") + text;
  try {
    const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value").set;
    setter.call(box, value);
    box.dispatchEvent(new window.parent.Event("input", { bubbles: true }));
  } catch (e) {}
}

/* ---------- retrieval ---------- */
function words(t) { return t.trim() ? t.trim().split(/\s+/) : []; }

async function sendRetrieval(isFinal) {
  if (inflight) { pendingFinal = pendingFinal || isFinal; return; }   // one request at a time
  const snapshot = words(fullTranscript).length;
  const delta = words(fullTranscript).slice(sentWords).join(" ");
  if (!delta) return;

  inflight = true;
  sequence += 1;
  const seq = sequence;
  const ctrl = new AbortController();
  const abort = setTimeout(() => ctrl.abort(), 90000);
  try {
    const response = await fetch(API + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION, text: delta, is_final: isFinal, sequence_id: seq }),
      signal: ctrl.signal
    });
    if (!response.ok) {
      let msg = "";
      try { msg = (await response.json()).detail || ""; } catch (e) {}
      throw new Error("API error " + response.status + (msg ? ": " + msg : ""));
    }
    const data = await response.json();
    sentWords = snapshot;                       // only advance on success
    clearError();
    const action = String(data.action || "WAIT").toUpperCase();
    chip("chunk " + seq + " → " + action, action === "RETRIEVE" ? "go" : "");
  } catch (e) {
    const network = e instanceof TypeError || e.name === "AbortError";
    showError(network
      ? "Can't reach " + API + " (server down, wrong URL or blocked)"
      : String(e.message).slice(0, 120));
    checkApi();
  } finally {
    clearTimeout(abort);
    inflight = false;
    if (pendingFinal !== null) {                // catch up on anything said meanwhile
      const f = pendingFinal; pendingFinal = null;
      sendRetrieval(f);
    }
  }
}

function scheduleRetrieval() {
  clearTimeout(timer);
  timer = setTimeout(() => sendRetrieval(false), 900);
}

/* ---------- speech ---------- */
function startSpeech() {
  if (!SR) {
    statusEl.textContent = "Speech recognition not supported (use Chrome or Edge).";
    statusEl.style.color = "#ff7b8b";
    return;
  }
  const box = answerBox();
  prefix = box ? box.value : "";
  base = ""; fullTranscript = ""; sentWords = 0;

  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-IN";
  shouldListen = true;

  recognition.onstart = () => {
    listening = true;
    mic.textContent = "⏹ Stop speaking";
    mic.className = "stop";
    bars.classList.add("on");
    statusEl.textContent = "Listening · live retrieval on";
    statusEl.style.color = "#63e6be";
  };
  recognition.onresult = (event) => {
    let text = "";
    for (let i = 0; i < event.results.length; i++) text += event.results[i][0].transcript + " ";
    fullTranscript = (base + " " + text).replace(/\s+/g, " ").trim();
    transcriptEl.textContent = fullTranscript || "Listening…";
    pushToAnswerBox(fullTranscript);
    scheduleRetrieval();
  };
  recognition.onerror = (event) => {
    if (event.error !== "no-speech" && event.error !== "aborted") chip("voice error: " + event.error, "err");
  };
  recognition.onend = () => {
    listening = false;
    if (shouldListen) {
      base = fullTranscript;
      setTimeout(() => { if (shouldListen && !listening) { try { recognition.start(); } catch (e) {} } }, 150);
    }
  };
  recognition.start();
}

function stopSpeech() {
  shouldListen = false;
  clearTimeout(timer);
  sendRetrieval(true);
  if (recognition) { try { recognition.stop(); } catch (e) {} }
  listening = false;
  mic.textContent = "🎙 Start speaking";
  mic.className = "primary";
  bars.classList.remove("on");
  statusEl.textContent = "Stopped · edit your answer below if needed";
  statusEl.style.color = "#8fa3bd";
}

mic.onclick = () => (listening ? stopSpeech() : startSpeech());
</script>
"""

def build_widget(template: str, **extra: str) -> str:
    rendered = (
        template.replace("__API__", json.dumps(BROWSER_API_URL))
        .replace("__SESSION__", json.dumps(session_id))
    )
    for key, value in extra.items():
        rendered = rendered.replace(key, value)
    return rendered


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1.05, 1.25], gap="large")


# ------------------------------------------------------------
# LEFT — question, voice, answer
# ------------------------------------------------------------

with left:
    st.markdown(
        h(
            f"""
<div class="q-card">
  <div class="q-meta">
    <span class="chip">Question {question_index + 1} of {total_questions}</span>
    <span class="chip topic">{esc(topic)}</span>
  </div>
  <div class="q-text">{esc(question)}</div>
</div>
<div class="notice">🔓 PRISM assistance is <b>authorized</b> for this assessment. You may ask questions about
the indexed company-policy documents while answering.</div>
"""
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Voice input</div>', unsafe_allow_html=True)
    components.html(
        build_widget(SPEECH_HTML, __QIDX__=str(question_index)),
        height=175,
        scrolling=False,
    )

    st.markdown('<div class="section-title">Your answer</div>', unsafe_allow_html=True)
    answer = st.text_area(
        "Candidate answer",
        key=f"answer_box_{question_index}",
        height=230,
        placeholder="Type your answer here, or use the microphone above — dictation appears in this box.",
        label_visibility="collapsed",
    )

    is_last = question_index == total_questions - 1
    if st.button(
        "Submit answer & finish" if is_last else "Submit answer → next question",
        use_container_width=True,
        type="primary",
    ):
        submit_answer(answer)

    st.caption("Once you submit, you can't return to this question.")


# ------------------------------------------------------------
# RIGHT — assistant + integrity
# ------------------------------------------------------------

@st.fragment(run_every="2s")
def assistant_panel() -> None:
    latest = fetch_latest_answer()
    notice = st.session_state.ask_notice

    if notice:
        st.warning(notice)

    if not latest:
        if not notice:
            st.markdown(
                h(
                    """
<div class="empty">
  <div class="ico">◈</div>
  <div class="t">Waiting for a policy question</div>
  <div class="s">Ask below, or start speaking — PRISM retrieves from the policy documents while you talk
  and its grounded answer will appear here.</div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )
        return

    grounded = latest.get("grounded")
    score = latest.get("groundedness")
    citations = latest.get("citations") or []

    if grounded is True:
        badge = '<span class="tag ok">● Grounded</span>'
    elif grounded is False:
        badge = '<span class="tag warn">● Low grounding</span>'
    else:
        badge = '<span class="tag info">● Answer</span>'

    st.markdown(
        h(
            f"""
<div class="answer-head">
  <span class="section-title" style="margin:0;flex:1">Grounded answer</span>&nbsp;{badge}
</div>
<div class="answer-query">Query: <b>{esc(latest.get('query') or '—')}</b></div>
"""
        ),
        unsafe_allow_html=True,
    )

    with st.container(height=380, border=True):
        st.markdown(str(latest.get("answer", "")))

    c1, c2, c3 = st.columns(3)
    c1.metric("Action", str(latest.get("action") or "—"))
    c2.metric(
        "Groundedness",
        f"{score:.2f}" if isinstance(score, (int, float)) else "—",
    )
    c3.metric("Sources", str(len(citations)))

    if citations:
        with st.expander(f"Evidence ({len(citations)})", expanded=False):
            for citation in citations[:8]:
                st.markdown(
                    f"**[{esc(citation.get('citation_id', ''))}]** · `{citation.get('source', '')}`"
                )
                st.caption(citation.get("excerpt", ""))
                st.divider()


def stat_tile(label: str, value: str, kind: str) -> str:
    return (
        f'<div class="stat {kind}"><div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div></div>'
    )


@st.fragment(run_every="3s")
def seconds_since(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()))
    except ValueError:
        return None


@st.fragment(run_every="3s")
def integrity_panel() -> None:
    error = None
    try:
        response = requests.get(f"{API_URL}/monitoring/{session_id}", timeout=6)
        response.raise_for_status()
        monitor = response.json()
        st.session_state.last_monitor = monitor
    except (requests.RequestException, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        monitor = st.session_state.get("last_monitor", {})

    if error:
        st.warning(f"Can't reach the monitoring API at `{API_URL}` — retrying.\n\n`{error[:180]}`")

    camera = monitor.get("camera_status", "NOT_STARTED")
    focus = monitor.get("focus_status", "ACTIVE")
    flags = int(monitor.get("alert_count", 0) or 0)
    age = seconds_since(monitor.get("last_seen"))

    if age is None:
        link_text, link_kind = "NO SIGNAL", "idle"
    elif age <= 15:
        link_text, link_kind = "LIVE", "ok"
    else:
        link_text, link_kind = f"{age}s AGO", "warn"

    st.markdown(
        h(
            '<div class="stats four">'
            + stat_tile("Camera", camera.replace("_", " "), {"ACTIVE": "ok", "INTERRUPTED": "warn"}.get(camera, "idle"))
            + stat_tile("Page focus", focus, "ok" if focus == "ACTIVE" else "bad")
            + stat_tile("Flags", str(flags), "ok" if flags == 0 else "bad")
            + stat_tile("Browser link", link_text, link_kind)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )

    # ---- Network identity (always visible) ----
    initial_ip = monitor.get("initial_ip")
    current_ip = monitor.get("current_ip")
    history = ", ".join(monitor.get("ip_history") or []) or "—"
    if not initial_ip:
        consistency, c_cls = "— waiting for first browser signal", ""
    elif monitor.get("ip_consistent", True):
        consistency, c_cls = "✓ Consistent", "ok"
    else:
        consistency, c_cls = "⚠ Changed during session", "warn"

    st.markdown(
        h(
            f"""
<div class="kv">
  <span class="k">Initial IP</span><span class="v">{esc(initial_ip or "Not observed yet")}</span>
  <span class="k">Current IP</span><span class="v">{esc(current_ip or "Not observed yet")}</span>
  <span class="k">IP history</span><span class="v">{esc(history)}</span>
  <span class="k">Consistency</span><span class="v {c_cls}">{esc(consistency)}</span>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

    # ---- Monitor agent ----
    agent = monitor.get("monitor_agent", {}) or {}
    agent_status = str(agent.get("status", "WAITING_FOR_IP"))
    tag_kind = {"IP_STABLE": "ok", "IP_CHANGED": "warn"}.get(agent_status, "info")
    st.markdown(
        h(
            f"""
<div class="agent">
  <div>
    <div class="name">MonitorAgent · event-driven · LangGraph</div>
    <div class="flow">Observe → decide → act → continue monitoring</div>
  </div>
  <span class="tag {tag_kind}">{esc(agent_status)}</span>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

    # ---- Activity timeline ----
    hidden = {"camera_heartbeat", "page_heartbeat"}
    events = [e for e in monitor.get("events", []) if e.get("event_type") not in hidden]
    st.markdown('<div class="section-title" style="margin-top:14px">Activity log</div>', unsafe_allow_html=True)
    if events:
        rows = []
        for event in reversed(events[-10:]):
            stamp = str(event.get("timestamp", ""))[11:19]
            rows.append(
                f'<div class="ev {esc(event.get("severity", "info"))}">'
                f'<span class="dot"></span><span class="t">{esc(stamp)}</span>'
                f'<span class="n">{esc(event.get("event_type", ""))}</span>'
                f'<span class="d">{esc(event.get("detail", ""))}</span></div>'
            )
        st.markdown(f'<div class="timeline">{"".join(rows)}</div>', unsafe_allow_html=True)
    else:
        st.caption("No activity recorded yet.")
with right:
    st.markdown('<div class="section-title" style="margin-top:0">PRISM assistant</div>', unsafe_allow_html=True)

    with st.form("ask_form", clear_on_submit=True):
        input_col, button_col = st.columns([4, 1.3])
        with input_col:
            ask_text = st.text_input(
                "Ask PRISM",
                placeholder="Ask a company-policy question…  e.g. “How many days of casual leave are allowed?”",
                label_visibility="collapsed",
            )
        with button_col:
            asked = st.form_submit_button("Ask PRISM", use_container_width=True)
    if asked:
        ask_prism(ask_text)

    assistant_panel()

    st.markdown('<div class="section-title">Camera & session integrity</div>', unsafe_allow_html=True)
    components.html(build_widget(CAMERA_HTML), height=140, scrolling=False)
    integrity_panel()


# ============================================================
# SUBMISSION HISTORY
# ============================================================

if st.session_state.submitted_answers:
    st.divider()
    with st.expander(f"Submitted answers ({len(st.session_state.submitted_answers)})"):
        for item in reversed(st.session_state.submitted_answers[-10:]):
            st.markdown(f"**Question {item['number']} · {item['topic']}**")
            st.caption(item["question"])
            st.write(item["answer"])
            st.divider()