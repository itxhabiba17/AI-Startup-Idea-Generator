"""
AI Startup Idea Generator
=========================
A Streamlit dashboard that uses the Groq API (LLM) to help entrepreneurs find
market gaps, generate startup ideas, compare them, and pick the best one.

HOW TO RUN (VS Code / terminal)
-------------------------------
1.  pip install -r requirements.txt
2.  Set your Groq API key (see the "API KEY CONFIGURATION" section below).
3.  streamlit run app.py

HOW TO RUN (Google Colab)
-------------------------
    !pip install -r requirements.txt
    import os; os.environ["GROQ_API_KEY"] = "gsk_your_key_here"
    !streamlit run app.py &>/content/logs.txt &
    !npx localtunnel --port 8501      # open the URL it prints
(The password shown by localtunnel is your Colab public IP: !curl ipv4.icanhazip.com)

Get a free API key at https://console.groq.com/keys
"""

import html
import json
import os
import re
import time
from datetime import datetime

import markdown as md_lib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    RateLimitError,
)

# =============================================================================
# 1. PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="AI Startup Idea Generator",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. API KEY CONFIGURATION  <-- ADD YOUR GROQ API KEY HERE (choose ONE option)
# =============================================================================
# Option A (recommended) - environment variable:
#     Windows PowerShell :  $env:GROQ_API_KEY="gsk_xxx"
#     macOS / Linux      :  export GROQ_API_KEY="gsk_xxx"
#
# Option B - Streamlit secrets. Create the file  .streamlit/secrets.toml  next to
#     app.py and put this single line inside it:
#     GROQ_API_KEY = "gsk_xxx"
#
# Option C - paste the key into the sidebar box while the app is running.
#     (It is kept only in memory for your current session.)
#
# NEVER write your real key directly in this file or upload it to GitHub.
# =============================================================================

# Models available on Groq change over time. If you see a "model not found" or
# "decommissioned" error, check https://console.groq.com/docs/models and either
# pick another model in the sidebar or type its ID into "Custom model ID".
MODEL_OPTIONS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", MODEL_OPTIONS[0])

INDUSTRIES = ["Healthcare", "Education", "E-commerce", "Finance", "Agriculture", "Technology"]

# Scoring criteria: (key used in JSON, label shown in the UI)
CRITERIA = [
    ("market_demand", "Market Demand"),
    ("innovation", "Innovation"),
    ("feasibility", "Feasibility"),
    ("scalability", "Scalability"),
    ("revenue_potential", "Revenue Potential"),
]
COST_COL = "Startup Cost (USD)"

PALETTE = ["#4FA89B", "#6C8EBF", "#E9B872", "#9CC5D9", "#A8D5BA", "#C9A7D8", "#E59F9F", "#7D8FA3"]


def get_api_key() -> str:
    """Read the Groq key from the environment first, then from Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:  # no secrets file present
            key = ""
    return key


# =============================================================================
# 3. STYLING
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Public+Sans:wght@400;500;600;700&display=swap');
:root{
  color-scheme: light;
  --bg:#F7F9FB; --surface:#FFFFFF; --ink:#16293A; --text:#1F3447; --muted:#4D6578;
  --line:#DCE5EC; --field:#C6D4DF;
  --sky:#E1EEF7; --mint:#DDF1EC; --peach:#FFF3E0; --gold:#F0B35E;
  --accent:#2E8B7E; --accent-dark:#236F65;
}
.stApp, [data-testid="stAppViewContainer"]{background:var(--bg) !important;color:var(--text);}
[data-testid="stHeader"]{background:transparent !important;}
html, body, .stApp, [data-testid="stMarkdownContainer"]{font-family:'Public Sans',sans-serif;}
h1,h2,h3,h4{font-family:'Sora',sans-serif !important;color:var(--ink) !important;font-weight:700 !important;letter-spacing:-0.01em;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{color:var(--text);font-size:1rem;line-height:1.6;}
strong, b{color:var(--ink);font-weight:700;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{color:var(--muted) !important;}
section[data-testid="stSidebar"]{background:var(--surface) !important;border-right:1px solid var(--line);}
.block-container{padding-top:1.6rem;max-width:1250px;}

/* ---------- form widgets: soft white fields with dark, readable text ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, label{color:var(--ink) !important;font-weight:600 !important;font-size:.95rem !important;}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"], [data-baseweb="select"] > div{
  background:#FFFFFF !important;border:1px solid var(--field) !important;border-radius:8px !important;}
input, textarea{background:#FFFFFF !important;color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important;font-weight:500;}
input::placeholder, textarea::placeholder{color:#6F8499 !important;-webkit-text-fill-color:#6F8499 !important;opacity:1;font-weight:400;}
[data-baseweb="select"] *{color:var(--ink) !important;font-weight:500;}
[data-baseweb="popover"] ul, [data-baseweb="menu"]{background:#FFFFFF !important;}
[data-baseweb="popover"] li{color:var(--ink) !important;}
[data-baseweb="popover"] li:hover{background:var(--sky) !important;}
[data-testid="stNumberInput"] button{background:#EEF3F7 !important;color:var(--ink) !important;}
[data-testid="stNumberInput"] button svg{fill:var(--ink) !important;}
div[data-testid="stForm"]{background:var(--surface);border:1px solid var(--line);border-radius:12px;}

/* sliders */
[data-baseweb="slider"] [role="slider"]{background:var(--accent) !important;border-color:var(--accent) !important;}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *{color:var(--ink) !important;font-weight:600;}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"]{color:var(--muted) !important;}

/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"], button[kind="primary"], button[kind="primaryFormSubmit"]{
  background:var(--accent) !important;border:1px solid var(--accent) !important;border-radius:8px !important;}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover{background:var(--accent-dark) !important;}
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] *, button[kind="primary"] *, button[kind="primaryFormSubmit"] *{
  color:#FFFFFF !important;font-weight:700 !important;font-size:1rem !important;}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"], button[kind="secondary"]{
  background:#FFFFFF !important;border:1px solid var(--field) !important;border-radius:8px !important;}
[data-testid="stBaseButton-secondary"] *, button[kind="secondary"] *{color:var(--ink) !important;font-weight:600 !important;}

/* alerts: soft tints with dark text */
[data-testid="stAlert"], [data-testid="stAlertContainer"]{background:var(--sky) !important;border:1px solid #C9DDEC;border-radius:10px;}
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] *{color:var(--ink) !important;font-weight:500;}
[data-baseweb="notification"][kind="positive"]{background:#E3F5EC !important;}
[data-baseweb="notification"][kind="warning"]{background:#FFF3DD !important;}
[data-baseweb="notification"][kind="negative"]{background:#FDE8E8 !important;}

/* tabs */
button[data-baseweb="tab"] p{color:var(--muted) !important;font-weight:600 !important;font-size:1rem !important;}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink) !important;font-weight:700 !important;}
[data-baseweb="tab-highlight"]{background:var(--accent) !important;}
[data-baseweb="tab-border"]{background:var(--line) !important;}

/* expanders, status, metrics */
[data-testid="stExpander"], [data-testid="stStatus"]{background:var(--surface) !important;border:1px solid var(--line) !important;border-radius:10px !important;}
[data-testid="stExpander"] summary p, [data-testid="stStatus"] summary p, [data-testid="stStatus"] p{color:var(--ink) !important;font-weight:600;}
[data-testid="stExpander"] summary svg, [data-testid="stStatus"] summary svg{color:var(--ink) !important;}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;}
[data-testid="stMetricLabel"] *{color:var(--muted) !important;font-weight:600;}
[data-testid="stMetricValue"] *{color:var(--ink) !important;font-weight:700;}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line) !important;}
[data-testid="stMarkdownContainer"] table th{background:var(--mint);color:var(--ink);}
[data-testid="stMarkdownContainer"] table td, [data-testid="stMarkdownContainer"] table th{border-color:var(--line);color:var(--ink);}

/* ---------- custom components ---------- */
.hero{background:linear-gradient(120deg,#DCEBF5 0%,#DDF1EC 100%);border:1px solid #CADFEA;
      padding:2rem 2.2rem;border-radius:14px;margin-bottom:1.2rem;}
.hero h1{color:var(--ink) !important;margin:0 0 .5rem 0;font-size:2.4rem;}
.hero p{color:#2B4A60 !important;margin:0;max-width:64ch;font-size:1.08rem;font-weight:500;}

.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:1rem;color:var(--text);}
.card h4{margin:0 0 .5rem 0;font-size:1.12rem;color:var(--ink) !important;}
.card .label{font-weight:700;color:var(--accent-dark);font-size:.9rem;margin-top:.8rem;}
.card ul{margin:.2rem 0 0 1.1rem;padding:0;}
.card li, .card div{color:var(--text);}
.chip{display:inline-block;background:var(--mint);color:#1B5A50;border-radius:999px;
      padding:3px 12px;margin:3px 5px 3px 0;font-size:.85rem;font-weight:600;}
.pick{border:2px solid var(--gold);background:var(--peach);}
.badge{background:var(--gold);color:#3A2600;font-weight:700;border-radius:6px;
       padding:2px 10px;font-size:.8rem;margin-right:.5rem;font-family:'Public Sans',sans-serif;}
.sev{float:right;color:var(--muted);font-size:.85rem;font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# 4. SMALL HELPERS (text, numbers, HTML)
# =============================================================================
def txt(v) -> str:
    """Turn any value returned by the LLM into clean text."""
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(as_list(v))
    if isinstance(v, dict):
        return ", ".join(f"{k}: {val}" for k, val in v.items())
    return str(v).strip()


def as_list(v) -> list:
    """Turn any value returned by the LLM into a list of strings."""
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v.strip()]
    if isinstance(v, list):
        return [txt(x) for x in v if txt(x)]
    return [txt(v)]


def to_num(v, default=5.0, lo=None, hi=None) -> float:
    """Safely convert a value (maybe '$12,000' or '8/10') to a number."""
    try:
        if isinstance(v, str):
            match = re.search(r"-?\d[\d,]*\.?\d*", v.replace("$", ""))
            v = float(match.group().replace(",", "")) if match else default
        v = float(v)
    except Exception:
        v = float(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def dict_list(v, keys) -> list:
    """Normalise a list of dicts so every item has every expected key."""
    out = []
    for item in v if isinstance(v, list) else []:
        if isinstance(item, dict):
            out.append({k: item.get(k, "") for k in keys})
        else:
            d = {k: "" for k in keys}
            d[keys[0]] = txt(item)
            out.append(d)
    return out


def esc(v) -> str:
    """Escape text for safe use inside HTML cards."""
    return html.escape(txt(v)).replace("\n", "<br>")


def ul(items) -> str:
    items = as_list(items)
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else ""


def chips(items) -> str:
    return "".join(f'<span class="chip">{esc(i)}</span>' for i in as_list(items))


def money(v) -> str:
    return f"${v:,.0f}"


# =============================================================================
# 5. GROQ / LLM LAYER
# =============================================================================
class AppError(Exception):
    """An error whose message is safe and helpful to show to the user."""


SYSTEM_PROMPT = (
    "You are a senior startup strategist, market analyst and AI product expert. "
    "You give specific, realistic, practical analysis (no vague buzzwords). "
    "Always reply with ONE valid JSON object only: no markdown, no commentary, no code fences."
)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply, even if it added extra text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def call_llm_json(client, model, user_prompt, temperature, max_tokens) -> dict:
    """Call Groq and return parsed JSON. Retries on rate limits and bad JSON."""
    json_mode = True
    for attempt in range(4):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return extract_json(response.choices[0].message.content or "")
        except ValueError:
            continue  # invalid JSON -> try again
        except RateLimitError:
            st.write(f"Rate limit reached - waiting {12 * (attempt + 1)}s before retrying...")
            time.sleep(12 * (attempt + 1))
        except BadRequestError as err:
            msg = str(err).lower()
            if json_mode and ("json" in msg or "response_format" in msg):
                json_mode = False  # this model may not support JSON mode; retry without it
                continue
            raise
    raise AppError(
        "The model did not return a usable answer after several tries. "
        "Wait a minute, or choose a different model in the sidebar, then try again."
    )


def explain_error(err: Exception) -> str:
    """Convert technical exceptions into friendly messages."""
    if isinstance(err, AppError):
        return str(err)
    if isinstance(err, AuthenticationError):
        return "Groq rejected your API key. Check the key in your environment variable, secrets file or the sidebar."
    if isinstance(err, RateLimitError):
        return "You hit Groq's rate limit. Wait a minute and try again, or choose another model in the sidebar."
    if isinstance(err, APIConnectionError):
        return "Could not reach the Groq API. Check your internet connection and try again."
    if isinstance(err, (NotFoundError, BadRequestError)):
        return (
            f"Groq did not accept the request (the model may be retired or unavailable). "
            f"Pick a different model in the sidebar or enter a Custom model ID. Details: {err}"
        )
    if isinstance(err, APIStatusError):
        return f"Groq returned an error (status {err.status_code}). Please try again shortly."
    return f"Something unexpected happened: {err}"


# =============================================================================
# 6. NORMALISERS (make LLM output safe to display)
# =============================================================================
def normalize_market(m: dict) -> dict:
    return {
        "industry_overview": txt(m.get("industry_overview")),
        "existing_problems": dict_list(m.get("existing_problems"), ["problem", "description", "severity"]),
        "market_gaps": dict_list(m.get("market_gaps"), ["gap", "underserved_segment", "opportunity"]),
        "ai_solvable_problems": dict_list(m.get("ai_solvable_problems"), ["problem", "ai_approach"]),
        "market_trends": as_list(m.get("market_trends")),
    }


def normalize_idea(d: dict, i: int) -> dict:
    d = d if isinstance(d, dict) else {}
    return {
        "name": txt(d.get("name")) or f"Startup Idea {i + 1}",
        "business_problem": txt(d.get("business_problem")),
        "ai_solution": txt(d.get("ai_solution")),
        "target_customers": txt(d.get("target_customers")),
        "usp": txt(d.get("usp")),
        "ai_technologies": as_list(d.get("ai_technologies")),
        "revenue_model": txt(d.get("revenue_model")),
        "customer_pain_points": as_list(d.get("customer_pain_points")),
        "customer_needs": txt(d.get("customer_needs")),
        "why_they_would_use": txt(d.get("why_they_would_use")),
        "customer_benefits": as_list(d.get("customer_benefits")),
    }


def normalize_recommendation(r: dict) -> dict:
    roadmap = []
    for item in r.get("roadmap") if isinstance(r.get("roadmap"), list) else []:
        if isinstance(item, dict):
            roadmap.append(
                {
                    "phase": txt(item.get("phase")),
                    "timeline": txt(item.get("timeline")),
                    "goals": as_list(item.get("goals")),
                }
            )
    return {
        "recommended_idea": txt(r.get("recommended_idea")),
        "why_best": txt(r.get("why_best")),
        "business_concept": txt(r.get("business_concept")),
        "target_market": txt(r.get("target_market")),
        "suggested_technologies": as_list(r.get("suggested_technologies")),
        "monetization_strategy": as_list(r.get("monetization_strategy")),
        "roadmap": roadmap,
        "risks": dict_list(r.get("risks"), ["risk", "mitigation"]),
    }


# =============================================================================
# 7. SCORING + COMPARISON TABLE
# =============================================================================
def find_eval(name: str, index: int, evals: list) -> dict:
    """Match an evaluation to an idea by name, falling back to position."""
    for ev in evals:
        if isinstance(ev, dict) and txt(ev.get("name")).lower() == name.lower():
            return ev
    if index < len(evals) and isinstance(evals[index], dict):
        return evals[index]
    return {}


def build_comparison(ideas: list, evals: list, budget: float, weights: dict) -> pd.DataFrame:
    """Create the comparison table, including a weighted overall score (0-10)."""
    total_weight = sum(weights.values()) or 1
    rows = []
    for i, idea in enumerate(ideas):
        ev = find_eval(idea["name"], i, evals)
        row = {"Startup": idea["name"]}
        for key, label in CRITERIA:
            row[label] = round(to_num(ev.get(key), 5, 1, 10))
        cost = to_num(ev.get("estimated_cost_usd"), budget, 0)
        row[COST_COL] = round(cost)

        # Cost fit: 10 when cost <= half the budget, falling to 1 at twice the budget.
        ratio = cost / max(budget, 1)
        cost_fit = 10 if ratio <= 0.5 else max(1.0, 10 - (ratio - 0.5) * 6)
        row["Cost Fit"] = round(cost_fit, 1)

        score = sum(row[label] * weights[key] for key, label in CRITERIA)
        score += cost_fit * weights["cost_fit"]
        row["Overall Score"] = round(score / total_weight, 2)
        row["Rationale"] = txt(ev.get("rationale"))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Overall Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def resolve_pick(rec_name: str, ideas: list, df: pd.DataFrame) -> str:
    """Make sure the AI's recommended name matches a real idea; else use the top score."""
    names = [i["name"] for i in ideas]
    for n in names:
        if n.lower() == rec_name.lower():
            return n
    for n in names:
        if rec_name and (rec_name.lower() in n.lower() or n.lower() in rec_name.lower()):
            return n
    return df.iloc[0]["Startup"]


# =============================================================================
# 8. THE AI PIPELINE (5 LLM calls)
# =============================================================================
def run_pipeline(client, model, profile, temperature, weights, log):
    budget = profile["budget"]
    profile_text = (
        f"Industry: {profile['industry']}\n"
        f"Interests: {profile['interests']}\n"
        f"Skills: {profile['skills']}\n"
        f"Budget (USD): {budget:,.0f}\n"
        f"Target audience: {profile['audience']}\n"
        f"Target region: {profile['region']}"
    )

    # ---- Step 1: market gap analysis ---------------------------------------
    log("Step 1/5 - Analysing market gaps...")
    market_raw = call_llm_json(
        client,
        model,
        f"""Analyse the {profile['industry']} industry for this founder profile:
{profile_text}

Return JSON with exactly these keys:
{{
 "industry_overview": "3-4 sentences on the current state of the industry",
 "existing_problems": [{{"problem": "...", "description": "1-2 sentences", "severity": 1-10 integer}}],
 "market_gaps": [{{"gap": "...", "underserved_segment": "...", "opportunity": "1-2 sentences"}}],
 "ai_solvable_problems": [{{"problem": "...", "ai_approach": "how AI can solve it"}}],
 "market_trends": ["short trend", "..."]
}}
Give 5 existing_problems, 4 market_gaps, 4 ai_solvable_problems and 4 market_trends. Be specific and concise.""",
        temperature,
        max_tokens=3500,
    )
    market = normalize_market(market_raw)

    # ---- Step 2: startup ideas + customer needs ----------------------------
    log("Step 2/5 - Generating startup ideas and customer needs...")
    n = profile["num_ideas"]
    ideas_raw = call_llm_json(
        client,
        model,
        f"""Founder profile:
{profile_text}

Market analysis summary:
{json.dumps({'gaps': market['market_gaps'], 'ai_problems': market['ai_solvable_problems']}, ensure_ascii=False)}

Generate exactly {n} innovative, AI-driven technology startup ideas that fit the founders' skills
and can realistically be started with a budget of about ${budget:,.0f}. Make the ideas clearly different from each other.
Return JSON: {{"ideas": [ {{
 "name": "catchy unique startup name",
 "business_problem": "1-2 sentences",
 "ai_solution": "2 sentences on the proposed AI-based solution",
 "target_customers": "who exactly will buy",
 "usp": "one-sentence unique selling proposition",
 "ai_technologies": ["e.g. LLM", "computer vision", "..."],
 "revenue_model": "how it makes money, with example pricing",
 "customer_pain_points": ["3 short pain points"],
 "customer_needs": "1-2 sentences on what customers need",
 "why_they_would_use": "1-2 sentences on why they would adopt this solution",
 "customer_benefits": ["3 short benefits"]
}} ]}}
Keep every field concise.""",
        temperature,
        max_tokens=6000,
    )
    raw_list = ideas_raw.get("ideas") if isinstance(ideas_raw.get("ideas"), list) else []
    ideas = [normalize_idea(d, i) for i, d in enumerate(raw_list)]
    if len(ideas) < 2:
        raise AppError("The model returned too few startup ideas. Please try again.")

    # ---- Step 3: scoring ----------------------------------------------------
    log("Step 3/5 - Scoring and comparing the opportunities...")
    compact = [
        {"name": i["name"], "problem": i["business_problem"], "solution": i["ai_solution"],
         "customers": i["target_customers"], "revenue_model": i["revenue_model"]}
        for i in ideas
    ]
    eval_raw = call_llm_json(
        client,
        model,
        f"""You are evaluating startup ideas for a founding team with a budget of ${budget:,.0f}.
Ideas:
{json.dumps(compact, ensure_ascii=False)}

Score EVERY idea (same order, exact same names). Use integers 1-10 (10 = best). For feasibility,
10 means very easy to build. Avoid giving all ideas the same scores; differentiate honestly.
estimated_cost_usd = realistic total cost of building an MVP and running the first 12 months (plain integer, no symbols).
Return JSON: {{"evaluations": [ {{"name": "...", "market_demand": 1-10, "innovation": 1-10,
"feasibility": 1-10, "scalability": 1-10, "revenue_potential": 1-10,
"estimated_cost_usd": 12000, "rationale": "one sentence"}} ]}}""",
        0.3,
        max_tokens=3000,
    )
    evals = eval_raw.get("evaluations") if isinstance(eval_raw.get("evaluations"), list) else []
    df = build_comparison(ideas, evals, budget, weights)

    # ---- Step 4: recommendation --------------------------------------------
    log("Step 4/5 - Choosing the most promising startup...")
    ranking = df[["Startup", "Overall Score", COST_COL]].to_dict("records")
    rec_raw = call_llm_json(
        client,
        model,
        f"""Founder profile:
{profile_text}

Startup ideas:
{json.dumps(compact, ensure_ascii=False)}

Weighted scoring results (0-10, higher is better):
{json.dumps(ranking, ensure_ascii=False)}

Select the single most promising startup. It should normally be a top scorer, but you may pick another if you can justify it.
Return JSON:
{{
 "recommended_idea": "exact name from the list",
 "why_best": "4-5 sentences explaining why it is the best opportunity",
 "business_concept": "a detailed 5-6 sentence business concept",
 "target_market": "specific target market description with segments",
 "suggested_technologies": ["technology and tool", "..."],
 "monetization_strategy": ["revenue stream with pricing idea", "..."],
 "roadmap": [{{"phase": "Phase name", "timeline": "e.g. Months 1-2", "goals": ["goal", "goal", "goal"]}}],
 "risks": [{{"risk": "...", "mitigation": "..."}}]
}}
Give 5-7 suggested_technologies, 3-4 monetization streams, 4-5 roadmap phases and 3 risks.""",
        temperature,
        max_tokens=4000,
    )
    rec = normalize_recommendation(rec_raw)
    rec["recommended_idea"] = resolve_pick(rec["recommended_idea"], ideas, df)

    # ---- Step 5: executive summary -----------------------------------------
    log("Step 5/5 - Writing the executive summary...")
    summary_raw = call_llm_json(
        client,
        model,
        f"""Write an executive summary for a business report.
Industry: {profile['industry']}. Audience: {profile['audience']}. Budget: ${budget:,.0f}.
Market overview: {market['industry_overview']}
Ideas considered: {', '.join(i['name'] for i in ideas)}
Recommended idea: {rec['recommended_idea']} - {rec['why_best']}
Return JSON: {{"executive_summary": "one paragraph, 5-6 sentences", "key_takeaways": ["3-4 short bullets"]}}""",
        0.4,
        max_tokens=1500,
    )
    summary = {
        "executive_summary": txt(summary_raw.get("executive_summary")),
        "key_takeaways": as_list(summary_raw.get("key_takeaways")),
    }

    return {
        "profile": profile,
        "model": model,
        "market": market,
        "ideas": ideas,
        "evals": evals,
        "recommendation": rec,
        "summary": summary,
        "generated_at": datetime.now().strftime("%d %B %Y, %H:%M"),
    }


# =============================================================================
# 9. CHARTS
# =============================================================================
def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color="#1F3447"),
        margin=dict(l=10, r=10, t=40, b=10),
        colorway=PALETTE,
    )
    return fig


def chart_problem_severity(problems):
    data = sorted(problems, key=lambda p: to_num(p["severity"], 5))
    fig = go.Figure(
        go.Bar(
            x=[to_num(p["severity"], 5, 1, 10) for p in data],
            y=[txt(p["problem"])[:55] for p in data],
            orientation="h",
            marker_color=PALETTE[0],
        )
    )
    fig.update_layout(title="How severe is each problem? (1-10)", xaxis=dict(range=[0, 10]))
    return style_fig(fig, 320)


def chart_radar(df):
    labels = [label for _, label in CRITERIA]
    fig = go.Figure()
    for i, row in df.iterrows():
        vals = [row[l] for l in labels]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
                opacity=0.35, name=row["Startup"], line=dict(color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.update_layout(
        title="Idea profiles across the five criteria",
        polar=dict(radialaxis=dict(range=[0, 10], showline=False)),
        legend=dict(orientation="h", y=-0.12),
    )
    return style_fig(fig, 460)


def chart_overall(df, pick):
    d = df.sort_values("Overall Score")
    colors = ["#F0B35E" if s == pick else PALETTE[0] for s in d["Startup"]]
    fig = go.Figure(go.Bar(x=d["Overall Score"], y=d["Startup"], orientation="h", marker_color=colors))
    fig.update_layout(title="Weighted overall score (0-10)", xaxis=dict(range=[0, 10]))
    return style_fig(fig, 420)


def chart_cost(df, budget):
    fig = go.Figure(go.Bar(x=df["Startup"], y=df[COST_COL], marker_color=PALETTE[1]))
    fig.add_hline(y=budget, line_dash="dash", line_color="#F0B35E",
                  annotation_text="Your budget", annotation_position="top left")
    fig.update_layout(title="Estimated first-year cost vs. your budget (USD)")
    return style_fig(fig, 340)


# =============================================================================
# 10. HTML CARD BUILDERS
# =============================================================================
def idea_card(idea, picked=False):
    badge = '<span class="badge">Recommended</span>' if picked else ""
    parts = [
        f'<div class="card{" pick" if picked else ""}">',
        f"<h4>{badge}{esc(idea['name'])}</h4>",
        '<div class="label">Business problem</div>', f"<div>{esc(idea['business_problem'])}</div>",
        '<div class="label">Proposed AI-based solution</div>', f"<div>{esc(idea['ai_solution'])}</div>",
        '<div class="label">Target customers</div>', f"<div>{esc(idea['target_customers'])}</div>",
        '<div class="label">Unique selling proposition</div>', f"<div>{esc(idea['usp'])}</div>",
        '<div class="label">Required AI technologies</div>', f"<div>{chips(idea['ai_technologies'])}</div>",
        '<div class="label">Potential revenue model</div>', f"<div>{esc(idea['revenue_model'])}</div>",
        "</div>",
    ]
    return "".join(parts)


# =============================================================================
# 11. REPORT BUILDER
# =============================================================================
def md_table(df: pd.DataFrame, cols) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(money(v) if c == COST_COL else str(v).replace("|", "/"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def bullets(items) -> str:
    return "\n".join(f"- {i}" for i in as_list(items)) or "- n/a"


def build_report_md(res: dict, df: pd.DataFrame) -> str:
    p, m, rec, s = res["profile"], res["market"], res["recommendation"], res["summary"]
    L = []
    L.append(f"# AI Startup Opportunity Report: {p['industry']}")
    L.append(f"*Generated on {res['generated_at']} with Groq model `{res['model']}`*\n")
    L.append("## Founder profile")
    L.append(
        f"- **Industry:** {p['industry']}\n- **Interests:** {p['interests']}\n- **Skills:** {p['skills']}\n"
        f"- **Budget:** {money(p['budget'])}\n- **Target audience:** {p['audience']}\n- **Region:** {p['region']}\n"
    )
    L.append("## Executive summary")
    L.append(s["executive_summary"] + "\n")
    L.append("**Key takeaways**\n")
    L.append(bullets(s["key_takeaways"]) + "\n")

    L.append("## 1. Market analysis")
    L.append(m["industry_overview"] + "\n")
    L.append("### Existing problems")
    for x in m["existing_problems"]:
        L.append(f"- **{x['problem']}** (severity {x['severity']}/10): {x['description']}")
    L.append("\n### Market gaps and underserved needs")
    for x in m["market_gaps"]:
        L.append(f"- **{x['gap']}**. Underserved segment: {x['underserved_segment']}. {x['opportunity']}")
    L.append("\n### Problems AI can solve")
    for x in m["ai_solvable_problems"]:
        L.append(f"- **{x['problem']}**: {x['ai_approach']}")
    L.append("\n### Market trends")
    L.append(bullets(m["market_trends"]) + "\n")

    L.append("## 2. Customer needs analysis")
    for i in res["ideas"]:
        L.append(f"### {i['name']}")
        L.append("**Pain points**\n")
        L.append(bullets(i["customer_pain_points"]))
        L.append(f"\n**What customers need:** {i['customer_needs']}\n")
        L.append(f"**Why they would use it:** {i['why_they_would_use']}\n")
        L.append("**Customer benefits**\n")
        L.append(bullets(i["customer_benefits"]) + "\n")

    L.append("## 3. Startup ideas")
    for n, i in enumerate(res["ideas"], 1):
        L.append(f"### 3.{n} {i['name']}")
        L.append(f"- **Business problem:** {i['business_problem']}")
        L.append(f"- **Proposed AI-based solution:** {i['ai_solution']}")
        L.append(f"- **Target customers:** {i['target_customers']}")
        L.append(f"- **USP:** {i['usp']}")
        L.append(f"- **Required AI technologies:** {', '.join(i['ai_technologies'])}")
        L.append(f"- **Revenue model:** {i['revenue_model']}\n")

    L.append("## 4. Opportunity comparison")
    L.append("Scores run from 1 (weak) to 10 (strong). Feasibility 10 means easiest to build.\n")
    cols = ["Rank", "Startup"] + [l for _, l in CRITERIA[:3]] + [COST_COL] + [l for _, l in CRITERIA[3:]] + ["Overall Score"]
    L.append(md_table(df, cols) + "\n")

    L.append("## 5. Final recommendation")
    L.append(f"### {rec['recommended_idea']}")
    L.append(f"**Why it is the best opportunity:** {rec['why_best']}\n")
    L.append(f"**Business concept:** {rec['business_concept']}\n")
    L.append(f"**Target market:** {rec['target_market']}\n")
    L.append("**Suggested technologies**\n")
    L.append(bullets(rec["suggested_technologies"]) + "\n")
    L.append("**Monetization strategy**\n")
    L.append(bullets(rec["monetization_strategy"]) + "\n")
    L.append("### Initial development roadmap")
    for ph in rec["roadmap"]:
        L.append(f"**{ph['phase']}** ({ph['timeline']})\n")
        L.append(bullets(ph["goals"]) + "\n")
    L.append("### Key risks and mitigation")
    for r in rec["risks"]:
        L.append(f"- **{r['risk']}**: {r['mitigation']}")
    L.append(
        "\n---\n*This report is AI-generated for brainstorming. Validate every assumption with real "
        "customer interviews and market research before investing.*"
    )
    return "\n".join(L)


def report_to_html(md_text: str) -> str:
    body = md_lib.markdown(md_text, extensions=["tables"])
    style = (
        "body{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;"
        "color:#1E3A4F;line-height:1.6}h1,h2,h3{color:#12324A}table{border-collapse:collapse;width:100%;"
        "font-size:.85rem}th,td{border:1px solid #D5DEE5;padding:6px 8px;text-align:left}th{background:#E3F1F2}"
    )
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Startup Report</title><style>{style}</style></head><body>{body}</body></html>"


# =============================================================================
# 12. SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### Settings")
    typed_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="Optional if already set in environment / secrets",
        help="Kept only in memory for this session. Get a key at console.groq.com/keys",
    )
    API_KEY = typed_key.strip() or get_api_key()
    if API_KEY:
        st.success("API key found", icon="🔑")
    else:
        st.warning("No API key yet. Add one above or set GROQ_API_KEY.", icon="⚠️")

    default_index = MODEL_OPTIONS.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODEL_OPTIONS else 0
    chosen_model = st.selectbox("Groq model", MODEL_OPTIONS, index=default_index)
    custom_model = st.text_input("Custom model ID (optional)", placeholder="overrides the choice above")
    MODEL = custom_model.strip() or chosen_model

    temperature = st.slider("Creativity", 0.2, 1.0, 0.7, 0.05, help="Higher = more adventurous ideas")

    with st.expander("Scoring weights"):
        st.caption("How much each factor counts in the overall score. Changes update the comparison instantly.")
        WEIGHTS = {
            "market_demand": st.slider("Market demand", 0, 5, 3),
            "innovation": st.slider("Innovation", 0, 5, 2),
            "feasibility": st.slider("Feasibility", 0, 5, 3),
            "scalability": st.slider("Scalability", 0, 5, 2),
            "revenue_potential": st.slider("Revenue potential", 0, 5, 3),
            "cost_fit": st.slider("Cost fit with budget", 0, 5, 2),
        }
    if sum(WEIGHTS.values()) == 0:
        WEIGHTS = {k: 1 for k in WEIGHTS}

    if st.button("Clear results", width="stretch"):
        st.session_state.pop("result", None)
        st.rerun()

    st.caption(
        "Free-tier Groq accounts have token limits. If you see a rate-limit message, "
        "the app waits and retries automatically."
    )

# =============================================================================
# 13. HEADER + USER INPUT SECTION
# =============================================================================
st.markdown(
    """<div class="hero"><h1>AI Startup Idea Generator</h1>
<p>Tell us about your team. We will map market gaps in your industry, generate startup concepts,
compare them side by side and recommend the one worth building first.</p></div>""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Your profile")
    with st.form("profile_form"):
        left, right = st.columns(2)
        with left:
            industry = st.selectbox("Preferred industry", INDUSTRIES, index=INDUSTRIES.index("Healthcare"))
            custom_industry = st.text_input("Other industry (optional)", placeholder="e.g., Logistics - overrides the choice above")
            audience = st.text_input("Target audience", placeholder="e.g., rural clinics, university students, small retailers")
            region = st.text_input("Target region (optional)", value="Global")
        with right:
            interests = st.text_area("Interests", height=95, placeholder="e.g., preventive health, voice assistants, data storytelling")
            skills = st.text_area("Skills in your team", height=95, placeholder="e.g., Python, machine learning, UX design, sales")
        b1, b2 = st.columns(2)
        with b1:
            budget = st.number_input("Startup budget (USD)", min_value=500, max_value=5_000_000, value=10_000, step=500)
        with b2:
            num_ideas = st.slider("Number of ideas to generate", 5, 8, 5)
        submitted = st.form_submit_button("Generate startup ideas", type="primary", width="stretch")

# =============================================================================
# 14. RUN THE PIPELINE
# =============================================================================
if submitted:
    if not (interests.strip() or skills.strip()):
        st.warning("Add at least your interests or your skills so the ideas can be tailored to you.")
    elif not API_KEY:
        st.error("No Groq API key found. Add it in the sidebar, an environment variable or Streamlit secrets.")
    else:
        profile = {
            "industry": custom_industry.strip() or industry,
            "interests": interests.strip() or "Not specified",
            "skills": skills.strip() or "Not specified",
            "budget": float(budget),
            "audience": audience.strip() or "General consumers and businesses",
            "region": region.strip() or "Global",
            "num_ideas": int(num_ideas),
        }
        error_message = None
        with st.status("Working with Groq. This usually takes 30-90 seconds...", expanded=True) as status:
            try:
                client = Groq(api_key=API_KEY, timeout=120.0, max_retries=1)
                st.session_state["result"] = run_pipeline(client, MODEL, profile, temperature, WEIGHTS, st.write)
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as exc:  # show a friendly message instead of a traceback
                error_message = explain_error(exc)
                status.update(label="Analysis failed", state="error")
        if error_message:
            st.error(error_message)

# =============================================================================
# 15. RESULTS DASHBOARD
# =============================================================================
result = st.session_state.get("result")

if not result:
    st.info("Fill in your profile above and select **Generate startup ideas** to see your dashboard.")
    st.stop()

profile = result["profile"]
ideas = result["ideas"]
market = result["market"]
rec = result["recommendation"]

# Recompute the table with the current weights so sliders update instantly.
df = build_comparison(ideas, result["evals"], profile["budget"], WEIGHTS)
pick_name = resolve_pick(rec["recommended_idea"], ideas, df)
pick_idea = next(i for i in ideas if i["name"] == pick_name)
pick_row = df[df["Startup"] == pick_name].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ideas generated", len(ideas))
m2.metric("Recommended startup", pick_name)
m3.metric("Its overall score", f"{pick_row['Overall Score']:.1f} / 10")
m4.metric("Est. first-year cost", money(pick_row[COST_COL]), delta=f"{money(profile['budget'] - pick_row[COST_COL])} vs budget", delta_color="normal")

tab_market, tab_ideas, tab_customers, tab_compare, tab_pick, tab_report = st.tabs(
    ["Market gaps", "Startup ideas", "Customer needs", "Comparison", "Recommendation", "Report"]
)

# ---- Market gaps -------------------------------------------------------------
with tab_market:
    st.markdown(f"### {profile['industry']}: where the opportunities are")
    st.markdown(f'<div class="card">{esc(market["industry_overview"])}</div>', unsafe_allow_html=True)

    if market["existing_problems"]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(chart_problem_severity(market["existing_problems"]), width="stretch")
        with c2:
            st.markdown("#### Existing problems")
            for p in market["existing_problems"]:
                st.markdown(
                    f'<div class="card"><h4>{esc(p["problem"])}<span class="sev">severity {esc(p["severity"])}/10</span></h4>'
                    f'<div>{esc(p["description"])}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### Market gaps and underserved needs")
    gap_cols = st.columns(2)
    for idx, g in enumerate(market["market_gaps"]):
        gap_cols[idx % 2].markdown(
            f'<div class="card"><h4>{esc(g["gap"])}</h4>'
            f'<div class="label">Underserved segment</div><div>{esc(g["underserved_segment"])}</div>'
            f'<div class="label">Opportunity</div><div>{esc(g["opportunity"])}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Problems that AI can solve")
    if market["ai_solvable_problems"]:
        st.dataframe(
            pd.DataFrame(market["ai_solvable_problems"]).rename(columns={"problem": "Problem", "ai_approach": "How AI helps"}),
            hide_index=True,
            width="stretch",
        )
    if market["market_trends"]:
        st.markdown("#### Market trends")
        st.markdown(f"<div>{chips(market['market_trends'])}</div>", unsafe_allow_html=True)

# ---- Startup ideas -----------------------------------------------------------
with tab_ideas:
    st.markdown(f"### {len(ideas)} startup concepts")
    idea_cols = st.columns(2)
    for idx, idea in enumerate(ideas):
        idea_cols[idx % 2].markdown(idea_card(idea, picked=(idea["name"] == pick_name)), unsafe_allow_html=True)

# ---- Customer needs ----------------------------------------------------------
with tab_customers:
    st.markdown("### What customers struggle with and what they gain")
    for idea in ideas:
        with st.expander(idea["name"], expanded=(idea["name"] == pick_name)):
            a, b = st.columns(2)
            with a:
                st.markdown("**Customer pain points**")
                for x in idea["customer_pain_points"]:
                    st.markdown(f"- {x}")
                st.markdown("**What customers need**")
                st.write(idea["customer_needs"])
            with b:
                st.markdown("**Why they would use it**")
                st.write(idea["why_they_would_use"])
                st.markdown("**Benefits for customers**")
                for x in idea["customer_benefits"]:
                    st.markdown(f"- {x}")

# ---- Comparison --------------------------------------------------------------
with tab_compare:
    st.markdown("### Opportunity comparison")
    st.caption("Scores run from 1 to 10 (higher is better). For feasibility, 10 means easiest to build. "
               "Overall score uses the weights in the sidebar.")
    score_cfg = {l: st.column_config.ProgressColumn(l, min_value=0, max_value=10, format="%d") for _, l in CRITERIA}
    score_cfg[COST_COL] = st.column_config.NumberColumn(COST_COL, format="$%d")
    score_cfg["Cost Fit"] = st.column_config.ProgressColumn("Cost Fit", min_value=0, max_value=10, format="%.1f")
    score_cfg["Overall Score"] = st.column_config.ProgressColumn("Overall Score", min_value=0, max_value=10, format="%.2f")
    table_cols = (
        ["Rank", "Startup"]
        + [l for _, l in CRITERIA[:3]]
        + [COST_COL]
        + [l for _, l in CRITERIA[3:]]
        + ["Cost Fit", "Overall Score"]
    )
    st.dataframe(df[table_cols], hide_index=True, width="stretch", column_config=score_cfg)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_radar(df), width="stretch")
    with c2:
        st.plotly_chart(chart_overall(df, pick_name), width="stretch")
    st.plotly_chart(chart_cost(df, profile["budget"]), width="stretch")

    with st.expander("Why each idea scored the way it did"):
        for _, row in df.iterrows():
            st.markdown(f"**{row['Startup']}**: {row['Rationale'] or 'No rationale provided.'}")

# ---- Recommendation ----------------------------------------------------------
with tab_pick:
    st.markdown("### Best startup recommendation")
    st.markdown(idea_card(pick_idea, picked=True), unsafe_allow_html=True)

    st.markdown("#### Why this is the best opportunity")
    st.write(rec["why_best"])

    a, b = st.columns(2)
    with a:
        st.markdown("#### Business concept")
        st.write(rec["business_concept"])
        st.markdown("#### Target market")
        st.write(rec["target_market"])
    with b:
        st.markdown("#### Suggested technologies")
        st.markdown(f"<div>{chips(rec['suggested_technologies'])}</div>", unsafe_allow_html=True)
        st.markdown("#### Monetization strategy")
        for x in rec["monetization_strategy"]:
            st.markdown(f"- {x}")

    st.markdown("#### Initial development roadmap")
    if rec["roadmap"]:
        road_cols = st.columns(len(rec["roadmap"]))
        for col, ph in zip(road_cols, rec["roadmap"]):
            col.markdown(
                f'<div class="card"><h4>{esc(ph["phase"])}</h4>'
                f'<div class="label">{esc(ph["timeline"])}</div>{ul(ph["goals"])}</div>',
                unsafe_allow_html=True,
            )
    if rec["risks"]:
        st.markdown("#### Key risks and how to handle them")
        st.dataframe(
            pd.DataFrame(rec["risks"]).rename(columns={"risk": "Risk", "mitigation": "Mitigation"}),
            hide_index=True,
            width="stretch",
        )

# ---- Report ------------------------------------------------------------------
with tab_report:
    st.markdown("### AI-generated business report")
    report_md = build_report_md(result, df)
    report_html = report_to_html(report_md)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download as Markdown", report_md, file_name=f"startup_report_{stamp}.md",
                       mime="text/markdown", width="stretch")
    d2.download_button("Download as HTML", report_html, file_name=f"startup_report_{stamp}.html",
                       mime="text/html", width="stretch")
    st.caption("Open the HTML file in a browser and use Print > Save as PDF if you need a PDF.")

    with st.container(border=True):
        st.markdown(report_md)

st.caption("Ideas and scores are AI-generated starting points. Validate them with real customer research before investing.")
