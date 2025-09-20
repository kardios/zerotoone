# app.py
import os
import time
import json
from io import StringIO

import streamlit as st
import fitz  # PyMuPDF
import anthropic
import pandas as pd
import matplotlib.pyplot as plt

# Optional GPT-5
import openai

# ---------------------------
# Config & secrets helper
# ---------------------------
MODEL_CLAUDE = "claude-opus-4-1-20250805"
MAX_DEEPEN_INSIGHTS = 8  # performance cap

def _get_secret(name: str):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None

API_KEY_CLAUDE = _get_secret("ANTHROPIC_API_KEY")
API_KEY_OPENAI = _get_secret("OPENAI_API_KEY")
if not API_KEY_CLAUDE and not API_KEY_OPENAI:
    st.error("No API keys found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")
    st.stop()

client_claude = anthropic.Anthropic(api_key=API_KEY_CLAUDE) if API_KEY_CLAUDE else None

st.set_page_config(page_title="Zero-to-One Analysis", layout="wide")
st.title("Zero-to-One Analysis")

# ---------------------------
# Helpers
# ---------------------------
def safe_join_textblocks(content):
    try:
        if isinstance(content, list):
            parts = []
            for tb in content:
                if hasattr(tb, "text"):
                    parts.append(tb.text)
                else:
                    parts.append(str(tb))
            return "\n\n".join(parts)
        return str(content)
    except Exception:
        return str(content)

def parse_json_fallback(text: str):
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # attempt extracting JSON from inside text
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass
    s, e = text.find("["), text.rfind("]")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            pass
    return None

def adapt_tokens_for_doc(base_tokens: int):
    length = len(st.session_state.get("doc_text", "").split())
    length_factor = min(2.0, max(0.5, length / 20000))
    scale = st.session_state.get("token_scale", 1.0)
    return int(base_tokens * length_factor * scale)

def call_model(prompt: str, max_tokens: int, step_name: str, model_choice: str):
    placeholder = st.session_state.get("step_placeholder") or st.empty()
    placeholder.info(f"{step_name}: running {model_choice}…")

    start = time.time()
    with st.spinner(f"{step_name} — waiting for model..."):
        progress = st.progress(0)
        for i in range(1, 60):
            time.sleep(0.005)
            progress.progress(i)

        if model_choice.startswith("Claude"):
            if not client_claude:
                placeholder.error("Claude API key not available.")
                return "", 0
            response = client_claude.messages.create(
                model=MODEL_CLAUDE,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            text = safe_join_textblocks(response.content)
        elif model_choice.startswith("GPT-5"):
            if not API_KEY_OPENAI:
                placeholder.error("GPT-5 API key not available.")
                return "", 0
            client_gpt5 = openai.OpenAI(api_key=API_KEY_OPENAI)
            result = client_gpt5.responses.create(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"}
            )
            text = result.output_text
        else:
            text = f"Unknown model: {model_choice}"

    elapsed = time.time() - start
    progress.progress(100)
    placeholder.success(f"{step_name} completed in {elapsed:.1f}s using {model_choice}")
    return text, elapsed

# ---------------------------
# Renderers
# ---------------------------
def render_summary(data):
    st.subheader("📖 Expanded Summary")
    if isinstance(data, dict):
        for key, title in [("key_arguments", "📌 Key Arguments"),
                           ("key_assumptions", "💡 Key Assumptions"),
                           ("contradictions", "⚖️ Contradictions / Gaps"),
                           ("strategic_choices", "🚀 Strategic Choices")]:
            items = data.get(key) or []
            st.markdown(f"### {title}")
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.info(f"No {title.lower()}.")
    else:
        st.markdown(data)

    with st.expander("Raw JSON / Raw Text"):
        if isinstance(data, dict):
            st.json(data)
        else:
            st.code(str(data)[:10000])

def render_insights(data):
    st.subheader("💡 Candidate Contrarian Insights")
    if not data:
        st.warning("No insights available.")
        return
    if isinstance(data, list):
        try:
            df = pd.DataFrame(data)
            for col in ["contrarian_impact", "plausibility"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            display_cols = [c for c in ["id", "statement", "conventional_assumption", "contrarian_impact", "plausibility"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols])
            if "plausibility" in df.columns and "contrarian_impact" in df.columns:
                st.markdown("### 🔍 Impact vs Plausibility")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df["plausibility"], df["contrarian_impact"])
                for _, row in df.iterrows():
                    label = str(row.get("id") or row.name)
                    ax.text(row["plausibility"], row["contrarian_impact"], label, fontsize=8)
                ax.set_xlabel("Plausibility")
                ax.set_ylabel("Contrarian Impact")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not render insights: {e}")
            with st.expander("Raw JSON"):
                st.json(data)
    else:
        st.info("Insights not in list form; showing raw output.")
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

def render_deepened_insights(data):
    st.subheader("🔍 Deepened Insights")
    if not data:
        st.warning("No deepened insights available.")
        return
    for insight in data:
        statement = str(insight.get("statement", "(no statement)"))
        expanded = insight.get("expanded_rationale") or "(no expanded rationale)"
        example = "\n".join(insight.get("example_use_cases") or [])
        market = insight.get("potential_market") or "(not specified)"
        risks = insight.get("risks") or "(not specified)"

        with st.expander(statement):
            st.markdown(f"**Expanded Rationale:** {expanded}")
            st.markdown(f"**Example Use Cases:**\n{example}")
            st.markdown(f"**Potential Market:** {market}")
            st.markdown(f"**Risks:** {risks}")

def render_hypotheses(data):
    st.subheader("🧪 Zero-to-One Hypotheses")
    if not data:
        st.warning("No hypotheses available.")
        return

    if isinstance(data, list):
        for hyp in data:
            hyp_id = str(hyp.get("id", "?"))
            statement = str(hyp.get("statement", "(no statement)"))
            st.markdown(f"### {hyp_id}: {statement}")

            feasibility = hyp.get("feasibility_score")
            try:
                feasibility = float(feasibility)
            except (ValueError, TypeError):
                feasibility = 0
            st.caption(f"Feasibility: {feasibility}/10")

            risks = hyp.get("primary_risk") or ""
            st.caption(f"Risks: {risks or 'None listed'}")

            linked = hyp.get("linked_insights") or []
            linked_str = ", ".join(map(str, linked)) if linked else "None"
            st.caption(f"Linked insights: {linked_str}")

            actions = hyp.get("first_3_action_steps") or []
            actions_str = "\n".join([f"- {str(a)}" for a in actions])
            if actions_str:
                st.markdown(f"**First Action Steps:**\n{actions_str}")

        # Overview table
        try:
            df = pd.DataFrame(data)
            cols = [c for c in ["id", "statement", "feasibility_score", "market_potential"] if c in df.columns]
            if cols:
                st.markdown("### Overview Table")
                st.dataframe(df[cols].astype(str))
        except Exception as e:
            st.error(f"Could not render overview table: {e}")
            with st.expander("Raw JSON"):
                st.json(data)
    else:
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

# ---------------------------
# UI Controls
# ---------------------------
col_control, col_out = st.columns([1, 2])
with col_control:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.session_state["token_scale"] = st.slider("Token scaling", 0.5, 2.0, 1.0, 0.1)
    model_choice = st.selectbox("Select model", ["Claude Opus 4.1", "GPT-5"])
    run_btn = st.button("Run Full Analysis")

with col_out:
    st.markdown("### Progress & outputs")
    if "step_placeholder" not in st.session_state:
        st.session_state["step_placeholder"] = st.empty()

# ---------------------------
# Main flow
# ---------------------------
if uploaded and run_btn:
    st.session_state["outputs"] = {}
    st.session_state["timings"] = {}
    st.session_state["final_text"] = ""

    # Step 0: PDF → Text
    st.session_state["step_placeholder"].info("Step 0: Extracting text from PDF...")
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    full_text = "\n\n".join(pages)
    words = full_text.split()
    HARD_WORD_CAP = 120_000
    if len(words) > HARD_WORD_CAP:
        st.warning(f"Document large ({len(words)} words). Truncating to {HARD_WORD_CAP} words.")
        full_text = " ".join(words[:HARD_WORD_CAP])
    st.session_state["doc_text"] = full_text
    st.session_state["step_placeholder"].success(f"Step 0: Text extracted ({len(full_text.split())} words).")

    def adapt_tokens(base_key):
        base_map = {"summary":6000, "insights":5000, "per_insight":1200, "hypotheses":4500, "refinement":6000}
        base = base_map.get(base_key, 3000)
        return adapt_tokens_for_doc(base)

    # ---------------------------
    # Pass 1: Expanded Summary
    prompt1 = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Produce a structured JSON summary with keys:
 - key_arguments: list of short strings
 - key_assumptions: list of short strings
 - contradictions: list of short strings
 - strategic_choices: list of short strings
Document:
{st.session_state['doc_text']}
"""
    raw1, t1 = call_model(prompt1, adapt_tokens("summary"), "Pass 1 — Expanded Summary", model_choice)
    st.session_state["timings"]["pass1"] = t1
    parsed1 = parse_json_fallback(raw1)
    st.session_state["outputs"]["Expanded Summary"] = parsed1 or {"raw": raw1}
    render_summary(st.session_state["outputs"]["Expanded Summary"])

    # ---------------------------
    # Pass 2: Candidate Insights
    prompt2 = f"""
Using the structured summary below, output JSON: an array of 6-12 insight objects with:
- id
- statement
- conventional_assumption
- reasoning
- contrarian_impact (1-10)
- plausibility (1-10)
- actionable_note (optional)
Summary JSON:
{json.dumps(st.session_state['outputs']['Expanded Summary'], ensure_ascii=False)}
"""
    raw2, t2 = call_model(prompt2, adapt_tokens("insights"), "Pass 2 — Candidate Insights", model_choice)
    st.session_state["timings"]["pass2"] = t2
    candidate_insights = parse_json_fallback(raw2)
    candidate_insights = candidate_insights if isinstance(candidate_insights, list) else []
    st.session_state["outputs"]["Candidate Insights"] = candidate_insights
    render_insights(candidate_insights)

    # ---------------------------
    # Pass 2a: Per-Insight Deepening (mandatory)
    deepened_insights = []
    for idx, insight in enumerate(candidate_insights[:MAX_DEEPEN_INSIGHTS], start=1):
        prompt = f"""
Deepen this insight for strategic business analysis:
{insight.get('statement','')}

Output JSON with keys:
- expanded_rationale
- example_use_cases
- potential_market
- risks
"""
        text, tdeep = call_model(prompt, adapt_tokens("per_insight"), f"Deepening Insight {idx}", model_choice)
        insight_update = parse_json_fallback(text) or {}
        insight.update(insight_update)
        deepened_insights.append(insight)
    if len(candidate_insights) > MAX_DEEPEN_INSIGHTS:
        deepened_insights += candidate_insights[MAX_DEEPEN_INSIGHTS:]
        st.info(f"{len(candidate_insights) - MAX_DEEPEN_INSIGHTS} insights not deepened due to performance cap.")
    st.session_state["outputs"]["Deepened Insights"] = deepened_insights
    render_deepened_insights(deepened_insights)

    # ---------------------------
    # Pass 3: Hypotheses
    prompt3 = f"""
From the deepened insights below, generate 3-6 high-leverage Zero-to-One hypotheses.
Output JSON array of objects with keys:
- id
- statement
- feasibility_score (1-10)
- primary_risk
- linked_insights (array of ids)
- first_3_action_steps
Deepened Insights JSON:
{json.dumps(st.session_state['outputs']['Deepened Insights'], ensure_ascii=False)}
"""
    raw3, t3 = call_model(prompt3, adapt_tokens("hypotheses"), "Pass 3 — Hypotheses", model_choice)
    st.session_state["timings"]["pass3"] = t3
    parsed3 = parse_json_fallback(raw3)
    st.session_state["outputs"]["Hypotheses"] = parsed3 if isinstance(parsed3, list) else []
    render_hypotheses(st.session_state["outputs"]["Hypotheses"])

    # ---------------------------
    # Download results
    final_text = json.dumps(st.session_state["outputs"], indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Results as JSON",
        data=final_text,
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )
