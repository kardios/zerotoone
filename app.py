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

# ---------------------------
# Config & secrets helper
# ---------------------------
MODEL_NAME = "claude-opus-4-1-20250805"

def _get_secret(name: str):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None

API_KEY = _get_secret("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Claude API key not found. Set ANTHROPIC_API_KEY in env or Streamlit secrets.")
    st.stop()

client = anthropic.Anthropic(api_key=API_KEY)

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
    text_strip = text.strip()
    try:
        return json.loads(text_strip)
    except Exception:
        pass
    # try to find first JSON block
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            candidate = text[start:end+1]
            return json.loads(candidate)
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        try:
            candidate = text[start:end+1]
            return json.loads(candidate)
        except Exception:
            pass
    return None

def run_claude(prompt: str, max_tokens: int, step_name: str):
    placeholder = st.session_state.get("step_placeholder")
    if placeholder is None:
        placeholder = st.empty()
        st.session_state["step_placeholder"] = placeholder

    placeholder.info(f"{step_name}: running…")
    with st.spinner(f"{step_name} — waiting for model..."):
        progress = st.progress(0)
        start = time.time()
        for i in range(1, 60):
            time.sleep(0.005)
            progress.progress(i)
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            progress.progress(100)
            elapsed = time.time() - start
            placeholder.error(f"{step_name}: API call failed: {e}")
            return f"Error: {e}", elapsed

        progress.progress(100)
        elapsed = time.time() - start

    text = safe_join_textblocks(response.content)
    placeholder.success(f"{step_name}: completed in {elapsed:.1f}s")
    return text, elapsed

def adapt_tokens_for_doc(base_tokens: int):
    length = len(st.session_state.get("doc_text", "").split())
    length_factor = min(2.0, max(0.5, length / 20000))
    scale = st.session_state.get("token_scale", 1.0)
    return int(base_tokens * length_factor * scale)

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
            # Quadrant chart
            if "plausibility" in df.columns and "contrarian_impact" in df.columns:
                st.markdown("### 🔍 Impact vs Plausibility")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df["plausibility"], df["contrarian_impact"])
                for _, row in df.iterrows():
                    label = (row.get("id") or "") or str(row.name)
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

def render_hypotheses(data):
    st.subheader("🧪 Zero-to-One Hypotheses")
    if not data:
        st.warning("No hypotheses available.")
        return
    if isinstance(data, list):
        for hyp in data:
            st.markdown(f"### {hyp.get('id','?')}: {hyp.get('statement','(no statement)')}")
            feas = float(hyp.get("feasibility_score", 0) or 0)
            st.caption(f"Feasibility: {feas}/10")
            risks = hyp.get("primary_risk") or ""
            st.caption(f"Risks: {risks or 'None listed'}")
            linked = hyp.get("linked_insights") or []
            st.caption(f"Linked insights: {', '.join(linked) if linked else 'None'}")
        try:
            df = pd.DataFrame(data)
            cols = [c for c in ["id", "statement", "feasibility_score", "market_potential"] if c in df.columns]
            if cols:
                st.markdown("### Overview")
                st.dataframe(df[cols])
        except Exception:
            pass
    else:
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

# ---------------------------
# UI controls
# ---------------------------
col_control, col_out = st.columns([1, 2])
with col_control:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.session_state["token_scale"] = st.slider("Token scaling", 0.5, 2.0, 1.0, 0.1)
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
    tok = adapt_tokens("summary")
    raw1, t1 = run_claude(prompt1, tok, "Pass 1 — Expanded Summary")
    st.session_state["timings"]["pass1"] = t1
    parsed1 = parse_json_fallback(raw1)
    st.session_state["outputs"]["Expanded Summary"] = parsed1 or {"raw": raw1}
    render_summary(st.session_state["outputs"]["Expanded Summary"])

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
    tok = adapt_tokens("insights")
    raw2, t2 = run_claude(prompt2, tok, "Pass 2 — Candidate Insights")
    st.session_state["timings"]["pass2"] = t2
    parsed2 = parse_json_fallback(raw2)
    if parsed2 and isinstance(parsed2, list):
        st.session_state["outputs"]["Candidate Insights"] = parsed2
    else:
        st.session_state["outputs"]["Candidate Insights"] = []
    render_insights(st.session_state["outputs"].get("Candidate Insights") or raw2)

    # Pass 3: Hypotheses & Refinement
    accepted_insights = st.session_state["outputs"].get("Candidate Insights", [])
    prompt3 = f"""
Generate 3-6 high-leverage hypotheses using these insights.
Output JSON array with:
- id
- statement
- linked_insights (list of ids)
- strategic_rationale
- market_potential
- feasibility_score (1-10)
- primary_risk
- first_3_action_steps (list)
Candidate insights:
{json.dumps(accepted_insights, ensure_ascii=False)}
"""
    tok = adapt_tokens("hypotheses")
    raw3, t3 = run_claude(prompt3, tok, "Pass 3 — Hypotheses & Refinement")
    st.session_state["timings"]["pass3"] = t3
    parsed3 = parse_json_fallback(raw3)
    if parsed3 and isinstance(parsed3, list):
        st.session_state["outputs"]["Hypotheses"] = parsed3
    else:
        st.session_state["outputs"]["Hypotheses"] = []
    render_hypotheses(st.session_state["outputs"].get("Hypotheses") or raw3)

    # Final download
    final_text = json.dumps(st.session_state["outputs"], indent=2, ensure_ascii=False)
    st.session_state["final_text"] = final_text
    st.download_button(
        label="Download Results as JSON",
        data=final_text,
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )
