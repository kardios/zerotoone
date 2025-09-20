import os
import json
import time
import streamlit as st
import fitz  # PyMuPDF
from typing import List

# ---------------------------
# Model API helpers
# ---------------------------
import anthropic
from openai import OpenAI

def call_model(prompt: str, model: str = "claude-opus-4-1"):
    """Call Claude Opus 4.1 or GPT-5 depending on model selection."""
    if model.startswith("claude"):
        client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content
    elif model.startswith("gpt"):
        client = OpenAI(api_key=_get_secret("OPENAI_API_KEY"))
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"},
        )
        return resp.output_text
    else:
        raise ValueError(f"Unknown model {model}")

# ---------------------------
# Secrets helper
# ---------------------------
def _get_secret(name: str):
    v = os.environ.get(name)
    if not v:
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
    return v

# ---------------------------
# PDF extraction
# ---------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ---------------------------
# Normalization & JSON helpers
# ---------------------------
def normalize_item(item):
    if hasattr(item, "text"):  # TextBlock
        return {"statement": item.text}
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"statement": item}
    return {"statement": str(item)}

def normalize_list(lst: List):
    return [normalize_item(x) for x in lst]

def parse_json_fallback(text):
    if isinstance(text, list):
        return normalize_list(text)
    if isinstance(text, dict):
        return text
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text}
    return {"raw_text": str(text)}

# ---------------------------
# Rendering functions
# ---------------------------
def render_insights(insights, title="Insights"):
    if not insights:
        st.info("No insights available yet.")
        return
    st.subheader(title)
    for insight in normalize_list(insights):
        statement = insight.get("statement", "No statement")
        impact = insight.get("contrarian_impact", "")
        plaus = insight.get("plausibility", "")
        actions = insight.get("action_steps", [])
        with st.expander(statement):
            st.caption(f"Contrarian Impact: {impact}, Plausibility: {plaus}")
            if actions:
                st.markdown("**Action Steps:**")
                for step in actions:
                    st.markdown(f"- {step}")

def render_hypotheses(hypotheses):
    if not hypotheses:
        st.info("No hypotheses generated yet.")
        return
    st.subheader("Hypotheses")
    for hyp in normalize_list(hypotheses):
        statement = hyp.get("statement", "No statement")
        feasibility = hyp.get("feasibility_score", "")
        risks = hyp.get("primary_risk", "None listed")
        linked = hyp.get("linked_insights", [])
        actions = hyp.get("first_3_action_steps", [])
        with st.expander(statement):
            st.caption(f"Feasibility: {feasibility}, Risks: {risks}")
            if linked:
                st.markdown(f"**Linked Insights:** {', '.join(str(x) for x in linked)}")
            if actions:
                st.markdown("**Next Steps:**")
                for step in actions:
                    st.markdown(f"- {step}")

# ---------------------------
# Streamlit app
# ---------------------------
st.title("Zero-to-One Analysis")

model_choice = st.selectbox("Select model", ["claude-opus-4-1", "gpt-5"])
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    st.session_state.setdefault("outputs", {})
    st.session_state.setdefault("timings", {})

    # PDF extraction
    with st.spinner("Extracting text from PDF..."):
        start = time.time()
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state["timings"]["pdf_extract"] = time.time() - start

    # ---------------------------
    # Pass 1: Summary
    # ---------------------------
    with st.spinner("Generating summary..."):
        start = time.time()
        summary_prompt = f"Summarize this document in detail and extract key points:\n{pdf_text}"
        summary_text = call_model(summary_prompt, model=model_choice)
        summary_json = parse_json_fallback(summary_text)
        st.session_state["outputs"]["summary"] = summary_json
        st.session_state["timings"]["summary"] = time.time() - start

    with st.expander("Document Summary"):
        st.text(summary_json.get("raw_text") or json.dumps(summary_json, indent=2, ensure_ascii=False))

    # ---------------------------
    # Pass 2: Counter-intuitive insights
    # ---------------------------
    with st.spinner("Generating counter-intuitive insights..."):
        start = time.time()
        insights_prompt = f"Generate counter-intuitive insights from this summary:\n{summary_json.get('raw_text', '')}"
        insights_text = call_model(insights_prompt, model=model_choice)
        insights_json = parse_json_fallback(insights_text)
        st.session_state["outputs"]["insights"] = insights_json
        st.session_state["timings"]["insights"] = time.time() - start

    render_insights(insights_json, title="Counter-Intuitive Insights")

    # ---------------------------
    # Pass 3: Zero-to-One hypotheses
    # ---------------------------
    with st.spinner("Generating Zero-to-One hypotheses..."):
        start = time.time()
        hypotheses_prompt = f"Generate Zero-to-One hypotheses based on these insights:\n{json.dumps(insights_json)}"
        hyp_text = call_model(hypotheses_prompt, model=model_choice)
        hyp_json = parse_json_fallback(hyp_text)
        st.session_state["outputs"]["hypotheses"] = hyp_json
        st.session_state["timings"]["hypotheses"] = time.time() - start

    render_hypotheses(hyp_json)

    # ---------------------------
    # Download results
    # ---------------------------
    final_json = json.dumps(st.session_state["outputs"], indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Results as JSON",
        data=final_json,
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )

    # ---------------------------
    # Show timings
    # ---------------------------
    st.subheader("Processing Timings (seconds)")
    st.json(st.session_state["timings"])
