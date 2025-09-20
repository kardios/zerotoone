import streamlit as st
import fitz  # PyMuPDF
import json
import time
import os

# Optional: APIs
import anthropic
from openai import OpenAI

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
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF UploadedFile"""
    text = ""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # page.get_text("blocks") returns list of tuples, block[4] is text
            blocks = page.get_text("blocks")
            text += "\n".join(str(block[4]) for block in blocks) + "\n"
    return text

# ---------------------------
# JSON parsing helper
# ---------------------------
def parse_json_fallback(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}

# ---------------------------
# LLM call abstraction
# ---------------------------
def call_model(prompt, model="claude-opus-4-1", max_tokens=2048):
    if model.startswith("gpt"):
        client = OpenAI(api_key=_get_secret("OPENAI_API_KEY"))
        result = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"}
        )
        return result.output_text
    else:
        client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.content

# ---------------------------
# Renderers
# ---------------------------
def render_insights(insights, title="Insights"):
    if not insights:
        st.info("No insights available yet.")
        return
    st.subheader(title)
    for insight in insights:
        statement = insight.get("statement", "No statement")
        impact = insight.get("contrarian_impact", "")
        plaus = insight.get("plausibility", "")
        actions = insight.get("action_steps", [])
        with st.expander(f"{statement}"):
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
    for hyp in hypotheses:
        statement = hyp.get("statement", "No statement")
        feasibility = hyp.get("feasibility_score", "")
        risks = hyp.get("primary_risk", "None listed")
        linked = hyp.get("linked_insights", [])
        actions = hyp.get("first_3_action_steps", [])
        with st.expander(f"{statement}"):
            st.caption(f"Feasibility: {feasibility}, Risks: {risks}")
            if linked:
                st.markdown(f"**Linked Insights:** {', '.join(linked)}")
            if actions:
                st.markdown("**Next Steps:**")
                for step in actions:
                    st.markdown(f"- {step}")

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Zero-to-One Analysis")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
model_choice = st.selectbox("Model", ["claude-opus-4-1", "gpt-5"])

if uploaded_file:
    st.session_state.setdefault("outputs", {})
    st.session_state.setdefault("timings", {})

    # Extract PDF text
    start = time.time()
    text = extract_text_from_pdf(uploaded_file)
    st.session_state["timings"]["pdf_extraction"] = time.time() - start

    # ---------------------------
    # Pass 1: Summary
    # ---------------------------
    start = time.time()
    summary_prompt = f"Summarize this document in detail and extract initial counter-intuitive insights:\n\n{text}"
    summary_text = call_model(summary_prompt, model=model_choice)
    summary_json = parse_json_fallback(summary_text)
    st.session_state["outputs"]["summary"] = summary_json
    st.session_state["timings"]["summary"] = time.time() - start

    render_insights(summary_json.get("insights") if isinstance(summary_json, dict) else [])

    # ---------------------------
    # Pass 2: Deepened insights
    # ---------------------------
    start = time.time()
    insights_for_deepening = summary_json.get("insights", []) if isinstance(summary_json, dict) else []
    deepening_prompt = f"Deepen and refine these insights for Zero-to-One hypotheses:\n{json.dumps(insights_for_deepening)}"
    deep_text = call_model(deepening_prompt, model=model_choice)
    deep_json = parse_json_fallback(deep_text)
    st.session_state["outputs"]["deepened_insights"] = deep_json.get("insights") if isinstance(deep_json, dict) else []
    st.session_state["timings"]["deepening"] = time.time() - start

    render_insights(st.session_state["outputs"]["deepened_insights"], title="Deepened Insights")

    # ---------------------------
    # Pass 3: Hypotheses generation
    # ---------------------------
    start = time.time()
    deepened_insights = st.session_state["outputs"]["deepened_insights"]
    hyp_prompt = f"Generate Zero-to-One hypotheses from these insights:\n{json.dumps(deepened_insights)}"
    hyp_text = call_model(hyp_prompt, model=model_choice)
    hyp_json = parse_json_fallback(hyp_text)
    st.session_state["outputs"]["hypotheses"] = hyp_json if isinstance(hyp_json, list) else [hyp_json]
    st.session_state["timings"]["hypotheses"] = time.time() - start

    render_hypotheses(st.session_state["outputs"]["hypotheses"])

    # ---------------------------
    # Download button
    # ---------------------------
    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(st.session_state["outputs"], indent=2, ensure_ascii=False),
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )

    # ---------------------------
    # Show timings
    # ---------------------------
    st.subheader("Timings (seconds)")
    for key, t in st.session_state["timings"].items():
        st.markdown(f"- **{key}**: {t:.2f}s")
