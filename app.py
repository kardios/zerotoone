import os
import time
import json
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd

# ----------------------------------------
# Secrets helper
# ----------------------------------------
def _get_secret(name: str):
    v = os.environ.get(name)
    if not v:
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
    return v

# ----------------------------------------
# LLM clients
# ----------------------------------------
# Claude
import anthropic
claude_client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))

# GPT-5
from openai import OpenAI
gpt5_client = OpenAI(api_key=_get_secret("OPENAI_API_KEY"))

# ----------------------------------------
# Utils
# ----------------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_json_fallback(text):
    try:
        # If text is already a list or dict, return as is
        if isinstance(text, (list, dict)):
            return text
        return json.loads(text)
    except Exception:
        return {"raw_text": str(text)}

def call_model(prompt, model="claude-opus-4-1", max_tokens=2000):
    if "claude" in model.lower():
        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content
    elif "gpt" in model.lower():
        result = gpt5_client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        return result.output_text
    else:
        raise ValueError("Unknown model")

# ----------------------------------------
# Deepen Insights
# ----------------------------------------
MAX_DEEPEN_INSIGHTS = 5  # Limit for performance

def deepen_insights(insights_list, model="claude-opus-4-1"):
    deepened = []
    for i, insight in enumerate(insights_list[:MAX_DEEPEN_INSIGHTS], start=1):
        prompt = f"Deepen this insight with actionable, contrarian, zero-to-one thinking:\n\n{insight.get('text', insight.get('raw_text',''))}"
        start_time = time.time()
        deep_text = call_model(prompt, model=model)
        deep_json = parse_json_fallback(deep_text)
        elapsed = time.time() - start_time
        insight_deepened = {
            "original": insight,
            "deepened": deep_json,
            "elapsed_sec": elapsed
        }
        deepened.append(insight_deepened)
    return deepened

# ----------------------------------------
# Render functions
# ----------------------------------------
def render_summary(summary_json):
    with st.expander("Document Summary"):
        if isinstance(summary_json, list):
            for item in summary_json:
                st.text(item.get("raw_text", ""))
        elif isinstance(summary_json, dict):
            st.text(summary_json.get("raw_text", ""))
        else:
            st.text(str(summary_json))

def render_insights(insights_json):
    with st.expander("Counter-Intuitive Insights"):
        for i, insight in enumerate(insights_json, start=1):
            text = insight.get("raw_text") if isinstance(insight, dict) else str(insight)
            st.markdown(f"**Insight {i}:**")
            st.text(text)

def render_deepened_insights(deepened_list):
    st.subheader("Deepened Insights")
    for i, insight in enumerate(deepened_list, start=1):
        with st.expander(f"Insight {i} (took {insight['elapsed_sec']:.1f}s)"):
            st.markdown("**Original:**")
            st.text(insight["original"].get("raw_text", ""))
            st.markdown("**Deepened:**")
            st.text(insight["deepened"].get("raw_text", ""))

def render_hypotheses(hypotheses_json):
    st.subheader("Zero-to-One Hypotheses")
    for hyp in hypotheses_json:
        statement = hyp.get("statement", "No statement") if isinstance(hyp, dict) else str(hyp)
        feasibility = hyp.get("feasibility_score", "") if isinstance(hyp, dict) else ""
        risks = hyp.get("primary_risk", "None listed") if isinstance(hyp, dict) else ""
        linked = hyp.get("linked_insights", []) if isinstance(hyp, dict) else []
        st.markdown(f"**Statement:** {statement}")
        st.markdown(f"**Feasibility:** {feasibility}")
        st.markdown(f"**Risks:** {risks}")
        st.markdown(f"**Linked insights:** {', '.join(linked) if linked else 'None'}")
        st.markdown("---")

# ----------------------------------------
# Streamlit App
# ----------------------------------------
st.title("Zero-to-One Analysis")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
model_choice = st.selectbox("Select Model", ["Claude Opus 4.1", "GPT-5"])

if uploaded_file:
    # Determine model string
    model = "claude-opus-4-1" if model_choice.startswith("Claude") else "gpt-5"

    # Initialize session state
    st.session_state.setdefault("outputs", {})
    st.session_state.setdefault("timings", {})

    # ---------------------------
    # Step 1: Extract text
    # ---------------------------
    start = time.time()
    text = extract_text_from_pdf(uploaded_file)
    st.session_state["timings"]["extract"] = time.time() - start

    # ---------------------------
    # Step 2: Summary
    # ---------------------------
    start = time.time()
    summary_prompt = f"Summarize this document in detail and extract insights:\n\n{text}"
    summary_text = call_model(summary_prompt, model=model)
    summary_json = parse_json_fallback(summary_text)
    st.session_state["outputs"]["summary"] = summary_json
    st.session_state["timings"]["summary"] = time.time() - start

    render_summary(summary_json)

    # ---------------------------
    # Step 3: Counter-intuitive Insights
    # ---------------------------
    start = time.time()
    insights_prompt = f"Extract counter-intuitive insights in JSON array form from the document summary:\n\n{summary_text}"
    insights_text = call_model(insights_prompt, model=model)
    insights_json = parse_json_fallback(insights_text)
    st.session_state["outputs"]["insights"] = insights_json
    st.session_state["timings"]["insights"] = time.time() - start

    render_insights(insights_json)

    # ---------------------------
    # Step 4: Deepen Insights
    # ---------------------------
    start = time.time()
    deepened_insights = deepen_insights(insights_json, model=model)
    st.session_state["outputs"]["deepened_insights"] = deepened_insights
    st.session_state["timings"]["deepen"] = time.time() - start

    render_deepened_insights(deepened_insights)

    # ---------------------------
    # Step 5: Hypotheses
    # ---------------------------
    start = time.time()
    hyp_prompt = f"Generate Zero-to-One hypotheses based on these deepened insights:\n\n{json.dumps(deepened_insights, indent=2)}"
    hyp_text = call_model(hyp_prompt, model=model)
    hyp_json = parse_json_fallback(hyp_text)
    st.session_state["outputs"]["hypotheses"] = hyp_json
    st.session_state["timings"]["hypotheses"] = time.time() - start

    render_hypotheses(hyp_json)

    # ---------------------------
    # Step 6: Download
    # ---------------------------
    st.download_button(
        label="Download Full JSON Results",
        data=json.dumps(st.session_state["outputs"], indent=2, ensure_ascii=False),
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )
