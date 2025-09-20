import os
import json
import time
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from openai import OpenAI
import anthropic

# ---------------------------
# Helper functions
# ---------------------------

def _get_secret(name: str):
    """Safely retrieve a secret from environment or st.secrets"""
    v = os.environ.get(name)
    if not v:
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
    return v

def extract_text_from_pdf(file_path):
    """Extract text from a PDF using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_json_fallback(text):
    """Try to parse JSON from text, return None if fails"""
    try:
        return json.loads(text)
    except Exception:
        return None

def call_model(prompt, max_tokens, desc, model_choice):
    """Call either Claude or GPT-5 depending on selection"""
    start = time.time()
    if model_choice == "Claude":
        client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        output = resp.content
    else:  # GPT-5
        client = OpenAI(api_key=_get_secret("OPENAI_API_KEY"))
        resp = client.responses.create(
            model="gpt-5",
            input=prompt,
            reasoning={"effort":"medium"},
            text={"verbosity":"medium"},
        )
        output = resp.output_text
    elapsed = time.time() - start
    return output, elapsed

def adapt_tokens(stage):
    """Return max tokens per stage"""
    mapping = {
        "summary": 2500,
        "insights": 3500,
        "per_insight": 1000,
        "hypotheses": 2500
    }
    return mapping.get(stage, 1500)

# ---------------------------
# Renderers
# ---------------------------

def render_summary(summary):
    st.markdown("## Document Summary")
    st.markdown(summary)

def render_insights(insights):
    st.markdown("## Candidate Insights")
    for i, ins in enumerate(insights, 1):
        st.markdown(f"**Insight {i}:** {ins.get('statement','')}")
        rationale = ins.get("rationale") or ""
        if rationale:
            st.caption(f"Rationale: {rationale}")

def render_deepened_insights(deepened):
    st.markdown("## Deepened Insights")
    for i, ins in enumerate(deepened, 1):
        st.markdown(f"**Insight {i}:** {ins.get('statement','')}")
        st.markdown(f"Expanded Rationale: {ins.get('expanded_rationale','')}")
        st.markdown(f"Example Use Cases: {ins.get('example_use_cases','')}")
        st.markdown(f"Potential Market: {ins.get('potential_market','')}")
        st.markdown(f"Risks: {ins.get('risks','')}")

def render_hypotheses(hypotheses):
    st.markdown("## Hypotheses")
    for hyp in hypotheses:
        st.markdown(f"**{hyp.get('id','')} - {hyp.get('statement','')}**")
        st.markdown(f"Feasibility: {hyp.get('feasibility_score','')}")
        st.caption(f"Risks: {hyp.get('primary_risk','None')}")
        linked = hyp.get("linked_insights") or []
        st.caption(f"Linked insights: {', '.join(map(str, linked)) if linked else 'None'}")
        actions = hyp.get("first_3_action_steps") or []
        st.markdown(f"Next Steps: {', '.join(actions)}")

# ---------------------------
# Main App
# ---------------------------

st.title("Zero-to-One Analysis")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
model_choice = st.selectbox("Select Model", ["Claude", "GPT-5"])

if uploaded_file:
    # Initialize session state
    st.session_state.setdefault("outputs", {})
    st.session_state.setdefault("timings", {})

    text = extract_text_from_pdf(uploaded_file)

    # ---------------------------
    # Pass 1: Summary
    with st.spinner("Generating Summary…"):
        prompt1 = f"Summarize the document and highlight contrarian insights:\n{text}"
        summary, t1 = call_model(prompt1, adapt_tokens("summary"), "Pass 1 — Summary", model_choice)
        st.session_state["timings"]["pass1"] = t1
        st.session_state["outputs"]["Summary"] = summary
    render_summary(summary)

    # ---------------------------
    # Pass 2: Candidate Insights
    with st.spinner("Generating Candidate Insights…"):
        prompt2 = f"Extract 5-10 counter-intuitive insights from the summary:\n{summary}"
        raw2, t2 = call_model(prompt2, adapt_tokens("insights"), "Pass 2 — Insights", model_choice)
        st.session_state["timings"]["pass2"] = t2
        candidate_insights = parse_json_fallback(raw2) or []
        st.session_state["outputs"]["Insights"] = candidate_insights
    render_insights(candidate_insights)

    # ---------------------------
    # Pass 2a: Per-Insight Deepening
    MAX_DEEPEN_INSIGHTS = 8
    deepened_insights = []
    st.session_state["timings"]["deepening"] = []
    deepen_placeholder = st.sidebar.empty()

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
        with st.spinner(f"Deepening Insight {idx} of {len(candidate_insights[:MAX_DEEPEN_INSIGHTS])}…"):
            start_time = time.time()
            text_deep, _ = call_model(prompt, adapt_tokens("per_insight"), f"Deepening Insight {idx}", model_choice)
            elapsed = time.time() - start_time
            st.session_state["timings"]["deepening"].append(elapsed)

        # Update sidebar progress
        total_elapsed_deep = sum(st.session_state["timings"]["deepening"])
        deepen_placeholder.markdown(
            f"**Deepening Insights:** {idx}/{len(candidate_insights[:MAX_DEEPEN_INSIGHTS])} "
            f"⏱ Total elapsed: {total_elapsed_deep:.1f}s"
        )

        # Safe JSON update
        insight_update = parse_json_fallback(text_deep)
        if isinstance(insight_update, dict):
            insight.update(insight_update)
        else:
            st.warning(f"Insight {idx} deepening returned non-dict type ({type(insight_update)}). Storing raw output instead.")
            insight["expanded_rationale"] = text_deep

        deepened_insights.append(insight)

        # Raw output expander
        with st.expander(f"Insight {idx} Deepening (took {elapsed:.1f}s)"):
            st.code(text_deep[:5000])
            st.markdown(f"**Elapsed time:** {elapsed:.1f} seconds")

    # Handle remaining insights
    if len(candidate_insights) > MAX_DEEPEN_INSIGHTS:
        deepened_insights += candidate_insights[MAX_DEEPEN_INSIGHTS:]
        st.info(f"{len(candidate_insights) - MAX_DEEPEN_INSIGHTS} insights not deepened due to performance cap.")

    st.session_state["outputs"]["Deepened Insights"] = deepened_insights
    render_deepened_insights(deepened_insights)

    # ---------------------------
    # Pass 3: Hypotheses
    prompt3 = f"""
From the deepened insights below, generate 3-6 high-leverage Zero-to-One hypotheses.
Output JSON array with keys:
- id
- statement
- feasibility_score (1-10)
- primary_risk
- linked_insights (array of ids)
- first_3_action_steps

Deepened Insights JSON:
{json.dumps(deepened_insights, ensure_ascii=False)}
"""
    with st.spinner("Generating Hypotheses…"):
        start_time = time.time()
        raw3, _ = call_model(prompt3, adapt_tokens("hypotheses"), "Pass 3 — Hypotheses", model_choice)
        elapsed3 = time.time() - start_time
        st.session_state["timings"]["pass3"] = elapsed3

    # Safe parsing
    parsed3 = parse_json_fallback(raw3)
    if isinstance(parsed3, list):
        st.session_state["outputs"]["Hypotheses"] = parsed3
    else:
        st.warning("Hypotheses generation did not return a valid JSON array. Storing raw output instead.")
        st.session_state["outputs"]["Hypotheses"] = [{"statement": raw3}]
    render_hypotheses(st.session_state["outputs"]["Hypotheses"])

    # Raw output expander
    with st.expander(f"Raw Hypotheses Output (took {elapsed3:.1f}s)"):
        st.code(raw3[:5000])
        st.markdown(f"**Elapsed time:** {elapsed3:.1f} seconds")

    # ---------------------------
    # Step Counter Sidebar
    with st.sidebar:
        st.title("Pipeline Progress")
        steps = [
            ("Pass 1: Summary", "pass1", 1),
            ("Pass 2: Candidate Insights", "pass2", 1),
            ("Pass 2a: Deepened Insights", "deepening", len(candidate_insights[:MAX_DEEPEN_INSIGHTS])),
            ("Pass 3: Hypotheses", "pass3", 1),
        ]
        total_elapsed = 0.0
        for idx, (label, key, count) in enumerate(steps, start=1):
            elapsed = st.session_state["timings"].get(key)
            if elapsed is None:
                st.markdown(f"**{idx}. {label}** — pending ⏳")
            elif isinstance(elapsed, list):
                step_time = sum(elapsed)
                total_elapsed += step_time
                st.markdown(f"**{idx}. {label}** — {len(elapsed)}/{count} items, {step_time:.1f}s")
            else:
                total_elapsed += elapsed
                st.markdown(f"**{idx}. {label}** — {elapsed:.1f}s")
        st.markdown("---")
        st.markdown(f"**Total elapsed:** {total_elapsed:.1f}s")

    # ---------------------------
    # Download Results
    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(st.session_state["outputs"], ensure_ascii=False, indent=2),
        file_name="zero_to_one_analysis.json",
        mime="application/json"
    )
