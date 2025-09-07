import os
import time
import streamlit as st
import fitz  # PyMuPDF
import anthropic

# ------------------------------------------------------------------
# Secrets helper
# ------------------------------------------------------------------
def _get_secret(name: str):
    v = os.environ.get(name)
    if not v:
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
    return v

# Initialize Claude client
api_key = _get_secret("ANTHROPIC_API_KEY")
if not api_key:
    st.error("Claude API key not found. Please set ANTHROPIC_API_KEY in env or Streamlit secrets.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# --- Streamlit UI ---
st.title("Zero-to-One Insight Pipeline (Claude Opus 4.1) - PDF Only")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
MAX_WORDS = 80000

# Initialize session_state to store outputs
if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}
if "final_text" not in st.session_state:
    st.session_state["final_text"] = ""

def run_claude_pass(prompt, step_name, step_num):
    """Run a Claude API call with spinner, progress bar, and timing."""
    step_placeholder = st.session_state.get("step_placeholder", None)
    if step_placeholder is None:
        step_placeholder = st.empty()
        st.session_state["step_placeholder"] = step_placeholder

    step_placeholder.info(f"Step {step_num}: {step_name}...")

    with st.spinner(f"{step_name} in progress..."):
        progress_bar = st.progress(0)
        start_time = time.time()
        # Simulate progress until API returns
        for i in range(1, 80):
            time.sleep(0.01)
            progress_bar.progress(i)
        response = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        progress_bar.progress(100)
        elapsed = time.time() - start_time

    # Handle TextBlock output
    if isinstance(response.content, list):
        output_text = "\n\n".join([tb.text for tb in response.content])
    else:
        output_text = str(response.content)

    step_placeholder.success(f"Step {step_num}: {step_name} complete in {elapsed:.1f}s")
    return output_text, elapsed

if uploaded_file and st.button("Analyze PDF"):

    # Step counter
    step = 0

    # Step 1: Extract text
    step += 1
    step_placeholder = st.empty()
    step_placeholder.info(f"Step {step}: Extracting text from PDF...")
    time.sleep(0.2)

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "".join([page.get_text("text") + "\n" for page in doc])

    words = full_text.split()
    if len(words) > MAX_WORDS:
        st.warning(f"Document is very large. Truncating to first {MAX_WORDS} words.")
        words = words[:MAX_WORDS]
    full_text = " ".join(words)
    step_placeholder.success(f"Step {step}: Text extraction complete. ({len(words)} words)")

    # --- Pass 1: Expanded Summary ---
    step += 1
    prompt_summary = f"""
Summarize the main points of the following document in 2–3 paragraphs.
Include key arguments, assumptions, and notable examples.
This summary will be used for subsequent analysis.

Document text:
{full_text}
"""
    summary_text, pass1_time = run_claude_pass(prompt_summary, "Expanded Summary", step)
    st.session_state["outputs"]["summary"] = summary_text

    with st.expander("Pass 1: Expanded Summary"):
        st.markdown(summary_text)

    # --- Pass 2: Counter-Intuitive Insights ---
    step += 1
    prompt_insights = f"""
Based on the following summary, extract 5–10 counter-intuitive insights:
- Each insight should be true but contrary to common assumptions.
- Explain why each insight is counter-intuitive and why it might be true.

Summary:
{summary_text}
"""
    insights_text, pass2_time = run_claude_pass(prompt_insights, "Counter-Intuitive Insights", step)
    st.session_state["outputs"]["insights"] = insights_text

    with st.expander("Pass 2: Counter-Intuitive Insights"):
        st.markdown(insights_text)

    # --- Pass 3: Zero-to-One Hypotheses ---
    step += 1
    prompt_hypotheses = f"""
Based on the summary and counter-intuitive insights, generate 3–5 high-leverage Zero-to-One hypotheses:
- Explain market potential and feasibility.
- Include actionable concepts where possible.

Summary:
{summary_text}

Insights:
{insights_text}
"""
    hypotheses_text, pass3_time = run_claude_pass(prompt_hypotheses, "Zero-to-One Hypotheses", step)
    st.session_state["outputs"]["hypotheses"] = hypotheses_text

    with st.expander("Pass 3: Zero-to-One Hypotheses"):
        st.markdown(hypotheses_text)

    # --- Pass 4: Refinement and Ranking ---
    step += 1
    prompt_refine = f"""
Refine and rank all insights and hypotheses for clarity, impact, and plausibility.
Maintain the summary at the top, and produce a clean, structured output.

Summary:
{summary_text}

Insights:
{insights_text}

Hypotheses:
{hypotheses_text}
"""
    refined_text, pass4_time = run_claude_pass(prompt_refine, "Refinement and Ranking", step)
    st.session_state["outputs"]["refined"] = refined_text

    with st.expander("Pass 4: Refined Insights & Hypotheses"):
        st.markdown(refined_text)

    # --- Pass 5: Final Synthesis ---
    step += 1
    st.session_state["final_text"] = refined_text
    with st.expander("Pass 5: Final Synthesized Report"):
        st.markdown(st.session_state["final_text"])

    # Download button reads from session_state to avoid rerun
    st.download_button(
        label="Download Full Report as TXT",
        data=st.session_state["final_text"],
        file_name="zero_to_one_full_report.txt",
        mime="text/plain"
    )

    total_time = pass1_time + pass2_time + pass3_time + pass4_time
    st.info(f"Total processing time (excluding final download prep): {total_time:.1f}s")
