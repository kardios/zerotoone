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
st.title("Zero-to-One Analysis")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
MAX_WORDS = 80000

# Initialize session_state to store outputs
if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}
if "final_text" not in st.session_state:
    st.session_state["final_text"] = ""

# Max tokens per pass configuration
MAX_TOKENS_PER_PASS = {
    "summary": 6000,
    "insights": 5000,
    "hypotheses": 5000,
    "refinement": 6000
}

def run_claude_pass(prompt, step_name, step_num, max_tokens):
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
            max_tokens=max_tokens,
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
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Summarize the main points of the following document in 2–3 paragraphs using structured headings:

- Key Arguments
- Key Assumptions
- Notable Examples

Highlight any non-obvious points, contradictions, or hidden assumptions.
This summary will serve as the foundation for extracting counter-intuitive insights and generating high-leverage hypotheses.

Document text:
{full_text}
"""
    summary_text, pass1_time = run_claude_pass(prompt_summary, "Expanded Summary", step, MAX_TOKENS_PER_PASS["summary"])
    st.session_state["outputs"]["summary"] = summary_text

    with st.expander("Pass 1: Expanded Summary"):
        st.markdown(summary_text)

    # --- Pass 2: Counter-Intuitive Insights ---
    step += 1
    prompt_insights = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Based on the summary below, extract 5–10 counter-intuitive insights that are:
- True or strongly plausible
- Contrary to common assumptions or conventional wisdom
- Relevant for innovation or high-leverage opportunities

For each insight, include:
1. A concise statement
2. Explanation of why it is counter-intuitive
3. Supporting reasoning from the summary
4. Optional: actionable recommendation

Rank insights by impact × plausibility.

Summary:
{summary_text}
"""
    insights_text, pass2_time = run_claude_pass(prompt_insights, "Counter-Intuitive Insights", step, MAX_TOKENS_PER_PASS["insights"])
    st.session_state["outputs"]["insights"] = insights_text

    with st.expander("Pass 2: Counter-Intuitive Insights"):
        st.markdown(insights_text)

    # --- Pass 3: Zero-to-One Hypotheses ---
    step += 1
    prompt_hypotheses = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Based on the summary and counter-intuitive insights below, generate 3–5 high-leverage Zero-to-One hypotheses.

For each hypothesis, include:
1. Concise statement
2. How it connects to the insights and summary
3. Estimated market potential or opportunity size
4. Feasibility or technical/practical considerations
5. Optional actionable steps or go-to-market ideas

Summary:
{summary_text}

Counter-Intuitive Insights:
{insights_text}
"""
    hypotheses_text, pass3_time = run_claude_pass(prompt_hypotheses, "Zero-to-One Hypotheses", step, MAX_TOKENS_PER_PASS["hypotheses"])
    st.session_state["outputs"]["hypotheses"] = hypotheses_text

    with st.expander("Pass 3: Zero-to-One Hypotheses"):
        st.markdown(hypotheses_text)

    # --- Pass 4: Refinement and Ranking ---
    step += 1
    prompt_refine = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Refine and rank all insights and hypotheses for clarity, precision, and impact.

- For each insight: include statement, Contrarian Impact (1–10), Plausibility (1–10), actionable note
- For each hypothesis: include statement, Market Potential, Feasibility, actionable steps
- Rank all outputs by impact × plausibility × potential ROI
- Ensure consistency with the summary

Summary:
{summary_text}

Insights:
{insights_text}

Hypotheses:
{hypotheses_text}
"""
    refined_text, pass4_time = run_claude_pass(prompt_refine, "Refinement and Ranking", step, MAX_TOKENS_PER_PASS["refinement"])
    st.session_state["outputs"]["refined"] = refined_text

    with st.expander("Pass 4: Refined Insights & Hypotheses"):
        st.markdown(refined_text)

    # --- Pass 5: Final Synthesis & Download ---
    step += 1
    st.session_state["final_text"] = refined_text
    with st.expander("Pass 5: Final Synthesized Report"):
        st.markdown(st.session_state["final_text"])

    st.download_button(
        label="Download Full Report as TXT",
        data=st.session_state["final_text"],
        file_name="zero_to_one_full_report.txt",
        mime="text/plain"
    )

    total_time = pass1_time + pass2_time + pass3_time + pass4_time
    st.info(f"Total processing time (excluding final download prep): {total_time:.1f}s")
