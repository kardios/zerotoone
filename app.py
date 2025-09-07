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

if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}
if "final_text" not in st.session_state:
    st.session_state["final_text"] = ""

# Max tokens per pass
MAX_TOKENS_PER_PASS = {
    "summary": 6000,
    "insights": 5000,
    "hypotheses": 5000,
    "refinement": 6000,
    "self_critique": 3000,
    "alt_perspective": 4000
}

# Modular pass configuration
PASS_SEQUENCE = [
    ("Expanded Summary", True),
    ("Counter-Intuitive Insights", True),
    ("Zero-to-One Hypotheses", True),
    ("Refinement & Ranking", True),
    ("Final Synthesis", True),
    ("Self-Critique", False),         # Optional
    ("Alternative Perspective", False) # Optional
]

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

    if isinstance(response.content, list):
        output_text = "\n\n".join([tb.text for tb in response.content])
    else:
        output_text = str(response.content)

    step_placeholder.success(f"Step {step_num}: {step_name} complete in {elapsed:.1f}s")
    return output_text, elapsed

if uploaded_file and st.button("Analyze PDF"):

    # Step 0: Extract text
    step = 0
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

    # --- Run modular passes ---
    pass_results = {}
    total_time = 0
    for idx, (pass_name, enabled) in enumerate(PASS_SEQUENCE, start=1):
        if not enabled:
            continue
        step = idx
        # Construct prompt dynamically based on prior outputs
        if pass_name == "Expanded Summary":
            prompt = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Summarize the main points of the following document in 2–3 paragraphs using structured headings:

- Key Arguments
- Key Assumptions
- Notable Examples

Highlight contradictions, non-obvious points, and implied strategic choices.

Document text:
{full_text}
"""
            max_tokens = MAX_TOKENS_PER_PASS["summary"]

        elif pass_name == "Counter-Intuitive Insights":
            prompt = f"""
Based on the summary below, extract 5–10 counter-intuitive insights that are:
- True or plausible
- Contrary to common assumptions
- Actionable if possible

For each insight, include:
1. Statement
2. Conventional assumption being challenged
3. Reasoning from summary
4. Contrarian impact (1–10)
5. Plausibility (1–10)
6. Optional actionable note

Summary:
{pass_results.get('Expanded Summary','')}
"""
            max_tokens = MAX_TOKENS_PER_PASS["insights"]

        elif pass_name == "Zero-to-One Hypotheses":
            prompt = f"""
Based on the summary and insights below, generate 3–5 high-leverage Zero-to-One hypotheses.

For each hypothesis, include:
1. Statement
2. Linked insights
3. Strategic rationale
4. Market potential
5. Feasibility / technical considerations
6. Risk assessment
7. Actionable steps

Summary:
{pass_results.get('Expanded Summary','')}

Insights:
{pass_results.get('Counter-Intuitive Insights','')}
"""
            max_tokens = MAX_TOKENS_PER_PASS["hypotheses"]

        elif pass_name == "Refinement & Ranking":
            prompt = f"""
Refine and rank insights and hypotheses for clarity, precision, and impact.
Include normalized metrics for Contrarian Impact, Plausibility, Feasibility, Market Potential.
Detect conflicts or overlaps and suggest merges/splits.

Summary:
{pass_results.get('Expanded Summary','')}

Insights:
{pass_results.get('Counter-Intuitive Insights','')}

Hypotheses:
{pass_results.get('Zero-to-One Hypotheses','')}
"""
            max_tokens = MAX_TOKENS_PER_PASS["refinement"]

        elif pass_name == "Final Synthesis":
            prompt = f"""
Produce a final report combining summary, ranked insights, and hypotheses.
Include a top 3 actionable insights/hypotheses and strategic recommendations.
Format for readability in Markdown.
"""
            max_tokens = MAX_TOKENS_PER_PASS["refinement"]

        elif pass_name == "Self-Critique":
            prompt = f"""
Review all previous outputs for inconsistencies, implausible claims, or conflicts.
Suggest corrections or refinements while maintaining reasoning integrity.
"""
            max_tokens = MAX_TOKENS_PER_PASS["self_critique"]

        elif pass_name == "Alternative Perspective":
            prompt = f"""
Re-evaluate insights and hypotheses from an alternative perspective (e.g., competitor, regulator, engineer).
Highlight any overlooked opportunities or risks.
"""
            max_tokens = MAX_TOKENS_PER_PASS["alt_perspective"]

        else:
            continue

        output_text, elapsed = run_claude_pass(prompt, pass_name, step, max_tokens)
        pass_results[pass_name] = output_text
        total_time += elapsed

        with st.expander(f"Pass {step}: {pass_name}"):
            st.markdown(output_text)

    # Store final output
    st.session_state["final_text"] = pass_results.get("Final Synthesis", "")

    st.download_button(
        label="Download Full Report as TXT",
        data=st.session_state["final_text"],
        file_name="zero_to_one_full_report.txt",
        mime="text/plain"
    )

    st.info(f"Total processing time (excluding download prep): {total_time:.1f}s")
