import os
import streamlit as st
import fitz  # PyMuPDF
import anthropic

# ------------------------------------------------------------------
# Secrets helper
# ------------------------------------------------------------------
def _get_secret(name: str):
    """Retrieve a secret from env first, then Streamlit secrets."""
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
st.title("Zero-to-One Insight Extractor (Claude Opus 4.1) - PDF Only")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Internal maximum word cap
MAX_WORDS = 80000

if uploaded_file and st.button("Analyze PDF"):

    st.info("Extracting text from PDF...")

    # --- PDF extraction ---
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "".join([page.get_text("text") + "\n" for page in doc])

    # --- Internal truncation if document is too long ---
    words = full_text.split()
    if len(words) > MAX_WORDS:
        st.warning(f"Document is very large. Truncating to first {MAX_WORDS} words.")
        words = words[:MAX_WORDS]
    full_text = " ".join(words)

    st.success("Text extraction complete.")

    # --- Pass 1: Initial extraction ---
    st.info("Running initial analysis...")

    prompt_pass1 = f"""
    You are a contrarian thinker trained in Peter Thiel's Zero to One framework.
    Analyze the following document:

    1. Summarize the main points in 3–5 sentences.
    2. Extract 5–10 counter-intuitive insights (true, but most people would disagree). 
       For each insight, explain why it is counter-intuitive and why it might be true.
    3. Suggest 3–5 high-leverage Zero-to-One hypotheses implied by the document.

    Document text:
    {full_text}
    """

    response_pass1 = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt_pass1}]
    )
    initial_output = response_pass1.content  # <-- corrected

    # --- Pass 2: Meta-refinement ---
    st.info("Running meta-refinement to sharpen insights...")

    prompt_pass2 = f"""
    You have the following initial analysis of a document:

    {initial_output}

    Please refine and rank all counter-intuitive insights and Zero-to-One hypotheses:
    - Make insights sharper and more actionable.
    - Rank by contrarian impact and plausibility.
    - Maintain the summary for context at the top.
    - Produce the final output in a clear structured format.
    """

    response_pass2 = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt_pass2}]
    )
    final_output = response_pass2.content  # <-- corrected

    # --- Display single output ---
    st.subheader("Refined Zero-to-One Insights")
    st.text_area("", final_output, height=800)

    # --- Download as TXT ---
    st.download_button(
        label="Download Results as TXT",
        data=final_output,
        file_name="zero_to_one_insights.txt",
        mime="text/plain"
    )
