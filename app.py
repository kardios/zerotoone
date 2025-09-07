import streamlit as st
import fitz  # PyMuPDF
import anthropic

# Initialize Claude client
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

st.title("Zero-to-One Insight Extractor (Claude Opus 4.1) - PDF Only")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
chunk_size = st.number_input("Chunk size (words)", min_value=200, max_value=2000, value=1200, step=100)

if uploaded_file and st.button("Analyze PDF"):

    st.info("Extracting text from PDF...")

    # --- Extract text ---
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "".join([page.get_text("text") + "\n" for page in doc])
    st.success("Text extraction complete.")

    # --- Chunk the text ---
    words = full_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    section_summaries = []
    chunk_insights_list = []

    # --- Process each chunk ---
    for idx, chunk in enumerate(chunks):
        st.write(f"Processing chunk {idx+1}/{len(chunks)}...")

        # --- Summary ---
        prompt_summary = f"""
        You are a contrarian thinker trained in Peter Thiel's Zero to One framework.
        Summarize the following text chunk in 2-3 sentences.

        Text chunk:
        {chunk}
        """
        response_summary = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt_summary}]
        )
        summary = response_summary["content"]
        section_summaries.append(f"Chunk {idx+1} Summary:\n{summary}\n")

        # --- Chunk-level insights ---
        prompt_insights = f"""
        Analyze the following text chunk and extract 2-5 counter-intuitive insights:
        - Explain why each is counter-intuitive.
        - Explain why it might be true.
        - Optionally, suggest 1-2 Zero-to-One hypotheses.

        Text chunk:
        {chunk}
        """
        response_insights = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt_insights}]
        )
        chunk_insights = response_insights["content"]
        chunk_insights_list.append(f"Chunk {idx+1} Insights:\n{chunk_insights}\n")

    # --- Global Synthesis ---
    global_prompt = f"""
    You have the following chunk-level insights:

    {'\n\n'.join(chunk_insights_list)}

    1. Combine these insights and identify any high-leverage contrarian ideas that challenge broad assumptions.
    2. Rank all insights and hypotheses by contrarian impact and plausibility.
    3. Produce a final, refined, ranked list of Zero-to-One insights and hypotheses.
    """
    global_response = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=3000,
        messages=[{"role": "user", "content": global_prompt}]
    )
    global_insights = global_response["content"]

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Section Summaries", "Chunk-Level Insights", "Global Synthesis"])

    with tab1:
        st.subheader("Section Summaries")
        for summary in section_summaries:
            st.text_area("", summary, height=120)

    with tab2:
        st.subheader("Chunk-Level Counter-Intuitive Insights")
        for chunk_text in chunk_insights_list:
            st.text_area("", chunk_text, height=300)

    with tab3:
        st.subheader("Global Synthesis & Ranked Zero-to-One Insights")
        st.text_area("", global_insights, height=600)

    # --- Download ---
    full_output = "\n\n".join(section_summaries + chunk_insights_list + [f"Global Synthesis:\n{global_insights}"])
    st.download_button(
        label="Download as TXT",
        data=full_output,
        file_name="zero_to_one_insights.txt",
        mime="text/plain"
    )
