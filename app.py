import os
import time
import streamlit as st
import fitz  # PyMuPDF
import anthropic
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

# ------------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------------
MAX_WORDS = 80000
MAX_TOKENS_PER_PASS = {
    "summary": 6000,
    "insights": 5000,
    "hypotheses": 5000,
    "refinement": 6000
}

MODEL_OPTIONS = {
    "Claude Opus 4": "claude-opus-4-1-20250805",
    "Claude Sonnet 4": "claude-sonnet-4-20250514"
}

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def _get_secret(name: str) -> Optional[str]:
    """Get secret from environment or Streamlit secrets."""
    v = os.environ.get(name)
    if not v:
        try:
            v = st.secrets.get(name)
        except Exception:
            v = None
    return v

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "outputs": {},
        "final_text": "",
        "processing_complete": False,
        "analysis_metadata": {},
        "step_placeholder": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def estimate_cost(model: str, total_tokens: int) -> float:
    """Estimate API cost based on model and token usage."""
    # Rough cost estimates (adjust based on actual pricing)
    costs_per_1k_tokens = {
        "claude-opus-4-1-20250805": 0.015,
        "claude-sonnet-4-20250514": 0.003
    }
    return (total_tokens / 1000) * costs_per_1k_tokens.get(model, 0.01)

def extract_pdf_text(uploaded_file) -> Tuple[str, Dict]:
    """Extract text from PDF with metadata."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Extract text and metadata
    full_text = ""
    metadata = {
        "total_pages": doc.page_count,
        "title": doc.metadata.get("title", "Unknown"),
        "author": doc.metadata.get("author", "Unknown"),
        "extraction_time": datetime.now().isoformat()
    }
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    
    doc.close()
    
    # Handle word limit
    words = full_text.split()
    if len(words) > MAX_WORDS:
        st.warning(f"Document is large ({len(words)} words). Truncating to first {MAX_WORDS} words.")
        words = words[:MAX_WORDS]
        metadata["truncated"] = True
        metadata["original_word_count"] = len(full_text.split())
    
    metadata["final_word_count"] = len(words)
    return " ".join(words), metadata

def run_claude_pass(
    client: anthropic.Anthropic,
    prompt: str, 
    step_name: str, 
    step_num: int, 
    max_tokens: int,
    model: str
) -> Tuple[str, float, int]:
    """Run a Claude API call with enhanced error handling and metrics."""
    step_placeholder = st.session_state.get("step_placeholder")
    if step_placeholder is None:
        step_placeholder = st.empty()
        st.session_state["step_placeholder"] = step_placeholder

    step_placeholder.info(f"Step {step_num}: {step_name}...")

    with st.spinner(f"{step_name} in progress..."):
        progress_bar = st.progress(0)
        start_time = time.time()
        
        try:
            # Simulate progress
            for i in range(1, 80):
                time.sleep(0.01)
                progress_bar.progress(i)
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7  # Slight creativity boost
            )
            
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            
            # Extract text from response
            if isinstance(response.content, list):
                output_text = "\n\n".join([tb.text for tb in response.content])
            else:
                output_text = str(response.content)
            
            # Estimate tokens used (rough approximation)
            tokens_used = len(prompt.split()) + len(output_text.split())
            
            step_placeholder.success(f"Step {step_num}: {step_name} complete in {elapsed:.1f}s")
            return output_text, elapsed, tokens_used
            
        except Exception as e:
            step_placeholder.error(f"Step {step_num}: {step_name} failed - {str(e)}")
            st.error(f"API Error in {step_name}: {str(e)}")
            return "", 0, 0

def create_enhanced_prompts(full_text: str, summary_text: str = "", insights_text: str = "", hypotheses_text: str = "") -> Dict[str, str]:
    """Create enhanced prompts for each analysis pass."""
    
    prompts = {}
    
    # Pass 1: Enhanced Summary
    prompts["summary"] = f"""
You are an expert analyst trained in Peter Thiel's Zero-to-One framework and contrarian thinking methodologies.

Analyze the following document and create a comprehensive summary with these sections:

## Document Overview
- Brief description of document type, scope, and main purpose
- Key stakeholders or audience mentioned

## Core Arguments & Thesis
- Primary arguments presented (numbered list)
- Central thesis or main proposition
- Evidence or data supporting key claims

## Underlying Assumptions
- Explicit assumptions stated by the author
- Implicit assumptions that underpin the arguments
- Conventional wisdom being challenged or reinforced

## Notable Examples & Case Studies
- Specific examples, case studies, or anecdotes
- Data points, statistics, or research cited
- Success/failure stories mentioned

## Contrarian Signals
- Statements that go against conventional wisdom
- Contradictions within the document
- Gaps in logic or unsupported claims
- Areas where the author might be wrong

Focus on extracting insights that could lead to non-obvious business opportunities or challenge mainstream thinking.

Document text:
{full_text}
"""

    # Pass 2: Enhanced Insights
    prompts["insights"] = f"""
You are a contrarian business strategist specializing in identifying non-obvious opportunities.

Based on the comprehensive summary below, extract 7-12 counter-intuitive insights that meet these criteria:
- Challenge conventional business wisdom
- Reveal hidden assumptions in the market/industry
- Suggest unexploited opportunities
- Are defensible with reasoning from the document

For each insight, provide:

**Insight [#]: [One-line statement]**
- **Contrarian Angle**: Why this challenges conventional thinking
- **Supporting Evidence**: Key points from the document that support this
- **Market Blindspot**: What most people/companies miss about this
- **Contrarian Score**: (1-10) How strongly this contradicts mainstream views
- **Confidence Level**: (1-10) How confident you are in this insight

Rank insights by: (Contrarian Score × Confidence Level × Market Potential)

Summary:
{summary_text}
"""

    # Pass 3: Enhanced Hypotheses
    prompts["hypotheses"] = f"""
You are a startup founder and investor trained in Zero-to-One thinking, focused on monopolistic opportunities.

Using the summary and counter-intuitive insights below, generate 4-6 high-leverage Zero-to-One business hypotheses.

For each hypothesis, provide:

**Hypothesis [#]: [Company/Product Concept in one line]**
- **The Opportunity**: What monopolistic advantage could be built
- **Why Now**: Timing factors that make this opportunity ripe
- **Market Dynamics**: Size, competition, and barriers to entry
- **Zero-to-One Factor**: How this creates something entirely new vs. incremental improvement
- **Technical/Execution Risk**: Key challenges and feasibility concerns
- **Go-to-Market Strategy**: Initial customer acquisition approach
- **Success Metrics**: How you'd measure early traction
- **10-Year Vision**: Where this could lead if successful

Rank hypotheses by: (Market Size × Defensibility × Feasibility × Zero-to-One Potential)

Summary:
{summary_text}

Counter-Intuitive Insights:
{insights_text}
"""

    # Pass 4: Enhanced Refinement
    prompts["refinement"] = f"""
You are a seasoned venture capitalist and strategic advisor, expert in Peter Thiel's frameworks.

Create a final, investment-grade analysis by refining and ranking all insights and hypotheses. 

## Executive Summary
- 2-3 sentence overview of the most compelling opportunities identified
- Key themes that emerged from the analysis

## Top-Ranked Insights
For the top 5 insights, provide:
- **Refined Statement**: Clear, actionable insight
- **Business Implication**: Direct impact on strategy/operations  
- **Validation Approach**: How to test this insight
- **Risk Factors**: What could make this insight wrong

## Top-Ranked Hypotheses  
For the top 3 hypotheses, provide:
- **Refined Pitch**: Elevator pitch version (2-3 sentences)
- **Competitive Moat**: How to build defensible advantages
- **Resource Requirements**: Key resources/capabilities needed
- **Timeline to Market**: Realistic development and launch timeline
- **Success Probability**: (%) Estimated likelihood of success
- **Potential ROI**: Conservative and optimistic return scenarios

## Strategic Recommendations
- Immediate next steps for validation
- Key questions to investigate further
- Potential partnerships or talent needs
- Market research priorities

## Risk Assessment
- Key assumptions that could be wrong
- Market/technology risks
- Execution challenges

Ensure all recommendations are specific, actionable, and grounded in the original document's insights.

Summary:
{summary_text}

Insights:
{insights_text}

Hypotheses:
{hypotheses_text}
"""

    return prompts

# ------------------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Zero-to-One Analysis",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Zero-to-One Analysis")
    st.markdown("*Transform documents into counter-intuitive insights and monopolistic business opportunities*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()), index=1)
        model_name = MODEL_OPTIONS[selected_model]
        
        st.header("📊 Analysis Settings")
        enable_detailed_prompts = st.checkbox("Enhanced Prompts", value=True, help="Use more detailed, structured prompts")
        custom_focus = st.text_area("Custom Focus Areas (optional)", 
                                   placeholder="e.g., fintech, sustainability, B2B SaaS")
        
        if st.session_state.get("processing_complete"):
            st.header("📈 Analysis Metrics")
            metadata = st.session_state.get("analysis_metadata", {})
            if metadata:
                st.metric("Total Time", f"{metadata.get('total_time', 0):.1f}s")
                st.metric("Est. Cost", f"${metadata.get('estimated_cost', 0):.3f}")
                st.metric("Pages Processed", metadata.get('pages', 'N/A'))
    
    # Initialize Claude client
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("🔑 Claude API key not found. Please set ANTHROPIC_API_KEY in environment variables or Streamlit secrets.")
        st.info("Get your API key from: https://console.anthropic.com/")
        st.stop()
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # File upload
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
    with col2:
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if uploaded_file and st.button("🔍 Analyze Document", type="primary", use_container_width=True):
        # Reset processing state
        st.session_state["processing_complete"] = False
        st.session_state["outputs"] = {}
        
        total_start_time = time.time()
        total_tokens = 0
        
        # Step 1: Extract text
        with st.spinner("📖 Extracting text from PDF..."):
            full_text, pdf_metadata = extract_pdf_text(uploaded_file)
            st.success(f"✅ Extracted {pdf_metadata['final_word_count']} words from {pdf_metadata['total_pages']} pages")
        
        # Add custom focus to prompts if provided
        focus_context = f"\n\nSPECIAL FOCUS: Pay particular attention to insights related to: {custom_focus}" if custom_focus else ""
        
        # Create prompts
        prompts = create_enhanced_prompts(full_text + focus_context) if enable_detailed_prompts else {
            "summary": f"Summarize key points: {full_text}",
            "insights": f"Extract counter-intuitive insights: {full_text}",
            "hypotheses": f"Generate business hypotheses: {full_text}",
            "refinement": f"Refine and rank: {full_text}"
        }
        
        # Execute analysis pipeline
        step = 1
        
        # Pass 1: Summary
        summary_text, time1, tokens1 = run_claude_pass(
            client, prompts["summary"], "Enhanced Summary", step, 
            MAX_TOKENS_PER_PASS["summary"], model_name
        )
        if summary_text:
            st.session_state["outputs"]["summary"] = summary_text
            total_tokens += tokens1
            
            with st.expander("📋 Pass 1: Enhanced Summary", expanded=False):
                st.markdown(summary_text)
        
        # Pass 2: Insights
        step += 1
        prompts["insights"] = create_enhanced_prompts("", summary_text)["insights"]
        insights_text, time2, tokens2 = run_claude_pass(
            client, prompts["insights"], "Counter-Intuitive Insights", step,
            MAX_TOKENS_PER_PASS["insights"], model_name
        )
        if insights_text:
            st.session_state["outputs"]["insights"] = insights_text
            total_tokens += tokens2
            
            with st.expander("💡 Pass 2: Counter-Intuitive Insights", expanded=False):
                st.markdown(insights_text)
        
        # Pass 3: Hypotheses
        step += 1
        prompts["hypotheses"] = create_enhanced_prompts("", summary_text, insights_text)["hypotheses"]
        hypotheses_text, time3, tokens3 = run_claude_pass(
            client, prompts["hypotheses"], "Zero-to-One Hypotheses", step,
            MAX_TOKENS_PER_PASS["hypotheses"], model_name
        )
        if hypotheses_text:
            st.session_state["outputs"]["hypotheses"] = hypotheses_text
            total_tokens += tokens3
            
            with st.expander("🎯 Pass 3: Zero-to-One Hypotheses", expanded=False):
                st.markdown(hypotheses_text)
        
        # Pass 4: Refinement
        step += 1
        prompts["refinement"] = create_enhanced_prompts("", summary_text, insights_text, hypotheses_text)["refinement"]
        refined_text, time4, tokens4 = run_claude_pass(
            client, prompts["refinement"], "Final Analysis & Ranking", step,
            MAX_TOKENS_PER_PASS["refinement"], model_name
        )
        if refined_text:
            st.session_state["outputs"]["refined"] = refined_text
            st.session_state["final_text"] = refined_text
            total_tokens += tokens4
            
            with st.expander("🏆 Pass 4: Final Analysis & Strategic Recommendations", expanded=True):
                st.markdown(refined_text)
        
        # Calculate final metrics
        total_time = time.time() - total_start_time
        estimated_cost = estimate_cost(model_name, total_tokens)
        
        # Store metadata
        st.session_state["analysis_metadata"] = {
            "total_time": total_time,
            "estimated_cost": estimated_cost,
            "total_tokens": total_tokens,
            "pages": pdf_metadata["total_pages"],
            "model_used": selected_model,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state["processing_complete"] = True
        
        # Success summary
        st.success(f"🎉 Analysis complete! Total time: {total_time:.1f}s | Est. cost: ${estimated_cost:.3f}")
        
        # Download options
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Full Report",
                data=st.session_state["final_text"],
                file_name=f"zero_to_one_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            # Create JSON export with all outputs
            json_data = {
                "metadata": st.session_state["analysis_metadata"],
                "pdf_info": pdf_metadata,
                "outputs": st.session_state["outputs"]
            }
            st.download_button(
                label="📊 Download JSON Data",
                data=json.dumps(json_data, indent=2),
                file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col3:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                for key in ["outputs", "final_text", "processing_complete", "analysis_metadata"]:
                    st.session_state[key] = {} if key in ["outputs", "analysis_metadata"] else "" if key == "final_text" else False
                st.rerun()

if __name__ == "__main__":
    main()
