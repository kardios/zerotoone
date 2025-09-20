# app.py
import os
import time
import json
import math
import streamlit as st
import fitz  # PyMuPDF
import anthropic
import pandas as pd
from io import StringIO

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

# ------------------------------------------------------------------
# Claude client init
# ------------------------------------------------------------------
API_KEY = _get_secret("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Claude API key not found. Set ANTHROPIC_API_KEY in env or Streamlit secrets.")
    st.stop()

client = anthropic.Anthropic(api_key=API_KEY)
MODEL_NAME = "claude-opus-4-1-20250805"

# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------
st.set_page_config(page_title="Zero-to-One Analysis", layout="wide")
st.title("Zero-to-One Analysis")

# Internal constants
MAX_WORDS_HARD = 120_000  # absolute cap (words) to protect context window
DEFAULT_MAX_TOKENS = {
    "summary": 6000,
    "insights": 5000,
    "per_insight": 1200,      # used for per-insight focused reasoning
    "merge_refine": 3000,
    "hypotheses": 4500,
    "refinement": 6000,
    "self_critique": 2500,
    "alt_perspective": 3500
}

# Session-state slots
if "doc_text" not in st.session_state:
    st.session_state["doc_text"] = ""
if "outputs" not in st.session_state:
    st.session_state["outputs"] = {}
if "timings" not in st.session_state:
    st.session_state["timings"] = {}
if "final_text" not in st.session_state:
    st.session_state["final_text"] = ""
if "pass_status" not in st.session_state:
    st.session_state["pass_status"] = {}

# Utilities
def safe_join_textblocks(content):
    """Handle Anthropic's TextBlock list vs string."""
    if isinstance(content, list):
        return "\n\n".join([getattr(tb, "text", str(tb)) for tb in content])
    return str(content)

def run_claude(prompt, max_tokens, step_name=""):
    """Run Claude with spinner/progress, return plain text and latency."""
    placeholder = st.session_state.get("step_placeholder")
    if not placeholder:
        placeholder = st.empty()
        st.session_state["step_placeholder"] = placeholder

    placeholder.info(f"{step_name}: running...")
    with st.spinner(f"{step_name} — waiting for model..."):
        progress = st.progress(0)
        start = time.time()
        # light simulated progress until model returns
        for i in range(1, 70):
            time.sleep(0.005)
            progress.progress(i)
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        progress.progress(100)
        elapsed = time.time() - start

    text = safe_join_textblocks(response.content)
    placeholder.success(f"{step_name}: completed in {elapsed:.1f}s")
    return text, elapsed

def parse_json_fallback(text):
    """Try to parse JSON; if fails, return None."""
    try:
        return json.loads(text)
    except Exception:
        # attempt to find a JSON block in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return None

# UI: Left controls, right outputs
col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    token_scale = st.slider("Context caution (larger → more tokens per pass)", 0.5, 2.0, 1.0, 0.1)
    enable_self_critique = st.checkbox("Enable Self-Critique Pass (recommended)", value=True)
    enable_alt_persp = st.checkbox("Enable Alternative Perspective Pass", value=False)
    run_button = st.button("Run Full Analysis")

    # control: regenerate a pass or per-insight (enabled after run)
    st.markdown("---")
    st.write("Retry controls (after run):")
    if "outputs" in st.session_state and st.session_state["outputs"]:
        retry_pass = st.selectbox("Retry which pass?", ["(none)"] + list(st.session_state["outputs"].keys()))
        if retry_pass != "(none)":
            if st.button("Retry Selected Pass"):
                st.session_state["retry_pass"] = retry_pass

with col_right:
    st.markdown("### Progress & Outputs")
    if "step_placeholder" not in st.session_state:
        st.session_state["step_placeholder"] = st.empty()

# Main flow
if uploaded and run_button:
    # Reset state for new run
    st.session_state["outputs"].clear()
    st.session_state["timings"].clear()
    st.session_state["final_text"] = ""
    st.session_state["pass_status"].clear()
    st.session_state.pop("retry_pass", None)

    # Extract text
    st.session_state["step_placeholder"].info("Step 0: Extracting text from PDF...")
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    full_text = "\n\n".join(pages_text)
    words = full_text.split()
    if len(words) > MAX_WORDS_HARD:
        st.warning(f"Document is large ({len(words)} words). Truncating to {MAX_WORDS_HARD} words.")
        full_text = " ".join(words[:MAX_WORDS_HARD])
    st.session_state["doc_text"] = full_text
    st.session_state["step_placeholder"].success(f"Step 0: Text extracted ({len(full_text.split())} words)")

    # Adaptive token function based on document length and scale slider
    def adapt_tokens(base_key):
        base = DEFAULT_MAX_TOKENS.get(base_key, 3000)
        length_factor = min(2.0, max(0.5, len(st.session_state["doc_text"].split()) / 20000))
        tokens = int(base * token_scale * length_factor)
        return tokens

    # PASS 1: Expanded Summary (structured)
    prompt_summary = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework. 
Produce a structured, machine-readable summary of the document. Output JSON with keys:
  - key_arguments: list of short strings
  - key_assumptions: list of short strings
  - notable_examples: list of short strings
  - contradictions_or_gaps: list of short strings
  - implied_strategic_choices: list of short strings

Be concise but include enough detail to support insight-generation.

Document:
{st.session_state['doc_text']}
"""
    max_toks = adapt_tokens("summary")
    summary_raw, t = run_claude(prompt_summary, max_toks, "Pass 1 — Expanded Summary")
    st.session_state["timings"]["pass1"] = t

    # Try to parse JSON; fallback to raw markdown
    parsed = parse_json_fallback(summary_raw)
    if parsed:
        st.session_state["outputs"]["Expanded Summary"] = parsed
        with st.expander("Pass 1: Expanded Summary (structured JSON)"):
            st.json(parsed)
    else:
        st.session_state["outputs"]["Expanded Summary"] = {"raw": summary_raw}
        with st.expander("Pass 1: Expanded Summary (raw)"):
            st.markdown(summary_raw)

    # PASS 2: Candidate Counter-Intuitive Insights (batch)
    prompt_insights = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.

Based on the structured summary (JSON) below, produce a JSON array of 6-12 candidate insights. Each insight object must include:
 - id: short unique id (e.g., insight_1)
 - statement: concise one-line insight
 - conventional_assumption: the conventional belief this challenges
 - reasoning: supporting reasoning referencing summary items
 - contrarian_impact: integer 1-10
 - plausibility: integer 1-10
 - actionable_note: short suggestion (optional)

Use only information that is supported by the summary. Output STRICT JSON (array of objects).

Summary JSON:
{json.dumps(st.session_state['outputs']['Expanded Summary'], ensure_ascii=False)}
"""
    max_toks = adapt_tokens("insights")
    insights_raw, t = run_claude(prompt_insights, max_toks, "Pass 2 — Candidate Insights")
    st.session_state["timings"]["pass2"] = t

    insights_parsed = parse_json_fallback(insights_raw)
    insights_list = []
    if isinstance(insights_parsed, list):
        insights_list = insights_parsed
        st.session_state["outputs"]["Candidate Insights"] = insights_list
        # show table
        df = pd.DataFrame([{"id": i.get("id",""), "statement": i.get("statement",""),
                            "impact": i.get("contrarian_impact",0), "plausibility": i.get("plausibility",0)}
                           for i in insights_list])
        with st.expander("Pass 2: Candidate Insights (parsed)"):
            st.table(df)
            st.bar_chart(df.set_index("id")[["impact","plausibility"]])
    else:
        # fallback: show raw and keep as text for per-insight parsing later
        st.session_state["outputs"]["Candidate Insights Raw"] = insights_raw
        with st.expander("Pass 2: Candidate Insights (raw output - failed JSON parse)"):
            st.markdown(insights_raw)
        # we will attempt to extract numbered items heuristically later

    # Allow user review + selection (human-in-the-loop)
    st.markdown("### Review candidate insights")
    accept_ids = []
    if "Candidate Insights" in st.session_state["outputs"]:
        # show each insight with a checkbox
        for ins in st.session_state["outputs"]["Candidate Insights"]:
            key = f"accept_{ins['id']}"
            default = True
            accepted = st.checkbox(f"{ins['id']}: {ins['statement']}", value=default, key=key)
            if accepted:
                accept_ids.append(ins['id'])
    else:
        st.info("Insights are not parsed into JSON. You can still proceed but manual selection is not available.")

    # If no insights selected, warn
    if "Candidate Insights" in st.session_state["outputs"] and not accept_ids:
        st.warning("No insights selected — you must accept at least one to proceed to hypotheses.")
    else:
        # PASS 2.5: Per-Insight Deepening (one-by-one focused reasoning)
        st.markdown("### Focused per-insight deepening (optional)")
        per_insight_results = {}
        if "Candidate Insights" in st.session_state["outputs"]:
            for ins in st.session_state["outputs"]["Candidate Insights"]:
                if ins["id"] not in accept_ids:
                    continue
                # For each accepted insight, run a focused pass to elaborate (assumptions, counter-arguments, quick experiments)
                prompt_per_insight = f"""
You are a methodical contrarian analyst. Deepen the following insight with a focused, structured JSON object.

Input insight:
{json.dumps(ins, ensure_ascii=False)}

Produce JSON:
{{
  "id": "{ins['id']}",
  "expanded_statement": "...",
  "assumptions_challenged": ["..."],
  "most_plausible_counterarguments": ["..."],
  "evidence_links_to_summary": ["..."],    # short references to summary items
  "actionable_experiments": ["..."],       # 2-3 small experiments or checks
  "confidence": 1-10
}}
Keep output as valid JSON.
"""
                max_toks = DEFAULT_MAX_TOKENS["per_insight"]
                per_raw, t = run_claude(prompt_per_insight, max_toks, f"Per-Insight: {ins['id']}")
                st.session_state["timings"][f"per_{ins['id']}"] = t
                parsed = parse_json_fallback(per_raw)
                if parsed:
                    per_insight_results[ins['id']] = parsed
                    with st.expander(f"Deepen {ins['id']}"):
                        st.json(parsed)
                else:
                    # fallback to raw
                    per_insight_results[ins['id']] = {"raw": per_raw}
                    with st.expander(f"Deepen {ins['id']} (raw)"):
                        st.markdown(per_raw)
        st.session_state["outputs"]["Per-Insight"] = per_insight_results

        # PASS 3: Zero-to-One Hypotheses (only if there are accepted insights)
        if "Candidate Insights" in st.session_state["outputs"] and accept_ids:
            accepted_insights = [i for i in st.session_state["outputs"]["Candidate Insights"] if i["id"] in accept_ids]
            prompt_hypotheses = f"""
You are a contrarian Zero-to-One strategist. Using the accepted insights below, generate 3-6 high-leverage hypotheses.
Output JSON array of objects with fields:
- id
- statement
- linked_insights: list of insight ids
- strategic_rationale
- market_potential_estimate (short)
- feasibility_score (1-10)
- primary_risk
- first_3_action_steps

Accepted insights JSON:
{json.dumps(accepted_insights, ensure_ascii=False)}
"""
            max_toks = adapt_tokens("hypotheses")
            hypotheses_raw, t = run_claude(prompt_hypotheses, max_toks, "Pass 3 — Hypotheses")
            st.session_state["timings"]["pass3"] = t

            hyp_parsed = parse_json_fallback(hypotheses_raw)
            if isinstance(hyp_parsed, list):
                st.session_state["outputs"]["Hypotheses"] = hyp_parsed
                with st.expander("Pass 3: Hypotheses (parsed)"):
                    st.json(hyp_parsed)
                    # present a quick table
                    dfh = pd.DataFrame([{"id": h.get("id",""), "statement": h.get("statement",""),
                                         "feasibility": h.get("feasibility_score",0)} for h in hyp_parsed])
                    st.table(dfh)
            else:
                st.session_state["outputs"]["Hypotheses Raw"] = hypotheses_raw
                with st.expander("Pass 3: Hypotheses (raw)"):
                    st.markdown(hypotheses_raw)

        # PASS 4: Refinement & Ranking (merge, dedupe, normalize)
        # Build prompt with everything
        step_name = "Pass 4 — Refinement & Ranking"
        prompt_refine = f"""
You are a contrarian Zero-to-One strategist. Refine and normalize the candidate insights and hypotheses.
Input:
- Summary JSON: {json.dumps(st.session_state['outputs'].get('Expanded Summary', {}), ensure_ascii=False)}
- Candidate insights (selected): {json.dumps([i for i in st.session_state['outputs'].get('Candidate Insights',[]) if i.get('id') in accept_ids], ensure_ascii=False)}
- Per-insight deepening: {json.dumps(st.session_state['outputs'].get('Per-Insight',{}), ensure_ascii=False)}
- Hypotheses (if any): {json.dumps(st.session_state['outputs'].get('Hypotheses',[]), ensure_ascii=False)}

Produce a JSON object with:
- insights: array with normalized fields (id, statement, impact 1-10, plausibility 1-10, confidence, actionable_note)
- hypotheses: array with normalized fields (id, statement, linked_insights, market_potential, feasibility 1-10, primary_risk, top_actions)
- rankings: arrays of top 3 insights and top 3 hypotheses by impact*plausibility
- merge_suggestions: list of suggestions to merge/split items if overlaps detected

Ensure output is valid JSON.
"""
        max_toks = adapt_tokens("refinement")
        refine_raw, t = run_claude(prompt_refine, max_toks, step_name)
        st.session_state["timings"]["pass4"] = t
        refine_parsed = parse_json_fallback(refine_raw)
        if refine_parsed:
            st.session_state["outputs"]["Refined"] = refine_parsed
            with st.expander("Pass 4: Refined & Ranked (structured)"):
                st.json(refine_parsed)
                # create DataFrame for visualization
                df_ins = pd.DataFrame(refine_parsed.get("insights", []))
                if not df_ins.empty:
                    df_ins = df_ins.set_index("id")
                    st.table(df_ins[["statement","impact","plausibility","confidence"]])
                    st.bar_chart(df_ins[["impact","plausibility"]])
        else:
            st.session_state["outputs"]["Refined Raw"] = refine_raw
            with st.expander("Pass 4: Refined & Ranked (raw)"):
                st.markdown(refine_raw)

        # Optional PASS 5: Self-critique
        if enable_self_critique:
            prompt_critique = f"""
You are a critical reviewer. Review the JSON outputs (summary, insights, refined) for internal inconsistencies, unsupported claims, or hallucinations.
Return JSON:
{{
  "issues": [ "..." ],
  "suggested_fixes": [ "..." ],
  "confidence": 1-10
}}
"""
            crit_raw, t = run_claude(prompt_critique, DEFAULT_MAX_TOKENS["self_critique"], "Self-Critique")
            st.session_state["timings"]["self_critique"] = t
            crit_parsed = parse_json_fallback(crit_raw)
            if crit_parsed:
                st.session_state["outputs"]["Self-Critique"] = crit_parsed
                with st.expander("Self-Critique"):
                    st.json(crit_parsed)
            else:
                st.session_state["outputs"]["Self-Critique Raw"] = crit_raw
                with st.expander("Self-Critique (raw)"):
                    st.markdown(crit_raw)

        # Optional PASS 6: Alternative perspective
        if enable_alt_persp:
            prompt_alt = f"""
Re-evaluate the top insights and hypotheses from the perspective of a skeptical regulator/competitor/engineer.
Return JSON array: [{{"id":"...","concerns":["..."], "opportunities":["..."]}}]
"""
            alt_raw, t = run_claude(prompt_alt, DEFAULT_MAX_TOKENS["alt_perspective"], "Alternative Perspective")
            st.session_state["timings"]["alt_persp"] = t
            alt_parsed = parse_json_fallback(alt_raw)
            if alt_parsed:
                st.session_state["outputs"]["Alt Perspective"] = alt_parsed
                with st.expander("Alternative Perspective"):
                    st.json(alt_parsed)
            else:
                st.session_state["outputs"]["Alt Perspective Raw"] = alt_raw
                with st.expander("Alternative Perspective (raw)"):
                    st.markdown(alt_raw)

        # Final synthesized report (human-readable)
        # Prefer structured refined output if present
        final_report = ""
        if "Refined" in st.session_state["outputs"]:
            final_report = json.dumps(st.session_state["outputs"]["Refined"], ensure_ascii=False, indent=2)
        else:
            # fallback compile
            pieces = []
            for k, v in st.session_state["outputs"].items():
                pieces.append(f"--- {k} ---\n{json.dumps(v, ensure_ascii=False, indent=2) if not isinstance(v,str) else v}\n")
            final_report = "\n\n".join(pieces)
        st.session_state["final_text"] = final_report
        with st.expander("Final Synthesized Report (structured JSON)"):
            st.code(final_report)

        st.download_button(
            "Download Full Report (TXT)",
            data=st.session_state["final_text"],
            file_name="zero_to_one_full_report.txt",
            mime="text/plain"
        )

        # Display timings summary
        times_df = pd.DataFrame.from_dict(st.session_state["timings"], orient="index", columns=["seconds"])
        with st.expander("Pass timings"):
            st.table(times_df)

# Retry pass handler (user clicked retry earlier)
if "retry_pass" in st.session_state:
    rp = st.session_state.pop("retry_pass")
    st.info(f"User requested retry of: {rp} — not implemented in this snippet, but you can re-run that pass by rerunning the app or adding a targeted re-run.")
