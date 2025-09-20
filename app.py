# app.py
import os
import time
import json
from io import StringIO

import streamlit as st
import fitz  # PyMuPDF
import anthropic
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config & secrets helper
# ---------------------------
MODEL_NAME = "claude-opus-4-1-20250805"

def _get_secret(name: str):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)
    except Exception:
        return None

API_KEY = _get_secret("ANTHROPIC_API_KEY")
if not API_KEY:
    st.error("Claude API key not found. Set ANTHROPIC_API_KEY in env or Streamlit secrets.")
    st.stop()

client = anthropic.Anthropic(api_key=API_KEY)

st.set_page_config(page_title="Zero-to-One Analysis", layout="wide")
st.title("Zero-to-One Analysis")

# ---------------------------
# Helpers
# ---------------------------
def safe_join_textblocks(content):
    """
    Anthropic responses sometimes return a list of TextBlock objects or a string.
    Convert into a single string safely.
    """
    try:
        if isinstance(content, list):
            parts = []
            for tb in content:
                # TextBlock objects often have .text
                if hasattr(tb, "text"):
                    parts.append(tb.text)
                else:
                    parts.append(str(tb))
            return "\n\n".join(parts)
        return str(content)
    except Exception:
        return str(content)

def parse_json_fallback(text: str):
    """
    Try strict JSON parse; if fails, attempt to locate the first JSON object/array substring.
    Returns Python object or None.
    """
    if not text or not isinstance(text, str):
        return None
    text_strip = text.strip()
    # direct parse
    try:
        return json.loads(text_strip)
    except Exception:
        pass
    # try to find first JSON block
    start = None
    end = None
    # try for array first
    a = text.find("[")
    b = text.rfind("]")
    if 0 <= a < b:
        try:
            candidate = text[a:b+1]
            return json.loads(candidate)
        except Exception:
            pass
    # try for object
    a = text.find("{")
    b = text.rfind("}")
    if 0 <= a < b:
        try:
            candidate = text[a:b+1]
            return json.loads(candidate)
        except Exception:
            pass
    return None

def run_claude(prompt: str, max_tokens: int, step_name: str):
    """
    Run Claude with spinner and a lightweight progress indicator.
    Returns (text_output, elapsed_seconds).
    """
    placeholder = st.session_state.get("step_placeholder")
    if placeholder is None:
        placeholder = st.empty()
        st.session_state["step_placeholder"] = placeholder

    placeholder.info(f"{step_name}: running…")
    with st.spinner(f"{step_name} — waiting for model..."):
        progress = st.progress(0)
        start = time.time()
        # simulated progress until response arrives
        for i in range(1, 60):
            time.sleep(0.005)
            progress.progress(i)
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            progress.progress(100)
            elapsed = time.time() - start
            placeholder.error(f"{step_name}: API call failed: {e}")
            return f"Error: {e}", elapsed

        progress.progress(100)
        elapsed = time.time() - start

    text = safe_join_textblocks(response.content)
    placeholder.success(f"{step_name}: completed in {elapsed:.1f}s")
    return text, elapsed

def adapt_tokens_for_doc(base_tokens: int):
    """
    Scale tokens based on document length and a user-selected scale factor.
    """
    length = len(st.session_state.get("doc_text", "").split())
    length_factor = min(2.0, max(0.5, length / 20000))  # doc 20k words => factor ~1
    scale = st.session_state.get("token_scale", 1.0)
    return int(base_tokens * length_factor * scale)

# ---------------------------
# Renderers (robust)
# ---------------------------
def render_pass1_summary(data):
    st.subheader("📖 Expanded Summary")
    # If data is raw string show it, otherwise treat as dict
    if isinstance(data, dict):
        ka = data.get("key_arguments") or data.get("key_arguments", []) or []
        ka = ka or []
        st.markdown("### 📌 Key Arguments")
        if ka:
            for arg in ka:
                st.markdown(f"- {arg}")
        else:
            st.info("No key arguments found.")

        asum = data.get("key_assumptions") or []
        st.markdown("### 💡 Key Assumptions")
        if asum:
            for a in asum:
                st.markdown(f"- {a}")
        else:
            st.info("No key assumptions found.")

        contradictions = data.get("contradictions") or data.get("contradictions_or_gaps") or []
        st.markdown("### ⚖️ Contradictions / Gaps")
        if contradictions:
            for c in contradictions:
                st.markdown(f"- {c}")
        else:
            st.info("No contradictions listed.")

        choices = data.get("strategic_choices") or data.get("implied_strategic_choices") or []
        st.markdown("### 🚀 Implied Strategic Choices")
        if choices:
            for sc in choices:
                st.markdown(f"- {sc}")
        else:
            st.info("No strategic choices listed.")
    else:
        st.markdown(data)

    with st.expander("Raw JSON / Raw Text"):
        if isinstance(data, dict):
            st.json(data)
        else:
            st.code(str(data)[:10000])  # limit length for display

def render_pass2_insights(data):
    st.subheader("💡 Candidate Contrarian Insights")
    if not data:
        st.warning("No insights available.")
        return
    # data expected list of dicts
    if isinstance(data, list):
        try:
            df = pd.DataFrame(data)
            # ensure numeric columns exist
            for col in ["contrarian_impact", "plausibility"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            display_cols = [c for c in ["id", "statement", "conventional_assumption", "contrarian_impact", "plausibility"] if c in df.columns]
            if display_cols:
                st.markdown("### 📊 Insights Table")
                st.dataframe(df[display_cols])
            else:
                st.table(df.head())

            # Quadrant chart if possible
            if "plausibility" in df.columns and "contrarian_impact" in df.columns:
                st.markdown("### 🔍 Impact vs Plausibility")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df["plausibility"], df["contrarian_impact"])
                for _, row in df.iterrows():
                    label = (row.get("id") or "") or str(row.name)
                    ax.text(row["plausibility"], row["contrarian_impact"], label, fontsize=8)
                ax.set_xlabel("Plausibility")
                ax.set_ylabel("Contrarian Impact")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not render insights: {e}")
            with st.expander("Raw JSON"):
                st.json(data)
    else:
        st.info("Insights not in list form; showing raw output.")
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

def render_pass3_hypotheses(data):
    st.subheader("🧪 Zero-to-One Hypotheses")
    if not data:
        st.warning("No hypotheses available.")
        return
    if isinstance(data, list):
        for hyp in data:
            st.markdown(f"### {hyp.get('id','?')}: {hyp.get('statement','(no statement)')}")
            feas = float(hyp.get("feasibility_score", 0) or 0)
            feas = max(0, min(feas, 10))
            st.progress(feas / 10)
            st.caption(f"Feasibility: {feas}/10")
            risks = hyp.get("primary_risk") or hyp.get("risks") or []
            if isinstance(risks, list):
                risks = ", ".join(risks)
            st.caption(f"Risks: {risks or 'None listed'}")
            linked = hyp.get("linked_insights") or []
            st.caption(f"Linked insights: {', '.join(linked) if linked else 'None'}")
        # table
        try:
            df = pd.DataFrame(data)
            cols = [c for c in ["id", "statement", "feasibility_score", "market_potential"] if c in df.columns]
            if cols:
                st.markdown("### Overview")
                st.dataframe(df[cols])
        except Exception:
            pass
    else:
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

def render_pass4_refinement(data):
    st.subheader("🏆 Refined & Ranked Insights/Hypotheses")
    if not data:
        st.warning("No refined output.")
        return
    # data could be dict with 'insights' and 'hypotheses' or a list
    if isinstance(data, dict):
        insights = data.get("insights", []) or []
        hypotheses = data.get("hypotheses", []) or []
        rankings = data.get("rankings", {}) or {}
        # Top 3 highlights
        st.markdown("### 🏅 Top 3 Insights")
        top_ins = rankings.get("top_insights") or []
        if top_ins:
            for tid in top_ins:
                # find insight by id
                found = next((i for i in insights if i.get("id") == tid), None)
                if found:
                    st.success(f"{found.get('id')}: {found.get('statement')}")
        else:
            if insights:
                for i in insights[:3]:
                    st.success(f"{i.get('id')}: {i.get('statement')}")
            else:
                st.info("No insights to highlight.")

        st.markdown("### 🔁 Insights — Before / After (sample)")
        for ins in insights:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original:**\n{ins.get('original_statement', ins.get('statement','(missing)'))}")
            with col2:
                st.markdown(f"**Refined:**\n{ins.get('refined_statement', ins.get('statement','(missing)'))}")

        if hypotheses:
            st.markdown("### 🧭 Hypotheses Summary")
            for h in hypotheses:
                st.markdown(f"- **{h.get('id')}**: {h.get('statement')} (feasibility {h.get('feasibility', h.get('feasibility_score','?'))})")
    elif isinstance(data, list):
        # treat list as refined items
        for item in data:
            st.markdown(f"- {item.get('refined_statement', item.get('statement','(no statement)'))}")
    else:
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

    with st.expander("Raw JSON / Raw Text"):
        if isinstance(data, dict) or isinstance(data, list):
            st.json(data)
        else:
            st.code(str(data)[:10000])

def render_pass5_critique(data):
    st.subheader("🧐 Self-Critique")
    if not data:
        st.info("No critique provided.")
        return
    if isinstance(data, dict):
        issues = data.get("issues", []) or []
        fixes = data.get("suggested_fixes", []) or []
        if issues:
            st.markdown("### ⚠️ Issues")
            for issue in issues:
                st.error(issue)
        else:
            st.info("No issues found.")
        if fixes:
            st.markdown("### ✅ Suggested Fixes")
            for fix in fixes:
                st.success(fix)
    else:
        with st.expander("Raw Output"):
            st.markdown(str(data)[:10000])

def render_pass6_altperspective(data):
    st.subheader("🔄 Alternative Perspective")
    if not data:
        st.info("No alternative perspective provided.")
        return
    concerns = data.get("concerns", []) if isinstance(data, dict) else []
    opportunities = data.get("opportunities", []) if isinstance(data, dict) else []
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❗ Concerns")
        if concerns:
            for c in concerns:
                st.error(c)
        else:
            st.info("No concerns.")
    with col2:
        st.markdown("### 🌱 Opportunities")
        if opportunities:
            for o in opportunities:
                st.success(o)
        else:
            st.info("No opportunities.")
    with st.expander("Raw JSON"):
        if isinstance(data, dict):
            st.json(data)
        else:
            st.markdown(str(data)[:10000])

# ---------------------------
# UI controls
# ---------------------------
col_control, col_out = st.columns([1, 2])
with col_control:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.session_state["token_scale"] = st.slider("Token scaling", 0.5, 2.0, 1.0, 0.1)
    enable_self_critique = st.checkbox("Enable Self-Critique pass", value=True)
    enable_alt = st.checkbox("Enable Alternative Perspective pass", value=False)
    run_btn = st.button("Run Full Analysis")

with col_out:
    st.markdown("### Progress & outputs")
    if "step_placeholder" not in st.session_state:
        st.session_state["step_placeholder"] = st.empty()

# ---------------------------
# Main flow
# ---------------------------
if uploaded and run_btn:
    # reset session outputs for fresh run
    st.session_state["outputs"] = {}
    st.session_state["timings"] = {}
    st.session_state["final_text"] = ""

    # extract text
    st.session_state["step_placeholder"].info("Step 0: Extracting text from PDF...")
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    full_text = "\n\n".join(pages)
    words = full_text.split()
    HARD_WORD_CAP = 120_000
    if len(words) > HARD_WORD_CAP:
        st.warning(f"Document large ({len(words)} words). Truncating to {HARD_WORD_CAP} words.")
        full_text = " ".join(words[:HARD_WORD_CAP])
    st.session_state["doc_text"] = full_text
    st.session_state["step_placeholder"].success(f"Step 0: Text extracted ({len(full_text.split())} words).")

    # helper for tokens
    def adapt_tokens(base_key):
        base_map = {
            "summary": 6000,
            "insights": 5000,
            "per_insight": 1200,
            "hypotheses": 4500,
            "refinement": 6000,
            "self_critique": 2500,
            "alt": 3500
        }
        base = base_map.get(base_key, 3000)
        return adapt_tokens_for_doc(base)

    # PASS 1: Expanded Summary (JSON)
    st.session_state["step_placeholder"].info("Pass 1: Expanded Summary")
    prompt1 = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Produce a structured JSON summary with keys:
 - key_arguments: list of short strings
 - key_assumptions: list of short strings
 - contradictions: list of short strings (or contradictions_or_gaps)
 - strategic_choices: list of short strings (or implied_strategic_choices)

Be concise but capture enough detail to support insight generation.

Document:
{st.session_state['doc_text']}
"""
    tok = adapt_tokens("summary")
    raw1, t1 = run_claude(prompt1, tok, "Pass 1 — Expanded Summary")
    st.session_state["timings"]["pass1"] = t1
    parsed1 = parse_json_fallback(raw1)
    st.session_state["outputs"]["Expanded Summary Raw"] = raw1
    st.session_state["outputs"]["Expanded Summary"] = parsed1 or {"raw": raw1}

    # Render pass 1
    render_pass1_summary(st.session_state["outputs"]["Expanded Summary"])

    # PASS 2: Candidate Insights (JSON array)
    st.session_state["step_placeholder"].info("Pass 2: Candidate Insights")
    prompt2 = f"""
You are a contrarian thinker trained in Peter Thiel’s Zero-to-One framework.
Using the structured summary below, output STRICT JSON: an array of 6-12 insight objects with fields:
 - id (short unique id)
 - statement (one-line)
 - conventional_assumption (brief)
 - reasoning (brief, referencing summary)
 - contrarian_impact (1-10)
 - plausibility (1-10)
 - actionable_note (optional)

Use only content supported by the summary.

Summary JSON:
{json.dumps(st.session_state['outputs']['Expanded Summary'], ensure_ascii=False)}
"""
    tok = adapt_tokens("insights")
    raw2, t2 = run_claude(prompt2, tok, "Pass 2 — Candidate Insights")
    st.session_state["timings"]["pass2"] = t2
    parsed2 = parse_json_fallback(raw2)
    if parsed2 and isinstance(parsed2, list):
        st.session_state["outputs"]["Candidate Insights"] = parsed2
    else:
        # store raw fallback
        st.session_state["outputs"]["Candidate Insights Raw"] = raw2
        st.session_state["outputs"]["Candidate Insights"] = []  # keep structure

    render_pass2_insights(st.session_state["outputs"].get("Candidate Insights") or st.session_state["outputs"].get("Candidate Insights Raw"))

    # Let user accept insights
    accepted_ids = []
    if st.session_state["outputs"].get("Candidate Insights"):
        st.markdown("### Review & select insights to keep")
        for ins in st.session_state["outputs"]["Candidate Insights"]:
            default = True
            cb = st.checkbox(f"{ins.get('id')}: {ins.get('statement')}", value=default, key=f"accept_{ins.get('id')}")
            if cb:
                accepted_ids.append(ins.get("id"))

    # If none accepted, warn and allow proceed but later passes will be skipped
    if st.session_state["outputs"].get("Candidate Insights") and not accepted_ids:
        st.warning("No insights selected. Select at least one insight to proceed to hypothesis generation.")

    # Optional focused per-insight deepening
    per_insight_results = {}
    if st.session_state["outputs"].get("Candidate Insights") and accepted_ids:
        st.markdown("### Focused deepening for accepted insights")
        for ins in st.session_state["outputs"]["Candidate Insights"]:
            if ins.get("id") not in accepted_ids:
                continue
            pid = ins.get("id")
            prompt_pi = f"""
You are a methodical contrarian analyst. Deepen this insight into a structured JSON object.

Input insight:
{json.dumps(ins, ensure_ascii=False)}

Return JSON:
{{
  "id": "{pid}",
  "expanded_statement": "...",
  "assumptions_challenged": ["..."],
  "plausible_counterarguments": ["..."],
  "evidence_links_to_summary": ["..."],
  "actionable_experiments": ["..."],
  "confidence": 1-10
}}
"""
            tok = adapt_tokens("per_insight")
            raw_pi, tpi = run_claude(prompt_pi, tok, f"Per-Insight {pid}")
            st.session_state["timings"][f"per_{pid}"] = tpi
            ppi = parse_json_fallback(raw_pi)
            per_insight_results[pid] = ppi or {"raw": raw_pi}
            if ppi:
                with st.expander(f"Deepen: {pid}"):
                    st.json(ppi)
            else:
                with st.expander(f"Deepen (raw): {pid}"):
                    st.markdown(raw_pi[:10000])
    st.session_state["outputs"]["Per-Insight"] = per_insight_results

    # PASS 3: Hypotheses (if we have accepted insights)
    if st.session_state["outputs"].get("Candidate Insights") and accepted_ids:
        st.session_state["step_placeholder"].info("Pass 3: Zero-to-One Hypotheses")
        accepted_ins = [i for i in st.session_state["outputs"]["Candidate Insights"] if i.get("id") in accepted_ids]
        prompt3 = f"""
You are a contrarian Zero-to-One strategist. Using the accepted insights below, generate 3-6 high-leverage hypotheses.
Output STRICT JSON array of objects with:
- id
- statement
- linked_insights (list of ids)
- strategic_rationale
- market_potential (short)
- feasibility_score (1-10)
- primary_risk
- first_3_action_steps (list)

Accepted insights:
{json.dumps(accepted_ins, ensure_ascii=False)}
"""
        tok = adapt_tokens("hypotheses")
        raw3, t3 = run_claude(prompt3, tok, "Pass 3 — Hypotheses")
        st.session_state["timings"]["pass3"] = t3
        parsed3 = parse_json_fallback(raw3)
        if parsed3 and isinstance(parsed3, list):
            st.session_state["outputs"]["Hypotheses"] = parsed3
        else:
            st.session_state["outputs"]["Hypotheses Raw"] = raw3
            st.session_state["outputs"]["Hypotheses"] = []
        render_pass3_hypotheses(st.session_state["outputs"].get("Hypotheses") or st.session_state["outputs"].get("Hypotheses Raw"))
    else:
        st.info("Skipping hypothesis generation (no accepted insights).")

    # PASS 4: Refinement & Ranking
    st.session_state["step_placeholder"].info("Pass 4: Refinement & Ranking")
    prompt4 = f"""
You are a contrarian Zero-to-One strategist. Refine and normalize the candidate insights and hypotheses.
Input:
- Summary: {json.dumps(st.session_state['outputs'].get('Expanded Summary', {}), ensure_ascii=False)}
- Candidate Insights (selected): {json.dumps([i for i in st.session_state.get('outputs',{}).get('Candidate Insights',[]) if i.get('id') in accepted_ids], ensure_ascii=False)}
- Per-Insight deepening: {json.dumps(st.session_state['outputs'].get('Per-Insight',{}), ensure_ascii=False)}
- Hypotheses: {json.dumps(st.session_state['outputs'].get('Hypotheses',[]), ensure_ascii=False)}

Produce a VALID JSON object:
{{
  "insights": [{{"id","statement","impact","plausibility","confidence","actionable_note"}}],
  "hypotheses": [{{"id","statement","linked_insights","market_potential","feasibility","primary_risk","top_actions"}}],
  "rankings": {{"top_insights": [], "top_hypotheses": []}},
  "merge_suggestions": []
}}
"""
    tok = adapt_tokens("refinement")
    raw4, t4 = run_claude(prompt4, tok, "Pass 4 — Refinement & Ranking")
    st.session_state["timings"]["pass4"] = t4
    parsed4 = parse_json_fallback(raw4)
    if parsed4:
        st.session_state["outputs"]["Refined"] = parsed4
    else:
        st.session_state["outputs"]["Refined Raw"] = raw4
        st.session_state["outputs"]["Refined"] = {}

    render_pass4_refinement(st.session_state["outputs"].get("Refined") or st.session_state["outputs"].get("Refined Raw"))

    # PASS 5: Self-Critique (optional)
    if enable_self_critique:
        st.session_state["step_placeholder"].info("Pass 5: Self-Critique")
        prompt5 = f"""
You are a critical reviewer. Given the summary, refined outputs, and hypotheses, identify inconsistencies, unsupported claims, hallucinations, or weak reasoning.
Return VALID JSON:
{{ "issues": [...], "suggested_fixes": [...], "confidence": 1-10 }}
"""
        tok = adapt_tokens("self_critique")
        raw5, t5 = run_claude(prompt5, tok, "Pass 5 — Self-Critique")
        st.session_state["timings"]["pass5"] = t5
        parsed5 = parse_json_fallback(raw5)
        st.session_state["outputs"]["Self-Critique"] = parsed5 or {"raw": raw5}
        render_pass5_critique(st.session_state["outputs"]["Self-Critique"])
    else:
        st.info("Self-critique pass disabled.")

    # PASS 6: Alternative Perspective (optional)
    if enable_alt:
        st.session_state["step_placeholder"].info("Pass 6: Alternative Perspective")
        prompt6 = f"""
Re-evaluate top insights and hypotheses from the perspective of a skeptical regulator/competitor/engineer.
Return JSON array of objects: [{{"id":"...","concerns":[...],"opportunities":[...]}}]
"""
        tok = adapt_tokens("alt")
        raw6, t6 = run_claude(prompt6, tok, "Pass 6 — Alternative Perspective")
        st.session_state["timings"]["pass6"] = t6
        parsed6 = parse_json_fallback(raw6)
        st.session_state["outputs"]["Alt Perspective"] = parsed6 or {"raw": raw6}
        render_pass6_altperspective(st.session_state["outputs"]["Alt Perspective"])
    else:
        st.info("Alternative perspective pass disabled.")

    # Final report (structured if available)
    final_report = ""
    if st.session_state["outputs"].get("Refined"):
        final_report = json.dumps(st.session_state["outputs"]["Refined"], ensure_ascii=False, indent=2)
    else:
        final_report = json.dumps(st.session_state["outputs"], ensure_ascii=False, indent=2)
    st.session_state["final_text"] = final_report

    with st.expander("Final Synthesized Report (JSON)"):
        st.code(final_report[:20000])

    # Download buttons (use session state to avoid re-run)
    st.download_button("Download structured JSON", data=final_report, file_name="zero_to_one_report.json", mime="application/json")
    st.download_button("Download human-readable TXT", data=final_report, file_name="zero_to_one_report.txt", mime="text/plain")

    # Show timings
    if st.session_state.get("timings"):
        times = st.session_state["timings"]
        df_times = pd.DataFrame.from_dict(times, orient="index", columns=["seconds"])
        with st.expander("Pass timings"):
            st.table(df_times)

# If user clicked Retry pass (optional extension placeholder)
if "retry_pass" in st.session_state:
    # placeholder: advanced targeted re-run not implemented here
    st.info(f"Retry requested for {st.session_state.pop('retry_pass')}. (Targeted retry not implemented in this build.)")
