import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PASS 1: Expanded Summary
# -----------------------------
def render_pass1_summary(data: dict):
    st.subheader("📖 Expanded Summary")

    st.markdown("### 📌 Key Arguments")
    for arg in data.get("key_arguments", []) or []:
        st.markdown(f"- {arg}")

    st.markdown("### 💡 Key Assumptions")
    for a in data.get("key_assumptions", []) or []:
        st.markdown(f"- {a}")

    st.markdown("### ⚖️ Contradictions")
    for c in data.get("contradictions", []) or []:
        st.markdown(f"- {c}")

    st.markdown("### 🚀 Strategic Choices")
    for sc in data.get("strategic_choices", []) or []:
        st.markdown(f"- {sc}")

    with st.expander("Raw JSON"):
        st.json(data)


# -----------------------------
# PASS 2: Candidate Insights
# -----------------------------
def render_pass2_insights(data: list):
    st.subheader("💡 Candidate Contrarian Insights")

    if not data:
        st.warning("No insights returned.")
        return

    try:
        df = pd.DataFrame(data)
        # Show table
        st.markdown("### 📊 Insights Table")
        cols = [c for c in ["id", "statement", "conventional_assumption", "contrarian_impact", "plausibility"] if c in df.columns]
        st.dataframe(df[cols])

        # Quadrant scatter plot (if both scores available)
        if "plausibility" in df.columns and "contrarian_impact" in df.columns:
            st.markdown("### 🔍 Contrarian Impact vs Plausibility")
            fig, ax = plt.subplots()
            ax.scatter(df["plausibility"], df["contrarian_impact"], alpha=0.7)
            for _, row in df.iterrows():
                label = str(row.get("id", row.name))
                ax.text(row["plausibility"], row["contrarian_impact"], label)
            ax.set_xlabel("Plausibility")
            ax.set_ylabel("Contrarian Impact")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render insights table/chart: {e}")

    with st.expander("Raw JSON"):
        st.json(data)


# -----------------------------
# PASS 3: Hypotheses
# -----------------------------
def render_pass3_hypotheses(data: list):
    st.subheader("🧪 Zero-to-One Hypotheses")

    if not data:
        st.warning("No hypotheses returned.")
        return

    for hyp in data:
        st.markdown(f"### {hyp.get('id', '?')}. {hyp.get('statement', 'No statement')}")
        feas = hyp.get("feasibility_score", 0)
        st.progress(min(max(feas, 0), 10) / 10)  # clamp between 0–10
        st.caption(f"Feasibility: {feas}/10")
        st.caption(f"Risks: {', '.join(hyp.get('risks', [])) or 'None'}")
        st.caption(f"Linked Insights: {', '.join(map(str, hyp.get('linked_insights', []))) or 'None'}")

    try:
        df = pd.DataFrame(data)
        st.markdown("### 📊 Hypotheses Overview")
        cols = [c for c in ["id", "statement", "feasibility_score", "risks", "linked_insights", "market_potential"] if c in df.columns]
        st.dataframe(df[cols])

        if "feasibility_score" in df.columns and "market_potential" in df.columns:
            st.markdown("### 📈 Feasibility vs Market Potential")
            fig, ax = plt.subplots()
            ax.scatter(df["feasibility_score"], df["market_potential"], alpha=0.7)
            for _, row in df.iterrows():
                label = str(row.get("id", row.name))
                ax.text(row["feasibility_score"], row["market_potential"], label)
            ax.set_xlabel("Feasibility")
            ax.set_ylabel("Market Potential")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not render hypotheses table/chart: {e}")

    with st.expander("Raw JSON"):
        st.json(data)


# -----------------------------
# PASS 4: Refinement & Ranking
# -----------------------------
def render_pass4_refinement(data: list):
    st.subheader("🏆 Refined & Ranked Insights/Hypotheses")

    if not data:
        st.warning("No refined outputs returned.")
        return

    try:
        # Highlight Top 3 by rank if present
        st.markdown("### 🏅 Top 3 Highlights")
        ranked = [d for d in data if d.get("rank") is not None]
        top = sorted(ranked, key=lambda x: x.get("rank", 999))[:3]
        if top:
            for item in top:
                st.success(f"**{item.get('refined_statement','(no statement)')}** (Rank {item.get('rank')})")

        # Side-by-side comparison
        st.markdown("### 🔄 Before/After Refinement")
        for item in data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original:**\n{item.get('original_statement', '(missing)')}")
            with col2:
                st.markdown(f"**Refined:**\n{item.get('refined_statement', '(missing)')}")
    except Exception as e:
        st.error(f"Could not render refinement stage: {e}")

    with st.expander("Raw JSON"):
        st.json(data)


# -----------------------------
# PASS 5: Self-Critique
# -----------------------------
def render_pass5_critique(data: dict):
    st.subheader("🧐 Self-Critique")

    for issue in data.get("issues", []) or []:
        st.error(f"⚠️ {issue}")

    for fix in data.get("suggested_fixes", []) or []:
        st.success(f"✅ {fix}")

    if not data.get("issues") and not data.get("suggested_fixes"):
        st.info("No critique provided.")

    with st.expander("Raw JSON"):
        st.json(data)


# -----------------------------
# PASS 6: Alternative Perspective
# -----------------------------
def render_pass6_altperspective(data: dict):
    st.subheader("🔄 Alternative Perspective")

    concerns = data.get("concerns", []) or []
    opportunities = data.get("opportunities", []) or []

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❗ Concerns")
        if concerns:
            for c in concerns:
                st.error(c)
        else:
            st.info("No concerns listed.")

    with col2:
        st.markdown("### 🌱 Opportunities")
        if opportunities:
            for o in opportunities:
                st.success(o)
        else:
            st.info("No opportunities listed.")

    with st.expander("Raw JSON"):
        st.json(data)
