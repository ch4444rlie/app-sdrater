import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
from scipy.stats import ks_2samp
import json

# ============================================
# QUALITY EVALUATOR
# ============================================

def evaluate_quality_lite(real, synth):
    scores = {"Column Shapes": {}, "Column Pair Trends": {}}
    issues = []

    for col in real.columns:
        if real[col].dtype.kind in 'bifc':  # numeric
            real_min, real_max = real[col].min(), real[col].max()
            synth_min, synth_max = synth[col].min(), synth[col].max()
            if synth_min < real_min or synth_max > real_max:
                issues.append({
                    "Type": "Range Violation",
                    "Column": col,
                    "Details": f"Synth range [{synth_min:.2f}, {synth_max:.2f}] exceeds real [{real_min:.2f}, {real_max:.2f}]",
                    "Fix": "Clip synthetic values or retrain with min/max constraints"
                })
            ks_stat, _ = ks_2samp(real[col].dropna(), synth[col].dropna())
            score = max(0.0, 1.0 - ks_stat)
        else:  # categorical
            real_counts = real[col].value_counts(normalize=True, dropna=False)
            synth_counts = synth[col].value_counts(normalize=True, dropna=False)
            real_cats = set(real_counts.index)
            synth_cats = set(synth_counts.index)
            novel_cats = synth_cats - real_cats
            missing_cats = real_cats - synth_cats

            if novel_cats:
                issues.append({
                    "Type": "Novel Categories",
                    "Column": col,
                    "Details": f"Synthetic data contains {len(novel_cats)} new categories",
                    "Fix": "Use TVAE or increase discriminator training"
                })
            if missing_cats:
                missing_prop = sum(real_counts.get(c, 0) for c in missing_cats)
                if missing_prop > 0.01:
                    issues.append({
                        "Type": "Missing Categories",
                        "Column": col,
                        "Details": f"{len(missing_cats)} real categories missing ({missing_prop:.1%} of data)",
                        "Fix": "Increase epochs or use conditional sampling"
                    })
            all_cats = real_cats | synth_cats
            tvd = sum(abs(real_counts.get(c, 0) - synth_counts.get(c, 0)) for c in all_cats) * 0.5
            score = max(0.0, 1.0 - tvd)

        # Null rate check
        null_diff = abs(real[col].isna().mean() - synth[col].isna().mean())
        if null_diff > 0.05:
            issues.append({
                "Type": "Null Rate Mismatch",
                "Column": col,
                "Details": f"Real: {real[col].isna().mean():.1%}, Synth: {synth[col].isna().mean():.1%} nulls",
                "Fix": "Adjust null imputation or generator settings"
            })

        scores["Column Shapes"][col] = round(score, 4)

    # Correlations
    numeric_cols = real.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                r_corr = real[[col1, col2]].corr().iloc[0, 1]
                s_corr = synth[[col1, col2]].corr().iloc[0, 1]
                if pd.isna(r_corr) or pd.isna(s_corr):
                    score = 0.0
                else:
                    diff = abs(r_corr - s_corr)
                    score = max(0.0, 1.0 - (diff ** 1.5) / 2.0)
                    score = round(score, 4)
                    if score < 0.7:
                        issues.append({
                            "Type": "Correlation Loss",
                            "Column": f"{col1} vs {col2}",
                            "Details": f"Real: {r_corr:.3f} → Synth: {s_corr:.3f}",
                            "Fix": "Use TVAE, GReaT, or train 10k+ epochs"
                        })
                scores["Column Pair Trends"][f"{col1} vs {col2}"] = score

    shape_avg = sum(scores["Column Shapes"].values()) / len(scores["Column Shapes"]) if scores["Column Shapes"] else 1.0
    pair_avg = sum(scores["Column Pair Trends"].values()) / len(scores["Column Pair Trends"]) if scores["Column Pair Trends"] else 1.0
    overall = 0.6 * shape_avg + 0.4 * pair_avg

    class Report:
        def __init__(self, score, details, issues):
            self.score = score
            self.details = details
            self.issues = issues
        def get_score(self): return round(self.score, 3)
        def get_details(self, name):
            return pd.DataFrame([{"Metric": k, "Score": round(v, 3)} for k, v in self.details.get(name, {}).items()])
        def get_issues(self):
            return pd.DataFrame(self.issues) if self.issues else pd.DataFrame(columns=["Type", "Column", "Details", "Fix"])

    return Report(overall, scores, issues)


def get_score_status(score):
    if score >= 0.95: return "Excellent", "#16a085"
    elif score >= 0.85: return "Great", "#28a745"
    elif score >= 0.70: return "Good", "#ffc107"
    elif score >= 0.55: return "Fair", "#ff9800"
    else: return "Needs Work", "#dc3545"


def get_interpretation(score):
    if score >= 0.95: return "Outstanding statistical match. Ready for production analytics and ML."
    elif score >= 0.85: return "High quality. Safe for analytics, reporting, and most ML use cases."
    elif score >= 0.70: return "Acceptable quality. Good for testing and general analytics."
    elif score >= 0.55: return "Moderate quality. Safe for development/testing, verify before production."
    else: return "Significant quality issues. Review flagged columns before using."


# ============================================
# STREAMLIT APP
# ============================================

st.set_page_config(page_title="Synthetic Data Quality Rater", layout="wide")
st.title("Synthetic Data Quality Rater")

st.markdown("Upload your **real** and **synthetic** CSV files for an instant, honest quality report.")

col1, col2 = st.columns(2)
with col1:
    real_file = st.file_uploader("Original (real) dataset", type=["csv"])
with col2:
    synth_file = st.file_uploader("Synthetic dataset", type=["csv"])

if real_file and synth_file:
    try:
        real_full = pd.read_csv(real_file)
        synth_full = pd.read_csv(synth_file)
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    if real_full.empty or synth_full.empty:
        st.error("One or both files are empty.")
        st.stop()

    if real_full.shape[1] != synth_full.shape[1]:
        st.error("Files must have the same number of columns.")
        st.stop()

    if not real_full.columns.equals(synth_full.columns):
        st.warning("Column names differ — aligning by position.")
        synth_full.columns = real_full.columns

    # === SMART ID EXCLUSION ===
    id_keywords = ['id', 'customer', 'patient', 'account', 'user', 'record', 'key', '_id', 'index']
    id_columns = [c for c in real_full.columns if any(kw in c.lower() for kw in id_keywords)]

    real = real_full.copy()
    synth = synth_full.copy()

    if id_columns:
        st.info(f"Ignoring ID-like columns in scoring: {', '.join(id_columns)}")
        real = real.drop(columns=id_columns, errors='ignore')
        synth = synth.drop(columns=id_columns, errors='ignore')

    with st.spinner("Analyzing quality..."):
        report = evaluate_quality_lite(real, synth)

    overall = report.get_score()
    shapes_df = report.get_details("Column Shapes")
    pairs_df = report.get_details("Column Pair Trends")
    issues_df = report.get_issues()

    # Header
    status_text, status_color = get_score_status(overall)
    st.markdown(f"<h2 style='color: {status_color};'>Overall Quality Score: {overall:.3f} — {status_text}</h2>", unsafe_allow_html=True)
    st.info(get_interpretation(overall))

    if overall >= 0.97:
        st.warning("Score ≥ 0.97 — Datasets are nearly identical. Possible data leakage or memorization!")

    # Quick metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Columns Scored", len(shapes_df))
    c2.metric("Issues Found", len(issues_df), delta="Review" if len(issues_df)>0 else None, delta_color="inverse")
    strong = len(pairs_df[pairs_df["Score"] >= 0.80]) if not pairs_df.empty else 0
    total = len(pairs_df)
    c3.metric("Strong Correlations", f"{strong}/{total}" if total else "N/A")

    # === ISSUES (main focus) ===
    if not issues_df.empty:
        st.divider()
        st.subheader("Issues That Need Attention")
        for typ in issues_df["Type"].unique():
            subset = issues_df[issues_df["Type"] == typ]
            with st.expander(f"{typ} ({len(subset)})", expanded=True):
                for _, row in subset.iterrows():
                    st.markdown(f"**{row['Column']}**")
                    st.write(row["Details"])
                    st.info(f"Fix: {row['Fix']}")
                    st.markdown("---")
    else:
        st.success("No major issues detected!")

    st.divider()
    st.subheader("Distribution Inspector")

    low_shape = shapes_df[shapes_df["Score"] < 0.70]["Metric"].tolist()
    options = (["[Needs Attention] " + c for c in low_shape] + real_full.columns.tolist()) if low_shape else real_full.columns.tolist()
    selected = st.selectbox("Select column", options, index=0)
    col_name = selected.replace("[Needs Attention] ", "")

    # Safe score display
    if col_name in id_columns:
        st.markdown(f"<h4 style='color: #666'>{col_name} — Excluded (ID column)</h4>", unsafe_allow_html=True)
        st.caption("Identifier columns are ignored in scoring — expected to differ.")
    else:
        score = shapes_df.loc[shapes_df["Metric"] == col_name, "Score"].iloc[0]
        status_text, color = get_score_status(score)
        st.markdown(f"<h4 style='color: {color};'>{col_name} — Score: {score:.3f} ({status_text})</h4>", unsafe_allow_html=True)

    # Plot using full data
    if col_name in real_full.select_dtypes(include="number").columns:
        dfp = pd.concat([real_full[[col_name]].assign(Source="Real"), synth_full[[col_name]].assign(Source="Synthetic")])
        fig = px.histogram(dfp, x=col_name, color="Source", barmode="overlay", opacity=0.7, marginal="box",
                           color_discrete_map={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    else:
        top15 = pd.concat([real_full[col_name], synth_full[col_name]]).value_counts().head(15).index
        dfp = pd.DataFrame({
            "Category": list(top15)*2,
            "Count": [real_full[col_name].value_counts().get(c,0) for c in top15] + [synth_full[col_name].value_counts().get(c,0) for c in top15],
            "Source": ["Real"]*len(top15) + ["Synthetic"]*len(top15)
        })
        fig = px.bar(dfp, x="Category", y="Count", color="Source", barmode="group",
                     color_discrete_map={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    with st.expander("Detailed Scores", expanded=False):
        if not shapes_df.empty:
            st.markdown("### Column Shapes")
            st.dataframe(shapes_df.style.format({"Score": "{:.3f}"}).background_gradient(subset=["Score"], cmap="RdYlGn"), use_container_width=True)
        if not pairs_df.empty:
            st.markdown("### Correlations")
            st.dataframe(pairs_df.style.format({"Score": "{:.3f}"}).background_gradient(subset=["Score"], cmap="RdYlGn"), use_container_width=True)

    # Export
    report_json = {
        "overall_score": overall,
        "status": status_text,
        "issues": issues_df.to_dict('records'),
        "shapes": shapes_df.to_dict('records'),
        "correlations": pairs_df.to_dict('records')
    }
    st.download_button("Download Report (JSON)", data=json.dumps(report_json, indent=2), file_name="synthetic_quality_report.json", mime="application/json")

else:
    st.info("Upload both real and synthetic CSV files to begin.")
    st.markdown("**Tips:** Same columns • ≥1,000 rows • Remove IDs • No leakage")