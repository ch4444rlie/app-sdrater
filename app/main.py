import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib  # Required for background_gradient

from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")


def evaluate_quality_lite(real, synth):
    """SDV-inspired lightweight quality evaluator with improved mathematical rigor."""
    scores = {"Column Shapes": {}, "Column Pair Trends": {}}
    issues = []  # Track data quality issues

    # --- Column Shapes ---
    for col in real.columns:
        if real[col].dtype.kind in 'bifc':  # numeric
            # Check for range violations
            real_min, real_max = real[col].min(), real[col].max()
            synth_min, synth_max = synth[col].min(), synth[col].max()
            
            if synth_min < real_min or synth_max > real_max:
                issues.append({
                    "Type": "Range Violation",
                    "Column": col,
                    "Details": f"Synth range [{synth_min:.2f}, {synth_max:.2f}] exceeds real [{real_min:.2f}, {real_max:.2f}]"
                })
            
            ks_stat, _ = ks_2samp(real[col].dropna(), synth[col].dropna())
            score = max(0.0, 1.0 - ks_stat)
        else:  # categorical
            real_counts = real[col].value_counts(normalize=True, dropna=False)
            synth_counts = synth[col].value_counts(normalize=True, dropna=False)
            
            # Check for novel categories
            real_cats = set(real_counts.index)
            synth_cats = set(synth_counts.index)
            novel_cats = synth_cats - real_cats
            missing_cats = real_cats - synth_cats
            
            if novel_cats:
                issues.append({
                    "Type": "Novel Categories",
                    "Column": col,
                    "Details": f"Synthetic data contains {len(novel_cats)} categories not in real data"
                })
            
            if missing_cats:
                # Only flag if missing categories represent >1% of real data
                missing_proportion = sum(real_counts.get(c, 0) for c in missing_cats)
                if missing_proportion > 0.01:
                    issues.append({
                        "Type": "Missing Categories",
                        "Column": col,
                        "Details": f"{len(missing_cats)} categories from real data are missing ({missing_proportion:.1%} of data)"
                    })
            
            all_cats = real_cats | synth_cats
            tvd = sum(abs(real_counts.get(c, 0) - synth_counts.get(c, 0)) for c in all_cats) * 0.5
            score = max(0.0, 1.0 - tvd)
        
        # Check null rate differences
        real_null_rate = real[col].isna().mean()
        synth_null_rate = synth[col].isna().mean()
        null_diff = abs(real_null_rate - synth_null_rate)
        
        if null_diff > 0.05:  # More than 5% difference
            issues.append({
                "Type": "Null Rate Mismatch",
                "Column": col,
                "Details": f"Real: {real_null_rate:.1%} nulls, Synth: {synth_null_rate:.1%} nulls (Δ {null_diff:.1%})"
            })
        
        scores["Column Shapes"][col] = round(score, 4)

    # --- Column Pair Trends (IMPROVED) ---
    numeric_cols = real.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) > 1:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                r_corr = real[[col1, col2]].corr().iloc[0, 1]
                s_corr = synth[[col1, col2]].corr().iloc[0, 1]
                
                if pd.isna(r_corr) or pd.isna(s_corr):
                    score = 0.0
                else:
                    # Non-linear penalty: penalizes large deviations more heavily
                    corr_diff = abs(r_corr - s_corr)
                    score = 1.0 - (corr_diff ** 1.5) / 2.0
                    score = max(0.0, round(score, 4))
                
                scores["Column Pair Trends"][f"{col1} vs {col2}"] = score

    # --- Overall Score ---
    shape_avg = sum(scores["Column Shapes"].values()) / len(scores["Column Shapes"]) if scores["Column Shapes"] else 1.0
    pair_avg = sum(scores["Column Pair Trends"].values()) / len(scores["Column Pair Trends"]) if scores["Column Pair Trends"] else 1.0
    overall = 0.6 * shape_avg + 0.4 * pair_avg

    class Report:
        def __init__(self, score, details, issues):
            self.score = score
            self.details = details
            self.issues = issues
        def get_score(self):
            return round(self.score, 3)
        def get_details(self, name):
            items = self.details.get(name, {})
            return pd.DataFrame([{"Metric": k, "Score": round(v, 3)} for k, v in items.items()])
        def get_issues(self):
            return pd.DataFrame(self.issues) if self.issues else pd.DataFrame(columns=["Type", "Column", "Details"])

    return Report(overall, scores, issues)


def get_score_status(score):
    if score >= 0.95:
        return "Excellent", "#16a085"
    elif score >= 0.85:
        return "Great", "#28a745"
    elif score >= 0.70:
        return "Good", "#ffc107"
    elif score >= 0.55:
        return "Fair", "#ff9800"
    else:
        return "Needs Work", "#dc3545"


def get_interpretation(score):
    if score >= 0.95:
        return "Outstanding statistical match. Suitable for most analytical and ML applications."
    elif score >= 0.85:
        return "High quality with minor deviations. Safe for analytics, reporting, and most ML use cases."
    elif score >= 0.70:
        return "Acceptable quality. Suitable for general analytics and testing, but verify for sensitive ML applications."
    elif score >= 0.55:
        return "Moderate quality issues. Safe for development/testing environments, but not recommended for production ML."
    else:
        return "Significant quality issues detected. Review flagged columns carefully before using this data."


# ============================================
# Streamlit UI
# ============================================

st.set_page_config(page_title="Synthetic Data Quality Rater", layout="wide")
st.title("Synthetic Data Quality Rater")

st.markdown("""
Upload your **real** and **synthetic** CSV files to get an honest, SDV-style quality assessment.
""")

col1, col2 = st.columns(2)
with col1:
    real_file = st.file_uploader("Original (real) dataset", type=["csv"])
with col2:
    synth_file = st.file_uploader("Synthetic dataset", type=["csv"])

if real_file and synth_file:
    real = pd.read_csv(real_file)
    synth = pd.read_csv(synth_file)

    if real.shape[1] != synth.shape[1]:
        st.error("Both files must have the same number of columns.")
        st.stop()
    if not real.columns.equals(synth.columns):
        st.warning("Column names differ — aligning by position.")
        synth.columns = real.columns


    id_keywords = ['id', 'customer', 'patient', 'account', 'user', 'record', 'key', '_id', 'index']
    id_columns = [col for col in real.columns if any(kw in col.lower() for kw in id_keywords)]

    
    real_full = real.copy()
    synth_full = synth.copy()

    if id_columns:
        st.info(f"Ignoring ID-like columns from scoring (expected to differ): {', '.join(id_columns)}")
        real = real.drop(columns=id_columns)
        synth = synth.drop(columns=id_columns)


    with st.spinner("Evaluating synthetic data quality..."):
        report = evaluate_quality_lite(real, synth)

    overall = report.get_score()
    shapes_df = report.get_details("Column Shapes")
    pairs_df = report.get_details("Column Pair Trends")
    issues_df = report.get_issues()

    # Metrics
    cols_attention = len(shapes_df[shapes_df["Score"] < 0.70])
    strong_corr_count = len(pairs_df[pairs_df["Score"] >= 0.80]) if not pairs_df.empty else 0
    total_pairs = len(pairs_df)

    # Header
    st.success(f"**Overall Quality Score: {overall:.3f}**")
    st.info(get_interpretation(overall))

    # Overfitting / leakage warning
    if overall >= 0.99:
        st.warning("⚠️ Score ≥ 0.99 — Datasets are essentially identical. Possible data leakage or overfitting if this is meant to be synthetic!")

    # Quick stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Columns Analyzed", len(shapes_df))
    c2.metric("Columns Need Attention", cols_attention,
              delta="Review" if cols_attention > 0 else None, delta_color="inverse")
    c3.metric("Strong Correlations Preserved", f"{strong_corr_count}/{total_pairs}" if total_pairs else "N/A")

    # Data Issues Section
    if not issues_df.empty:
        st.divider()
        st.subheader("⚠️ Data Quality Issues Detected")
        
        issue_counts = issues_df["Type"].value_counts()
        st.markdown(f"Found **{len(issues_df)}** issues across {len(issue_counts)} categories:")
        
        with st.expander(f"View All {len(issues_df)} Issues", expanded=len(issues_df) <= 10):
            st.dataframe(issues_df, use_container_width=True)

    st.divider()
    st.subheader("Distribution Comparison")

    numeric_cols = real.select_dtypes(include="number").columns.tolist()
    low_shape = shapes_df[shapes_df["Score"] < 0.70]["Metric"].tolist()

    options = (["[Needs Attention] " + c for c in low_shape] + real_full.columns.tolist()) if low_shape else real_full.columns.tolist()
    selected = st.selectbox("Select column to inspect", options, index=0)
    col_name = selected.replace("[Needs Attention] ", "")
        # Handle score safely — ID columns have no score
    if col_name in id_columns:
        score = None
        display_score = "Excluded"
        status_text = "ID Column"
    else:
        score_row = shapes_df[shapes_df["Metric"] == col_name]
        if score_row.empty:
            score = None
            display_score = "N/A"
            status_text = "Not Scored"
        else:
            score = score_row["Score"].values[0]
            display_score = f"{score:.3f}"
            status_text, _ = get_score_status(score)

    st.markdown(f"**{col_name}** — Score: **{display_score}** ({status_text})")
    
    if col_name in id_columns:
        st.caption("This is an identifier column — automatically excluded from quality scoring (expected to differ).")

    # Plot
    if col_name in real_full.select_dtypes(include="number").columns:
        df_plot = pd.concat([
            real_full[[col_name]].assign(Source="Real"),
            synth_full[[col_name]].assign(Source="Synthetic")
        ])
        fig = px.histogram(df_plot, x=col_name, color="Source", barmode="overlay",
                           opacity=0.7, marginal="box",
                           color_discrete_map={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    else:
        top15 = pd.concat([real_full[col_name], synth_full[col_name]]).value_counts().head(15).index
        plot_df = pd.DataFrame({
            "Category": list(top15) * 2,
            "Count": [real_full[col_name].value_counts().get(c, 0) for c in top15] + 
                     [synth_full[col_name].value_counts().get(c, 0) for c in top15],
            "Source": ["Real"] * len(top15) + ["Synthetic"] * len(top15)
        })
        fig = px.bar(plot_df, x="Category", y="Count", color="Source", barmode="group",
                     color_discrete_map={"Real": "#1f77b4", "Synthetic": "#ff7f0e"})
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    if numeric_cols:
        with st.expander("Summary Statistics (first 8 numeric columns)"):
            summary = []
            for c in numeric_cols[:8]:
                summary.append({
                    "Column": c,
                    "Real Mean": f"{real[c].mean():.3f}",
                    "Real Std": f"{real[c].std():.3f}",
                    "Synth Mean": f"{synth[c].mean():.3f}",
                    "Synth Std": f"{synth[c].std():.3f}",
                })
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

    st.divider()

    with st.expander("Detailed Quality Scores", expanded=False):
        st.markdown("### Column Shapes")
        shape_disp = shapes_df.copy()
        shape_disp["Status"] = shape_disp["Score"].apply(lambda x: get_score_status(x)[0])
        st.dataframe(
            shape_disp.sort_values("Score")
            .style.format({"Score": "{:.3f}"})
            .background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True
        )

        if not pairs_df.empty:
            st.markdown("### Column Pair Correlations")
            pair_disp = pairs_df.copy()
            pair_disp["Status"] = pair_disp["Score"].apply(lambda x: get_score_status(x)[0])
            st.dataframe(
                pair_disp.sort_values("Score")
                .style.format({"Score": "{:.3f}"})
                .background_gradient(subset=["Score"], cmap="RdYlGn"),
                use_container_width=True
            )

            with st.expander("How to improve low correlation scores"):
                st.markdown("""
                **Low scores here mean important relationships were lost during synthesis.**

                **Recommended actions:**
                - Switch to **TVAE** (much better at preserving correlations than CTGAN)
                - Train CTGAN for **50,000+ epochs** with lower learning rate
                - Try modern models: **GReaT**, **REaLTabFormer**, **TabDDPM**, or **STASY**
                - Use **conditional generation** (e.g. generate `account_balance` conditioned on `income`)
                - Apply post-processing with Gaussian copulas
                """)

    # ============================================
    # DOCUMENTATION SECTION
    # ============================================
    st.divider()
    st.subheader("📚 Understanding Your Score")
    
    with st.expander("How Scores Are Calculated", expanded=False):
        st.markdown("""
        ### Scoring Methodology
        
        Your overall quality score is calculated using two main components:
        
        #### 1. Column Shapes (60% weight)
        Measures how well individual column distributions match between real and synthetic data.
        
        **For numeric columns:**
        - Uses the **Kolmogorov-Smirnov (KS) test** to compare distributions
        - KS statistic measures the maximum distance between cumulative distribution functions
        - Score = 1.0 - KS_statistic (range: 0 to 1)
        - A score of 1.0 means distributions are identical; 0.0 means completely different
        
        **For categorical columns:**
        - Uses **Total Variation Distance (TVD)** to compare category frequencies
        - TVD = 0.5 × sum of absolute differences in proportions
        - Score = 1.0 - TVD
        - Penalizes both missing categories and incorrect proportions
        
        #### 2. Column Pair Trends (40% weight)
        Measures how well pairwise correlations are preserved between numeric columns.
        
        **Correlation preservation:**
        - Compares Pearson correlations between all numeric column pairs
        - Score = 1.0 - (|correlation_difference|^1.5) / 2.0
        - This formula applies a **non-linear penalty** that punishes large deviations more heavily than small ones
        - Example: A correlation difference of 0.2 yields a score of ~0.94, while a difference of 0.5 yields ~0.82
        
        #### Overall Score Formula
        `Overall = 0.6 × (average column shape score) + 0.4 × (average correlation score)`
        
        The 60/40 weighting prioritizes getting individual distributions right while still accounting for relationships.
        
        ### Score Interpretation Guidelines
        
        - **≥ 0.95** — Excellent: Outstanding statistical match
        - **0.85 - 0.94** — Great: High quality, minor deviations only
        - **0.70 - 0.84** — Good: Acceptable for most analytics uses
        - **0.55 - 0.69** — Fair: Suitable for testing, verify before production
        - **< 0.55** — Needs Work: Significant quality issues present
        """)
    
    with st.expander("⚠️ Important Limitations - What This Tool CANNOT Detect", expanded=False):
        st.markdown("""
        ### Critical Disclaimer
        
        This tool provides a **lightweight statistical assessment** focused on basic distributional properties. 
        It has significant limitations and should be supplemented with domain expertise and downstream testing.
        
        ---
        
        #### ❌ What We DON'T Evaluate:
        
        **1. Complex Multivariate Relationships**
        - We only check **pairwise correlations** (2 columns at a time)
        - Cannot detect if **3+ column interactions** are broken
        - **Example:** Real data might have a rule like "if Age > 65 AND Income < 30k then Healthcare_Costs are high" 
          - Even if Age, Income, and Healthcare_Costs individually look fine, and pairwise correlations are preserved, 
            this three-way interaction could be completely wrong
        
        **2. Non-Linear Relationships**
        - Pearson correlation only measures **linear relationships**
        - Curved, exponential, polynomial, or other non-linear patterns may be poorly captured
        - **Example:** If Salary increases exponentially with Years_Experience in real data, but linearly in synthetic data, 
          we might still show a high correlation score
        
        **3. Conditional Distributions**
        - Don't verify if relationships hold **within subgroups**
        - **Example:** The correlation between Height and Weight might be 0.7 for males and 0.6 for females in real data, 
          but 0.65 for both genders in synthetic data — we'd report good overall correlation without catching this issue
        
        **4. Time Series & Sequential Properties**
        - No **autocorrelation** analysis
        - No **trend or seasonality** preservation checks
        - No verification of **temporal ordering** or dependencies
        - **Example:** Stock prices might have realistic individual day values but impossible day-to-day changes
        
        **5. Rare Events & Tail Behavior**
        - KS test may not reliably detect poor synthesis of **extreme values** or **outliers**
        - **Rare combinations** of values might be completely implausible
        - **Example:** Synthetic data might have a 95-year-old pregnant woman, or a $5M salary for a janitor position
        
        **6. Domain-Specific Constraints & Business Rules**
        - Cannot verify **logical consistency** (e.g., "end_date must be after start_date")
        - No **referential integrity** validation (e.g., every order must have a valid customer_id)
        - No detection of **impossible combinations** specific to your domain
        - **Example:** A patient record with "never_smoked = True" but also "pack_years = 20"
        
        **7. Privacy & Re-identification Risk**
        - This tool does **NOT assess privacy preservation** at all
        - **High quality scores could indicate memorization** (which is a privacy risk!)
        - No measurement of k-anonymity, differential privacy, or re-identification risk
        - **Warning:** A score of 0.99+ often means your synthetic data is too similar to the real data
        
        **8. Causality & Causal Relationships**
        - Correlation ≠ causation, and we only measure correlation
        - Cannot verify if **causal mechanisms** are preserved
        - **Example:** In real data, Education → Income → Health. In synthetic data, the correlation structure 
          might look similar but the causal flow could be completely broken
        
        ---
        
        #### ✅ What You Should Do:
        
        **Always Supplement With:**
        - **Domain expert review** - Have people who understand the data manually inspect samples
        - **Downstream task testing** - Train ML models on synthetic data and evaluate on real test data (TSTR)
        - **Business rule validation** - Check that domain-specific constraints hold
        - **Privacy auditing** - Use specialized tools to assess re-identification risk
        - **Manual inspection** - Look at actual synthetic records for plausibility and edge cases
        
        **When to Seek Advanced Evaluation:**
        - High-stakes applications (healthcare, finance, legal, government)
        - Complex datasets with many interdependencies (>20 columns)
        - Data with known business rules or strict constraints
        - Privacy-sensitive applications requiring formal guarantees
        - When synthetic data will be shared publicly or sold
        - ML applications where model performance is critical
        
        **Recommended Advanced Tools:**
        - **SDMetrics** (by MIT) - Comprehensive evaluation suite
        - **TSTR (Train on Synthetic, Test on Real)** - ML-based validation
        - **Privacy auditing tools** - For re-identification risk assessment
        - **Domain-specific validators** - Custom tests for your use case
        
        ---
        
        ### Bottom Line
        
        **This tool gives you a useful starting point**, but a high score here does NOT guarantee your synthetic 
        data is production-ready. Always validate with domain experts and test in your actual use case before 
        relying on synthetic data for important decisions.
        """)
    
    with st.expander("Why These Specific Metrics?", expanded=False):
        st.markdown("""
        ### Design Rationale
        
        **Why Kolmogorov-Smirnov (KS) Test for numeric columns?**
        - Non-parametric: works for any distribution shape (normal, skewed, multimodal, etc.)
        - Sensitive to differences in both location (mean/median) and shape (spread, skewness)
        - Well-established statistical test with clear interpretation
        - Fast to compute even on large datasets
        
        **Why Total Variation Distance (TVD) for categorical columns?**
        - Simple, interpretable measure of distribution difference
        - Directly measures how much probability mass needs to be moved
        - Naturally handles any number of categories
        - Symmetric: TVD(A,B) = TVD(B,A)
        
        **Why Pearson correlation for relationships?**
        - Industry standard for measuring linear relationships
        - Easily interpretable (values from -1 to +1)
        - Fast to compute for all column pairs
        - Despite limitations (only linear), it catches most common synthesis failures
        
        **Why 60/40 weighting (shapes vs. correlations)?**
        - Individual distributions are more fundamental than pairwise relationships
        - Many synthesis methods get distributions right but fail on correlations
        - In practice, distribution mismatches are often more problematic than correlation differences
        - Adjustable if your use case prioritizes relationships differently
        
        **Why non-linear penalty for correlation differences?**
        - Small correlation errors (0.1-0.2) are often acceptable
        - Large correlation errors (0.5+) indicate serious synthesis problems
        - The 1.5 exponent provides a balanced middle ground between linear and quadratic penalties
        - Prevents overly harsh scoring while still penalizing major deviations
        """)

else:
    st.info("Please upload both real and synthetic CSV files to begin.")
    with st.expander("Tips for best results"):
        st.markdown("""
        - Use identical column names and order
        - At least 1,000 rows recommended for stable scores
        - Same data types in both files
        - Remove identifier columns (IDs, names) before upload
        - Ensure no data leakage between real and synthetic datasets
        """)