# Synthetic Data Quality Rater

## Overview

This is a lightweight web application built with **Streamlit** for evaluating the quality of synthetic datasets compared to their real counterparts. It provides a quick, statistical assessment of how well synthetic data mimics the original in terms of distributions, correlations, and other key properties.



### Key Features
- Upload real and synthetic CSV files for instant analysis.
- Computes an overall quality score (0–1 scale).
- Flags common issues like range violations, novel/missing categories, correlation losses, and null rate mismatches.
- Interactive distribution visualizations (histograms or bar charts).
- Detailed issue explanations with suggested fixes.
- Exportable JSON report.

The app is fully containerized with Docker for easy local or server deployment.

## Installation & Running

### Prerequisites
- Docker & Docker Compose (recommended)
- Or Python 3.11+ for local run

### With Docker (Recommended)
```bash
git clone https://github.com/ch4444rlie/app-sdrater.git
cd app-sdrater
docker-compose up -build
```
Open http://localhost:8501 in your browser.



## Usage

Upload your real and synthetic CSV files.
The app instantly shows:
Overall quality score + status (Excellent, Great, etc.)
Interpretation of suitability for analytics/ML
Key metrics and flagged issues with fix suggestions
Interactive column distribution inspector
Detailed tables for univariate and bivariate scores

Download the JSON report for records.

Tips (shown in-app):

Datasets should have the same columns and order
≥1,000 rows recommended for stable statistics
Remove explicit ID columns before upload
Avoid data leakage (real data mixed into synthetic training)

## How the Scores Are Calculated
The evaluation is performed in evaluate_quality_lite() — a fast, model-free statistical comparison.
# Preprocessing
Aligns columns by position if names differ

Automatically excludes ID-like columns (containing keywords like 'id', 'customer', 'key', etc.)

# Column Shapes (Univariate) — 60% of overall score

Numeric: KS test → score = max(0, 1 - KS_statistic)

Categorical: Total Variation Distance → score = max(0, 1 - TVD)

Flags: range violations, novel/missing categories, null rate mismatch (>5%)

# Column Pair Trends (Bivariate) — 40% of overall score

Pearson correlation difference on numeric pairs

Penalty emphasizes larger deviations

Flags significant correlation loss

Overall score = 0.6 × avg(univariate) + 0.4 × avg(bivariate)
Score thresholds:

≥0.95 → Excellent
≥0.85 → Great
≥0.70 → Good
≥0.55 → Fair
<0.55 → Needs Work

A warning appears for scores ≥0.97 (possible leakage/memorization).

## Limitations
This is a lite evaluator built with limited advanced knowledge in synthetic data metrics:

Only univariate + simple Pearson bivariate checks (no multivariate, no ML utility tests)
No support for time-series, text, images, or complex hierarchies
Basic statistical tests (KS/TVD/Pearson) — may miss higher moments, non-linear relationships, or rare events
Hardcoded thresholds and weights
Best for small–medium datasets (<10k–50k rows)
No fairness, privacy, or downstream task validation
High score ≠ perfect for all use cases — always manually verify

For more comprehensive evaluation, consider SDV's full metrics suite or SynthCity.



