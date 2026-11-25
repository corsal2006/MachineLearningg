# Week 4 — Mall Customer Segmentation (K-Means Clustering)

A business analytics-oriented customer segmentation project inside `week4/` built on the Kaggle Mall Customers dataset ([`vjchoudhary7/customer-segmentation-tutorial-in-python`](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)).

## Workflow Coverage

1. **K-Means Customer Clustering**: Standardizes `Annual Income (k$)` and `Spending Score (1-100)` (with optional `Age` inclusion in the interactive app), validates `k = 5` via Elbow & Silhouette analysis (`Silhouette = 0.5547`), and fits `KMeans(n_clusters=5)`.
2. **Business Spender Tier Labeling**: Labels every cluster into **`Low Spenders`**, **`Medium Spenders`**, and **`High Spenders`**, enriched with actionable customer personas (e.g., *Affluent VIPs*, *Young Trendsetters*, *Careful High-Earners*, *Core Steady Shoppers*, *Budget-Conscious Shoppers*).
3. **Visualizations**: Plots customer groups across **Age**, **Annual Income**, and **Spending Score**, plus Elbow, Silhouette, and Spender Tier distribution charts.

## Project Structure

```text
week4/
├── data/
│   └── Mall_Customers.csv                  # Raw Kaggle dataset (200 × 5)
├── outputs/
│   ├── mall_customers_segmented.csv        # Customers with Cluster, Spender_Group & Persona
│   ├── customer_segments_overview.png      # 2x2 Age / Income / Spending Score scatter views
│   ├── segmentation_diagnostics.png        # Elbow, Silhouette & Spender Group breakdown
│   └── summary_report.txt                  # Business segment KPIs and cluster summary
├── customer_segmentation.py                # Core segmentation, labeling & plotting pipeline
├── customer_segmentation.ipynb             # Executed Jupyter Notebook
├── app.py                                  # Interactive Streamlit business analytics dashboard
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd week4
pip install -r requirements.txt

# Run K-Means segmentation and generate CSV, charts & report
python customer_segmentation.py

# Launch interactive Streamlit analytics dashboard
streamlit run app.py
```
