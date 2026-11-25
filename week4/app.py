"""
Week 4 — Mall Customer Segmentation Business Analytics Dashboard
================================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from customer_segmentation import (
    CLUSTER_FEATURES_2D,
    CLUSTER_FEATURES_3D,
    CLUSTER_PALETTE,
    PALETTE,
    SPENDER_ORDER,
    SPENDER_PALETTE,
    SegmentationResults,
    load_mall_customers,
    segment_customers,
)

st.set_page_config(
    page_title="Week 4: Customer Segmentation Analytics",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .metric-label { font-size: 0.85rem; color: #475569; margin-bottom: 0.25rem; }
    .metric-value { font-size: 1.65rem; font-weight: 700; color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_segmentation(n_clusters: int, use_age: bool) -> tuple[pd.DataFrame, SegmentationResults]:
    df_raw = load_mall_customers()
    cols = CLUSTER_FEATURES_3D if use_age else CLUSTER_FEATURES_2D
    res = segment_customers(df_raw, n_clusters=n_clusters, feature_cols=cols)
    return df_raw, res


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Customer Segmentation — Business Analytics Dashboard")
    st.caption("Week 4 Mini Project | Mall Customers Dataset (`vjchoudhary7/customer-segmentation-tutorial-in-python`)")

    ctrl1, ctrl2 = st.columns([2, 1])
    feature_mode = ctrl1.radio(
        "Clustering Feature Space",
        ["Annual Income + Spending Score (Recommended, Peak Silhouette)", "Age + Annual Income + Spending Score"],
        horizontal=True,
    )
    use_age = feature_mode.startswith("Age")
    k_val = ctrl2.slider("Number of K-Means Clusters (k)", min_value=3, max_value=8, value=5)

    df_raw, res = get_segmentation(k_val, use_age)
    df_seg = res["df_segmented"]
    tier_df = res["tier_summary"]

    high_row = tier_df.loc[tier_df["Spender_Group"] == "High Spenders"]
    high_pct = float(high_row["Share_Pct"].iloc[0]) if not high_row.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Customers", f"{len(df_raw):,}")
    with c2:
        metric_card("Silhouette Score", f"{res['silhouette_score']:.4f}")
    with c3:
        metric_card("Avg Annual Income", f"${df_raw['Annual Income (k$)'].mean():.1f}k")
    with c4:
        metric_card("High Spenders Share", f"{high_pct:.1f}%")

    st.markdown("---")
    tab_viz, tab_kpi, tab_pred = st.tabs(
        [
            "1. Segment Visualizations (Age / Income / Spend)",
            "2. Spender Tiers & Elbow Diagnostics",
            "3. Customer Segment Classifier",
        ]
    )

    with tab_viz:
        st.subheader("Customer Segments by Income, Age & Spending Score")
        color_by = st.radio(
            "Color Points By",
            ["Spender Group (Low / Medium / High Spenders)", "Granular K-Means Cluster"],
            horizontal=True,
        )

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)

        if color_by.startswith("Spender"):
            hue_col = "Spender_Group"
            hue_ord = [g for g in SPENDER_ORDER if g in set(df_seg["Spender_Group"].tolist())]
            pal = SPENDER_PALETTE
        else:
            hue_col = "Cluster"
            hue_ord = sorted(df_seg["Cluster"].unique().tolist())
            pal = {cid: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, cid in enumerate(hue_ord)}

        pairs = [
            ("Annual Income (k$)", "Spending Score (1-100)", "Income vs. Spending Score"),
            ("Age", "Spending Score (1-100)", "Age vs. Spending Score"),
            ("Age", "Annual Income (k$)", "Age vs. Annual Income"),
        ]
        for ax, (x_c, y_c, title) in zip(axes, pairs):
            sns.scatterplot(
                data=df_seg,
                x=x_c,
                y=y_c,
                hue=hue_col,
                hue_order=hue_ord,
                palette=pal,
                s=65,
                alpha=0.88,
                edgecolor="white",
                ax=ax,
            )
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8)

        sns.despine(fig=fig)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_kpi:
        st.subheader("Business Spender Tiers & Persona KPIs")
        k_left, k_right = st.columns(2)
        with k_left:
            st.markdown("**Spender Group Summary (`Low` / `Medium` / `High Spenders`)**")
            st.dataframe(tier_df, width="stretch", hide_index=True)
        with k_right:
            st.markdown("**Detailed Cluster Personas**")
            st.dataframe(res["cluster_summary"], width="stretch", hide_index=True)

        elbow_df = res["elbow_df"]
        sns.set_theme(style="whitegrid")
        fig_d, axes_d = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130)
        axes_d[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color=PALETTE["navy"], linewidth=2.0)
        axes_d[0].axvline(k_val, color=PALETTE["coral"], linestyle="--")
        axes_d[0].set_title("Elbow Curve (Inertia)", fontweight="bold")
        axes_d[0].set_xlabel("k")
        axes_d[0].set_ylabel("Inertia")

        sil_df = elbow_df.dropna(subset=["silhouette"])
        axes_d[1].plot(sil_df["k"], sil_df["silhouette"], marker="s", color=PALETTE["teal"], linewidth=2.0)
        axes_d[1].axvline(k_val, color=PALETTE["coral"], linestyle="--")
        axes_d[1].set_title("Silhouette Score by k", fontweight="bold")
        axes_d[1].set_xlabel("k")
        axes_d[1].set_ylabel("Silhouette")
        sns.despine(fig=fig_d)
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

        st.download_button(
            label="Download Segmented Customers (CSV)",
            data=df_seg.to_csv(index=False).encode("utf-8"),
            file_name="mall_customers_segmented.csv",
            mime="text/csv",
        )

    with tab_pred:
        st.subheader("Classify a New Customer & Recommend Strategy")
        p1, p2, p3 = st.columns(3)
        age = p1.slider("Customer Age", 18, 70, 32)
        income = p2.slider("Annual Income (k$)", 15, 140, 85)
        spend = p3.slider("Spending Score (1-100)", 1, 100, 80)

        sample_vals = [age, income, spend] if use_age else [income, spend]
        sample_scaled = res["scaler"].transform(np.array([sample_vals], dtype=float))
        pred_cluster = int(res["kmeans"].predict(sample_scaled)[0])
        pred_tier = res["cluster_to_tier"][pred_cluster]
        pred_persona = res["cluster_to_persona"][pred_cluster]

        action_map = {
            "High Spenders": "Enroll in VIP loyalty tier, early-access launches, and personalized concierge offers.",
            "Medium Spenders": "Target with seasonal bundle promotions and cross-category upsell incentives.",
            "Low Spenders": "Engage with value discounts, cashback campaigns, and targeted high-ROI reminders.",
        }
        st.success(
            f"**Assigned Group:** **{pred_tier}** (`Cluster {pred_cluster}` — {pred_persona})  \n"
            f"**Recommended Business Action:** {action_map[pred_tier]}"
        )


if __name__ == "__main__":
    main()
