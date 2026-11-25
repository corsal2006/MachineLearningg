"""
Week 4 — Mall Customer Segmentation using K-Means Clustering
=============================================================
Option 3: Business Analytics-Oriented Customer Segmentation

Covers:
  1. Load Kaggle Mall Customers dataset (`vjchoudhary7/customer-segmentation-tutorial-in-python`).
  2. Scan Elbow & Silhouette scores (k=2..10) and apply K-Means clustering (default k=5).
  3. Label customer groups into business spending tiers:
       - "Low Spenders"
       - "Medium Spenders"
       - "High Spenders"
     along with granular customer personas and segment KPIs.
  4. Visualize customer groups across Age, Annual Income, and Spending Score
     using Matplotlib & Seaborn.
  5. Export segmented dataset (`outputs/mall_customers_segmented.csv`) and report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

SEGMENTED_CSV_PATH = OUTPUT_DIR / "mall_customers_segmented.csv"
SEGMENTS_PLOT_PATH = OUTPUT_DIR / "customer_segments_overview.png"
DIAGNOSTICS_PLOT_PATH = OUTPUT_DIR / "segmentation_diagnostics.png"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

CLUSTER_FEATURES_2D = ["Annual Income (k$)", "Spending Score (1-100)"]
CLUSTER_FEATURES_3D = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

SPENDER_ORDER = ["Low Spenders", "Medium Spenders", "High Spenders"]

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#E69F00",
    "slate": "#5C6B73",
}

SPENDER_PALETTE = {
    "Low Spenders": PALETTE["coral"],
    "Medium Spenders": PALETTE["amber"],
    "High Spenders": PALETTE["teal"],
}

CLUSTER_PALETTE = [
    "#008080",
    "#E05A47",
    "#1B365D",
    "#E69F00",
    "#6B5B95",
    "#2A9D8F",
    "#D62828",
    "#457B9D",
]


class SegmentationResults(TypedDict):
    df_segmented: pd.DataFrame
    scaler: StandardScaler
    kmeans: KMeans
    feature_cols: list[str]
    n_clusters: int
    silhouette_score: float
    cluster_to_tier: dict[int, str]
    cluster_to_persona: dict[int, str]
    tier_summary: pd.DataFrame
    cluster_summary: pd.DataFrame
    elbow_df: pd.DataFrame


def load_mall_customers(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Mall Customers dataset locally, fetching from Kaggle if absent."""
    if not csv_path.exists():
        import kagglehub

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_dir = Path(kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python"))
        src = downloaded_dir / "Mall_Customers.csv"
        csv_path.write_bytes(src.read_bytes())
    return pd.read_csv(csv_path)


def _assign_business_labels(
    df_with_clusters: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, str]]:
    """Map each K-Means cluster to a Spender Tier and descriptive business persona."""
    stats = df_with_clusters.groupby("Cluster")[
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ].mean()

    cluster_to_tier: dict[int, str] = {}
    cluster_to_persona: dict[int, str] = {}

    median_income = float(df_with_clusters["Annual Income (k$)"].median())

    for cluster_id, row in stats.iterrows():
        cid = int(str(cluster_id))
        spend = float(row["Spending Score (1-100)"])
        income = float(row["Annual Income (k$)"])
        age = float(row["Age"])

        if spend < 35.0:
            tier = "Low Spenders"
            persona = "Careful High-Earners" if income >= median_income else "Budget-Conscious Shoppers"
        elif spend <= 65.0:
            tier = "Medium Spenders"
            persona = "Core Steady Shoppers"
        else:
            tier = "High Spenders"
            persona = "Affluent VIPs" if income >= median_income else ("Young Trendsetters" if age < 35.0 else "Loyal Enthusiasts")

        cluster_to_tier[cid] = tier
        cluster_to_persona[cid] = f"{persona} ({tier})"

    return cluster_to_tier, cluster_to_persona


def segment_customers(
    df_raw: pd.DataFrame,
    n_clusters: int = 5,
    feature_cols: list[str] | None = None,
) -> SegmentationResults:
    """Standardize selected features, scan Elbow/Silhouette, and fit K-Means segmentation."""
    cols = feature_cols if feature_cols is not None else CLUSTER_FEATURES_2D
    X = df_raw.loc[:, cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow & Silhouette scan across k = 1..10
    elbow_rows: list[dict[str, float]] = []
    for k in range(1, 11):
        km_k = KMeans(n_clusters=k, random_state=42)
        km_k.set_params(n_init=20)
        labels_k = km_k.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels_k)) if k >= 2 else float("nan")
        elbow_rows.append({"k": float(k), "inertia": float(km_k.inertia_ or 0.0), "silhouette": sil})
    elbow_df = pd.DataFrame(elbow_rows)

    # Final K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.set_params(n_init=20)
    clusters = kmeans.fit_predict(X_scaled)
    sil_final = float(silhouette_score(X_scaled, clusters))

    df_seg = df_raw.copy()
    df_seg["Cluster"] = clusters.astype(int)

    cluster_to_tier, cluster_to_persona = _assign_business_labels(df_seg)
    df_seg["Spender_Group"] = [cluster_to_tier[int(c)] for c in clusters]
    df_seg["Customer_Persona"] = [cluster_to_persona[int(c)] for c in clusters]

    tier_summary = (
        df_seg.groupby("Spender_Group", observed=False)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income_k=("Annual Income (k$)", "mean"),
            Avg_Spending_Score=("Spending Score (1-100)", "mean"),
        )
        .reindex(SPENDER_ORDER)
        .dropna(how="all")
        .round(2)
    )
    tier_summary["Share_Pct"] = (tier_summary["Customers"] / len(df_seg) * 100.0).round(1)
    tier_summary = tier_summary.reset_index()

    cluster_summary = (
        df_seg.groupby(["Cluster", "Spender_Group", "Customer_Persona"], observed=False)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Age=("Age", "mean"),
            Avg_Income_k=("Annual Income (k$)", "mean"),
            Avg_Spending_Score=("Spending Score (1-100)", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values(by="Avg_Spending_Score", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "df_segmented": df_seg,
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_cols": cols,
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_final, 4),
        "cluster_to_tier": cluster_to_tier,
        "cluster_to_persona": cluster_to_persona,
        "tier_summary": tier_summary,
        "cluster_summary": cluster_summary,
        "elbow_df": elbow_df,
    }


def plot_customer_segments(res: SegmentationResults, save_path: Path = SEGMENTS_PLOT_PATH) -> None:
    """Visualize customer groups by Income/Spending Score, Age/Spending Score, and Spender Tiers."""
    df_s = res["df_segmented"]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor("white")

    # 1. Annual Income vs Spending Score colored by K-Means Cluster
    ax = axes[0, 0]
    cluster_ids = sorted(df_s["Cluster"].unique().tolist())
    c_pal = {cid: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, cid in enumerate(cluster_ids)}
    sns.scatterplot(
        data=df_s,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette=c_pal,
        s=72,
        alpha=0.88,
        edgecolor="white",
        ax=ax,
    )
    centroids_orig = res["scaler"].inverse_transform(res["kmeans"].cluster_centers_)
    inc_idx = res["feature_cols"].index("Annual Income (k$)")
    spd_idx = res["feature_cols"].index("Spending Score (1-100)")
    ax.scatter(
        centroids_orig[:, inc_idx],
        centroids_orig[:, spd_idx],
        marker="*",
        s=280,
        color=PALETTE["amber"],
        edgecolor="black",
        linewidth=1.1,
        zorder=5,
        label="Centroids",
    )
    ax.set_title(f"K-Means Clusters (k={res['n_clusters']}): Income vs. Spending Score", fontsize=12.5, fontweight="bold")
    ax.legend(title="Cluster", frameon=True, facecolor="white", fontsize=8.5)

    # 2. Annual Income vs Spending Score colored by Business Spender Group
    ax = axes[0, 1]
    sns.scatterplot(
        data=df_s,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Spender_Group",
        hue_order=SPENDER_ORDER,
        palette=SPENDER_PALETTE,
        s=72,
        alpha=0.88,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Business Segments: Income vs. Spending Score", fontsize=12.5, fontweight="bold")
    ax.legend(title="Spender Group", frameon=True, facecolor="white", fontsize=9)

    # 3. Age vs Spending Score colored by Business Spender Group
    ax = axes[1, 0]
    sns.scatterplot(
        data=df_s,
        x="Age",
        y="Spending Score (1-100)",
        hue="Spender_Group",
        hue_order=SPENDER_ORDER,
        palette=SPENDER_PALETTE,
        s=72,
        alpha=0.88,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Customer Demographics: Age vs. Spending Score", fontsize=12.5, fontweight="bold")
    ax.legend(title="Spender Group", frameon=True, facecolor="white", fontsize=9)

    # 4. Age vs Annual Income colored by Business Spender Group
    ax = axes[1, 1]
    sns.scatterplot(
        data=df_s,
        x="Age",
        y="Annual Income (k$)",
        hue="Spender_Group",
        hue_order=SPENDER_ORDER,
        palette=SPENDER_PALETTE,
        s=72,
        alpha=0.88,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Customer Demographics: Age vs. Annual Income", fontsize=12.5, fontweight="bold")
    ax.legend(title="Spender Group", frameon=True, facecolor="white", fontsize=9)

    sns.despine(fig=fig)
    fig.suptitle(
        f"Mall Customer Segmentation — Age, Income & Spending Score (Silhouette = {res['silhouette_score']:.3f})",
        fontsize=15,
        fontweight="bold",
        y=0.99,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_diagnostics(res: SegmentationResults, save_path: Path = DIAGNOSTICS_PLOT_PATH) -> None:
    """Create Elbow Curve, Silhouette Score, and Spender Group KPI comparison charts."""
    elbow_df = res["elbow_df"]
    tier_df = res["tier_summary"]

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    fig.patch.set_facecolor("white")

    # 1. Elbow Curve
    ax = axes[0]
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o", color=PALETTE["navy"], linewidth=2.2)
    k_opt = res["n_clusters"]
    k_inertia = float(elbow_df.loc[elbow_df["k"] == k_opt, "inertia"].iloc[0])
    ax.scatter([k_opt], [k_inertia], color=PALETTE["coral"], s=130, zorder=5, label=f"Selected k = {k_opt}")
    ax.set_title("Elbow Method (WCSS Inertia)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=True, facecolor="white")

    # 2. Silhouette Score
    ax = axes[1]
    sil_df = elbow_df.dropna(subset=["silhouette"])
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="s", color=PALETTE["teal"], linewidth=2.2)
    ax.axvline(k_opt, color=PALETTE["coral"], linestyle="--", linewidth=1.6, label=f"k = {k_opt} (Sil = {res['silhouette_score']:.3f})")
    ax.set_title("Silhouette Score by k", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(range(2, 11))
    ax.legend(frameon=True, facecolor="white")

    # 3. Spender Group Customer Count & Share
    ax = axes[2]
    sns.barplot(
        data=tier_df,
        x="Spender_Group",
        y="Customers",
        hue="Spender_Group",
        order=SPENDER_ORDER,
        hue_order=SPENDER_ORDER,
        palette=SPENDER_PALETTE,
        legend=False,
        ax=ax,
    )
    for idx, row in tier_df.iterrows():
        x_pos = SPENDER_ORDER.index(str(row["Spender_Group"]))
        ax.text(
            x_pos,
            float(row["Customers"]) + 1.5,
            f"{int(row['Customers'])} ({row['Share_Pct']:.1f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Customer Distribution by Spender Group", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, float(tier_df["Customers"].max()) * 1.2)

    sns.despine(fig=fig)
    fig.suptitle(
        "Week 4 — Clustering Validation & Spender Tier Breakdown",
        fontsize=14.5,
        fontweight="bold",
        y=1.02,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pipeline() -> None:
    """Execute full Mall Customer Segmentation workflow and save CSV, plots, and report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_mall_customers()
    res = segment_customers(df_raw, n_clusters=5, feature_cols=CLUSTER_FEATURES_2D)
    res["df_segmented"].to_csv(SEGMENTED_CSV_PATH, index=False)

    plot_customer_segments(res)
    plot_segmentation_diagnostics(res)

    report_lines = [
        "=" * 74,
        "WEEK 4 — MALL CUSTOMER SEGMENTATION (K-MEANS CLUSTERING) REPORT",
        "=" * 74,
        f"Dataset Shape       : {df_raw.shape}",
        f"Clustering Features : {', '.join(res['feature_cols'])}",
        f"Optimal Clusters (k): {res['n_clusters']}",
        f"Silhouette Score    : {res['silhouette_score']:.4f}",
        "",
        "--- Business Spender Group Summary ---",
        res["tier_summary"].to_string(index=False),
        "",
        "--- Detailed Cluster Persona Breakdown ---",
        res["cluster_summary"].to_string(index=False),
        "",
        "--- Generated Artifacts ---",
        f"Segmented CSV    : {SEGMENTED_CSV_PATH}",
        f"Segments Plot    : {SEGMENTS_PLOT_PATH}",
        f"Diagnostics Plot : {DIAGNOSTICS_PLOT_PATH}",
    ]
    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    run_pipeline()
