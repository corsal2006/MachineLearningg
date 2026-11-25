"""
Week 3 — Iris Flower Clustering (K-Means & PCA)
===============================================
Covers:
  1. Assignment 1 & Mini Project 3:
     - Load Kaggle/UCI Iris dataset (`uciml/iris`).
     - Apply K-Means clustering (k=3) on standardized floral measurements.
     - Compare predicted clusters vs. true species labels using optimal
       Hungarian matching, Adjusted Rand Index (ARI), NMI, and Silhouette Score.
     - Visualize predicted clusters vs. true labels (scatter plots + contingency).
  2. Assignment 2:
     - Apply PCA to reduce 4D Iris features to 2D (`PC1`, `PC2`) and analyze
       explained variance across all principal components.
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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

CLUSTERED_CSV_PATH = OUTPUT_DIR / "iris_clustered.csv"
CLUSTERS_PLOT_PATH = OUTPUT_DIR / "iris_clusters_comparison.png"
ANALYSIS_PLOT_PATH = OUTPUT_DIR / "iris_pca_elbow_analysis.png"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#E69F00",
    "slate": "#5C6B73",
}
CLUSTER_COLORS = [PALETTE["teal"], PALETTE["coral"], PALETTE["navy"]]


class ClusteringResults(TypedDict):
    df_clustered: pd.DataFrame
    scaler: StandardScaler
    kmeans: KMeans
    pca_2d: PCA
    centroids_pca: np.ndarray
    cluster_to_species: dict[int, str]
    species_names: list[str]
    contingency_matrix: np.ndarray
    silhouette_k3: float
    ari_score: float
    nmi_score: float
    matched_accuracy: float
    explained_variance_ratio: list[float]
    cumulative_variance_2d: float
    elbow_df: pd.DataFrame


def load_iris_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Iris dataset locally, fetching from Kaggle (`uciml/iris`) if absent."""
    if not csv_path.exists():
        import kagglehub

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_dir = Path(kagglehub.dataset_download("uciml/iris"))
        src = downloaded_dir / "Iris.csv"
        csv_path.write_bytes(src.read_bytes())
    return pd.read_csv(csv_path)


def _match_clusters_to_species(
    true_labels: list[str], clusters: np.ndarray
) -> tuple[dict[int, str], np.ndarray, list[str], float]:
    """Use Hungarian linear sum assignment to align unsupervised cluster IDs to true species."""
    species_names = sorted(set(true_labels))
    species_to_idx = {name: i for i, name in enumerate(species_names)}
    y_true_idx = np.array([species_to_idx[s] for s in true_labels], dtype=int)

    cm = confusion_matrix(y_true_idx, clusters)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    cluster_to_species = {int(c): species_names[int(r)] for r, c in zip(row_ind, col_ind)}
    matched_accuracy = float(cm[row_ind, col_ind].sum()) / float(len(y_true_idx))

    # Reorder contingency matrix columns to align with species_names order
    ordered_cols = [col_ind[i] for i in np.argsort(row_ind)]
    aligned_cm = cm[:, ordered_cols]
    return cluster_to_species, aligned_cm, species_names, matched_accuracy


def run_kmeans_and_pca(df_iris: pd.DataFrame, n_clusters: int = 3) -> ClusteringResults:
    """Run K-Means (k=3) on Iris measurements (cm) and PCA (2D + full) on standardized features."""
    X = df_iris.loc[:, FEATURE_COLS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow & Silhouette scan (k = 1..8)
    elbow_rows: list[dict[str, float]] = []
    for k in range(1, 9):
        km_k = KMeans(n_clusters=k, random_state=42)
        km_k.set_params(n_init=20)
        labels_k = km_k.fit_predict(X)
        sil = float(silhouette_score(X, labels_k)) if k >= 2 else float("nan")
        elbow_rows.append({"k": float(k), "inertia": float(km_k.inertia_ or 0.0), "silhouette": sil})
    elbow_df = pd.DataFrame(elbow_rows)

    # Primary K-Means model (k=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.set_params(n_init=20)
    clusters = kmeans.fit_predict(X)

    # PCA full (for variance ratios) and 2D projection on standardized features
    pca_full = PCA(n_components=len(FEATURE_COLS), random_state=42)
    pca_full.fit(X_scaled)
    explained_ratio = [round(float(v), 4) for v in pca_full.explained_variance_ratio_]

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    centroids_pca = pca_2d.transform(scaler.transform(kmeans.cluster_centers_))

    true_species_list = [str(s) for s in df_iris["Species"].tolist()]
    cluster_to_species, aligned_cm, species_names, matched_acc = _match_clusters_to_species(
        true_species_list, clusters
    )

    df_out = df_iris.copy()
    df_out["PC1"] = np.round(X_pca[:, 0], 4)
    df_out["PC2"] = np.round(X_pca[:, 1], 4)
    df_out["Cluster"] = clusters.astype(int)
    df_out["Predicted_Species"] = [cluster_to_species[int(c)] for c in clusters]
    df_out["Is_Matched"] = df_out["Predicted_Species"] == df_out["Species"]

    sil_k3 = float(silhouette_score(X, clusters))
    ari = float(adjusted_rand_score(df_iris["Species"], clusters))
    nmi = float(normalized_mutual_info_score(df_iris["Species"], clusters))

    return {
        "df_clustered": df_out,
        "scaler": scaler,
        "kmeans": kmeans,
        "pca_2d": pca_2d,
        "centroids_pca": centroids_pca,
        "cluster_to_species": cluster_to_species,
        "species_names": species_names,
        "contingency_matrix": aligned_cm,
        "silhouette_k3": round(sil_k3, 4),
        "ari_score": round(ari, 4),
        "nmi_score": round(nmi, 4),
        "matched_accuracy": round(matched_acc, 4),
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance_2d": round(float(sum(explained_ratio[:2])), 4),
        "elbow_df": elbow_df,
    }


def plot_clusters_comparison(res: ClusteringResults, save_path: Path = CLUSTERS_PLOT_PATH) -> None:
    """Create side-by-side PCA scatter plots comparing K-Means clusters vs. true species."""
    df_c = res["df_clustered"]
    centroids = res["centroids_pca"]
    var_pct = [v * 100 for v in res["explained_variance_ratio"][:2]]

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    # Left: K-Means Predicted Clusters (k=3)
    ax1 = axes[0]
    plot_df = df_c.copy()
    plot_df["Cluster_Label"] = [
        f"Cluster {int(c)} ({res['cluster_to_species'][int(c)]})" for c in plot_df["Cluster"]
    ]
    species_order = res["species_names"]
    species_palette = {sp: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, sp in enumerate(species_order)}
    unique_labels = sorted(plot_df["Cluster_Label"].unique(), key=lambda lbl: res["cluster_to_species"][int(lbl.split()[1])])
    cluster_palette = {
        lbl: species_palette[res["cluster_to_species"][int(lbl.split()[1])]] for lbl in unique_labels
    }

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Cluster_Label",
        hue_order=unique_labels,
        palette=cluster_palette,
        s=68,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
        ax=ax1,
    )
    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="*",
        s=280,
        color=PALETTE["amber"],
        edgecolor="black",
        linewidth=1.1,
        zorder=5,
        label="Centroids",
    )
    ax1.set_title("K-Means Predicted Clusters (k = 3)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", fontsize=9)

    # Right: True Species Labels
    ax2 = axes[1]
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Species",
        hue_order=species_order,
        palette=species_palette,
        s=68,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
        ax=ax2,
    )
    ax2.set_title("True Species Labels (Ground Truth)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)", fontsize=11)
    ax2.legend(frameon=True, facecolor="white", fontsize=9.5)

    sns.despine(fig=fig)
    fig.suptitle(
        f"Mini Project 3: Iris K-Means (k=3) vs. True Species in PCA Space  |  ARI={res['ari_score']:.3f}, Match Acc={res['matched_accuracy']*100:.1f}%",
        fontsize=14,
        fontweight="bold",
        y=1.02,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_and_elbow_analysis(res: ClusteringResults, save_path: Path = ANALYSIS_PLOT_PATH) -> None:
    """Create 2x2 dashboard for Elbow Method, Silhouette Scores, PCA Scree, and Contingency Matrix."""
    elbow_df = res["elbow_df"]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")

    # 1. Elbow Curve (Inertia)
    ax = axes[0, 0]
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o", color=PALETTE["navy"], linewidth=2.2)
    k3_inertia = float(elbow_df.loc[elbow_df["k"] == 3, "inertia"].iloc[0])
    ax.scatter([3], [k3_inertia], color=PALETTE["coral"], s=130, zorder=5, label="Selected k = 3")
    ax.set_title("Elbow Method: Inertia vs. Number of Clusters (k)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_xticks(range(1, 9))
    ax.legend(frameon=True, facecolor="white")

    # 2. Silhouette Score
    ax = axes[0, 1]
    sil_df = elbow_df.dropna(subset=["silhouette"])
    ax.plot(sil_df["k"], sil_df["silhouette"], marker="s", color=PALETTE["teal"], linewidth=2.2)
    ax.axvline(3, color=PALETTE["coral"], linestyle="--", linewidth=1.6, label=f"k = 3 (Sil = {res['silhouette_k3']:.3f})")
    ax.set_title("Silhouette Score by Cluster Count (k)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_xticks(range(2, 9))
    ax.legend(frameon=True, facecolor="white")

    # 3. PCA Explained Variance (Scree Plot)
    ax = axes[1, 0]
    pcs = [f"PC{i}" for i in range(1, len(res["explained_variance_ratio"]) + 1)]
    var_pct = [v * 100 for v in res["explained_variance_ratio"]]
    cum_pct = np.cumsum(var_pct)
    bars = ax.bar(pcs, var_pct, color=PALETTE["teal"], alpha=0.82, label="Individual Variance (%)")
    ax.plot(pcs, cum_pct, marker="o", color=PALETTE["coral"], linewidth=2.2, label="Cumulative Variance (%)")
    for bar, val in zip(bars, var_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=9.5)
    ax.set_title(f"Assignment 2: PCA Explained Variance (2D = {res['cumulative_variance_2d']*100:.1f}%)", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=True, facecolor="white")

    # 4. Contingency Heatmap (True Species vs Matched Cluster)
    ax = axes[1, 1]
    sns.heatmap(
        res["contingency_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[f"Pred: {s.replace('Iris-', '')}" for s in res["species_names"]],
        yticklabels=[f"True: {s.replace('Iris-', '')}" for s in res["species_names"]],
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Cluster Alignment Matrix (True vs. Predicted)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Matched K-Means Cluster")
    ax.set_ylabel("True Species")

    sns.despine(fig=fig)
    fig.suptitle(
        "Week 3 — Iris K-Means Diagnostics & PCA Dimensionality Reduction",
        fontsize=15,
        fontweight="bold",
        y=0.99,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pipeline() -> None:
    """Execute Week 3 K-Means + PCA workflow and save CSV, plots, and report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_iris = load_iris_dataset()
    res = run_kmeans_and_pca(df_iris, n_clusters=3)
    res["df_clustered"].to_csv(CLUSTERED_CSV_PATH, index=False)

    plot_clusters_comparison(res)
    plot_pca_and_elbow_analysis(res)

    report_lines = [
        "=" * 70,
        "WEEK 3 — IRIS FLOWER CLUSTERING (K-MEANS & PCA) REPORT",
        "=" * 70,
        f"Dataset Shape             : {df_iris.shape}",
        f"Features Standardized     : {', '.join(FEATURE_COLS)}",
        f"PCA Variance Ratios       : {res['explained_variance_ratio']}",
        f"PCA 2D Cumulative Var     : {res['cumulative_variance_2d'] * 100:.2f}%",
        "",
        "--- K-Means (k=3) Evaluation vs. True Species ---",
        f"Cluster-to-Species Map    : {res['cluster_to_species']}",
        f"Silhouette Score (k=3)    : {res['silhouette_k3']:.4f}",
        f"Adjusted Rand Index (ARI) : {res['ari_score']:.4f}",
        f"Normalized Mutual Info    : {res['nmi_score']:.4f}",
        f"Matched Cluster Accuracy  : {res['matched_accuracy'] * 100:.2f}%",
        "",
        "Contingency Matrix (Rows=True Species, Cols=Aligned Clusters):",
        str(res["contingency_matrix"]),
        "",
        "--- Elbow & Silhouette Scan (k=1..8) ---",
        res["elbow_df"].round(4).to_string(index=False),
        "",
        "--- Generated Artifacts ---",
        f"Clustered CSV  : {CLUSTERED_CSV_PATH}",
        f"Clusters Plot  : {CLUSTERS_PLOT_PATH}",
        f"Analysis Plot  : {ANALYSIS_PLOT_PATH}",
    ]
    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    run_pipeline()
