"""
Week 3 — Iris Flower Clustering (K-Means & PCA) App
===================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from iris_clustering import (
    CLUSTER_COLORS,
    FEATURE_COLS,
    PALETTE,
    ClusteringResults,
    load_iris_dataset,
    run_kmeans_and_pca,
)

st.set_page_config(
    page_title="Week 3: Iris K-Means & PCA Clustering",
    page_icon="🌸",
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
def get_clustering_results(n_clusters: int) -> tuple[pd.DataFrame, ClusteringResults]:
    df_iris = load_iris_dataset()
    res = run_kmeans_and_pca(df_iris, n_clusters=n_clusters)
    return df_iris, res


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Iris Flower Clustering — K-Means & PCA")
    st.caption("Week 3 Mini Project | Unsupervised ML & Dimensionality Reduction (`uciml/iris`)")

    df_iris, res_k3 = get_clustering_results(3)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Silhouette Score (k=3)", f"{res_k3['silhouette_k3']:.4f}")
    with c2:
        metric_card("Adjusted Rand Index (ARI)", f"{res_k3['ari_score']:.4f}")
    with c3:
        metric_card("Cluster Match Accuracy", f"{res_k3['matched_accuracy'] * 100:.1f}%")
    with c4:
        metric_card("PCA 2D Variance Kept", f"{res_k3['cumulative_variance_2d'] * 100:.1f}%")

    st.markdown("---")
    tab_cmp, tab_pca, tab_sim = st.tabs(
        [
            "1. Clusters vs. True Labels",
            "2. Elbow, Silhouette & PCA",
            "3. Interactive Flower Cluster Predictor",
        ]
    )

    with tab_cmp:
        st.subheader("Mini Project 3: K-Means (k=3) vs. True Species")
        axis_mode = st.radio(
            "Scatter Plot Axes",
            ["PCA 2D Projection (PC1 vs PC2)", "Petal Dimensions (PetalLengthCm vs PetalWidthCm)", "Sepal Dimensions (SepalLengthCm vs SepalWidthCm)"],
            horizontal=True,
        )
        if axis_mode.startswith("PCA"):
            x_col, y_col = "PC1", "PC2"
        elif axis_mode.startswith("Petal"):
            x_col, y_col = "PetalLengthCm", "PetalWidthCm"
        else:
            x_col, y_col = "SepalLengthCm", "SepalWidthCm"

        df_c = res_k3["df_clustered"].copy()
        df_c["Cluster_Label"] = [
            f"Cluster {int(c)} ({res_k3['cluster_to_species'][int(c)]})" for c in df_c["Cluster"]
        ]
        species_pal = {sp: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, sp in enumerate(res_k3["species_names"])}
        cluster_order = sorted(
            df_c["Cluster_Label"].unique(),
            key=lambda lbl: res_k3["cluster_to_species"][int(lbl.split()[1])],
        )
        cluster_pal = {
            lbl: species_pal[res_k3["cluster_to_species"][int(lbl.split()[1])]] for lbl in cluster_order
        }

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=130, sharex=True, sharey=True)
        sns.scatterplot(
            data=df_c,
            x=x_col,
            y=y_col,
            hue="Cluster_Label",
            hue_order=cluster_order,
            palette=cluster_pal,
            s=65,
            alpha=0.85,
            edgecolor="white",
            ax=axes[0],
        )
        if x_col == "PC1" and y_col == "PC2":
            cents = res_k3["centroids_pca"]
            axes[0].scatter(
                cents[:, 0],
                cents[:, 1],
                marker="*",
                s=240,
                color=PALETTE["amber"],
                edgecolor="black",
                zorder=5,
                label="Centroids",
            )
            axes[0].legend(fontsize=8.5)
        axes[0].set_title("K-Means Predicted Clusters (k = 3)", fontweight="bold")

        sns.scatterplot(
            data=df_c,
            x=x_col,
            y=y_col,
            hue="Species",
            hue_order=res_k3["species_names"],
            palette=species_pal,
            s=65,
            alpha=0.85,
            edgecolor="white",
            ax=axes[1],
        )
        axes[1].set_title("True Species Labels", fontweight="bold")
        sns.despine(fig=fig)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Cluster Alignment Matrix")
            cm_df = pd.DataFrame(
                res_k3["contingency_matrix"],
                index=pd.Index([f"True: {s}" for s in res_k3["species_names"]]),
                columns=pd.Index([f"Pred: {s}" for s in res_k3["species_names"]]),
            )
            st.dataframe(cm_df, width="stretch")
        with col_r:
            st.subheader("Clustered Dataset Preview")
            preview_cols = ["Id", "SepalLengthCm", "PetalLengthCm", "PC1", "PC2", "Cluster", "Predicted_Species", "Species", "Is_Matched"]
            st.dataframe(df_c.loc[:, preview_cols].head(12), width="stretch", hide_index=True)
            st.download_button(
                label="Download Clustered Dataset (CSV)",
                data=df_c.to_csv(index=False).encode("utf-8"),
                file_name="iris_clustered.csv",
                mime="text/csv",
            )

    with tab_pca:
        st.subheader("Assignments 1 & 2: Elbow, Silhouette & PCA Explained Variance")
        k_choice = st.slider("Explore K-Means Cluster Count (k)", min_value=2, max_value=8, value=3)
        _, res_k = get_clustering_results(k_choice)
        st.info(
            f"**k = {k_choice}** → Silhouette Score: **{res_k['silhouette_k3']:.4f}** | "
            f"ARI: **{res_k['ari_score']:.4f}** | NMI: **{res_k['nmi_score']:.4f}**"
        )

        elbow_df = res_k3["elbow_df"]
        sns.set_theme(style="whitegrid")
        fig_d, axes_d = plt.subplots(1, 3, figsize=(15, 4.3), dpi=130)

        axes_d[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color=PALETTE["navy"], linewidth=2.0)
        axes_d[0].axvline(k_choice, color=PALETTE["coral"], linestyle="--", linewidth=1.5)
        axes_d[0].set_title("Elbow Curve (Inertia)", fontweight="bold")
        axes_d[0].set_xlabel("k")
        axes_d[0].set_ylabel("Inertia")

        sil_df = elbow_df.dropna(subset=["silhouette"])
        axes_d[1].plot(sil_df["k"], sil_df["silhouette"], marker="s", color=PALETTE["teal"], linewidth=2.0)
        axes_d[1].axvline(k_choice, color=PALETTE["coral"], linestyle="--", linewidth=1.5)
        axes_d[1].set_title("Silhouette Score", fontweight="bold")
        axes_d[1].set_xlabel("k")
        axes_d[1].set_ylabel("Silhouette")

        pcs = [f"PC{i}" for i in range(1, len(res_k3["explained_variance_ratio"]) + 1)]
        var_pct = [v * 100 for v in res_k3["explained_variance_ratio"]]
        axes_d[2].bar(pcs, var_pct, color=PALETTE["teal"], alpha=0.82, label="Individual %")
        axes_d[2].plot(pcs, np.cumsum(var_pct), marker="o", color=PALETTE["coral"], linewidth=2.0, label="Cumulative %")
        axes_d[2].set_title(f"PCA Scree (2D = {res_k3['cumulative_variance_2d']*100:.1f}%)", fontweight="bold")
        axes_d[2].set_ylim(0, 112)
        axes_d[2].legend(fontsize=8.5)

        sns.despine(fig=fig_d)
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

    with tab_sim:
        st.subheader("Assign a Flower Sample to a K-Means Cluster (k=3)")
        s1, s2, s3, s4 = st.columns(4)
        sl = s1.slider("Sepal Length (cm)", float(df_iris["SepalLengthCm"].min()), float(df_iris["SepalLengthCm"].max()), 5.8, 0.1)
        sw = s2.slider("Sepal Width (cm)", float(df_iris["SepalWidthCm"].min()), float(df_iris["SepalWidthCm"].max()), 3.0, 0.1)
        pl = s3.slider("Petal Length (cm)", float(df_iris["PetalLengthCm"].min()), float(df_iris["PetalLengthCm"].max()), 4.3, 0.1)
        pw = s4.slider("Petal Width (cm)", float(df_iris["PetalWidthCm"].min()), float(df_iris["PetalWidthCm"].max()), 1.3, 0.1)

        sample_arr = np.array([[sl, sw, pl, pw]], dtype=float)
        sample_scaled = res_k3["scaler"].transform(sample_arr)
        pred_cluster = int(res_k3["kmeans"].predict(sample_arr)[0])
        pred_species = res_k3["cluster_to_species"][pred_cluster]
        sample_pca = res_k3["pca_2d"].transform(sample_scaled)[0]

        st.success(
            f"**Assigned Cluster:** `Cluster {pred_cluster}` → Matched Species: **{pred_species}**  \n"
            f"PCA Coordinates: `(PC1 = {sample_pca[0]:.3f}, PC2 = {sample_pca[1]:.3f})` (Features: {', '.join(FEATURE_COLS)})"
        )

        df_c = res_k3["df_clustered"]
        sns.set_theme(style="whitegrid")
        fig_s, ax_s = plt.subplots(figsize=(8.5, 4.5), dpi=130)
        sns.scatterplot(
            data=df_c,
            x="PC1",
            y="PC2",
            hue="Predicted_Species",
            hue_order=res_k3["species_names"],
            palette={sp: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, sp in enumerate(res_k3["species_names"])},
            s=55,
            alpha=0.75,
            ax=ax_s,
        )
        ax_s.scatter(
            [sample_pca[0]],
            [sample_pca[1]],
            marker="*",
            s=360,
            color=PALETTE["amber"],
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
            label="Your Flower Sample",
        )
        ax_s.set_title("Your Sample Projected onto PCA Cluster Map", fontweight="bold")
        ax_s.legend()
        sns.despine(fig=fig_s)
        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)


if __name__ == "__main__":
    main()
