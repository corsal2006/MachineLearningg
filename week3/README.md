# Week 3 — Iris Flower Clustering (K-Means & PCA)

A clean, modular, end-to-end unsupervised machine learning project inside `week3/`:

- **Assignment 1 & Mini Project 3 ("Iris Flower Clustering Project")**: Applies `KMeans(n_clusters=3)` to the Kaggle/UCI Iris dataset ([`uciml/iris`](https://www.kaggle.com/datasets/uciml/iris)), visualizes clusters via scatter plots, and compares predicted clusters against true species labels using Silhouette Score, Adjusted Rand Index (ARI), NMI, and optimal Hungarian alignment.
- **Assignment 2 ("PCA Dimensionality Reduction")**: Applies `PCA` to reduce the 4D floral measurements to 2D (`PC1`, `PC2`) and visualizes explained & cumulative variance.

## Project Structure

```text
week3/
├── data/
│   └── Iris.csv                        # Raw Kaggle/UCI Iris dataset (150 × 6)
├── outputs/
│   ├── iris_clustered.csv              # Iris features + PC1/PC2 + Cluster + Matched Species
│   ├── iris_clusters_comparison.png    # Side-by-side K-Means clusters vs. True Species
│   ├── iris_pca_elbow_analysis.png     # Elbow, Silhouette, PCA Scree, & Alignment Matrix
│   └── summary_report.txt              # Full clustering & PCA evaluation metrics
├── iris_clustering.py                  # Core K-Means, PCA, & plotting pipeline
├── iris_analysis.ipynb                 # Executed Jupyter Notebook
├── app.py                              # Interactive Streamlit clustering explorer
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd week3
pip install -r requirements.txt

# Run K-Means + PCA pipeline and generate plots & CSV
python iris_clustering.py

# Launch interactive Streamlit app
streamlit run app.py
```
