"""
Titanic Survival Prediction - Interactive Data Cleaning & Prediction App
========================================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_cleaning import (
    PALETTE,
    CleaningMetadata,
    DatasetSummary,
    ModelMetrics,
    clean_and_encode,
    evaluate_survival_models,
    load_dataset,
    summarize_dataset,
)

st.set_page_config(
    page_title="Titanic Data Cleaning & Survival Prediction",
    page_icon="🚢",
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


@st.cache_data
def get_processed_data() -> (
    tuple[pd.DataFrame, DatasetSummary, pd.DataFrame, CleaningMetadata, dict[str, ModelMetrics]]
):
    df_raw = load_dataset()
    stats = summarize_dataset(df_raw)
    df_clean, meta = clean_and_encode(df_raw)
    model_results = evaluate_survival_models(df_raw)
    return df_raw, stats, df_clean, meta, model_results


@st.cache_resource
def train_predictor(df_raw: pd.DataFrame) -> Pipeline:
    df = df_raw.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "HasCabin", "Embarked"]
    X = df.loc[:, features]
    y = df["Survived"]

    prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), ["Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin"]),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
                    ]
                ),
                ["Sex", "Embarked"],
            ),
        ]
    )
    model = Pipeline(
        [
            ("prep", prep),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Titanic Survival Prediction — Data Cleaning Project")
    st.caption("Week 1 Mini Project | Kaggle Dataset (`yasserh/titanic-dataset`)")

    df_raw, stats, df_clean, meta, model_results = get_processed_data()
    predictor = train_predictor(df_raw)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Passengers", f"{len(df_raw):,}")
    with c2:
        metric_card("Missing Cells (Before → After)", f"{int(df_raw.isnull().sum().sum())} → {meta['remaining_nulls']}")
    with c3:
        metric_card("Median Age Imputed", f"{meta['age_median_before']} yrs ({meta['age_imputed_count']} rows)")
    with c4:
        rf_acc = model_results["Random Forest"]["test_accuracy"] * 100
        metric_card("Random Forest Holdout Acc", f"{rf_acc:.1f}%")

    st.markdown("---")
    tab_data, tab_viz, tab_pred = st.tabs(
        ["1. Summary & Cleaned Data", "2. Age Distribution & EDA", "3. Survival Prediction"]
    )

    with tab_data:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Assignment 1: Dataset `.info()` & Missing Values")
            st.code(stats["info_text"], language="text")
            st.dataframe(stats["missing_summary"], width="stretch")
        with col_right:
            st.subheader("Assignment 1: Numeric `.describe()`")
            st.dataframe(stats["describe_numeric"], width="stretch")

        st.subheader("Assignments 2 & 3: Imputation & Encoding Summary")
        st.info(
            f"**Age**: Imputed {meta['age_imputed_count']} missing rows using **median** ({meta['age_median_before']} yrs; mean was {meta['age_mean_before']} yrs).  \n"
            f"**Embarked**: Imputed {meta['embarked_imputed_count']} missing rows using **mode** (`{meta['embarked_mode']}`), then **OneHotEncoded** into `{', '.join(meta['embarked_columns'])}`.  \n"
            f"**Sex**: Encoded via **LabelEncoder** `{meta['sex_mapping']}`.  \n"
            f"**Cabin**: Converted to binary `HasCabin` (0/1) to retain signal without sparse nulls."
        )

        st.subheader("Cleaned Dataset Preview (`titanic_cleaned.csv`)")
        st.dataframe(df_clean.head(15), width="stretch")
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name="titanic_cleaned.csv",
            mime="text/csv",
        )

    with tab_viz:
        st.subheader("Interactive Age Distribution (Matplotlib / Seaborn)")
        f_col1, f_col2 = st.columns(2)
        selected_pclass = f_col1.multiselect("Filter by Passenger Class", [1, 2, 3], default=[1, 2, 3])
        selected_sex = f_col2.multiselect("Filter by Sex", ["female", "male"], default=["female", "male"])

        filtered = df_raw.loc[df_raw["Pclass"].isin(selected_pclass) & df_raw["Sex"].isin(selected_sex)].copy()
        filtered["Age_Clean"] = pd.Series(filtered["Age"]).fillna(meta["age_median_before"])
        filtered["Outcome"] = pd.Series(filtered["Survived"]).map(
            lambda v: "Survived" if int(v) == 1 else "Did Not Survive"
        )

        if filtered.empty:
            st.warning("Select at least one class and sex to render the plot.")
        else:
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
            raw_age_df = filtered.dropna(subset=["Age"])
            sns.histplot(
                data=raw_age_df,
                x="Age",
                bins=25,
                kde=True,
                stat="density",
                color=PALETTE["slate"],
                alpha=0.35,
                label="Raw Age",
                ax=axes[0],
            )
            sns.kdeplot(
                data=filtered,
                x="Age_Clean",
                color=PALETTE["teal"],
                linewidth=2.2,
                label="Imputed Age",
                ax=axes[0],
            )
            axes[0].axvline(
                meta["age_median_before"],
                color=PALETTE["coral"],
                linestyle="--",
                label=f"Global Median ({meta['age_median_before']})",
            )
            axes[0].set_title("Age Distribution: Raw vs. Imputed", fontweight="bold")
            axes[0].set_xlabel("Age (Years)")
            axes[0].legend()

            sns.histplot(
                data=filtered,
                x="Age_Clean",
                hue="Outcome",
                bins=25,
                kde=True,
                element="step",
                palette={"Did Not Survive": PALETTE["coral"], "Survived": PALETTE["teal"]},
                alpha=0.35,
                ax=axes[1],
            )
            axes[1].set_title("Age Distribution by Survival", fontweight="bold")
            axes[1].set_xlabel("Age (Years)")
            sns.despine(fig=fig)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_pred:
        st.subheader("Model Benchmark (Leak-Free Evaluation)")
        bench_rows = [
            {
                "Model": name,
                "5-Fold CV Accuracy": f"{m['cv_accuracy_mean']:.4f} ± {m['cv_accuracy_std']:.4f}",
                "Holdout Accuracy": f"{m['test_accuracy']:.4f}",
                "Holdout ROC-AUC": f"{m['test_roc_auc']:.4f}",
            }
            for name, m in model_results.items()
        ]
        st.dataframe(pd.DataFrame(bench_rows), width="stretch", hide_index=True)

        st.subheader("Interactive Passenger Survival Simulator")
        p1, p2, p3, p4 = st.columns(4)
        pclass = p1.selectbox("Passenger Class", [1, 2, 3], index=0)
        sex = p2.selectbox("Sex", ["female", "male"], index=0)
        age = p3.slider("Age", 1, 80, 28)
        fare = p4.slider("Fare ($)", 0.0, 250.0, 32.0, step=1.0)

        p5, p6, p7, p8 = st.columns(4)
        sibsp = p5.number_input("Siblings / Spouses (SibSp)", min_value=0, max_value=8, value=0)
        parch = p6.number_input("Parents / Children (Parch)", min_value=0, max_value=6, value=0)
        embarked = p7.selectbox("Embarked Port", ["S", "C", "Q"], index=0)
        has_cabin = p8.selectbox("Has Recorded Cabin?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        sample = pd.DataFrame(
            [
                {
                    "Pclass": pclass,
                    "Sex": sex,
                    "Age": float(age),
                    "SibSp": int(sibsp),
                    "Parch": int(parch),
                    "Fare": float(fare),
                    "HasCabin": int(has_cabin),
                    "Embarked": embarked,
                }
            ]
        )
        prob_survive = float(predictor.predict_proba(sample)[0, 1])
        pred_label = "Likely Survived" if prob_survive >= 0.5 else "Likely Did Not Survive"
        st.success(f"**Prediction:** {pred_label} — Estimated Survival Probability: **{prob_survive * 100:.1f}%**")


if __name__ == "__main__":
    main()
