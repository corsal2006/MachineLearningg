"""
Week 2 — House Price & Titanic Survival Prediction App
======================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from week2_models import (
    PALETTE,
    HousingResults,
    TitanicLogisticResults,
    load_housing_dataset,
    load_titanic_dataset,
    train_and_evaluate_housing,
    train_and_evaluate_titanic_logistic,
)

st.set_page_config(
    page_title="Week 2: House Price & Titanic Prediction",
    page_icon="🏠",
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
def load_and_train_models() -> tuple[pd.DataFrame, HousingResults, pd.DataFrame, TitanicLogisticResults]:
    df_housing = load_housing_dataset()
    housing_res = train_and_evaluate_housing(df_housing)
    df_titanic = load_titanic_dataset()
    titanic_res = train_and_evaluate_titanic_logistic(df_titanic)
    return df_housing, housing_res, df_titanic, titanic_res


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("House Price & Titanic Survival Prediction")
    st.caption("Week 2 Mini Project | Linear Regression (`yasserh/housing-prices-dataset`) & Logistic Regression (`yasserh/titanic-dataset`)")

    df_housing, housing_res, df_titanic, titanic_res = load_and_train_models()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("House Price Test R²", f"{housing_res['r2_test']:.4f}")
    with c2:
        metric_card("5-Fold CV R² (Mean)", f"{housing_res['cv_r2_mean']:.4f}")
    with c3:
        metric_card("House Price Test RMSE", f"{housing_res['rmse_test']:,.0f}")
    with c4:
        metric_card("Titanic Logistic Accuracy", f"{titanic_res['test_accuracy'] * 100:.1f}%")

    st.markdown("---")
    tab_housing, tab_calc, tab_titanic = st.tabs(
        [
            "1. House Price Model & Plots",
            "2. Interactive House Price Predictor",
            "3. Titanic Logistic Regression",
        ]
    )

    with tab_housing:
        st.subheader("Mini Project 2 & Assignment 1: Linear Regression Evaluation")
        preds_df = housing_res["predictions_df"]
        plot_df = pd.DataFrame(
            {
                "Actual (Millions)": preds_df["Actual_Price"] / 1e6,
                "Predicted (Millions)": preds_df["Predicted_Price"] / 1e6,
                "Residual (Millions)": preds_df["Residual"] / 1e6,
            }
        )

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
        sns.regplot(
            data=plot_df,
            x="Actual (Millions)",
            y="Predicted (Millions)",
            scatter_kws={"color": PALETTE["teal"], "alpha": 0.72, "s": 48, "edgecolor": "white"},
            line_kws={"color": PALETTE["navy"], "linewidth": 2.0, "label": "Linear Fit"},
            ax=axes[0],
        )
        min_v = float(min(plot_df["Actual (Millions)"].min(), plot_df["Predicted (Millions)"].min()))
        max_v = float(max(plot_df["Actual (Millions)"].max(), plot_df["Predicted (Millions)"].max()))
        axes[0].plot([min_v, max_v], [min_v, max_v], linestyle="--", color=PALETTE["coral"], linewidth=1.6, label="Ideal (y = x)")
        axes[0].set_title(f"Predicted vs. Actual (R² = {housing_res['r2_test']:.4f})", fontweight="bold")
        axes[0].legend()

        sns.histplot(data=plot_df, x="Residual (Millions)", bins=20, kde=True, color=PALETTE["navy"], alpha=0.45, ax=axes[1])
        axes[1].axvline(0.0, color=PALETTE["coral"], linestyle="--", linewidth=1.6, label="Zero Error")
        axes[1].set_title("Residuals Distribution (Actual - Predicted)", fontweight="bold")
        axes[1].legend()

        sns.despine(fig=fig)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Test Set Predictions Preview")
            st.dataframe(preds_df.head(15), width="stretch", hide_index=True)
            st.download_button(
                label="Download Predictions (CSV)",
                data=preds_df.to_csv(index=False).encode("utf-8"),
                file_name="housing_predictions.csv",
                mime="text/csv",
            )
        with col_r:
            st.subheader("Top Linear Regression Coefficients")
            st.dataframe(housing_res["coefficients_df"], width="stretch", hide_index=True)

    with tab_calc:
        st.subheader("Predict House Price (Interactive Linear Regression)")
        r1, r2, r3, r4, r5 = st.columns(5)
        area = r1.slider("Area (sq ft)", int(df_housing["area"].min()), int(df_housing["area"].max()), 5000, step=100)
        bedrooms = r2.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = r3.selectbox("Bathrooms", [1, 2, 3, 4], index=0)
        stories = r4.selectbox("Stories", [1, 2, 3, 4], index=1)
        parking = r5.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)

        c_a, c_b, c_c, c_d = st.columns(4)
        mainroad = c_a.selectbox("Main Road", ["yes", "no"], index=0)
        guestroom = c_b.selectbox("Guest Room", ["no", "yes"], index=0)
        basement = c_c.selectbox("Basement", ["no", "yes"], index=0)
        hotwaterheating = c_d.selectbox("Hot Water Heating", ["no", "yes"], index=0)

        c_e, c_f, c_g = st.columns(3)
        airconditioning = c_e.selectbox("Air Conditioning", ["yes", "no"], index=0)
        prefarea = c_f.selectbox("Preferred Area", ["no", "yes"], index=0)
        furnishingstatus = c_g.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=1)

        sample_house = pd.DataFrame(
            [
                {
                    "area": int(area),
                    "bedrooms": int(bedrooms),
                    "bathrooms": int(bathrooms),
                    "stories": int(stories),
                    "parking": int(parking),
                    "mainroad": mainroad,
                    "guestroom": guestroom,
                    "basement": basement,
                    "hotwaterheating": hotwaterheating,
                    "airconditioning": airconditioning,
                    "prefarea": prefarea,
                    "furnishingstatus": furnishingstatus,
                }
            ]
        )
        pred_price = float(np.asarray(housing_res["pipeline"].predict(sample_house), dtype=float)[0])
        st.success(f"**Predicted House Price:** {max(0.0, pred_price):,.0f}")

    with tab_titanic:
        st.subheader("Assignment 2: Titanic Survival Prediction (Logistic Regression)")
        m1, m2, m3 = st.columns(3)
        m1.metric("5-Fold CV Accuracy", f"{titanic_res['cv_accuracy_mean']:.4f} ± {titanic_res['cv_accuracy_std']:.4f}")
        m2.metric("Holdout Accuracy", f"{titanic_res['test_accuracy']:.4f}")
        m3.metric("Holdout ROC-AUC", f"{titanic_res['test_roc_auc']:.4f}")

        sns.set_theme(style="whitegrid")
        fig_t, axes_t = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=130)
        sns.heatmap(
            titanic_res["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"],
            ax=axes_t[0],
        )
        axes_t[0].set_title("Confusion Matrix", fontweight="bold")
        axes_t[0].set_xlabel("Predicted")
        axes_t[0].set_ylabel("Actual")

        axes_t[1].plot(
            titanic_res["fpr"],
            titanic_res["tpr"],
            color=PALETTE["teal"],
            linewidth=2.2,
            label=f"AUC = {titanic_res['test_roc_auc']:.4f}",
        )
        axes_t[1].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["coral"], linewidth=1.5)
        axes_t[1].set_title("ROC Curve", fontweight="bold")
        axes_t[1].set_xlabel("False Positive Rate")
        axes_t[1].set_ylabel("True Positive Rate")
        axes_t[1].legend(loc="lower right")
        sns.despine(fig=fig_t)
        fig_t.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

        st.subheader("Interactive Titanic Survival Simulator")
        p1, p2, p3, p4 = st.columns(4)
        pclass = p1.selectbox("Passenger Class", [1, 2, 3], index=0, key="t_pclass")
        sex = p2.selectbox("Sex", ["female", "male"], index=0, key="t_sex")
        age = p3.slider("Age", 1, 80, 28, key="t_age")
        fare = p4.slider("Fare", 0.0, 250.0, 32.0, step=1.0, key="t_fare")

        p5, p6, p7, p8 = st.columns(4)
        sibsp = p5.number_input("SibSp", 0, 8, 0, key="t_sibsp")
        parch = p6.number_input("Parch", 0, 6, 0, key="t_parch")
        embarked = p7.selectbox("Embarked", ["S", "C", "Q"], index=0, key="t_emb")
        has_cabin = p8.selectbox("Has Cabin?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="t_cabin")

        sample_passenger = pd.DataFrame(
            [
                {
                    "Pclass": int(pclass),
                    "Age": float(age),
                    "SibSp": int(sibsp),
                    "Parch": int(parch),
                    "Fare": float(fare),
                    "HasCabin": int(has_cabin),
                    "Sex": sex,
                    "Embarked": embarked,
                }
            ]
        )
        prob_survive = float(titanic_res["pipeline"].predict_proba(sample_passenger)[0, 1])
        label = "Likely Survived" if prob_survive >= 0.5 else "Likely Did Not Survive"
        st.success(f"**Prediction:** {label} — Survival Probability: **{prob_survive * 100:.1f}%** (Dataset rows: {len(df_titanic)})")


if __name__ == "__main__":
    main()
