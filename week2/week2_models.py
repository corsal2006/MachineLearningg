"""
Week 2 — Linear Regression (House Price Prediction) & Logistic Regression (Titanic)
===================================================================================
Covers:
  1. Assignment 1 & Mini Project 2:
     - Load Kaggle House Prices dataset (`yasserh/housing-prices-dataset`).
     - Train a leak-free Linear Regression Pipeline to predict `price`.
     - Evaluate R^2, Adjusted R^2, RMSE, MAE, and 5-Fold CV R^2.
     - Plot Predicted vs. Actual values and Residuals (Matplotlib & Seaborn).
     - Export test predictions CSV (`outputs/housing_predictions.csv`).
  2. Assignment 2:
     - Load Kaggle Titanic dataset (`yasserh/titanic-dataset`).
     - Train a leak-free Logistic Regression Pipeline for survival prediction.
     - Evaluate Accuracy, ROC-AUC, 5-Fold CV Accuracy, Confusion Matrix & ROC Curve.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

HOUSING_DATA_PATH = DATA_DIR / "Housing.csv"
TITANIC_DATA_PATH = DATA_DIR / "Titanic-Dataset.csv"

HOUSING_PLOT_PATH = OUTPUT_DIR / "predicted_vs_actual.png"
TITANIC_PLOT_PATH = OUTPUT_DIR / "titanic_logistic_eval.png"
HOUSING_PRED_CSV_PATH = OUTPUT_DIR / "housing_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#E69F00",
    "slate": "#5C6B73",
}

HOUSING_NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
HOUSING_CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]


class HousingResults(TypedDict):
    pipeline: Pipeline
    dataset_shape: tuple[int, int]
    train_size: int
    test_size: int
    r2_train: float
    r2_test: float
    adj_r2_test: float
    cv_r2_mean: float
    cv_r2_std: float
    rmse_test: float
    mae_test: float
    predictions_df: pd.DataFrame
    coefficients_df: pd.DataFrame


class TitanicLogisticResults(TypedDict):
    pipeline: Pipeline
    dataset_shape: tuple[int, int]
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    test_roc_auc: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    classification_report: str


def _download_if_missing(target_path: Path, kaggle_slug: str, filename: str) -> Path:
    if not target_path.exists():
        import kagglehub

        target_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_dir = Path(kagglehub.dataset_download(kaggle_slug))
        src = downloaded_dir / filename
        target_path.write_bytes(src.read_bytes())
    return target_path


def load_housing_dataset(csv_path: Path = HOUSING_DATA_PATH) -> pd.DataFrame:
    """Load Housing Prices dataset locally, fetching from Kaggle if missing."""
    _download_if_missing(csv_path, "yasserh/housing-prices-dataset", "Housing.csv")
    return pd.read_csv(csv_path)


def load_titanic_dataset(csv_path: Path = TITANIC_DATA_PATH) -> pd.DataFrame:
    """Load Titanic dataset locally, fetching from Kaggle if missing."""
    _download_if_missing(csv_path, "yasserh/titanic-dataset", "Titanic-Dataset.csv")
    return pd.read_csv(csv_path)


def build_housing_pipeline() -> Pipeline:
    """Create a leak-free preprocessing + Linear Regression pipeline for house prices."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                HOUSING_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                HOUSING_CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("prep", preprocessor), ("regressor", LinearRegression())])


def train_and_evaluate_housing(df_housing: pd.DataFrame) -> HousingResults:
    """Train Linear Regression on housing dataset and evaluate R^2, RMSE, MAE, and CV R^2."""
    feature_cols = HOUSING_NUMERIC_FEATURES + HOUSING_CATEGORICAL_FEATURES
    X = df_housing.loc[:, feature_cols]
    y = df_housing["price"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_housing_pipeline()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")

    pipeline.fit(X_train, y_train)
    y_train_pred = np.asarray(pipeline.predict(X_train), dtype=float)
    y_test_pred = np.asarray(pipeline.predict(X_test), dtype=float)

    r2_train = float(r2_score(y_train, y_train_pred))
    r2_test = float(r2_score(y_test, y_test_pred))

    prep: ColumnTransformer = pipeline.named_steps["prep"]
    reg: LinearRegression = pipeline.named_steps["regressor"]
    feature_names = [str(name).split("__", 1)[-1] for name in prep.get_feature_names_out()]
    n_samples = len(y_test)
    n_features = len(feature_names)
    adj_r2_test = 1.0 - (1.0 - r2_test) * (n_samples - 1) / max(1, n_samples - n_features - 1)

    y_test_arr = np.asarray(y_test, dtype=float)
    residuals = y_test_arr - y_test_pred
    predictions_df = pd.DataFrame(
        {
            "Actual_Price": y_test_arr,
            "Predicted_Price": np.round(y_test_pred, 2),
            "Residual": np.round(residuals, 2),
            "Abs_Error": np.round(np.abs(residuals), 2),
            "Pct_Error": np.round(np.abs(residuals) / y_test_arr * 100.0, 2),
        }
    )

    coefficients_df = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": np.round(np.asarray(reg.coef_, dtype=float), 2)}
    ).sort_values(by="Coefficient", key=lambda s: s.abs(), ascending=False)

    return {
        "pipeline": pipeline,
        "dataset_shape": df_housing.shape,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "r2_train": round(r2_train, 4),
        "r2_test": round(r2_test, 4),
        "adj_r2_test": round(float(adj_r2_test), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "rmse_test": round(float(root_mean_squared_error(y_test, y_test_pred)), 2),
        "mae_test": round(float(mean_absolute_error(y_test, y_test_pred)), 2),
        "predictions_df": predictions_df,
        "coefficients_df": coefficients_df.reset_index(drop=True),
    }


def plot_predicted_vs_actual(results: HousingResults, save_path: Path = HOUSING_PLOT_PATH) -> None:
    """Plot Predicted vs. Actual house prices and residual distribution with Seaborn/Matplotlib."""
    preds_df = results["predictions_df"]
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    fig.patch.set_facecolor("white")

    # Left panel: Predicted vs Actual Scatter + Ideal Line + Regression Trend
    ax1 = axes[0]
    plot_df = pd.DataFrame(
        {
            "Actual (Millions)": preds_df["Actual_Price"] / 1e6,
            "Predicted (Millions)": preds_df["Predicted_Price"] / 1e6,
            "Residual (Millions)": preds_df["Residual"] / 1e6,
        }
    )
    sns.regplot(
        data=plot_df,
        x="Actual (Millions)",
        y="Predicted (Millions)",
        scatter_kws={"color": PALETTE["teal"], "alpha": 0.72, "s": 55, "edgecolor": "white"},
        line_kws={"color": PALETTE["navy"], "linewidth": 2.0, "label": "Linear Fit"},
        ax=ax1,
    )
    min_val = float(min(plot_df["Actual (Millions)"].min(), plot_df["Predicted (Millions)"].min()))
    max_val = float(max(plot_df["Actual (Millions)"].max(), plot_df["Predicted (Millions)"].max()))
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=PALETTE["coral"], linewidth=1.8, label="Ideal (y = x)")
    ax1.set_title("Predicted vs. Actual House Prices (Test Set)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Actual Price (Millions)", fontsize=11)
    ax1.set_ylabel("Predicted Price (Millions)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.text(
        0.04,
        0.84,
        f"$R^2$ = {results['r2_test']:.4f}\nRMSE = {results['rmse_test']:,.0f}\nMAE  = {results['mae_test']:,.0f}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    # Right panel: Residuals Distribution
    ax2 = axes[1]
    sns.histplot(
        data=plot_df,
        x="Residual (Millions)",
        bins=22,
        kde=True,
        color=PALETTE["navy"],
        alpha=0.45,
        ax=ax2,
    )
    ax2.axvline(0.0, color=PALETTE["coral"], linestyle="--", linewidth=1.8, label="Zero Error")
    ax2.set_title("Test Residuals Distribution (Actual - Predicted)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Residual (Millions)", fontsize=11)
    ax2.set_ylabel("House Count", fontsize=11)
    ax2.legend(frameon=True, facecolor="white")

    sns.despine(fig=fig)
    fig.suptitle(
        "Mini Project 2: House Price Prediction — Linear Regression Evaluation",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_titanic_logistic(df_titanic: pd.DataFrame) -> TitanicLogisticResults:
    """Train Logistic Regression on Titanic dataset for survival prediction (Assignment 2)."""
    df = df_titanic.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin"]
    categorical_features = ["Sex", "Embarked"]
    X = df.loc[:, numeric_features + categorical_features]
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    pipeline.fit(X_train, y_train)
    y_pred = np.asarray(pipeline.predict(X_test), dtype=int)
    y_prob = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "pipeline": pipeline,
        "dataset_shape": df_titanic.shape,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "classification_report": str(
            classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"])
        ),
    }


def plot_titanic_logistic(results: TitanicLogisticResults, save_path: Path = TITANIC_PLOT_PATH) -> None:
    """Plot Confusion Matrix and ROC Curve for Titanic Logistic Regression."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor("white")

    # Left: Confusion Matrix
    ax1 = axes[0]
    cm = results["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix (Holdout Set)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.set_ylabel("True Label", fontsize=11)

    # Right: ROC Curve
    ax2 = axes[1]
    ax2.plot(
        results["fpr"],
        results["tpr"],
        color=PALETTE["teal"],
        linewidth=2.4,
        label=f"Logistic Regression (AUC = {results['test_roc_auc']:.4f})",
    )
    ax2.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["coral"], linewidth=1.6, label="Random Baseline")
    ax2.set_title("ROC Curve — Titanic Survival Prediction", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.legend(loc="lower right", frameon=True, facecolor="white")

    sns.despine(fig=fig)
    fig.suptitle(
        "Assignment 2: Titanic Survival Prediction — Logistic Regression",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pipeline() -> None:
    """Execute both Week 2 models, generate visualizations, and save reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_housing = load_housing_dataset()
    housing_res = train_and_evaluate_housing(df_housing)
    housing_res["predictions_df"].to_csv(HOUSING_PRED_CSV_PATH, index=False)
    plot_predicted_vs_actual(housing_res)

    df_titanic = load_titanic_dataset()
    titanic_res = train_and_evaluate_titanic_logistic(df_titanic)
    plot_titanic_logistic(titanic_res)

    report_lines = [
        "=" * 72,
        "WEEK 2 — LINEAR & LOGISTIC REGRESSION REPORT",
        "=" * 72,
        "",
        "--- Mini Project 2 & Assignment 1: House Price Prediction (Linear Regression) ---",
        f"Dataset Shape      : {housing_res['dataset_shape']} (Train={housing_res['train_size']}, Test={housing_res['test_size']})",
        f"Train R^2 Score    : {housing_res['r2_train']:.4f}",
        f"Test R^2 Score     : {housing_res['r2_test']:.4f}",
        f"Adjusted Test R^2  : {housing_res['adj_r2_test']:.4f}",
        f"5-Fold CV R^2      : {housing_res['cv_r2_mean']:.4f} +/- {housing_res['cv_r2_std']:.4f}",
        f"Test RMSE          : {housing_res['rmse_test']:,.2f}",
        f"Test MAE           : {housing_res['mae_test']:,.2f}",
        "",
        "Top Feature Coefficients (Standardized Numeric + One-Hot Categoricals):",
        housing_res["coefficients_df"].to_string(index=False),
        "",
        "--- Assignment 2: Titanic Survival Prediction (Logistic Regression) ---",
        f"Dataset Shape      : {titanic_res['dataset_shape']}",
        f"5-Fold CV Accuracy : {titanic_res['cv_accuracy_mean']:.4f} +/- {titanic_res['cv_accuracy_std']:.4f}",
        f"Holdout Accuracy   : {titanic_res['test_accuracy']:.4f}",
        f"Holdout ROC-AUC    : {titanic_res['test_roc_auc']:.4f}",
        titanic_res["classification_report"].strip(),
        "",
        "--- Generated Artifacts ---",
        f"Predictions CSV : {HOUSING_PRED_CSV_PATH}",
        f"Housing Plot    : {HOUSING_PLOT_PATH}",
        f"Titanic Plot    : {TITANIC_PLOT_PATH}",
    ]
    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    run_pipeline()
