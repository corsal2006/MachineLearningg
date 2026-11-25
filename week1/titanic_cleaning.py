"""
Titanic Survival Prediction - Data Cleaning & Preprocessing Pipeline
===================================================================
Covers:
  1. Loading the Kaggle Titanic dataset (yasserh/titanic-dataset) with Pandas
     and summarizing basic statistics (.info(), .describe()).
  2. Handling missing values (median/mode imputation + Cabin indicator).
  3. Encoding categorical variables (`Sex` via LabelEncoder, `Embarked` via OneHotEncoder).
  4. Visualizing Age distributions & EDA summaries with Matplotlib and Seaborn.
  5. Exporting the cleaned dataset to CSV (`outputs/titanic_cleaned.csv`).
  6. Evaluating leak-free baseline survival prediction models.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Titanic-Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEANED_CSV_PATH = OUTPUT_DIR / "titanic_cleaned.csv"
AGE_PLOT_PATH = OUTPUT_DIR / "age_distribution.png"
EDA_PLOT_PATH = OUTPUT_DIR / "titanic_eda_summary.png"
REPORT_PATH = OUTPUT_DIR / "summary_report.txt"

# Consistent visual theme
PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#E69F00",
    "slate": "#5C6B73",
    "bg": "#F8FAFC",
}


class DatasetSummary(TypedDict):
    shape: tuple[int, int]
    info_text: str
    describe_numeric: pd.DataFrame
    describe_all: pd.DataFrame
    missing_summary: pd.DataFrame


class CleaningMetadata(TypedDict):
    age_mean_before: float
    age_median_before: float
    age_imputed_count: int
    embarked_mode: str
    embarked_imputed_count: int
    sex_mapping: dict[str, int]
    embarked_columns: list[str]
    remaining_nulls: int


class ModelMetrics(TypedDict):
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    test_roc_auc: float
    classification_report: str


def load_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Titanic dataset locally, downloading from Kaggle via kagglehub if absent."""
    if not csv_path.exists():
        import kagglehub

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_dir = Path(kagglehub.dataset_download("yasserh/titanic-dataset"))
        src = downloaded_dir / "Titanic-Dataset.csv"
        csv_path.write_bytes(src.read_bytes())
    return pd.read_csv(csv_path)


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    """Capture .info(), .describe(), and missing value statistics."""
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    return {
        "shape": df.shape,
        "info_text": info_buf.getvalue(),
        "describe_numeric": df.describe().round(3),
        "describe_all": df.describe(include="all").round(3),
        "missing_summary": missing_df.loc[missing_df["missing_count"] > 0],
    }


def clean_and_encode(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningMetadata]:
    """
    Clean missing values and encode `Sex` (LabelEncoder) and `Embarked` (OneHotEncoder).

    - Age: Median imputation via SimpleImputer (robust to right-skewed ages).
    - Embarked: Most-frequent (mode) imputation via SimpleImputer.
    - Cabin: Converted into binary `HasCabin` flag; sparse raw text column dropped.
    - Sex: Encoded with LabelEncoder (female -> 0, male -> 1).
    - Embarked: Encoded with OneHotEncoder -> Embarked_C, Embarked_Q, Embarked_S.
    """
    df = df_raw.copy()

    age_mean_before = float(df["Age"].mean())
    age_median_before = float(df["Age"].median())

    # 1. Handle missing data
    age_imputer = SimpleImputer(strategy="median")
    df["Age"] = age_imputer.fit_transform(df[["Age"]]).ravel()

    embarked_imputer = SimpleImputer(strategy="most_frequent")
    df["Embarked"] = embarked_imputer.fit_transform(df[["Embarked"]]).ravel()

    fare_imputer = SimpleImputer(strategy="median")
    df["Fare"] = fare_imputer.fit_transform(df[["Fare"]]).ravel()

    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # 2. Encode Sex with LabelEncoder
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    classes = [str(c) for c in (label_encoder.classes_ if label_encoder.classes_ is not None else [])]
    encoded_vals = [int(v) for v in np.asarray(label_encoder.transform(classes)).tolist()]
    sex_mapping = dict(zip(classes, encoded_vals))

    # 3. Encode Embarked with OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    embarked_encoded = np.asarray(ohe.fit_transform(df[["Embarked"]]), dtype=int)
    embarked_cols = [f"Embarked_{cat}" for cat in ohe.categories_[0]]
    embarked_df = pd.DataFrame(embarked_encoded, columns=pd.Index(embarked_cols), index=df.index)

    # Assemble clean, ML-ready DataFrame with zero missing values
    feature_cols = [
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "HasCabin",
    ]
    cleaned_df = pd.concat([df.loc[:, feature_cols], embarked_df], axis=1)

    embarked_stats = embarked_imputer.statistics_
    embarked_mode = str(embarked_stats[0]) if embarked_stats is not None else "S"

    metadata: CleaningMetadata = {
        "age_mean_before": round(age_mean_before, 2),
        "age_median_before": round(age_median_before, 2),
        "age_imputed_count": int(df_raw["Age"].isnull().sum()),
        "embarked_mode": embarked_mode,
        "embarked_imputed_count": int(df_raw["Embarked"].isnull().sum()),
        "sex_mapping": sex_mapping,
        "embarked_columns": embarked_cols,
        "remaining_nulls": int(cleaned_df.isnull().sum().sum()),
    }
    return cleaned_df, metadata


def plot_age_distribution(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame, meta: CleaningMetadata, save_path: Path = AGE_PLOT_PATH
) -> None:
    """Create a polished two-panel Matplotlib/Seaborn visualization of Age distribution."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    # Left panel: Before vs After Median Imputation
    ax1 = axes[0]
    raw_age_df = df_raw.dropna(subset=["Age"])
    sns.histplot(
        data=raw_age_df,
        x="Age",
        bins=30,
        kde=True,
        stat="density",
        color=PALETTE["slate"],
        alpha=0.35,
        label=f"Raw Age (n={len(raw_age_df)})",
        ax=ax1,
    )
    sns.kdeplot(
        data=df_clean,
        x="Age",
        color=PALETTE["teal"],
        linewidth=2.4,
        label=f"Imputed Age (n={len(df_clean)})",
        ax=ax1,
    )
    ax1.axvline(
        meta["age_median_before"],
        color=PALETTE["coral"],
        linestyle="--",
        linewidth=1.8,
        label=f"Median = {meta['age_median_before']} yrs",
    )
    ax1.axvline(
        meta["age_mean_before"],
        color=PALETTE["amber"],
        linestyle=":",
        linewidth=1.8,
        label=f"Mean = {meta['age_mean_before']} yrs",
    )
    ax1.set_title("Age Distribution: Raw vs. Median Imputed", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Age (Years)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", fontsize=9.5)

    # Right panel: Age Distribution by Survival Status
    ax2 = axes[1]
    plot_df = df_clean.loc[:, ["Age", "Survived"]].copy()
    plot_df["Outcome"] = plot_df["Survived"].map(lambda v: "Survived" if int(v) == 1 else "Did Not Survive")
    sns.histplot(
        data=plot_df,
        x="Age",
        hue="Outcome",
        bins=28,
        kde=True,
        element="step",
        palette={"Did Not Survive": PALETTE["coral"], "Survived": PALETTE["teal"]},
        alpha=0.35,
        ax=ax2,
    )
    ax2.set_title("Age Distribution by Survival Outcome", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Age (Years)", fontsize=11)
    ax2.set_ylabel("Passenger Count", fontsize=11)

    sns.despine(fig=fig)
    fig.suptitle(
        "Titanic Dataset — Passenger Age Distribution Analysis",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame, save_path: Path = EDA_PLOT_PATH) -> None:
    """Create a 2x2 EDA dashboard covering missing data, survival drivers, and correlations."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")

    # 1. Missing values before cleaning
    ax = axes[0, 0]
    missing_counts = df_raw.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
    bars = ax.barh(
        missing_counts.index,
        missing_counts.to_numpy(),
        color=[PALETTE["amber"], PALETTE["teal"], PALETTE["coral"]],
    )
    ax.set_title("Missing Values Before Cleaning", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Missing Row Count")
    for bar, val in zip(bars, missing_counts.to_numpy()):
        pct = float(val) / len(df_raw) * 100
        ax.text(float(val) + 10, bar.get_y() + bar.get_height() / 2, f"{int(val)} ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_xlim(0, float(missing_counts.max()) * 1.25)

    # 2. Survival rate by Sex and Passenger Class
    ax = axes[0, 1]
    sns.barplot(
        data=df_raw,
        x="Pclass",
        y="Survived",
        hue="Sex",
        palette={"female": PALETTE["teal"], "male": PALETTE["navy"]},
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Survival Rate by Passenger Class & Sex", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Passenger Class (1 = Upper, 3 = Lower)")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1.05)

    # 3. Survival rate by Embarked Port
    ax = axes[1, 0]
    emb_df = df_raw.copy()
    emb_df["Embarked"] = emb_df["Embarked"].fillna("S")
    port_names = {"C": "Cherbourg (C)", "Q": "Queenstown (Q)", "S": "Southampton (S)"}
    emb_df["Port"] = emb_df["Embarked"].map(lambda v: port_names.get(str(v), str(v)))
    sns.barplot(
        data=emb_df,
        x="Port",
        y="Survived",
        hue="Port",
        palette=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"]],
        legend=False,
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Survival Rate by Embarkation Port", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Embarkation Port")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 0.7)

    # 4. Correlation heatmap on cleaned numeric features
    ax = axes[1, 1]
    corr_df = df_clean.drop(columns=["PassengerId"])
    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.7,
        vmax=0.7,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Cleaned Data)", fontsize=12.5, fontweight="bold")

    sns.despine(fig=fig)
    fig.suptitle(
        "Titanic Survival Prediction — Exploratory & Cleaning Summary",
        fontsize=15,
        fontweight="bold",
        y=0.99,
        color=PALETTE["navy"],
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_survival_models(df_raw: pd.DataFrame) -> dict[str, ModelMetrics]:
    """
    Train and evaluate leak-free survival prediction pipelines on raw features.

    Imputers and encoders are fit strictly inside train/CV folds to prevent data leakage.
    """
    df = df_raw.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "HasCabin", "Embarked"]
    X = df.loc[:, feature_cols]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin"]
    categorical_features = ["Sex", "Embarked"]

    def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
        num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        cat_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(num_steps), numeric_features),
                ("cat", Pipeline(cat_steps), categorical_features),
            ]
        )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=True)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=False)),
                ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, ModelMetrics] = {}

    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        report_text = str(
            classification_report(y_test, preds, target_names=["Did Not Survive", "Survived"])
        )
        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "classification_report": report_text,
        }

    return results


def run_pipeline() -> None:
    """Execute the full Week 1 Titanic Data Cleaning & Survival Prediction workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset()
    stats = summarize_dataset(df_raw)
    df_clean, meta = clean_and_encode(df_raw)
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)

    plot_age_distribution(df_raw, df_clean, meta)
    plot_eda_summary(df_raw, df_clean)
    model_results = evaluate_survival_models(df_raw)

    report_lines = [
        "=" * 70,
        "TITANIC SURVIVAL PREDICTION - DATA CLEANING REPORT",
        "=" * 70,
        f"Raw Dataset Shape     : {stats['shape']}",
        f"Cleaned Dataset Shape : {df_clean.shape}",
        f"Remaining Null Values : {meta['remaining_nulls']}",
        "",
        "--- Assignment 1: Dataset .info() ---",
        stats["info_text"].strip(),
        "",
        "--- Assignment 1: Numeric Summary (.describe()) ---",
        stats["describe_numeric"].to_string(),
        "",
        "--- Missing Values Before Cleaning ---",
        stats["missing_summary"].to_string(),
        "",
        "--- Assignment 2 & 3: Cleaning & Encoding Summary ---",
        f"Age imputed ({meta['age_imputed_count']} rows) with Median = {meta['age_median_before']} (Mean was {meta['age_mean_before']})",
        f"Embarked imputed ({meta['embarked_imputed_count']} rows) with Mode = '{meta['embarked_mode']}'",
        "Cabin converted to binary indicator `HasCabin` (0/1); raw sparse column dropped.",
        f"Sex LabelEncoder mapping : {meta['sex_mapping']}",
        f"Embarked OneHotEncoder columns : {meta['embarked_columns']}",
        "",
        "--- Survival Prediction Model Evaluation (Leak-Free 5-Fold CV & 80/20 Holdout) ---",
    ]
    for model_name, metrics in model_results.items():
        report_lines.extend(
            [
                f"\n[{model_name}]",
                f"  5-Fold CV Accuracy : {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}",
                f"  Holdout Accuracy   : {metrics['test_accuracy']:.4f}",
                f"  Holdout ROC-AUC    : {metrics['test_roc_auc']:.4f}",
                metrics["classification_report"].strip(),
            ]
        )
    report_lines.extend(
        [
            "",
            "--- Generated Artifacts ---",
            f"Cleaned CSV  : {CLEANED_CSV_PATH}",
            f"Age Plot     : {AGE_PLOT_PATH}",
            f"EDA Dashboard: {EDA_PLOT_PATH}",
        ]
    )

    report_text = "\n".join(report_lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    run_pipeline()
