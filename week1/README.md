# Week 1 — Titanic Survival Prediction (Data Cleaning Project)

A clean, modular, end-to-end Python project built on the Kaggle Titanic dataset ([`yasserh/titanic-dataset`](https://www.kaggle.com/datasets/yasserh/titanic-dataset)).

## Project Structure

```text
week1/
├── data/
│   └── Titanic-Dataset.csv       # Raw Kaggle dataset (891 rows × 12 columns)
├── outputs/
│   ├── titanic_cleaned.csv       # Cleaned, encoded, zero-null ML-ready dataset
│   ├── age_distribution.png      # Raw vs. imputed Age & survival distributions
│   ├── titanic_eda_summary.png   # Missing data, survival drivers, & correlation heatmap
│   └── summary_report.txt        # Full .info(), .describe(), and model benchmark report
├── titanic_cleaning.py           # Core cleaning, encoding, visualization & baseline models
├── titanic_analysis.ipynb        # Step-by-step executed Jupyter notebook
├── app.py                        # Interactive Streamlit dashboard & survival predictor
├── requirements.txt              # Project dependencies
└── README.md
```

## Assignments & Mini Project Coverage

1. **Dataset Summary (`Assignment 1`)**: Loads `Titanic-Dataset.csv` via Pandas and prints `.info()`, `.describe()`, and missing-value statistics.
2. **Missing Data Handling (`Assignment 2` & `Task 1`)**:
   - `Age` (177 missing): Imputed with **median** (`28.0` years) via `SimpleImputer(strategy="median")` because age is right-skewed.
   - `Embarked` (2 missing): Imputed with **mode** (`'S'`) via `SimpleImputer(strategy="most_frequent")`.
   - `Cabin` (687 missing, 77.1%): Extracted into a binary `HasCabin` indicator (`0`/`1`) and raw sparse text dropped.
3. **Categorical Encoding (`Assignment 3` & `Task 2`)**:
   - `Sex`: Encoded with `LabelEncoder` (`female -> 0`, `male -> 1`).
   - `Embarked`: Encoded with `OneHotEncoder` (`Embarked_C`, `Embarked_Q`, `Embarked_S`).
4. **Visualization (`Task 3`)**: Generates publication-ready Matplotlib & Seaborn charts comparing raw vs. imputed `Age` distributions and survival patterns.
5. **Cleaned CSV & Baseline Models (`Task 4` & `Mini Project 1`)**: Exports `outputs/titanic_cleaned.csv` with `0` missing values and evaluates leak-free 5-fold CV & holdout Logistic Regression and Random Forest classifiers.

## Quick Start

```bash
cd week1
pip install -r requirements.txt

# Run the full cleaning, visualization, and model evaluation pipeline
python titanic_cleaning.py

# Launch the interactive Streamlit dashboard
streamlit run app.py
```
