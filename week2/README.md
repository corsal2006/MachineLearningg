# Week 2 — House Price & Titanic Survival Prediction

A clean, modular, end-to-end Week 2 machine learning project inside `week2/`:

- **Assignment 1 & Mini Project 2 ("House Price Prediction Model")**: Trains a leak-free `LinearRegression` pipeline on the Kaggle Housing Prices dataset ([`yasserh/housing-prices-dataset`](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)), evaluates $R^2$, Adjusted $R^2$, 5-Fold CV $R^2$, RMSE, and MAE, and plots **Predicted vs. Actual** house prices and residuals.
- **Assignment 2 ("Titanic Survival Prediction")**: Trains a leak-free `LogisticRegression` pipeline on the Kaggle Titanic dataset ([`yasserh/titanic-dataset`](https://www.kaggle.com/datasets/yasserh/titanic-dataset)), evaluating Accuracy, ROC-AUC, Confusion Matrix, and ROC Curve.

## Project Structure

```text
week2/
├── data/
│   ├── Housing.csv                # Raw Kaggle House Prices dataset (545 × 13)
│   └── Titanic-Dataset.csv        # Raw Kaggle Titanic dataset (891 × 12)
├── outputs/
│   ├── housing_predictions.csv    # Test set Actual vs. Predicted prices & residuals
│   ├── predicted_vs_actual.png    # Predicted vs. Actual & residuals visualization
│   ├── titanic_logistic_eval.png  # Confusion Matrix & ROC Curve for Titanic
│   └── summary_report.txt         # Full metrics & feature coefficients report
├── week2_models.py                # Core training, evaluation, and plotting pipeline
├── week2_analysis.ipynb           # Executed Jupyter Notebook
├── app.py                         # Interactive Streamlit dashboard & estimators
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd week2
pip install -r requirements.txt

# Run the full Week 2 pipeline and generate outputs
python week2_models.py

# Launch the interactive Streamlit app
streamlit run app.py
```
