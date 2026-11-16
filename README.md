# L&T Realty House Price Prediction (ML Regression)

A beginner-friendly end-to-end machine learning project using the included `train.csv` dataset for Indian urban house price prediction. The training target and app predictions are converted/displayed in Indian rupees (INR, ₹).

## What is included?

- Data loading and basic cleaning
- Automatic target-column detection (or manually specify the target)
- Missing-value imputation and categorical encoding
- Exploratory data analysis outputs
- Comparison of Linear Regression, Ridge, Random Forest and Histogram Gradient Boosting
- Cross-validation hyperparameter tuning for Random Forest
- MAE, RMSE and R² evaluation on a held-out test split
- Actual-vs-predicted plot
- Permutation feature importance
- Saved model pipeline and Streamlit demo app

## Important data note

The recommended Kaggle Indian Urban House Price Prediction dataset is synthetic. This is suitable for learning and demonstrating an ML workflow, but it is **not actual L&T Realty transaction data**. Do not claim that the model is trained or validated on L&T Realty's private sales. Real-world use requires authorized, representative transaction data and further validation.

## 1. Get the data

Download `train.csv` from:
https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data

Place `train.csv` in this project folder.

## 2. Install dependencies

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux/WSL:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
```

## 3. Train and evaluate

```bash
python train_model.py --data train.csv
```

The script attempts to detect a target column such as `Price_INR_Lakhs`. Explicit targets labelled in lakhs/crores are converted to rupees; explicitly USD/dollar targets use the requested approximate rate of **$1 = ₹100**. Targets labelled in INR/rupees are treated as already being rupees. A generic target with no clear unit is left unchanged, so verify its unit before relying on predictions. If the script cannot safely identify the target, use the exact column name printed by the error:

```bash
python train_model.py --data train.csv --target "EXACT_PRICE_COLUMN_NAME"
```

The script excludes obvious price-per-square-foot columns because they may leak information from the target. Inspect the output and `reports/` folder after training.

## 4. Run the app

After training:
```bash
streamlit run app.py
```

## 5. Project outputs

- `artifacts/lt_realty_price_model.pkl` — trained preprocessing + model pipeline
- `artifacts/model_metadata.json` — feature names, model and metrics
- `reports/data_summary.csv` — summary statistics
- `reports/missing_values.csv` — missing-value counts
- `reports/model_comparison.csv` — model metrics
- `reports/final_metrics.json` — selected model's test metrics
- `reports/test_predictions.csv` — actual and predicted test values
- `reports/actual_vs_predicted.png` — evaluation plot
- `reports/feature_importance.csv` and `.png` — permutation importance
- `reports/target_distribution.png` — target distribution
- `reports/numeric_correlations.csv` and heatmap — numeric correlations

## How to interpret the metrics

- MAE: average absolute prediction error, in the target's unit.
- RMSE: penalizes large prediction errors more than MAE.
- R²: compares the model against predicting the test-set mean; higher is generally better, but a high score alone does not prove real-world usefulness.

The held-out test split is used for final evaluation. Hyperparameter tuning uses training data only. For time-dependent or repeated-property data, consider a time-based or group-based split.

## Suggested report structure

1. Abstract and problem statement
2. Dataset source, fields, target definition and limitations
3. Preprocessing
4. EDA findings and plots
5. Models and training methodology
6. Evaluation table (MAE, RMSE, R²)
7. Hyperparameter optimization
8. Feature importance and interpretation
9. Streamlit app screenshots
10. Limitations, ethics and future scope

## Before submission

- Run the code against the actual downloaded CSV.
- Confirm the detected target column and unit.
- Inspect `reports/model_comparison.csv` and report only actual measured results.
- Never invent model accuracy or feature-importance findings.
- State clearly that synthetic data cannot validate actual L&T Realty pricing.
