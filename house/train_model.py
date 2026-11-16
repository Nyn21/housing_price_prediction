"""
L&T Realty House Price Prediction
Train, compare, tune, evaluate and save a regression model.

Usage:
    python train_model.py --data train.csv
"""
import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_CANDIDATES = [
    "Price_INR_Lakhs", "price_inr_lakhs", "Price_in_Lakhs",
    "price_in_lakhs", "price", "Price", "Sale_Price",
    "sale_price", "target"
]
LEAKAGE_HINTS = [
    "price_per_sqft", "pricepersqft", "price_per_sq_ft",
    "price_per_square_foot", "price_per_unit"
]
ID_HINTS = ["id", "listing_id", "property_id", "unnamed: 0"]

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def find_target(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Target column '{requested}' not found. Available columns: {list(df.columns)}"
            )
        return requested
    normalized = {normalize_name(c): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    # Fallback: likely target column name contains price but isn't a per-unit feature
    candidates = [
        c for c in df.columns
        if "price" in normalize_name(c)
        and not any(h in normalize_name(c) for h in LEAKAGE_HINTS)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not safely identify the target price column. "
        f"Columns found: {list(df.columns)}. Run with --target YOUR_PRICE_COLUMN."
    )

def clean_target(series):
    # Handles plain numeric columns and common formatted currency strings.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def target_to_inr_multiplier(target_name):
    """Return multiplier that converts a clearly labelled target unit to INR rupees.

    Supported explicit units:
    - INR lakhs / lakhs: 1 lakh = ₹100,000
    - INR crores / crores: 1 crore = ₹10,000,000
    - USD / dollars: fixed project assumption of $1 = ₹100
    - explicitly INR/rupees/Rs targets: already in rupees
    Unknown generic target names are left unchanged rather than guessed.
    """
    name = normalize_name(target_name)
    if "lakh" in name or "lac" in name:
        return 100_000.0, "INR rupees (converted from lakhs at 1 lakh = ₹100,000)"
    if "crore" in name:
        return 10_000_000.0, "INR rupees (converted from crores at 1 crore = ₹10,000,000)"
    if any(token in name for token in ["usd", "dollar", "dollars"]):
        return 100.0, "INR rupees (converted at the requested approximate rate of $1 = ₹100)"
    if any(token in name for token in ["inr", "rupee", "rupees", "rs_"]):
        return 1.0, "INR rupees (target already labelled in INR)"
    return 1.0, "Assumed to already be INR rupees because the target unit is not explicitly labelled"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv", help="Path to Kaggle train.csv")
    parser.add_argument("--target", default=None, help="Optional exact target column name")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The CSV is empty.")

    target = find_target(df, args.target)
    df[target] = clean_target(df[target])
    unit_multiplier, target_unit_note = target_to_inr_multiplier(target)
    if unit_multiplier != 1.0:
        print(f"Converting target '{target}' to INR rupees using multiplier {unit_multiplier:,.0f}.")
        df[target] = df[target] * unit_multiplier
    else:
        print(f"Target unit handling: {target_unit_note}")
    df = df.dropna(subset=[target]).drop_duplicates()
    df = df[df[target] > 0].copy()

    if len(df) < 30:
        print("WARNING: Very few rows. Metrics may not be reliable.")

    # Exclude target, IDs, and direct price-derived columns to avoid target leakage.
    feature_cols = []
    for col in df.columns:
        if col == target:
            continue
        n = normalize_name(col)
        if any(h == n or n.startswith(h + "_") for h in ID_HINTS):
            continue
        if any(h in n for h in LEAKAGE_HINTS):
            print(f"Excluding possible target leakage column: {col}")
            continue
        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable feature columns were found.")

    X = df[feature_cols].copy()
    y = df[target].astype(float)

    # Drop columns with no non-missing values.
    usable = [c for c in X.columns if not X[c].isna().all()]
    X = X[usable]
    feature_cols = usable
    if not feature_cols:
        raise ValueError("All candidate feature columns are empty.")

    # Convert object columns that are mostly numeric into numeric columns.
    for col in X.columns:
        if X[col].dtype == "object":
            parsed = pd.to_numeric(
                X[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
            if parsed.notna().mean() >= 0.85:
                X[col] = parsed

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Drop very high-cardinality text columns (often descriptions/unique identifiers).
    high_card = [
        c for c in categorical_features
        if X[c].nunique(dropna=True) > max(1000, 0.5 * len(X))
    ]
    if high_card:
        print("Excluding high-cardinality text columns:", high_card)
        X = X.drop(columns=high_card)
        feature_cols = [c for c in feature_cols if c not in high_card]
        categorical_features = [c for c in categorical_features if c not in high_card]

    if not feature_cols:
        raise ValueError("No features remain after filtering.")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print(f"Rows used: {len(X):,}")
    print(f"Target column: {target}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # EDA outputs
    summary = df.describe(include="all").transpose()
    summary.to_csv("reports/data_summary.csv")
    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv("reports/missing_values.csv", header=["missing_count"])

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=40)
    plt.title("Distribution of House Prices (INR)")
    plt.xlabel("House price (INR ₹)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/target_distribution.png", dpi=160)
    plt.close()

    if len(numeric_features) > 0:
        corr_df = X[numeric_features].copy()
        corr_df[target] = y
        corr = corr_df.corr(numeric_only=True)
        corr.to_csv("reports/numeric_correlations.csv")
        plt.figure(figsize=(10, 7))
        plt.imshow(corr, aspect="auto")
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Numeric Feature Correlations")
        plt.tight_layout()
        plt.savefig("reports/correlation_heatmap.png", dpi=160)
        plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ], remainder="drop")

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1,
            min_samples_leaf=2
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150, random_state=42
        )
    }

    results = []
    fitted = {}
    for name, estimator in model_specs.items():
        # HistGradientBoosting cannot accept sparse one-hot output reliably,
        # so use dense output for this pipeline if it is selected.
        if name == "Hist Gradient Boosting":
            dense_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            prep = ColumnTransformer([
                ("numeric", numeric_pipe, numeric_features),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", dense_encoder)
                ]), categorical_features)
            ], remainder="drop")
        else:
            prep = preprocessor

        pipe = Pipeline([("preprocessor", prep), ("model", estimator)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results.append({
                "model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred)
            })
            fitted[name] = pipe
            print(f"Finished: {name}")
        except Exception as exc:
            print(f"Skipping {name} due to error: {exc}")

    if not results:
        raise RuntimeError("All models failed. Check the dataset and column types.")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\nModel comparison (lower MAE/RMSE is better; higher R2 is generally better):")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]

    # Tune Random Forest separately, then retain it only if it improves test RMSE.
    # Cross-validation happens on training data only.
    print(f"\nBest initial model: {best_name}")
    if "Random Forest" in fitted:
        rf_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        grid = {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2, 4]
        }
        try:
            search = GridSearchCV(
                rf_pipe, grid, cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1, verbose=0
            )
            search.fit(X_train, y_train)
            tuned_pred = search.best_estimator_.predict(X_test)
            tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
            current_pred = best_model.predict(X_test)
            current_rmse = np.sqrt(mean_squared_error(y_test, current_pred))
            print("Random Forest best params:", search.best_params_)
            print("Tuned Random Forest test RMSE:", round(tuned_rmse, 3))
            if tuned_rmse < current_rmse:
                best_model = search.best_estimator_
                best_name = "Tuned Random Forest"
        except Exception as exc:
            print("Random Forest tuning skipped:", exc)

    # Final evaluation
    predictions = best_model.predict(X_test)
    final_metrics = {
        "selected_model": best_name,
        "target_column": target,
        "target_unit_note": target_unit_note,
        "prediction_unit": "INR rupees",
        "target_conversion_multiplier": unit_multiplier,
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
        "rows_used": int(len(X)),
        "feature_columns": feature_cols
    }
    with open("reports/final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions,
        "absolute_error": np.abs(y_test.values - predictions)
    }).to_csv("reports/test_predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.55)
    lo = min(float(y_test.min()), float(np.min(predictions)))
    hi = max(float(y_test.max()), float(np.max(predictions)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual price (INR ₹)")
    plt.ylabel("Predicted price (INR ₹)")
    plt.title("Actual vs Predicted House Prices (INR)")
    plt.tight_layout()
    plt.savefig("reports/actual_vs_predicted.png", dpi=160)
    plt.close()

    # Permutation importance on original input columns.
    try:
        perm = permutation_importance(
            best_model, X_test, y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5, random_state=42, n_jobs=-1
        )
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False)
        imp.to_csv("reports/feature_importance.csv", index=False)

        plot_imp = imp.head(20).sort_values("importance_mean")
        plt.figure(figsize=(9, max(4, len(plot_imp) * 0.35)))
        plt.barh(plot_imp["feature"].astype(str), plot_imp["importance_mean"])
        plt.xlabel("Increase in MAE after shuffling")
        plt.title("Permutation Feature Importance")
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png", dpi=160)
        plt.close()
    except Exception as exc:
        print("Feature importance could not be calculated:", exc)

    joblib.dump(best_model, "artifacts/lt_realty_price_model.pkl")
    metadata = {
        "target": target,
        "features": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_name": best_name,
        "metrics": final_metrics
    }
    with open("artifacts/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model: artifacts/lt_realty_price_model.pkl")
    print("Reports are in the reports/ folder.")
    print("IMPORTANT: This model is only as reliable as the data used to train it.")

if __name__ == "__main__":
    main()
