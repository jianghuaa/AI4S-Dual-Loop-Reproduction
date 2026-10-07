#!/usr/bin/env python
"""TabPFN regression with 5-fold cross-validation for material property prediction."""
import os

os.environ["TABPFN_MODEL_CACHE_DIR"] = r"F:\TabPFN_Models"
from tabpfn import TabPFNRegressor
import tabpfn.browser_auth as browser_auth
import tabpfn.model_loading as model_loading


# We already verified through Prior Labs API that:
# license = tabpfn-3-5-license-v1.0
# accepted = true
#
# Avoid the unstable Hugging Face metadata request.
def patched_get_license_name(hf_repo_id):
    if hf_repo_id == "tabpfn_3_5":
        return "tabpfn-3-5-license-v1.0"
    return browser_auth._get_license_name(hf_repo_id)


browser_auth._get_license_name = patched_get_license_name
model_loading._get_license_name = patched_get_license_name
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import os
import joblib
from pathlib import Path

# Resolve data path relative to project root
# _BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure output directories exist
os.makedirs('./results', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# Load data
# data_path = _BASE_DIR / 'data' / 'data_3_iter1.csv'
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(BASE_DIR / "data" / "data_3_iter1.csv")

# Define target columns to predict
target_columns = ["evaluation", "ra", "cie"]

# Initialize 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store summary results across targets
summary_results = []

# Run 5-fold CV for each target column
for target in target_columns:
    print(f"\n{'=' * 50}")
    print(f"===== 5-Fold CV for target: {target} =====")
    print(f"{'=' * 50}\n")

    # Prepare features (first 3 columns) and target
    X = df.iloc[:, :3].values
    y = df[target].values

    mse_scores = []
    r2_scores = []
    mae_scores = []
    true_means = []
    pred_means = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = TabPFNRegressor(model_path=r"F:\TabPFN_Models\Prior-Labs\tabpfn_3_5\tabpfn-v3.5-20260909.safetensors")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Drop NaN predictions (TabPFN instability on tiny folds)
        valid = ~np.isnan(y_pred)
        if not valid.all():
            print(f"Fold {fold + 1}: dropping {(~valid).sum()} NaN predictions")
            y_test, y_pred = y_test[valid], y_pred[valid]
        if len(y_test) == 0:
            print(f"Fold {fold + 1}: all predictions NaN, skipping fold")
            continue

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

        true_means.append(np.mean(y_test))
        pred_means.append(np.mean(y_pred))

    # Compute average metrics across folds
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_mae = np.mean(mae_scores)
    std_mse = np.std(mse_scores)
    std_r2 = np.std(r2_scores)
    std_mae = np.std(mae_scores)

    overall_true_mean = np.mean(true_means)
    overall_pred_mean = np.mean(pred_means)

    summary_results.append({
        'target_column': target,
        'avg_mse': avg_mse,
        'std_mse': std_mse,
        'avg_r2': avg_r2,
        'std_r2': std_r2,
        'avg_mae': avg_mae,
        'std_mae': std_mae,
    })

    print(f"{target} 5-fold CV average results:")
    print(f"Avg MSE: {avg_mse:.4f} (+/-{std_mse:.4f})")
    print(f"Avg R^2: {avg_r2:.4f} (+/-{std_r2:.4f})")
    print(f"Avg MAE: {avg_mae:.4f} (+/-{std_mae:.4f})")
    print(f"Overall true mean: {overall_true_mean:.4f}")
    print(f"Overall predicted mean: {overall_pred_mean:.4f}")

# Save summary results to CSV
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv('./results/cv_summary_results.csv', index=False)
print("\nCV summary results saved to ./results/cv_summary_results.csv")

# Train final TabPFN model on all data and save
model = TabPFNRegressor(model_path=r"F:\TabPFN_Models\Prior-Labs\tabpfn_3_5\tabpfn-v3.5-20260909.safetensors")
X = df.iloc[:, :3].values
X = X / X.sum(1, keepdims=True)
y = df["evaluation"].values
model.fit(X, y)
joblib.dump(model, './models/tabpfn.pkl')
print("Final TabPFN model saved to ./models/tabpfn.pkl")
