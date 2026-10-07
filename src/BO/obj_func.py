import json
import joblib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

from src.utils.model_utils import (
    plot_shap_summary, plot_bar_chart, plot_heatmap,
    save_shap_values, save_model_performance, save_correlation_matrix,
    save_feature_importance
)
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

class Config:
    file_path = None
    scaler_path = None
    figs_output_dir = None
    models_output_dir = None
    rf_feature_names = None


def load_and_process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    for data in dataset:
        x = np.array(data['CPL']['X'])
        y = np.array(data['CPL']['Y'])
        x = x[::-1]
        y = y[::-1]

        # Compute glum: max absolute value of CPL signal
        glum = y.max() if abs(y.max()) > abs(y.min()) else y.min()
        data['glum'] = glum
        # Compute fill factor via trapezoidal integration
        data['ff'] = np.trapezoid(y, x)

    # Save processed data back
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    return dataset


def prepare_data(dataset, target):
    X = []
    glums, ffs = [], []
    y = []

    for data in dataset:
        try:
            glum = abs(data['glum'])
            ff = abs(data['ff'])
        except KeyError:
            continue

        # Features: thickness + stretching + angle
        feature = [data['thickness'], data['stretching'], data['angle']]
        X.append(feature)
        glums.append(glum)
        ffs.append(ff)

    # Standardize features and targets
    scaler = StandardScaler()
    X = np.array(X)
    glums = np.array(glums)
    ffs = np.array(ffs)

    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(Config.scaler_path, 'scaler.pkl'))

    glum_scaler = MinMaxScaler()
    ff_scaler = MinMaxScaler()
    glums_scaled = glum_scaler.fit_transform(glums.reshape(-1, 1))
    joblib.dump(glum_scaler, os.path.join(Config.scaler_path, 'glum_minmax.pkl'))

    ffs_scaled = ff_scaler.fit_transform(ffs.reshape(-1, 1))
    joblib.dump(ff_scaler, os.path.join(Config.scaler_path, 'ff_minmax.pkl'))

    print(f"Scalers saved to: {Config.scaler_path}")

    # Select target variable
    if target == 'glum':
        y = glums  # Raw glum
    elif target == 'ff':
        y = ffs_scaled.ravel()  # Scaled ff
    elif target == 'G':
        y = 0.5 * ffs_scaled.ravel() + 0.5 * glums_scaled.ravel()  # Q = 0.5*FF + 0.5*|glum|
    else:
        raise ValueError(f"Unsupported target variable: {target}")

    return X_scaled, y


def plot_cv_pred_true(all_y_true, all_y_pred, avg_mse, avg_r2, avg_mae, model_name, output_dir):
    """
    Plot predicted vs true values from 5-fold CV, save plot and CSV data.

    Parameters
    ----------
    all_y_true : array
        True values from all CV validation folds.
    all_y_pred : array
        Predicted values from all CV validation folds.
    avg_mse, avg_r2, avg_mae : float
        Average CV metrics.
    model_name : str
        Model name for file naming.
    output_dir : str
        Output directory.
    """
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    plt.scatter(all_y_true, all_y_pred, alpha=0.7, color='darkblue', s=30,
                edgecolor='white', linewidth=0.5)

    # Ideal prediction line (y=x)
    min_val = min(np.min(all_y_true), np.min(all_y_pred))
    max_val = max(np.max(all_y_true), np.max(all_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Ideal Prediction (y=x)')

    plt.xlabel('Real Value (CV Validation Set)', fontsize=12, labelpad=10)
    plt.ylabel('Predicted Value (CV Validation Set)', fontsize=12, labelpad=10)
    plt.title(f'{model_name} - 5-Fold CV: Predicted vs Real Value', fontsize=14, pad=20)

    text_str = f'Avg R^2: {avg_r2:.4f}\nAvg MSE: {avg_mse:.4f}\nAvg MAE: {avg_mae:.4f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                                facecolor='wheat', alpha=0.8))

    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    img_path = os.path.join(output_dir, f'{model_name.lower()}_cv_pred_scatter.png')
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {model_name} CV prediction plot saved: {img_path}")

    # Save CV prediction data to CSV
    cv_data = pd.DataFrame({
        'CV_Real_Value': all_y_true,
        'CV_Predicted_Value': all_y_pred
    })
    csv_path = os.path.join(output_dir, f'{model_name.lower()}_cv_pred_data.csv')
    cv_data.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"[OK] {model_name} CV prediction data saved: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description='5-fold CV regression analysis with prediction visualization')
    parser.add_argument('--components_num', type=int, default=3, help='Number of components (3 or 4)')
    parser.add_argument('--target', type=str, default='G', help='Target variable (G/glum/ff)')
    args = parser.parse_args()
    components_num = args.components_num
    Config.feature_names = ["Thickness", "Stretching", "Angle"]
    target = args.target

    # Configure paths based on component count
    # Configure paths based on component count
    if components_num == 3:
        Config.file_path = str(BASE_DIR / "data" / "pl_cpl_3.json")
        Config.scaler_path = str(BASE_DIR / "models" / "3")
        Config.figs_output_dir = str(BASE_DIR / "figs" / "3")
        Config.models_output_dir = str(BASE_DIR / "models" / "3")
    elif components_num == 4:
        Config.file_path = str(BASE_DIR / "data" / "pl_cpl_4.json")
        Config.scaler_path = str(BASE_DIR / "models" / "4")
        Config.figs_output_dir = str(BASE_DIR / "figs" / "4")
        Config.models_output_dir = str(BASE_DIR / "models" / "4")
    else:
        raise ValueError(f"Unsupported components_num: {components_num} (only 3 or 4)")
    # Configure target-specific subdirectories
    if target == 'glum':
        Config.figs_output_dir = os.path.join(Config.figs_output_dir, 'glum')
        Config.models_output_dir = os.path.join(Config.models_output_dir, 'glum')
    elif target == 'ff':
        Config.figs_output_dir = os.path.join(Config.figs_output_dir, 'ff')
        Config.models_output_dir = os.path.join(Config.models_output_dir, 'ff')

    # Create output directories
    os.makedirs(Config.scaler_path, exist_ok=True)
    os.makedirs(Config.figs_output_dir, exist_ok=True)
    os.makedirs(Config.models_output_dir, exist_ok=True)
    os.makedirs(os.path.join(Config.figs_output_dir, 'data'), exist_ok=True)

    output_dir = Config.figs_output_dir
    model_save_path = Config.models_output_dir
    file_path = Config.file_path
    feature_names = Config.feature_names

    # 1. Load and preprocess data
    print("1. Loading and preprocessing data...")
    dataset = load_and_process_data(file_path)
    X, y = prepare_data(dataset, target=target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data preprocessing complete: features {X.shape}, target {y.shape}")

    # Initialize 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define models with fixed hyperparameters
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
        'SVR': SVR(C=1.0, kernel='rbf', gamma='scale'),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    objective='reg:squarederror', random_state=42, tree_method='hist'),
        'CatBoost': CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                      l2_leaf_reg=3, verbose=0, random_state=42),
        'TabPFN': TabPFNRegressor(model_path=r"F:\TabPFN_Models\Prior-Labs\tabpfn_3_5\tabpfn-v3.5-20260909.safetensors")
    }

    cv_results = {}

    # 2. Run 5-fold CV for each model
    print("\n2. Starting 5-fold cross-validation...\n")
    for model_name, model in models.items():
        print(f"=== Processing: {model_name} ===")

        fold_mse, fold_r2, fold_mae = [], [], []
        cv_all_y_true, cv_all_y_pred = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            model.fit(X_fold_train, y_fold_train)
            y_fold_pred = model.predict(X_fold_val)

            # Drop NaN predictions (e.g. TabPFN instability on tiny folds)
            valid = ~np.isnan(y_fold_pred)
            if valid.all():
                y_true_valid = y_fold_val
                y_pred_valid = y_fold_pred
            else:
                print(f"Fold {fold + 1}: dropping {(~valid).sum()} NaN predictions")
                y_true_valid = y_fold_val[valid]
                y_pred_valid = y_fold_pred[valid]
            if len(y_true_valid) == 0:
                print(f"Fold {fold + 1}: all predictions NaN, skipping fold")
                continue

            cv_all_y_true.extend(y_true_valid)
            cv_all_y_pred.extend(y_pred_valid)

            mse = mean_squared_error(y_true_valid, y_pred_valid)
            r2 = r2_score(y_true_valid, y_pred_valid)
            mae = mean_absolute_error(y_true_valid, y_pred_valid)

            fold_mse.append(mse)
            fold_r2.append(r2)
            fold_mae.append(mae)

            print(f"Fold {fold + 1}: R^2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

        avg_mse = np.mean(fold_mse)
        avg_r2 = np.mean(fold_r2)
        avg_mae = np.mean(fold_mae)

        cv_results[model_name] = {
            'MSE': avg_mse, 'R2': avg_r2, 'MAE': avg_mae,
            'fold_MSE': fold_mse, 'fold_R2': fold_r2, 'fold_MAE': fold_mae
        }

        print(f"\n{model_name} CV average: R^2={avg_r2:.4f}, MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")

        # Plot CV predicted-vs-true
        plot_cv_pred_true(
            all_y_true=cv_all_y_true,
            all_y_pred=cv_all_y_pred,
            avg_mse=avg_mse, avg_r2=avg_r2, avg_mae=avg_mae,
            model_name=model_name,
            output_dir=output_dir
        )

        # Train final model on all data and save
        model.fit(X, y)
        final_model_path = os.path.join(model_save_path, f'best_{model_name.lower()}_model.pkl')
        joblib.dump(model, final_model_path)
        print(f"{model_name} final model saved: {final_model_path}\n")

        # SHAP analysis for tree-based models
        if model_name in ['RandomForest', 'XGBoost', 'CatBoost']:
            try:
                print(f"Generating {model_name} SHAP analysis...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

                save_shap_values(shap_values, feature_names, output_dir, f'{model_name.lower()}')
                save_feature_importance(model.feature_importances_, feature_names, output_dir, model_name.lower())

                plot_shap_summary(
                    shap_values=shap_values, X=X, feature_names=feature_names,
                    title=f'{model_name} Feature Importance (SHAP)',
                    plot_type='bar',
                    file_path=os.path.join(output_dir, f'{model_name.lower()}_shap_bar.png')
                )
                plot_shap_summary(
                    shap_values=shap_values, X=X, feature_names=feature_names,
                    title=f'{model_name} Feature Impact (SHAP Dot Plot)',
                    plot_type='dot',
                    file_path=os.path.join(output_dir, f'{model_name.lower()}_shap_dot.png')
                )
                print(f"{model_name} SHAP analysis complete\n")
            except Exception as e:
                print(f"{model_name} SHAP analysis failed: {str(e)}\n")

    # 3. Save all models' CV performance metrics
    save_model_performance(
        models=list(cv_results.keys()),
        metrics={
            'R2': [cv_results[model]['R2'] for model in cv_results],
            'MSE': [cv_results[model]['MSE'] for model in cv_results],
            'MAE': [cv_results[model]['MAE'] for model in cv_results]
        },
        output_dir=output_dir,
        cv=True
    )

    # 4. Model CV performance comparison chart
    print("3. Plotting model CV performance comparison...")
    plot_bar_chart(
        categories=list(cv_results.keys()),
        values=[cv_results[model]['R2'] for model in cv_results],
        labels=[
            [cv_results[model]['MAE'] for model in cv_results],
            [cv_results[model]['MSE'] for model in cv_results]
        ],
        file_path=os.path.join(output_dir, 'model_cv_performance_comparison.png')
    )
    print("Model CV performance comparison saved\n")

    # 5. Feature correlation analysis
    print("4. Generating feature correlation heatmap...")
    corr_matrix = np.corrcoef(X, rowvar=False)
    save_correlation_matrix(corr_matrix, feature_names, output_dir)
    plot_heatmap(
        data=corr_matrix, feature_names=feature_names,
        file_path=os.path.join(output_dir, 'feature_correlation_heatmap.png')
    )
    print("Feature correlation heatmap saved\n")

    # 6. Output best model
    best_model = max(cv_results.keys(), key=lambda x: cv_results[x]['R2'])
    print("=" * 50)
    print(f"Final Result: Best model = {best_model}")
    print(f"Best model CV metrics: R^2={cv_results[best_model]['R2']:.4f}, "
          f"MSE={cv_results[best_model]['MSE']:.4f}, MAE={cv_results[best_model]['MAE']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
