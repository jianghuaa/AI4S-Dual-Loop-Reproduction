from pathlib import Path
import argparse
import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# TODO: user must place data_3_iter1.csv under data/ before running
df = pd.read_csv(BASE_DIR / "data" / "data_3_iter1.csv")

class Config:
    file_path = None
    model_path = None
    scaler_path = None
    figs_save_path = None
    components_num = None
    data_save_path = None
    angle = 45


# ==================== 1. Objective Function (Surrogate from ML Model) ====================
def get_dataset(file_path=Config.file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def obj_func(params):
    x1, x2 = params
    feature = np.concatenate(([x1, x2], [Config.angle]))
    scaler = joblib.load(Config.scaler_path)
    feature = scaler.transform([feature])
    model = joblib.load(Config.model_path)
    return model.predict(feature)[0]


# ==================== 2. Bayesian Optimization Core ====================
def bayesian_optimization(
    objective_func,
    param_bounds,
    n_initial=10,       # Number of initial design points
    n_iter=20,          # Total number of iterations
    plot_each_iter=True,
    random_seed=None,
    start_num=0         # Start index for file saving
):
    if random_seed is not None:
        np.random.seed(random_seed)

    d = len(param_bounds)

    # Initial sampling: LHS + random + boundary points for diversity
    sampler = qmc.LatinHypercube(d=d, seed=np.random.randint(0, 1000))
    lhs_samples = max(n_initial - 2 * d, 2)  # At least 2 LHS points
    X_lhs = sampler.random(lhs_samples)
    X_lhs = qmc.scale(X_lhs, param_bounds[:, 0], param_bounds[:, 1])

    # Boundary points
    X_bounds = []
    for i in range(d):
        lower_bound = [param_bounds[j, 0] if j == i else np.mean(param_bounds[j]) for j in range(d)]
        upper_bound = [param_bounds[j, 1] if j == i else np.mean(param_bounds[j]) for j in range(d)]
        X_bounds.append(lower_bound)
        X_bounds.append(upper_bound)
    X_bounds = np.array(X_bounds)

    # Random samples
    random_samples = max(n_initial - lhs_samples - len(X_bounds), 1)  # At least 1 random point
    X_random = np.random.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(random_samples, d))

    # Combine and deduplicate initial points
    X_init = np.vstack([X_lhs, X_bounds, X_random])
    _, unique_idx = np.unique(X_init, axis=0, return_index=True)
    X_init = X_init[unique_idx]

    # Ensure we don't exceed n_initial
    if len(X_init) > n_initial:
        X_init = X_init[:n_initial]

    y_init = np.array([objective_func(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()
    best_history = [np.max(y)]  # Track best value at each step

    # Store detailed data for each iteration
    all_iterations_data = []

    # Total number of BO iteration steps
    total_iter_steps = n_iter - len(X_init)

    for iter_num in range(len(X_init), n_iter):
        # Compute current iteration progress (0 to 1)
        progress = (iter_num - len(X_init)) / total_iter_steps if total_iter_steps > 0 else 0

        # Dynamic noise parameter
        y_std = np.std(y) if len(y) > 1 else 5.0
        alpha = max(y_std ** 2, 1e-3)  # Noise level

        # Hybrid kernel (RBF + Matern)
        kernel = (
            C(1.0, (0.01, 2000.0)) * RBF(
                length_scale=[(b[1] - b[0]) / 5 for b in param_bounds],
                length_scale_bounds=(1, 200)
            )
            + C(0.5, (0.01, 1000.0)) * Matern(
                length_scale=[(b[1] - b[0]) / 10 for b in param_bounds],
                nu=1.5,
                length_scale_bounds=(0.1, 100)
            )
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=30,
            random_state=np.random.randint(0, 1000)
        )
        gp.fit(X, y)

        # Expected Improvement acquisition function
        def ei(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = max(sigma, 1e-9)  # Avoid division by zero
            current_best = np.max(y)
            z = (mu - current_best) / sigma
            return (mu - current_best) * norm.cdf(z) + sigma * norm.pdf(z)

        # Upper Confidence Bound acquisition function
        def ucb(x, kappa):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = max(sigma, 1e-9)
            return mu + kappa * sigma

        # Dynamically decrease kappa from 10 to 2
        kappa = 10.0 - progress * 8.0

        # Generate candidate points via LHS
        X_candidate = qmc.scale(
            qmc.LatinHypercube(d=d).random(5000),
            param_bounds[:, 0],
            param_bounds[:, 1]
        )

        # Hybrid acquisition: EI + UCB
        ei_values = np.array([ei(x) for x in X_candidate])
        ucb_values = np.array([ucb(x, kappa) for x in X_candidate])

        # Normalize and weight
        ei_norm = (ei_values - ei_values.min()) / (ei_values.max() - ei_values.min() + 1e-9)
        ucb_norm = (ucb_values - ucb_values.min()) / (ucb_values.max() - ucb_values.min() + 1e-9)

        # UCB weight linearly decreases from 1 to 0
        weight_ucb = 1.0 - progress
        acq_values = (1 - weight_ucb) * ei_norm + weight_ucb * ucb_norm

        # Randomly select from top candidates (top 20%)
        top_k = int(len(X_candidate) * 0.2)
        top_indices = np.argsort(acq_values)[-top_k:]
        x_next = X_candidate[np.random.choice(top_indices.reshape(-1,))]

        # Evaluate the next point
        y_next = objective_func(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        best_history.append(np.max(y))

        # Collect current iteration data
        iter_data = {
            'iteration': iter_num + 1,
            'X': X.copy(),
            'y': y.copy(),
            'gp_model': gp,
            'best_value': np.max(y)
        }
        all_iterations_data.append(iter_data)

        if plot_each_iter:
            plot_2d_optimization(X, y, gp, param_bounds, iter_num, ei, iter_data, start_num)

    return X, y, best_history, all_iterations_data


# ==================== 3. 2D Parameter Space Visualization ====================
def plot_2d_optimization(X, y, gp, param_bounds, iter_num, acquisition_func, iter_data=None, start_num=0):
    """Plot 2D GP mean and acquisition function for each iteration."""
    start_figs_path = os.path.join(Config.figs_save_path, f'start_{start_num}')
    os.makedirs(start_figs_path, exist_ok=True)

    x1_min, x1_max = param_bounds[0]
    x2_min, x2_max = param_bounds[1]

    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.c_[X1.ravel(), X2.ravel()]

    mu, sigma = gp.predict(X_grid, return_std=True)
    mu = mu.reshape(X1.shape)
    sigma = sigma.reshape(X1.shape)

    acq_values = np.array([acquisition_func(x) for x in X_grid]).reshape(X1.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Angle={Config.angle}°, Start {start_num + 1}, Iter {iter_num + 1}: Thickness vs Strength', fontsize=14)
    im1 = ax1.imshow(
        mu,
        extent=[x1_min, x1_max, x2_min, x2_max],
        origin='lower',
        cmap='viridis',
        alpha=0.8
    )
    ax1.contour(X1, X2, mu, levels=8, colors='white', linestyles='--', linewidths=0.8)
    ax1.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap='viridis',
        edgecolor='k', s=60,
        label=f'Experimental Points (Best: {np.max(y):.2f})'
    )
    ax1.set_xlabel('Thickness (um)', fontsize=14)
    ax1.set_ylabel('Strength (%)', fontsize=14)
    ax1.set_title('Gaussian Process Mean Prediction', fontsize=16)
    fig.colorbar(im1, ax=ax1, label='G', fraction=0.046, pad=0.04)

    im2 = ax2.imshow(
        acq_values,
        extent=[x1_min, x1_max, x2_min, x2_max],
        origin='lower',
        cmap='plasma',
        alpha=0.8
    )
    ax2.scatter(
        X[-1, 0], X[-1, 1],
        c='red', edgecolor='k', s=100,
        marker='*', label='Latest Experimental Point'
    )
    ax2.set_xlabel('Thickness (um)', fontsize=14)
    ax2.set_ylabel('Strength (%)', fontsize=14)
    ax2.set_title('Acquisition Function (Hybrid)', fontsize=16)
    fig.colorbar(im2, ax=ax2, label='Acquisition Value', fraction=0.046, pad=0.04)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(start_figs_path, f"iter_{iter_num + 1}.png"))
    plt.close()

    if iter_data is not None:
        save_2d_optimization_data(iter_data, X1, X2, mu, acq_values, start_num)


def save_2d_optimization_data(iter_data, X1, X2, mu, acq_values, start_num=0):
    """Save 2D optimization plot data to CSVs for the given start."""
    start_data_path = os.path.join(Config.data_save_path, f'start_{start_num}')
    os.makedirs(start_data_path, exist_ok=True)

    iter_num = iter_data['iteration']
    X = iter_data['X']
    y = iter_data['y']

    # Save parameter points and target values
    df_points = pd.DataFrame({
        'Thickness (um)': X[:, 0],
        'Strength (%)': X[:, 1],
        'G': y
    })
    points_save_path = os.path.join(start_data_path, f'iter_{iter_num}_points.csv')
    df_points.to_csv(points_save_path, index=False)

    # Save GP mean prediction data
    df_mu = pd.DataFrame({
        'Thickness (um)': X1.ravel(),
        'Strength (%)': X2.ravel(),
        'GP Mean Prediction': mu.ravel()
    })
    mu_save_path = os.path.join(start_data_path, f'iter_{iter_num}_mu.csv')
    df_mu.to_csv(mu_save_path, index=False)

    # Save acquisition function data
    df_acq = pd.DataFrame({
        'Thickness (um)': X1.ravel(),
        'Strength (%)': X2.ravel(),
        'Acquisition Value': acq_values.ravel()
    })
    acq_save_path = os.path.join(start_data_path, f'iter_{iter_num}_acq.csv')
    df_acq.to_csv(acq_save_path, index=False)


# ==================== 4. Convergence Curve Visualization ====================
def plot_convergence(best_history, label="Best G", start_num=0):
    """Plot and save convergence curve for a single start."""
    start_figs_path = os.path.join(Config.figs_save_path, f'start_{start_num}')
    os.makedirs(start_figs_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_history) + 1), best_history, 'o-', color='tab:blue',
             linewidth=2, markersize=8, label=label)

    plt.xlabel('Number of Experiments')
    plt.ylabel('Maximum G')
    plt.title(f'Start {start_num + 1}: Convergence Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(start_figs_path, "convergence_curve.png"))
    plt.close()

    save_convergence_data(best_history, start_num)


def save_convergence_data(best_history, start_num=0):
    """Save convergence curve data to CSV."""
    start_data_path = os.path.join(Config.data_save_path, f'start_{start_num}')
    os.makedirs(start_data_path, exist_ok=True)

    df_convergence = pd.DataFrame({
        'Iteration': range(1, len(best_history) + 1),
        'Best G': best_history
    })
    convergence_save_path = os.path.join(start_data_path, 'convergence_curve_data.csv')
    df_convergence.to_csv(convergence_save_path, index=False)


# ==================== 5. Multi-Start Optimization ====================
def multi_start_bayesian_optimization(objective_func, param_bounds, n_starts=5, **kwargs):
    """Multi-start optimization: run BO multiple times and pick the best result."""
    all_results = []
    all_histories = []
    all_full_results = []  # Store complete results for each start

    os.makedirs(Config.figs_save_path, exist_ok=True)
    os.makedirs(Config.data_save_path, exist_ok=True)

    for start in range(n_starts):
        print(f"\n===== Start {start + 1}/{n_starts} =====")
        current_kwargs = kwargs.copy()
        current_kwargs['random_seed'] = start
        current_kwargs['plot_each_iter'] = True
        current_kwargs['start_num'] = start

        X_opt, y_opt, best_history, all_iter_data = bayesian_optimization(
            objective_func,
            param_bounds,
            **current_kwargs
        )

        save_optimization_data(X_opt, y_opt, best_history, all_iter_data, start)
        plot_convergence(best_history, start_num=start)

        best_idx = np.argmax(y_opt)
        all_results.append((X_opt[best_idx], y_opt[best_idx]))
        all_histories.append(best_history)
        all_full_results.append((X_opt, y_opt, best_history, all_iter_data))

        print(f"Start {start + 1} best result: G = {y_opt[best_idx]:.2f}")

    # Select the best across all starts
    best_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    best_X, best_y = best_results[0]

    # Plot convergence comparison across all starts
    plt.figure(figsize=(8, 5))
    for i, history in enumerate(all_histories):
        max_len = max(len(h) for h in all_histories)
        padded_history = history + [history[-1]] * (max_len - len(history))
        plt.plot(range(1, max_len + 1), padded_history, 'o-',
                 linewidth=1.5, markersize=5, label=f'Start {i + 1}')

    plt.xlabel('Number of Experiments')
    plt.ylabel('Maximum G')
    plt.title('Convergence Curves for All Starts')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.figs_save_path, "multi_start_convergence_comparison.png"))
    plt.close()

    save_all_starts_summary(all_results)

    return best_X, best_y, all_results, all_full_results


def save_all_starts_summary(all_results):
    """Save a summary of best results across all starts."""
    summary_data = []
    for i, (params, value) in enumerate(all_results):
        summary_data.append({
            'start_number': i + 1,
            'thickness': float(params[0]),
            'strength': float(params[1]),
            'best_g': float(value),
            'rank': i + 1
        })

    # Sort by best G value
    summary_data.sort(key=lambda x: x['best_g'], reverse=True)
    for i, item in enumerate(summary_data):
        item['rank'] = i + 1

    df_summary = pd.DataFrame(summary_data)
    summary_save_path = os.path.join(Config.data_save_path, 'all_starts_summary.csv')
    df_summary.to_csv(summary_save_path, index=False)

    summary_json_path = os.path.join(Config.data_save_path, 'all_starts_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)


# ==================== 6. Save Optimization Data ====================
def save_optimization_data(X, y, best_history, all_iterations_data, start_num=0):
    """Save all optimization results for a single start."""
    start_data_path = os.path.join(Config.data_save_path, f'start_{start_num}')
    os.makedirs(start_data_path, exist_ok=True)

    # Save all evaluated points
    df_points = pd.DataFrame({
        'Thickness (um)': X[:, 0],
        'Strength (%)': X[:, 1],
        'G': y
    })
    points_save_path = os.path.join(start_data_path, 'optimization_points.csv')
    df_points.to_csv(points_save_path, index=False)

    # Save convergence history
    df_convergence = pd.DataFrame({
        'Iteration': range(1, len(best_history) + 1),
        'Best G': best_history
    })
    convergence_save_path = os.path.join(start_data_path, 'convergence_data.csv')
    df_convergence.to_csv(convergence_save_path, index=False)

    # Save per-iteration detailed data
    iterations_data = []
    for _, iter_data in enumerate(all_iterations_data):
        iter_num = iter_data['iteration']
        iter_best_idx = np.argmax(iter_data['y'])

        iterations_data.append({
            'iteration': iter_num,
            'best_thickness': float(iter_data['X'][iter_best_idx, 0]),
            'best_strength': float(iter_data['X'][iter_best_idx, 1]),
            'best_g': float(iter_data['best_value']),
            'total_points': len(iter_data['X'])
        })

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_python(data):
        if isinstance(data, dict):
            return {k: convert_to_python(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_python(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        return data

    serialized_data = convert_to_python(iterations_data)
    iterations_data_path = os.path.join(start_data_path, 'all_iterations_summary.json')
    with open(iterations_data_path, 'w') as f:
        json.dump(serialized_data, f, indent=4)


# ==================== 7. Main Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--components_num', type=int, default=3, help='Number of components (3 or 4)')
    parser.add_argument('--n_starts', type=int, default=5, help='Number of multi-start runs')
    parser.add_argument('--model', type=str, default='Tab', help='Model type (Tab, XGB, RF, CatB, SVR)')
    args = parser.parse_args()

    components_num = args.components_num
    Config.components_num = components_num

    

    if components_num == 3:
        Config.file_path = str(BASE_DIR / "data" / "pl_cpl_3.json")
        Config.model_path = str(BASE_DIR / "models" / "3")
        Config.scaler_path = str(BASE_DIR / "models" / "3" / "scaler.pkl")
        Config.figs_save_path = str(BASE_DIR / "figs" / "Bayes" / "3")
        Config.data_save_path = str(BASE_DIR / "figs" / "Bayes" / "3" / "optimization_results")
    elif components_num == 4:
        Config.file_path = str(BASE_DIR / "data" / "pl_cpl_4.json")
        Config.model_path = str(BASE_DIR / "models" / "4")
        Config.scaler_path = str(BASE_DIR / "models" / "4" / "scaler.pkl")
        Config.figs_save_path = str(BASE_DIR / "figs" / "Bayes" / "4")
        Config.data_save_path = str(BASE_DIR / "figs" / "Bayes" / "4" / "optimization_results")
    # Parameter bounds: [Thickness (um), Strength (%)]
    # thickness lower bound changed 20 -> 30 to match training coverage (30,48,80)
    param_bounds = np.array([[30, 80], [10, 150]])

    # Select model filename based on model type
    model_name_map = {
        'Tab': 'best_tabpfn_model.pkl',
        'XGB': 'best_xgboost_model.pkl',
        'RF': 'best_randomforest_model.pkl',
        'CatB': 'best_catboost_model.pkl',
        'SVR': 'best_svr_model.pkl',
    }
    model_name = model_name_map.get(args.model, 'best_tabpfn_model.pkl')

    Config.model_path = os.path.join(Config.model_path, model_name)
    print("CWD:", os.getcwd())
    print("file_path:", Config.file_path, os.path.exists(Config.file_path))
    print("model_path:", Config.model_path, os.path.exists(Config.model_path))
    print("scaler_path:", Config.scaler_path, os.path.exists(Config.scaler_path))
    print("figs_save_path parent exists:", os.path.exists(os.path.dirname(Config.figs_save_path)))
    # Run multi-start Bayesian optimization
    for angle in [45, 135]:
        Config.angle = angle
        Config.figs_save_path = str(BASE_DIR / "figs" / "Bayes" / str(components_num) / f"angle_{angle}")
        Config.data_save_path = str(BASE_DIR / "figs" / "Bayes" / str(components_num) / f"angle_{angle}" / "optimization_results")

        print(f"\n########## BO for angle={angle} ##########")

        best_X, best_y, all_results, all_full_results = multi_start_bayesian_optimization(
            objective_func=obj_func,
            param_bounds=param_bounds,
            n_initial=15,
            n_iter=20,
            n_starts=args.n_starts
        )

        print(f"\nAngle {angle} best:")
        print(f"  Thickness: {best_X[0]:.2f} um")
        print(f"  Strength:  {best_X[1]:.2f} %")
        print(f"  Max G:     {best_y:.4f}")