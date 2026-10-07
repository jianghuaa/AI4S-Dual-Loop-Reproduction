#!/usr/bin/env python
# coding: utf-8

import joblib
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from src.GMM_MCMC.wae import WAE
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latent_representation(model, X):
    """Get latent representation of input data via WAE encoder"""
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = model._encode(X_tensor)
    return z.cpu().numpy()


def mcmc_sampling(z, n_components=3, n_samples=1500, proposal_cov=None):
    """
    Metropolis-Hastings based MCMC sampling with GMM as target distribution
    
    Parameters:
    z: Input latent data, shape (n_samples, n_features)
    n_components: Number of components in GMM
    n_samples: Number of samples to generate via MCMC
    proposal_cov: Covariance matrix of proposal distribution (Multivariate Normal)
    
    Returns:
    gm: Trained GMM model
    centers: Means of GMM components
    sampled_points: MCMC sampled points in latent space
    acceptance_rate: Acceptance rate of MCMC sampling
    """
    # Fit GMM to latent data
    gm = GaussianMixture(n_components=n_components, random_state=0).fit(z)
    centers = gm.means_
    n_features = z.shape[1]
    
    # Set default proposal covariance if not provided
    if proposal_cov is None:
        data_cov = np.cov(z, rowvar=False)
        proposal_cov = 0.1 * data_cov  # Scale to balance exploration-exploitation
    
    # Initialize MCMC with a sample from GMM
    current_sample = gm.sample(1)[0].squeeze()
    samples = np.zeros((n_samples, n_features))
    acceptances = 0
    
    # Metropolis-Hastings sampling loop
    for i in range(n_samples):
        # Generate candidate sample from proposal distribution
        candidate = np.random.multivariate_normal(current_sample, proposal_cov)
        # Calculate log probabilities under GMM
        log_prob_current = gm.score_samples([current_sample])[0]
        log_prob_candidate = gm.score_samples([candidate])[0]
        # Compute log acceptance ratio
        log_acceptance_ratio = log_prob_candidate - log_prob_current
        # Acceptance probability (clamped to 1)
        acceptance_prob = min(1, np.exp(log_acceptance_ratio))
        
        # Accept or reject candidate
        if np.random.rand() < acceptance_prob:
            current_sample = candidate
            acceptances += 1
        
        samples[i] = current_sample
    
    # Calculate acceptance rate
    acceptance_rate = acceptances / n_samples
    return gm, centers, samples, acceptance_rate


def plot_latent_space(z_original, z_sampled, centers, save_path):
    """
    Plot original latent points and MCMC sampled points in WAE latent space
    
    Parameters:
    z_original: Latent representation of original data, shape (n_original, n_features)
    z_sampled: MCMC sampled points in latent space, shape (n_sampled, n_features)
    centers: GMM component centers, shape (n_components, n_features)
    save_path: Path to save the plot (e.g., './latent_space_plot.png')
    """
    # Set plot style for clarity
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot original data points (blue circles)
    ax.scatter(
        z_original[:, 0], z_original[:, 1],  # Use first 2 latent dimensions (common for visualization)
        c='steelblue', marker='o', s=50, alpha=0.6, 
        label=f'Original Data Points (n={len(z_original)})'
    )
    
    # Plot MCMC sampled points (orange triangles)
    ax.scatter(
        z_sampled[:, 0], z_sampled[:, 1],
        c='darkorange', marker='^', s=50, alpha=0.6,
        label=f'MCMC Sampled Points (n={len(z_sampled)})'
    )
    
    # Plot GMM component centers (red stars, for reference)
    ax.scatter(
        centers[:, 0], centers[:, 1],
        c='crimson', marker='*', s=200, edgecolors='black', linewidth=1.5,
        label=f'GMM Component Centers (n={len(centers)})'
    )
    
    # Customize plot labels and title
    ax.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('WAE Latent Space: Original Data vs. MCMC Sampled Points', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend and grid
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout and save (high resolution)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent space plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--iter_num', type=int, default=1, help='iteration number')
    parser.add_argument('--composition', type=int, default=3, help='number of composition')
    parser.add_argument('--config', type=str, default='./mcmc_config.yaml', help='path to config file')
    args = parser.parse_args()

    # Read config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    try:
        cfg = config[args.composition][args.iter_num]
    except KeyError:
        raise ValueError(f"No config found for composition={args.composition}, iter_num={args.iter_num}")


    # Use cfg paths to load models/data
    data_path = cfg["data_path"]
    wae_model_path = cfg["wae_model_path"]
    regression_model_path = cfg["regression_model_path"]
    save_csv_path = cfg["save_csv_path"]
    save_plot_path = cfg["save_plot_path"]
    columns = cfg["columns"]

    # --------------------------
    # 1. Load and preprocess data
    # --------------------------
    df = pd.read_csv(data_path)
    X_original = df.iloc[:, :args.composition].values  # Original input data (4 features)
    X_original = X_original / X_original.sum(axis=1, keepdims=True)  # Normalization
    
    # --------------------------
    # 2. Load WAE model and get latent representations
    # --------------------------
    WAE_model = WAE(input_size=args.composition)
    WAE_model.load_state_dict(torch.load(wae_model_path, map_location=device))
    WAE_model.to(device)
    WAE_model.eval()  # Set to evaluation mode
    
    # Get latent space of original data
    z_original = get_latent_representation(WAE_model, X_original)
    print(f"Original data latent shape: {z_original.shape}")  # Should be (n_original, latent_dim)
    
    # --------------------------
    # 3. MCMC sampling in latent space
    # --------------------------
    n_components = 3  # Number of GMM components
    n_samples = 1500  # Number of MCMC samples
    _gm, centers, z_sampled, acceptance_rate = mcmc_sampling(
        z=z_original,
        n_components=n_components,
        n_samples=n_samples,
        proposal_cov=None
    )
    print(f"MCMC Sampling Complete | Acceptance Rate: {acceptance_rate:.4f}")
    print(f"MCMC sampled points latent shape: {z_sampled.shape}")  # Should be (2000, latent_dim)
    
    # --------------------------
    # 4. Decode sampled points to original input space
    # --------------------------
    sampled_points_tensor = torch.tensor(z_sampled, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_recon = WAE_model._decode(sampled_points_tensor).cpu().numpy()
    
    # --------------------------
    # 5. Predict with regression model and save results
    # --------------------------
    regression_model = joblib.load(regression_model_path)
    y_pred = regression_model.predict(X_recon)

    # Drop NaN predictions (TabPFN instability on out-of-distribution inputs)
    valid = ~np.isnan(y_pred)
    if not valid.all():
        print(f"Dropping {(~valid).sum()} NaN predictions")
        X_recon = X_recon[valid]
        y_pred = y_pred[valid]
    if len(X_recon) == 0:
        raise ValueError("All regression predictions are NaN; check the regression model and data")

    # Combine and save results
    y_reshaped = y_pred.reshape(-1, 1)
    combined_array = np.concatenate((X_recon, y_reshaped), axis=1)
    sampled_df = pd.DataFrame(
        combined_array,
        columns=columns
    )
    sampled_df.sort_values(by="evaluation", inplace=True, ascending=False)
    sampled_df.to_csv(save_csv_path, index=False)
    print(f"Sampled results saved to: {save_csv_path}")
    
    # --------------------------
    # 6. Plot latent space (original + sampled points)
    # --------------------------
    
    plot_latent_space(
        z_original=z_original,
        z_sampled=z_sampled,
        centers=centers,
        save_path=save_plot_path
    )


if __name__ == "__main__":
    main()