"""Visualization utilities for latent space, GMM elbow plots, and MCMC sampling results."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def plot_latent_space(X, z, color_data, title, save_path):
    """Plot 2D latent space with points colored by objective and shaped by composition."""

    # Mapping of non-zero patterns to marker shapes
    pattern_mapping = {
        (1, 1, 1): '^',
        (1, 1, 1, 0): '^',
        (1, 1, 1, 1): 'D',
        (1, 1, 1, 0, 0): '^',
        (1, 1, 1, 1, 0): 'D',
        (1, 1, 1, 0, 1): 'P',
        (1, 1, 1, 1, 1): 'o'
    }

    scatter_handles = {}
    for i in range(len(z)):
        non_zero_mask = tuple(int(x) for x in (X[i] != 0))
        shape = pattern_mapping[non_zero_mask]
        scatter = plt.scatter(z[i, 0], z[i, 1],
                              c=[color_data[i]],
                              cmap='jet',
                              marker=shape,
                              s=70,
                              edgecolor='w',
                              linewidth=0.5,
                              alpha=0.8,
                              vmin=0.5,
                              vmax=1)
        scatter_handles[shape] = scatter

    scatter = scatter_handles[next(iter(scatter_handles))]
    cbar = plt.colorbar(scatter, shrink=0.8, pad=0.02)
    cbar.set_label('Objective Value', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Legend
    legend_labels = {
        '^': 'Three components',
        'D': 'Four (RUB)',
        'P': 'Four (AIE)',
        'o': 'Five components'
    }
    handles = [scatter_handles[s] for s in scatter_handles]
    labels = [legend_labels[s] for s in scatter_handles]
    plt.legend(handles, labels, fontsize=16, loc='best')

    plt.title('Latent Space Visualization', fontsize=24, pad=20)
    plt.xlabel('Latent Dimension 1', fontsize=20)
    plt.ylabel('Latent Dimension 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_elbow(z, elbow_path):
    """Elbow method plot: negative log-likelihood vs. number of GMM components."""
    n_components_range = range(1, 10)
    scores = []

    for n in n_components_range:
        gm = GaussianMixture(n_components=n, random_state=0).fit(z)
        scores.append(-gm.score(z))

    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, scores, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Elbow Method for Selecting n_components")
    plt.savefig(elbow_path, dpi=600)
    plt.close()


def plot_sampling_results(centers, sampled_points, z, fig_path='./pics/MCMCsampling.png'):
    """Plot original latent points, MCMC sampled points, and GMM cluster centers."""
    plt.figure(figsize=(10, 8))
    plt.scatter(z[:, 0], z[:, 1], c='#D8BFD8', s=50, edgecolor='w',
                linewidth=0.5, label='Original Points')
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='#6B89C4', s=50,
                edgecolor='w', linewidth=0.5, label='Sampled Points')
    plt.scatter(centers[:, 0], centers[:, 1], c='#FFD700', s=500, marker='*',
                edgecolor='black', linewidth=1, label='Cluster Centers')
    plt.xlabel("Latent Dimension 1", fontsize=26, labelpad=8, fontweight='semibold')
    plt.ylabel("Latent Dimension 2", fontsize=26, labelpad=8, fontweight='semibold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=20, title_fontsize='12', frameon=True,
               framealpha=0.9, edgecolor='#404040')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_log_likelihood(gm, best_points, loglike_figpath):
    """Plot GMM log-likelihood contour map with best-ranked sample points."""
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    Z = gm.score_samples(XX)
    Z = Z.reshape(X.shape)

    plt.figure(figsize=(10, 8))

    levels = np.linspace(-30, 0, 10)
    contour = plt.contourf(X, Y, Z, levels=levels, cmap="RdBu_r",
                           alpha=0.75, extend='both', antialiased=True)

    plt.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5,
                linestyles='--', alpha=0.5)

    # Mark best-ranked samples
    plt.scatter(best_points[:, 0], best_points[:, 1],
                c='w', edgecolor='k', marker='o', s=100, linewidth=1,
                zorder=3, label='Best ranked samples')

    cbar = plt.colorbar(contour, shrink=0.8, pad=0.02)
    cbar.set_label('Log Likelihood', fontsize=20, labelpad=10)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_edgecolor('#404040')

    plt.xlabel("Dimension 1", fontsize=20, labelpad=10,
               fontweight='semibold', color='#333333')
    plt.ylabel("Dimension 2", fontsize=20, labelpad=10,
               fontweight='semibold', color='#333333')

    plt.legend(loc='lower left', fontsize=18, title_fontsize='12',
               frameon=True, framealpha=0.95, edgecolor='#404040',
               borderpad=1, scatterpoints=1, markerscale=1.2)

    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout(pad=2)
    plt.savefig(loglike_figpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
