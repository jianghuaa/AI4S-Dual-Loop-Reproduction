import joblib
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from sklearn.mixture import GaussianMixture
from src.utils.Ultility import compute_mmd
import os
import argparse
from pathlib import Path


# ======================
# 1. Centralized Configuration
# ======================
_BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
    DATA_PATH = None
    MODEL_SAVE_DIR = str(_BASE_DIR / 'src' / 'GMM_MCMC' / 'models' / 'WAE')
    FIG_SAVE_DIR = str(_BASE_DIR / 'src' / 'GMM_MCMC' / 'figs' / 'WAE')
    SAVE_NAME = '.pth'

    INPUT_SIZE = 3
    LATENT_DIM = 2
    NUM_EPOCHS = 4000
    LR = 0.0001
    WEIGHT_DECAY = 0.0001
    T_MAX = 2000  # CosineAnnealing half-period
    ETA_MIN = 1e-6
    PLOT_STYLE = {
        'figsize': (10, 8),
        'title_fontsize': 24,
        'label_fontsize': 20,
        'tick_fontsize': 16,
        'colorbar_tick_fontsize': 16
    }


# ======================
# 2. WAE Model Definition
# ======================
class WAE(nn.Module):
    def __init__(self, input_size):
        super(WAE, self).__init__()
        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Linear(80, self.input_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)
        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


# ======================
# 3. Data Processing
# ======================
def load_and_process_data(file_path, components):
    """Load and preprocess raw data."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :components].values
    X = X / X.sum(axis=1, keepdims=True)
    return df, X


# ======================
# 4. Model Training
# ======================
def train_model(model, X_tensor, config):
    """Execute the model training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.T_MAX,
                                  eta_min=config.ETA_MIN)

    best_loss = float('inf')
    losses = []
    lrs = []

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        x_recon, z = model(X_tensor)
        recon_loss = criterion(x_recon, X_tensor)
        prior_samples = torch.randn(X_tensor.size(0), config.LATENT_DIM).to(X_tensor.device)
        mmd_loss = compute_mmd(z, prior_samples)
        total_loss = recon_loss + 0.5 * mmd_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, config.SAVE_NAME))
            print(f"Epoch {epoch + 1}: Best loss {best_loss:.4f} - Model saved")

    return losses, lrs, best_loss


# ======================
# 5. Visualization
# ======================
def plot_latent_space(z, color_data, shapes, config, save_name):
    """Plot latent space scatter plot."""
    plt.figure(figsize=config.PLOT_STYLE['figsize'])
    for i in range(len(z)):
        plt.scatter(z[i, 0], z[i, 1],
                    c=color_data[i],
                    cmap='jet',
                    marker=shapes[i],
                    vmin=0.5,
                    vmax=1,
                    s=70)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=config.PLOT_STYLE['colorbar_tick_fontsize'])
    plt.title('Visualization of Latent Space', fontsize=config.PLOT_STYLE['title_fontsize'])
    plt.xlabel('Latent Dimension 1', fontsize=config.PLOT_STYLE['label_fontsize'])
    plt.ylabel('Latent Dimension 2', fontsize=config.PLOT_STYLE['label_fontsize'])
    plt.xticks(fontsize=config.PLOT_STYLE['tick_fontsize'])
    plt.yticks(fontsize=config.PLOT_STYLE['tick_fontsize'])
    plt.savefig(f"{config.FIG_SAVE_DIR}/{save_name}_latent_space.png", dpi=300)
    plt.close()


def plot_log_likelihood(z, config, save_name, n_components=3):
    """Plot GMM log-likelihood contour map."""
    gm = GaussianMixture(n_components=n_components,
                         covariance_type='full',
                         random_state=0).fit(z)

    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    XX = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = gm.score_samples(XX).reshape(X_grid.shape)
    print("Log likelihood range:")
    print("Z min:", Z.min())
    print("Z max:", Z.max())
    print("Z mean:", Z.mean())
    print("GMM means:")
    print(gm.means_)

    print("GMM covariances:")
    print(gm.covariances_)

    plt.figure(figsize=config.PLOT_STYLE['figsize'])
    # levels = np.linspace(-15, 0, 30)
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = plt.contourf(X_grid, Y_grid, Z, levels=levels, cmap="RdBu_r",
                          alpha=0.8, extend="both", antialiased=True)
    plt.contour(X_grid, Y_grid, Z, levels=Z.min() + np.linspace(0, 1, 5) * (Z.max() - Z.min()),
               colors='black', linewidths=0.5)

    cbar = plt.colorbar(contour)
    cbar.set_label("Log Likelihood", fontsize=config.PLOT_STYLE['label_fontsize'])
    cbar.locator = MaxNLocator(nbins=5)
    cbar.formatter = ScalarFormatter()
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=config.PLOT_STYLE['colorbar_tick_fontsize'])

    plt.xlabel("Dimension 1", fontsize=config.PLOT_STYLE['label_fontsize'] + 4)
    plt.ylabel("Dimension 2", fontsize=config.PLOT_STYLE['label_fontsize'] + 4)
    plt.xticks(fontsize=config.PLOT_STYLE['tick_fontsize'])
    plt.yticks(fontsize=config.PLOT_STYLE['tick_fontsize'])
    plt.savefig(f"{config.FIG_SAVE_DIR}/{save_name}_loglike.png",
               dpi=300, bbox_inches='tight')
    plt.close()


# ======================
# 6. Main Entry Point
# ======================
if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('1', 'true', 'yes', 'on')

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_num', type=int, default=0, help='iteration number')
    parser.add_argument('--n_components', type=int, default=3, help='number of components')
    parser.add_argument('--retrain', type=str2bool, default=False, help='whether to retrain the model')
    args = parser.parse_args()

    config = Config()
    iter_num = args.iter_num
    n_components = args.n_components

    # Configure data paths per component count and iteration
    data_dir = _BASE_DIR / 'data'
    if n_components == 3:
        if iter_num == 1:
            config.DATA_PATH = str(data_dir / 'data_3_iter1.csv')
            Config.SAVE_NAME = '3_1'
        elif iter_num == 2:
            config.DATA_PATH = str(data_dir / 'data_3_iter2.csv')
            Config.SAVE_NAME = '3_2'
        elif iter_num == 3:
            config.DATA_PATH = str(data_dir / 'data_3.csv')
            Config.SAVE_NAME = '3_3'
        config.INPUT_SIZE = 3
    elif n_components == 4:
        if iter_num == 1:
            config.DATA_PATH = str(data_dir / 'data_4_iter1.csv')
            Config.SAVE_NAME = '4_1'
        elif iter_num == 2:
            config.DATA_PATH = str(data_dir / 'data_4_iter2.csv')
            Config.SAVE_NAME = '4_2'
        elif iter_num == 3:
            config.DATA_PATH = str(data_dir / 'data_4.csv')
            Config.SAVE_NAME = '4_3'
        config.INPUT_SIZE = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data processing
    df_processed, X = load_and_process_data(config.DATA_PATH, components=n_components)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Model initialization
    model = WAE(input_size=config.INPUT_SIZE).to(device)

    # Train or load model
    if args.retrain:
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        losses, lrs, best_loss = train_model(model, X_tensor, config)
    else:
        model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, config.SAVE_NAME), map_location=device))

    # Evaluate model
    model.eval()
    with torch.no_grad():
        z = model._encode(X_tensor).cpu().numpy()

    # Prepare visualization data
    color_data = df_processed['evaluation'].values
    if n_components == 3:
        shapes = ['^' for _ in range(len(X_tensor))]
    elif n_components == 4:
        last_dim = X_tensor.cpu().numpy()[:, -1]
        shapes = ['^' if val == 0 else 'o' for val in last_dim]

    # Plot latent space
    plot_latent_space(z, color_data, shapes, config, save_name=Config.SAVE_NAME)

    # Plot log-likelihood map
    plot_log_likelihood(z, config, save_name=Config.SAVE_NAME, n_components=3)

    # Save latent space data points
    latent_space_df = pd.DataFrame(z, columns=['Latent Dimension 1', 'Latent Dimension 2'])
    latent_space_df['Color Data'] = color_data
    latent_space_df['Shapes'] = shapes
    latent_space_df.to_csv(f"{config.FIG_SAVE_DIR}/{Config.SAVE_NAME}_latent_space_data.csv", index=False)

    # Save log-likelihood grid data
    gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(z)
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    XX = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = gm.score_samples(XX).reshape(X_grid.shape)
    log_likelihood_df = pd.DataFrame({
        'Dimension 1': XX[:, 0],
        'Dimension 2': XX[:, 1],
        'Log Likelihood': Z.ravel()
    })
    log_likelihood_df.to_csv(f"{config.FIG_SAVE_DIR}/{Config.SAVE_NAME}_log_likelihood_data.csv", index=False)
