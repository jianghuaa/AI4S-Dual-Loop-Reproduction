"""Heatmap visualization of glum and fill factor across thickness-stretching space."""

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler

from src.utils.model_utils import calculate_fill_factor


def load_and_process_data(file_path):
    """Load JSON data and compute glum and fill factor for each sample."""
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    for data in dataset:
        try:
            x = np.array(data['CPL']['X'])
            y = np.array(data['CPL']['Y'])
            max_val = y.max()
            min_val = y.min()
            # glum is the value with larger absolute magnitude
            data['glum'] = max_val if abs(max_val) > abs(min_val) else min_val
            data['ff'] = calculate_fill_factor(x, y)
        except (KeyError, TypeError) as e:
            print(f"Error processing data: {e}, skipping sample")

    # Save updated data
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    return dataset


def extract_data_points(dataset):
    """Extract valid data points (thickness, stretching, glum, ff)."""
    thickness = []
    stretching = []
    glum_values = []
    ff_values = []

    for i, data in enumerate(dataset):
        try:
            thickness.append(float(data['thickness']))
            stretching.append(float(data['stretching']))
            glum_values.append(float(abs(data['glum'])))
            ff_values.append(float(data['ff']))
        except (KeyError, ValueError, TypeError) as e:
            print(f"Data point {i} has error: {e}, skipped")

    minmax = MinMaxScaler()
    ff_values = minmax.fit_transform(np.array(ff_values).reshape(-1, 1)).flatten()
    return np.array(thickness), np.array(stretching), np.array(glum_values), np.array(ff_values)


def save_grid_data(xi, yi, zi, output_file):
    """Save interpolated grid data points to CSV."""
    df = pd.DataFrame({
        'thickness': xi.reshape(-1),
        'stretching': yi.reshape(-1),
        'value': zi.reshape(-1)
    })
    df.to_csv(output_file, index=False)
    print(f"Grid data saved to: {os.path.abspath(output_file)}")


def _plot_heatmap(thickness, stretching, values, value_label, title, output_file):
    """Plot a single heatmap with contours for a given value type."""
    if len(thickness) < 2 or len(stretching) < 2:
        raise ValueError("Not enough data points for heatmap")

    # Create interpolation grid
    grid_resolution = 100j
    xi = np.linspace(thickness.min(), thickness.max(), int(grid_resolution.imag))
    yi = np.linspace(stretching.min(), stretching.max(), int(grid_resolution.imag))
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(
        (thickness, stretching), values, (xi, yi),
        method='cubic',
        fill_value=np.nan
    )
    save_grid_data(xi, yi, zi, output_file=f'{value_label.lower()}_grid_data_points.csv')

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        zi,
        extent=[thickness.min(), thickness.max(), stretching.min(), stretching.max()],
        origin='lower',
        cmap='coolwarm',
        aspect='auto',
        alpha=0.9
    )

    contour = ax.contour(
        xi, yi, zi,
        levels=MaxNLocator(nbins=10).tick_values(zi.min(), zi.max()),
        colors='black',
        linewidths=0.8,
        alpha=0.7
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xlabel('Thickness', fontsize=12)
    ax.set_ylabel('Stretching', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{value_label} Value', fontsize=12)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {os.path.abspath(output_file)}")


def main(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    dataset = load_and_process_data(file_path)
    thickness, stretching, glum, ff = extract_data_points(dataset)

    if len(thickness) == 0:
        print("No valid data points available for heatmap plotting")
        return

    _plot_heatmap(thickness, stretching, glum, 'glum',
                  'glum Heatmap', 'glum_heatmap.png')
    _plot_heatmap(thickness, stretching, ff, 'ff',
                  'ff Heatmap', 'ff_heatmap.png')


if __name__ == "__main__":
    json_file_path = "../../data/pl_cpl_3.json"
    try:
        main(json_file_path)
    except Exception as e:
        print(f"Program execution error: {e}")
