"""Utility functions for data loading, processing, and evaluation."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .Data import Data
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from scipy.integrate import simpson


def load_and_process_data(file_path):
    """Load CPL JSON data and compute glum and gRa for each sample."""
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    for data in dataset:
        x = np.array(data['CPL']['X'])
        y = np.array(data['CPL']['Y'])

        # Compute glum: maximum absolute value
        glum = y.max() if abs(y.max()) > abs(y.min()) else y.min()
        data['glum'] = glum

        # Compute gRa: integral area via Simpson's rule
        gRa = simpson(y, x)
        data['gRa'] = gRa

    # Save processed data back
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    return dataset


def prepare_data(dataset, components_num):
    """Prepare features and labels for regression tasks."""
    X = []
    y = []

    for data in dataset:
        if data['temperature'] == 200:
            try:
                CPL_values = data['glum']
            except KeyError:
                continue

            # Features: mix ratios + thickness + stretching + angle
            feature = data['mix_ration'][:components_num] + [
                float(data['thickness']),
                float(data['stretching']),
                float(data['angle']),
            ]
            X.append(feature)
            y.append(CPL_values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_data_list_to_json(data_list, file_path):
    """Serialize a list of Data objects to JSON."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json_data = []
        for data in data_list:
            json_data.append({
                'Thickness': data.thickness,
                'Stretching Degree': data.stretching,
                'Mixture': data.mixture,
                'Annealing Temperature': data.temperature,
                'CPL': data.CPL,
                'Fluorescence': data.fluorescence,
                'Angle': data.angle,
            })
        json.dump(json_data, file, ensure_ascii=False, indent=4)


def load_data_list_from_json(file_path):
    """Deserialize a list of Data objects from JSON."""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        for item in json_data:
            data_instance = Data()
            data_instance.thickness = item.get('Thickness')
            data_instance.stretching = item.get('Stretching Degree')
            data_instance.mixture = item.get('Mixture')
            data_instance.temperature = item.get('Annealing Temperature')
            data_instance.CPL = item.get('CPL', {'X': [], 'Y': []})
            data_instance.fluorescence = item.get('Fluorescence', {'X': [], 'Y': []})
            data_instance.angle = item.get('Angle')
            data_list.append(data_instance)
    return data_list


# ====================== CPL/TXT File Parsing ======================

def read_cpl_file(file_path):
    """Parse a CPL spectrum file and return a Data object."""
    parts = os.path.splitext(file_path)[0].split('/')[-1].split('-')
    thickness = int(file_path.split('/')[4].split('-')[0].replace('um', ''))
    stretching = int(file_path.split('/')[4].split('-')[1].replace('%', ''))
    mix_ration = [float(x) for x in parts[:5]]
    mix_total = sum(mix_ration)
    mix_ration = [x / mix_total for x in mix_ration]
    temperature = int(parts[-2])
    angle = int(parts[-1])

    data = Data()
    data.thickness = thickness
    data.stretching = stretching
    data.mix_ration = mix_ration
    data.temperature = temperature
    data.angle = angle

    # Read file content
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_index = None
        end_index = None
        for i, line in enumerate(lines):
            if line.startswith('XYDATA'):
                start_index = i + 1
            if line.startswith('##### Extended Information'):
                end_index = i
        if start_index is not None:
            for line in lines[start_index:end_index - 1]:
                values = line.split()
                if len(values) >= 2:
                    data.CPL['X'].append(float(values[0]))
                    data.CPL['Y'].append(float(values[1]))

    return data


def read_pl_file(file_path):
    """Parse a PL (photoluminescence) spectrum file."""
    parts = os.path.splitext(file_path)[0].split('/')[-1].split('-')
    mix_ration = [float(x) for x in parts[:5]]
    mix_total = sum(mix_ration)
    mix_ration = [x / mix_total for x in mix_ration]
    temperature = int(parts[5])

    fluorescence = {'X': [], 'Y': []}
    with open(file_path, 'r', encoding='gbk') as f:
        lines = f.readlines()
        start_index = None
        for i, line in enumerate(lines):
            if line.startswith('"波长(nm)"'):  # "wavelength(nm)" header in Chinese
                start_index = i + 1
                break
        if start_index is not None:
            for line in lines[start_index:]:
                values = line.strip().split(',')
                if len(values) == 2:
                    fluorescence['X'].append(float(values[0]))
                    fluorescence['Y'].append(float(values[1]))

    return mix_ration, temperature, fluorescence


def create_dataset(file_path):
    """Build a dataset by matching CPL and PL files from a directory."""
    dataset = []
    cpl_folder = os.path.join(file_path, 'CPL')
    pl_folder = os.path.join(file_path, 'PL')

    for root, dirs, files in os.walk(cpl_folder):
        for file in files:
            cpl_file_path = os.path.join(root, file)
            data = read_cpl_file(cpl_file_path)

            # Match corresponding PL file
            for pl_root, pl_dirs, pl_files in os.walk(pl_folder):
                for pl_file in pl_files:
                    pl_file_path = os.path.join(pl_root, pl_file)
                    pl_mix_ration, pl_temperature, pl_fluorescence = read_pl_file(pl_file_path)
                    if pl_mix_ration == data.mix_ration and pl_temperature == data.temperature:
                        data.fluorescence = pl_fluorescence
                        break

            dataset.append(data)

    return dataset


# ====================== Misc Utilities ======================

def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    Parameters
    ----------
    x : torch.Tensor, shape [batch_size, latent_dim]
    y : torch.Tensor, shape [batch_size, latent_dim]
    kernel : str, kernel type (currently only 'rbf')
    sigma : float, RBF kernel bandwidth

    Returns
    -------
    mmd : torch.Tensor, MMD^2 value
    """
    xx, yy, xy = torch.matmul(x, x.t()), torch.matmul(y, y.t()), torch.matmul(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    if kernel == 'rbf':
        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * xy
        kxx = torch.exp(-0.5 * dxx / sigma ** 2)
        kyy = torch.exp(-0.5 * dyy / sigma ** 2)
        kxy = torch.exp(-0.5 * dxy / sigma ** 2)
    else:
        raise ValueError("Unsupported kernel")

    mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return mmd


def load_and_preprocess_data(file_path, column_length=5):
    """Load CSV and filter out rows where all first N columns are 1."""
    df = pd.read_csv(file_path)
    df_subset = df.iloc[:, :column_length]
    df = df[(df_subset != 1).all(axis=1)]
    X = df.iloc[:, :column_length].values
    return X


def read_candidates(candidates_path):
    """Read candidate compositions from a text file (dash-separated values)."""
    arrays = []
    with open(candidates_path, 'r') as file:
        for line in file:
            numbers = line.strip().split('-')
            float_numbers = [float(num) for num in numbers]
            arrays.append(np.array(float_numbers))
    return np.array(arrays)
