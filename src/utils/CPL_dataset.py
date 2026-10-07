"""Build a CPL dataset from raw data files and save as JSON."""

import json
from Data import Data
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def read_cpl_file(file_path):
    """Parse a CPL spectrum file and return a Data object."""
    # 1. 先获取父文件夹的名字，比如 "30um-10%"
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    # 2. 从父文件夹名字里提取厚度和拉伸度
    thickness = int(parent_dir.split('-')[0].replace('um', ''))
    stretching = int(parent_dir.split('-')[1].replace('%', ''))

    # 3. 获取文件名，比如 "80-0.3-7-135" 并拆分
    filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    parts = filename_no_ext.split('-')
    
    # 4. 四元体系取前四个作为配比
    mix_ration = [float(x) for x in parts[:4]]
    mix_total = sum(mix_ration)
    mix_ration = [x / mix_total for x in mix_ration]
    
    # 5. 角度取最后一个
    angle = int(parts[-1])
    
    # 6. 温度这里如果文件名没有提供，可以暂时给个默认值，或者按你的真实数据来
    temperature = 200

    data = Data()
    data.thickness = thickness
    data.stretching = stretching
    data.mix_ration = mix_ration
    data.temperature = temperature
    data.angle = angle

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


def create_dataset(file_path):
    """Create a dataset by reading all CPL files under the given directory."""
    dataset = []
    cpl_folder = os.path.join(file_path, 'CPL')

    for root, dirs, files in os.walk(cpl_folder):
        for file in files:
            cpl_file_path = os.path.join(root, file)
            data = read_cpl_file(cpl_file_path)
            dataset.append(data)

    return dataset


def main():
    file_path = str(BASE_DIR / "data" / "4-CPL")
    data_path = str(BASE_DIR / "data" / "pl_cpl_4.json")

    dataset = create_dataset(file_path=file_path)

    # Convert Data objects to list of dicts for JSON serialization
    dataset_dict = []
    for data in dataset:
        data_dict = {
            'CPL': data.CPL,
            'thickness': data.thickness,
            'stretching': data.stretching,
            'angle': data.angle,
            'temperature': data.temperature,
            'mix_ration': data.mix_ration
        }
        dataset_dict.append(data_dict)

    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved to {data_path}")


if __name__ == "__main__":
    main()
