"""Convert raw experimental measurement data to CSV with computed evaluation metric."""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../../data/20250825.txt', sep=' ', header=None)
    df.columns = ['mix_ratio', 'x', 'y', 'z', 'CCT', 'ra']

    # Split the mix_ratio column into individual components
    split_cols = df['mix_ratio'].str.split('-', expand=True)
    split_cols.columns = [f'mr{i + 1}' for i in range(4)]

    def convert_value(val):
        num = float(val)
        return int(num) if num.is_integer() else num

    for col in split_cols.columns:
        split_cols[col] = split_cols[col].apply(convert_value)

    # Concatenate new columns
    new_df = pd.concat([split_cols, df.drop('mix_ratio', axis=1)], axis=1)

    # Compute derived metrics
    new_df['ra'] = new_df['ra'] / 100
    new_df['cie'] = 1 - np.sqrt((df['x'] - 0.33) ** 2 + (df['y'] - 0.33) ** 2) \
                    / np.sqrt((0.5 - 0.33) ** 2 + (0.5 - 0.33) ** 2)
    new_df['evaluation'] = 0.5 * new_df['cie'] + 0.5 * new_df['ra']

    new_df.columns = ['PFO', 'F8BT', 'PFO_DBT', 'RUB', 'CCT', 'ra', 'x', 'y', 'z', 'cie', 'evaluation']
    new_df.to_csv('../../data/20250825.csv', index=False)
    print("Converted data saved to ../../data/20250825.csv")


if __name__ == "__main__":
    main()
