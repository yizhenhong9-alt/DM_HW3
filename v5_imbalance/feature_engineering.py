import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def main():
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df = pd.read_csv('../train.csv').fillna(-1)
    
    features = [col for col in df.columns if col != 'ACTION']
    df_encoded = df[['ACTION']].copy()
    
    print("Applying Count Encoding...")
    for col in features:
        df_encoded[f'{col}_count'] = df[col].map(df[col].value_counts())

    print("Applying K-fold OOF Target Encoding...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in features:
        df_encoded[f'{col}_target'] = 0.0
        for train_idx, val_idx in skf.split(df, df['ACTION']):
            train_fold = df.iloc[train_idx]
            target_mean = train_fold.groupby(col)['ACTION'].mean()
            df_encoded.loc[df_encoded.index[val_idx], f'{col}_target'] = df.loc[df.index[val_idx], col].map(target_mean)
        
        global_mean = df['ACTION'].mean()
        df_encoded[f'{col}_target'] = df_encoded[f'{col}_target'].fillna(global_mean)

    output_path = os.path.join(output_dir, 'train_v5_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\nV5 特徵工程完成！產出檔案: v5_imbalance/{output_path}")

if __name__ == "__main__":
    main()
