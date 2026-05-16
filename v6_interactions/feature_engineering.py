import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def main():
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 讀取選定的組合
    selected_pairs = []
    if os.path.exists('results/selected_pairs.txt'):
        with open('results/selected_pairs.txt', 'r') as f:
            for line in f:
                selected_pairs.append(line.strip().split(','))
    
    print(f"Loading data and creating {len(selected_pairs)} interaction features...")
    df = pd.read_csv('../train.csv').fillna(-1)
    df_encoded = df[['ACTION']].copy()
    
    # 2. 基礎特徵 (Count Encoding for original features)
    original_features = [col for col in df.columns if col != 'ACTION']
    for col in original_features:
        df_encoded[f'{col}_count'] = df[col].map(df[col].value_counts())

    # 3. 準備 OOF Target Encoding 的欄位清單 (原始 + 組合)
    target_cols = original_features.copy()
    for p1, p2 in selected_pairs:
        new_col = f"{p1}_{p2}"
        df[new_col] = df[p1].astype(str) + "_" + df[p2].astype(str)
        target_cols.append(new_col)

    # 4. 執行 K-fold OOF Target Encoding
    print(f"Applying OOF Target Encoding to {len(target_cols)} features...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in target_cols:
        df_encoded[f'{col}_target'] = 0.0
        for train_idx, val_idx in skf.split(df, df['ACTION']):
            train_fold = df.iloc[train_idx]
            target_mean = train_fold.groupby(col)['ACTION'].mean()
            df_encoded.loc[df_encoded.index[val_idx], f'{col}_target'] = df.loc[df.index[val_idx], col].map(target_mean)
        
        global_mean = df['ACTION'].mean()
        df_encoded[f'{col}_target'] = df_encoded[f'{col}_target'].fillna(global_mean)

    # 儲存結果
    output_path = os.path.join(output_dir, 'train_v6_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\nV6 特徵工程完成！總特徵數: {df_encoded.shape[1]-1}")
    print(f"產出檔案: v6_interactions/{output_path}")

if __name__ == "__main__":
    main()
