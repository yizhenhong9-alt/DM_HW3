import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def main():
    # 建立輸出目錄
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df = pd.read_csv('../train.csv').fillna(-1)
    
    features = [col for col in df.columns if col != 'ACTION']
    df_encoded = df[['ACTION']].copy()
    
    # 1. Count Encoding (基礎特徵)
    print("Applying Count Encoding...")
    for col in features:
        df_encoded[f'{col}_count'] = df[col].map(df[col].value_counts())

    # 2. Out-of-Fold (OOF) Target Encoding (關鍵改進)
    print("Applying K-fold Out-of-Fold Target Encoding...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in features:
        # 建立一個暫存欄位存取均值
        df_encoded[f'{col}_target'] = 0.0
        
        # 將資料切成 5 份
        for train_idx, val_idx in skf.split(df, df['ACTION']):
            # 用 train_idx 的資料計算均值，填入 val_idx 的位置
            train_fold = df.iloc[train_idx]
            target_mean = train_fold.groupby(col)['ACTION'].mean()
            
            # 映射到驗證集片段
            df_encoded.loc[df_encoded.index[val_idx], f'{col}_target'] = df.loc[df.index[val_idx], col].map(target_mean)
        
        # 處理可能的缺失值 (若某類別在該 fold 沒出現過)
        global_mean = df['ACTION'].mean()
        df_encoded[f'{col}_target'] = df_encoded[f'{col}_target'].fillna(global_mean)

    # 儲存結果
    output_path = os.path.join(output_dir, 'train_cv_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\nV4 特徵工程完成！產出檔案: v4_cv/{output_path}")

if __name__ == "__main__":
    main()
