import pandas as pd
import numpy as np
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 讀取資料 (修正路徑)
    print("Loading train.csv...")
    df = pd.read_csv('../train.csv')
    
    # 2. 缺失值處理 (保留原有邏輯)
    print("\n--- Step 1: Missing Value Handling ---")
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("Missing values found and filled with -1.")
        df = df.fillna(-1)
    else:
        print("No missing values found.")

    # 3. 欄位基數分析與決策 (Cardinality-based Decision)
    print("\n--- Step 2: Cardinality Analysis & Encoding Decision ---")
    features = [col for col in df.columns if col != 'ACTION']
    df_encoded = df[['ACTION']].copy()
    global_mean = df['ACTION'].mean()
    
    # --- 關鍵參數 ---
    # smoothing_weight: 平滑係數
    smoothing_weight = 10 
    # cardinality_threshold: 門檻值
    cardinality_threshold = 50 

    for col in features:
        cardinality = df[col].nunique()
        counts = df[col].value_counts()
        
        # 所有欄位均執行 Count Encoding
        df_encoded[f'{col}_count'] = df[col].map(counts)
        
        # 根據基數判斷是否執行平滑化的 Target Encoding
        if cardinality > cardinality_threshold:
            print(f"Field [{col:18}]: Unique Values = {cardinality:5} (High) -> Applying Smoothed Target Encoding")
            group_stats = df.groupby(col)['ACTION'].agg(['mean', 'count'])
            smoothed_val = (group_stats['count'] * group_stats['mean'] + 
                            smoothing_weight * global_mean) / (group_stats['count'] + smoothing_weight)
            df_encoded[f'{col}_target'] = df[col].map(smoothed_val)
        else:
            print(f"Field [{col:18}]: Unique Values = {cardinality:5} (Low)  -> Skipping Target Encoding to avoid noise")

    # 4. 儲存結果到 results 目錄
    output_path = os.path.join(output_dir, 'train_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\n--- Feature Engineering Complete! ---")
    print(f"Total features generated: {df_encoded.shape[1] - 1}")
    print(f"Saved to: v3_final_optimized/{output_path}")

if __name__ == "__main__":
    main()
