import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import os

def main():
    print("Loading data for analysis...")
    df = pd.read_csv('../train.csv').fillna(-1)
    features = [col for col in df.columns if col != 'ACTION']
    
    pairs = list(combinations(features, 2))
    print(f"Analyzing {len(pairs)} possible pairs...")
    
    results = []
    for i, (f1, f2) in enumerate(pairs, 1):
        # 建立組合 ID
        combined = df[f1].astype(str) + "_" + df[f2].astype(str)
        # 轉換為數值編碼以便計算 MI
        combined_encoded = combined.astype('category').cat.codes.values.reshape(-1, 1)
        
        # 計算互資訊 (Mutual Information)
        mi = mutual_info_classif(combined_encoded, df['ACTION'], discrete_features=True, random_state=42)[0]
        results.append(((f1, f2), mi))
        if i % 5 == 0:
            print(f"Progress: {i}/{len(pairs)} pairs analyzed...")

    # 排序並取得前 10 名
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("\n--- Top 10 Feature Pairs (By Data Evidence) ---")
    top_10 = []
    for pair, score in sorted_results[:10]:
        print(f"{pair[0]} + {pair[1]}: {score:.4f}")
        top_10.append(pair)
    
    # 儲存篩選結果供特徵工程腳本讀取
    with open('results/selected_pairs.txt', 'w') as f:
        for p1, p2 in top_10:
            f.write(f"{p1},{p2}\n")
    print("\nSelected pairs saved to results/selected_pairs.txt")

if __name__ == "__main__":
    main()
