import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

def main():
    data_path = 'results/train_v6_encoded.csv'
    if not os.path.exists(data_path):
        print("請先執行 feature_engineering.py")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop('ACTION', axis=1)
    y = df['ACTION']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    print("\nStarting 5-Fold CV Training (V6 Interactions)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 採用穩健的森林參數 (V4 邏輯)
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)
        print(f"Fold {fold} ROC-AUC: {auc:.4f}")

    avg_auc = np.mean(auc_scores)
    print(f"\n--- V6 CV 模型評估結果 (Interactions) ---")
    print(f"Average ROC-AUC: {avg_auc:.4f}")
    
    with open('results/evaluation_v6.txt', 'w') as f:
        f.write(f"V6 5-Fold Average ROC-AUC: {avg_auc:.4f}\n")
    print(f"評估結果已記錄至: v6_interactions/results/evaluation_v6.txt")

if __name__ == "__main__":
    main()
