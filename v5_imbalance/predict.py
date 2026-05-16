import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    # 1. 讀取原始資料
    train_df = pd.read_csv('../train.csv').fillna(-1)
    test_df = pd.read_csv('../test.csv').fillna(-1)
    test_ids = test_df['id']
    
    # 2. 準備測試集特徵
    print("Preparing test features...")
    features = [col for col in train_df.columns if col != 'ACTION']
    global_mean = train_df['ACTION'].mean()
    X_test = pd.DataFrame()
    for col in features:
        counts = train_df[col].value_counts()
        X_test[f'{col}_count'] = test_df[col].map(counts).fillna(0)
    for col in features:
        target_means = train_df.groupby(col)['ACTION'].mean()
        X_test[f'{col}_target'] = test_df[col].map(target_means).fillna(global_mean)

    # 3. 讀取 V5 訓練特徵
    X_train_full = pd.read_csv('results/train_v5_encoded.csv').drop('ACTION', axis=1)
    X_train_full = X_train_full[X_test.columns]
    y_train_full = train_df['ACTION']

    # 4. 執行 5-Fold 預測並集成 (加入 Class Weighting)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_predictions = np.zeros(len(test_df))
    
    print("\nStarting CV Ensembling Prediction (V5)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_fold_train = X_train_full.iloc[train_idx]
        y_fold_train = y_train_full.iloc[train_idx]
        
        # 核心改進: class_weight='balanced'
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_fold_train, y_fold_train)
        
        fold_preds = model.predict_proba(X_test)[:, 1]
        final_predictions += fold_preds
        print(f"Fold {fold} prediction finished.")

    final_predictions /= 5
    output_path = 'results/submission_v5.csv'
    submission = pd.DataFrame({'Id': test_ids, 'Action': final_predictions})
    submission.to_csv(output_path, index=False)
    print(f"\nV5 預測完成！產出檔案: v5_imbalance/{output_path}")

if __name__ == "__main__":
    main()
