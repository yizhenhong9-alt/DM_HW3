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
    
    # 2. 準備測試集特徵 (必須嚴格遵守訓練時的欄位順序)
    print("Preparing test features...")
    features = [col for col in train_df.columns if col != 'ACTION']
    global_mean = train_df['ACTION'].mean()
    X_test = pd.DataFrame()
    
    # 先產出所有 count 欄位
    for col in features:
        counts = train_df[col].value_counts()
        X_test[f'{col}_count'] = test_df[col].map(counts).fillna(0)
    
    # 再產出所有 target 欄位
    for col in features:
        target_means = train_df.groupby(col)['ACTION'].mean()
        X_test[f'{col}_target'] = test_df[col].map(target_means).fillna(global_mean)

    # 3. 讀取已經做過 OOF 處理的訓練特徵
    # 確保欄位順序與 X_test 完全一致
    X_train_full = pd.read_csv('results/train_cv_encoded.csv').drop('ACTION', axis=1)
    # 重新對齊欄位 (以防萬一)
    X_train_full = X_train_full[X_test.columns]
    y_train_full = train_df['ACTION']

    # 4. 執行 5-Fold 預測並集成 (Ensembling)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_predictions = np.zeros(len(test_df))
    
    print("\nStarting CV Ensembling Prediction...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_fold_train = X_train_full.iloc[train_idx]
        y_fold_train = y_train_full.iloc[train_idx]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_fold_train, y_fold_train)
        
        fold_preds = model.predict_proba(X_test)[:, 1]
        final_predictions += fold_preds
        print(f"Fold {fold} prediction finished.")

    # 5. 取平均值並儲存
    final_predictions /= 5
    output_path = 'results/submission_v4.csv'
    submission = pd.DataFrame({'Id': test_ids, 'Action': final_predictions})
    submission.to_csv(output_path, index=False)
    print(f"\nV4 預測完成！產出檔案: v4_cv/{output_path}")

if __name__ == "__main__":
    main()
