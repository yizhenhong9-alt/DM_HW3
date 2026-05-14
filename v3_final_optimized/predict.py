import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 讀取原始資料 (修正路徑)
    print("Loading data...")
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')
    test_ids = test_df['id']
    
    # 處理缺失值
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)
    
    # 2. 準備特徵工程參數 (需與 feature_engineering.py 一致)
    features = [col for col in train_df.columns if col != 'ACTION']
    global_mean = train_df['ACTION'].mean()
    smoothing_weight = 10 
    cardinality_threshold = 50 
    
    X_train_full = pd.DataFrame()
    X_test = pd.DataFrame()
    
    print("Applying Smoothed Target Encoding and Count Encoding...")
    for col in features:
        # --- 基礎 Count Encoding ---
        counts = train_df[col].value_counts()
        X_train_full[f'{col}_count'] = train_df[col].map(counts)
        X_test[f'{col}_count'] = test_df[col].map(counts).fillna(0)
        
        # --- 判斷基數決定是否執行 Smoothed Target Encoding ---
        cardinality = train_df[col].nunique()
        
        if cardinality > cardinality_threshold:
            # 計算訓練集的統計資訊
            group_stats = train_df.groupby(col)['ACTION'].agg(['mean', 'count'])
            smoothed_map = (group_stats['count'] * group_stats['mean'] + 
                            smoothing_weight * global_mean) / (group_stats['count'] + smoothing_weight)
            
            # 套用到訓練集
            X_train_full[f'{col}_target'] = train_df[col].map(smoothed_map)
            # 套用到測試集 (沒看過的類別填 Global Mean)
            X_test[f'{col}_target'] = test_df[col].map(smoothed_map).fillna(global_mean)
            
    y_train_full = train_df['ACTION']
    
    # 3. 訓練模型 (採用優化後的參數)
    print(f"Training optimized Random Forest on {X_train_full.shape[1]} features...")
    model = RandomForestClassifier(
        n_estimators=500,      # 增加樹的數量
        max_depth=20,          # 稍微加深
        min_samples_leaf=5,    # 防止過擬合
        max_features=0.7,      # 增加特徵考慮比例
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_full, y_train_full)
    
    # 4. 進行預測
    print("Generating predictions...")
    predictions = model.predict_proba(X_test)[:, 1]
    
    # 5. 儲存結果到 results 目錄
    submission = pd.DataFrame({
        'Id': test_ids,
        'Action': predictions
    })
    output_path = os.path.join(output_dir, 'submission_v3.csv')
    submission.to_csv(output_path, index=False)
    print(f"\nPrediction complete! Result saved to: v3_final_optimized/{output_path}")

if __name__ == "__main__":
    main()
