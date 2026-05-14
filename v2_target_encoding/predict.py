import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 讀取原始資料 (修正路徑)
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')
    test_ids = test_df['id']
    
    # 處理缺失值
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)
    
    # 2. 準備特徵工程
    features = [col for col in train_df.columns if col != 'ACTION']
    global_mean = train_df['ACTION'].mean() # 用於填充測試集中新類別的 Target Encoding
    
    X_train_full = pd.DataFrame()
    X_test = pd.DataFrame()
    
    print("正在執行 Count + Target Encoding 特徵工程...")
    for col in features:
        # --- 訓練集特徵 ---
        counts = train_df[col].value_counts()
        target_means = train_df.groupby(col)['ACTION'].mean()
        
        X_train_full[f'{col}_count'] = train_df[col].map(counts)
        X_train_full[f'{col}_target'] = train_df[col].map(target_means)
        
        # --- 測試集特徵 ---
        # 若測試集出現訓練集沒有的類別：Count 填 0，Target 填全局平均值
        X_test[f'{col}_count'] = test_df[col].map(counts).fillna(0)
        X_test[f'{col}_target'] = test_df[col].map(target_means).fillna(global_mean)
        
    y_train_full = train_df['ACTION']
    
    # 3. 訓練隨機森林模型
    print("正在使用隨機森林訓練完整模型...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_full, y_train_full)
    
    # 4. 進行預測
    print("正在生成測試集預測結果...")
    predictions = model.predict_proba(X_test)[:, 1]
    
    # 5. 儲存結果到 results 目錄
    submission = pd.DataFrame({
        'Id': test_ids,
        'Action': predictions
    })
    output_path = os.path.join(output_dir, 'submission_v2.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n預測完成！結果已儲存至: v2_target_encoding/{output_path}")

if __name__ == "__main__":
    main()
