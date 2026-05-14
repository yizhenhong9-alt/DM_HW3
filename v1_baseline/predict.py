import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 讀取訓練集以取得 Count Encoding 的映射基準 (修正路徑)
    train_df = pd.read_csv('../train.csv')
    
    # 2. 讀取測試集 (修正路徑)
    test_df = pd.read_csv('../test.csv')
    test_ids = test_df['id']
    
    # 處理缺失值 (與訓練集一致)
    test_df = test_df.fillna(-1)
    
    # 3. 測試集特徵工程 (Count Encoding)
    features = [col for col in train_df.columns if col != 'ACTION']
    X_test = pd.DataFrame()
    for col in features:
        counts = train_df[col].value_counts()
        # 映射次數，若為新類別則填 0
        X_test[col] = test_df[col].map(counts).fillna(0)
    
    # 4. 使用完整訓練集訓練模型
    X_train_full = pd.DataFrame()
    for col in features:
        X_train_full[col] = train_df[col].map(train_df[col].value_counts())
    y_train_full = train_df['ACTION']
    
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train_full, y_train_full)
    
    # 5. 進行預測 (取得預測為 1 的機率值)
    predictions = model.predict_proba(X_test)[:, 1]
    
    # 6. 儲存結果到 results 目錄
    submission = pd.DataFrame({
        'Id': test_ids,
        'Action': predictions
    })
    output_path = os.path.join(output_dir, 'submission_v1.csv')
    submission.to_csv(output_path, index=False)
    print(f"預測完成！結果已儲存至: v1_baseline/{output_path}")

if __name__ == "__main__":
    main()
