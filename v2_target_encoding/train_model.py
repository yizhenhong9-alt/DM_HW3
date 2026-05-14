import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import os

def main():
    # 確保輸出目錄存在
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. 讀取包含 Count + Target Encoding 的特徵資料 (路徑修正)
    data_path = 'results/train_encoded.csv'
    if not os.path.exists(data_path):
        print(f"找不到檔案: {data_path}，請先執行 feature_engineering.py")
        return
        
    print(f"正在讀取資料: {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. 準備特徵 (X) 與標籤 (y)
    X = df.drop('ACTION', axis=1)
    y = df['ACTION']
    
    # 3. 切分訓練集與驗證集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"驗證集大小: {X_val.shape}")
    
    # 4. 初始化隨機森林模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. 訓練模型
    print("\n正在訓練隨機森林模型 (Random Forest)...")
    model.fit(X_train, y_train)
    
    # 6. 在驗證集上進行預測
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # 7. 計算評估指標
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_prob)
    
    print("\n模型評估結果:")
    print(f"Accuracy (準確度): {accuracy:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # 8. 顯示特徵重要性 (Top 10)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n特徵重要性 (Top 10):")
    print(feature_importance.head(10))
    
    # 將評估結果記錄下來
    with open('results/evaluation_v2.txt', 'w') as f:
        f.write(f"V2 Accuracy: {accuracy:.4f}\n")
        f.write(f"V2 ROC-AUC Score: {auc_score:.4f}\n")
    print(f"評估結果已記錄至: v2_target_encoding/results/evaluation_v2.txt")

if __name__ == "__main__":
    main()
