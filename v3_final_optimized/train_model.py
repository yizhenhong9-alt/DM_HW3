import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import os

def main():
    # 確保輸出目錄存在
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. 讀取特徵工程後的資料 (修正路徑)
    data_path = 'results/train_encoded.csv'
    if not os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        print(f"找不到檔案: {data_path}，請先執行 feature_engineering.py")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. 準備特徵 (X) 與標籤 (y)
    X = df.drop('ACTION', axis=1)
    y = df['ACTION']
    
    # 3. 切分訓練集與驗證集 (80% 訓練, 20% 驗證)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # 4. 初始化隨機森林模型 (依據優化建議)
    model = RandomForestClassifier(
        n_estimators=500,      # 增加樹的數量提升穩定性
        max_depth=20,          # 稍微加深以捕捉更多特徵組合
        min_samples_leaf=5,    # 限制葉子最小樣本數以防止過擬合
        max_features=0.7,      # 增加每次分裂考慮的特徵比例
        random_state=42,
        n_jobs=-1
    )
    
    # 5. 訓練模型
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    
    # 6. 在驗證集上進行預測
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] 
    
    # 7. 計算評估指標
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_prob)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # 8. 顯示特徵重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10))
    
    # 將評估結果記錄下來
    with open('results/evaluation_v3.txt', 'w') as f:
        f.write(f"V3 Accuracy: {accuracy:.4f}\n")
        f.write(f"V3 Optimized ROC-AUC Score: {auc_score:.4f}\n")
    print(f"評估結果已記錄至: v3_final_optimized/results/evaluation_v3.txt")

if __name__ == "__main__":
    main()
