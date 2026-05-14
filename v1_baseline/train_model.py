import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import os

def main():
    # 確保輸出目錄存在 (記錄評估結果用)
    if not os.path.exists('results'):
        os.makedirs('results')

    # 讀取特徵工程後的資料 (路徑修正為 results 目錄)
    data_path = 'results/train_count_encoded.csv'
    if not os.path.exists(data_path):
        print(f"找不到檔案: {data_path}，請先執行 feature_engineering.py")
        return
        
    df = pd.read_csv(data_path)
    
    # 準備特徵 (X) 與標籤 (y)
    X = df.drop('ACTION', axis=1)
    y = df['ACTION']
    
    # 切分訓練集與驗證集 (80% 訓練, 20% 驗證)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化決策樹模型 (設定最大深度為 10 以避免過度擬合)
    model = DecisionTreeClassifier(
        max_depth=10, 
        random_state=42
    )
    
    # 訓練模型
    print("正在訓練決策樹模型...")
    model.fit(X_train, y_train)
    
    # 在驗證集上進行預測
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # 計算評估指標 (Amazon 競賽通常看 AUC)
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_prob)
    
    print("\n模型評估結果:")
    print(f"Accuracy (準確度): {accuracy:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # 顯示最重要的前 5 個特徵
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n特徵重要性 (Top 5):")
    print(feature_importance.head(5))
    
    # 將評估結果記錄下來
    with open('results/evaluation_v1.txt', 'w') as f:
        f.write(f"V1 Accuracy: {accuracy:.4f}\n")
        f.write(f"V1 ROC-AUC Score: {auc_score:.4f}\n")
    print(f"評估結果已記錄至: v1_baseline/results/evaluation_v1.txt")

if __name__ == "__main__":
    main()
