import pandas as pd
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取訓練資料 (路徑修正)
    train_path = '../train.csv'
    df = pd.read_csv(train_path)
    
    # --- 1. 缺失值處理 (保留) ---
    print("正在檢查缺失值...")
    if df.isnull().sum().sum() > 0:
        df = df.fillna(-1)
        print("已將缺失值填充為 -1。")
    else:
        print("未發現缺失值。")

    # 找出需要進行編碼的欄位
    features = [col for col in df.columns if col != 'ACTION']
    
    # 建立一個新的 DataFrame 來存放結果，保留原始標籤 ACTION
    df_encoded = df[['ACTION']].copy()
    
    # 計算全局平均值 (Global Mean)
    global_mean = df['ACTION'].mean()

    print("\n正在進行併行特徵工程 (Count + Target Encoding)...")
    
    for col in features:
        # A. Count Encoding
        counts = df[col].value_counts()
        df_encoded[f'{col}_count'] = df[col].map(counts)
        
        # B. Target Encoding (計算該類別對應 ACTION 的平均值)
        target_means = df.groupby(col)['ACTION'].mean()
        df_encoded[f'{col}_target'] = df[col].map(target_means)
        
    # 顯示前幾列結果
    print("\n處理後的特徵 (前 5 列):")
    print(df_encoded.head())
    
    # 儲存處理後的資料到 results 目錄
    output_path = os.path.join(output_dir, 'train_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\n處理完成！結果已儲存至: v2_target_encoding/{output_path}")

if __name__ == "__main__":
    main()
