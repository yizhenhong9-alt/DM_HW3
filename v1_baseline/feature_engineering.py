import pandas as pd
import os

def main():
    # 確保輸出目錄存在
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 讀取訓練資料 (路徑修正為根目錄)
    train_path = '../train.csv'
    df = pd.read_csv(train_path)

    # --- 缺失值處理 ---
    print("正在檢查缺失值...")
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("發現缺失值情況：")
        print(missing_info[missing_info > 0])
        # 填充缺失值為 -1
        df = df.fillna(-1)
        print("已將缺失值填充為 -1。")
    else:
        print("未發現缺失值。")
    # ------------------
    
    # 找出需要進行編碼的欄位（除了 'ACTION' 以外的所有欄位）
    features = [col for col in df.columns if col != 'ACTION']
    
    print(f"原始欄位: {df.columns.tolist()}")
    print(f"進行 Count Encoding 的欄位: {features}")
    
    # 建立一個新的 DataFrame 來存放轉換後的結果
    df_encoded = df[['ACTION']].copy()
    
    for col in features:
        # 計算每個類別出現的次數
        counts = df[col].value_counts()
        # 將原始數值映射為出現次數
        df_encoded[col] = df[col].map(counts)
        
    # 顯示轉換後的前幾列數據
    print("\n轉換後的數據前 5 列:")
    print(df_encoded.head())
    
    # 儲存處理後的資料到 results 目錄
    output_path = os.path.join(output_dir, 'train_count_encoded.csv')
    df_encoded.to_csv(output_path, index=False)
    print(f"\n處理後的資料已儲存至: v1_baseline/{output_path}")

if __name__ == "__main__":
    main()
