# Amazon Employee Access Challenge - 專案演進全紀錄

本專案旨在解決 Amazon 員工資源存取權限的自動化預測問題。透過六個階段的迭代，從基礎的決策樹逐步進化到高階的特徵交互作用與模型集成系統，最終在 Kaggle 獲得了 **Public Score 0.91053 / Private Score 0.90033** 的優異成績。

---

## 📊 歷程版本與成效對照 (Summary Table)

| 版本 | 技術核心 | 特徵工程策略 | 模型演算法 | Public AUC | Private AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | 基準線 (Baseline) | Count Encoding | Decision Tree | 0.74296 | 0.75847 |
| **V2** | 特徵擴充 | Count + Target Encoding | Random Forest | 0.85139 | 0.83833 |
| **V3** | 平滑化優化 | Smoothed Target Encoding | RF (Tuned) | 0.80812 | 0.79748 |
| **V4** | **驗證機制革命** | **OOF Target Encoding** | **RF Ensemble** | 0.89260 | 0.88031 |
| **V5** | 不平衡處理 | Class Weighting (Balanced) | RF Ensemble | 0.87537 | 0.86907 |
| **V6** | **最強交互特徵** | **Top 10 MI Interactions** | **RF Ensemble** | **0.91053** | **0.90033** |

---

## 🚀 版本詳細說明

### 🟢 Version 1: 基礎基準線 (Baseline)
*   **技術特點**: 採用基礎 **Count Encoding**（計算類別頻率），將類別轉為出現次數。
*   **模型**: 單一 `DecisionTreeClassifier` (depth=10)。
*   **總結**: 雖然建立了預測流，但無法捕捉複雜 ID 之間的深層關聯，且單樹模型泛化能力有限。

### 🔵 Version 2: 特徵擴充與模型升級
*   **技術特點**: 併行使用 **Target Encoding**（目標均值編碼），將 ID 直接映射為 ACTION 的成功率。
*   **模型**: 升級為 `RandomForestClassifier` (100 棵樹)。
*   **成效**: 分數大幅躍升至 0.85，驗證了 Target Encoding 是處理高基數 ID 欄位的利器。

### 🟡 Version 3: 優化平滑化與參數調優
*   **特徵技術**: 引入 **Smoothing (平滑化)** 並加入 **基數門檻判斷 (Cardinality Threshold=50)**，避免低頻類別產生雜訊。
*   **模型調優**: 進行大幅度超參數調整：
    *   `n_estimators` 提升至 **500** 棵樹。
    *   `max_depth` 增加至 **20**。
    *   加入 `min_samples_leaf=5` 與 `max_features=0.7` 以平衡複雜度與過擬合。
*   **成效**: **分數不升反降 (0.80)**。分析顯示 Smoothing 權重過高稀釋了強訊號，且參數組合在單一切分下（無 CV）穩定度不足。

### 🔴 Version 4: 交叉驗證與 OOF 技術 (重大突破)
*   **技術核心**: 導入 **Out-of-Fold (OOF) Target Encoding**。透過 5-Fold 切分，確保訓練集中的編碼均由其他 Fold 產出，徹底根絕 Data Leakage。
*   **預測策略**: 採用 **模型集成 (Ensembling)**，取 5 個 CV 模型的平均機率。
*   **成效**: 分數顯著回升並超越 V2，達到 0.89。這證明了專業的驗證機制是穩定提升的關鍵。

### 🟠 Version 5: 類別權重調整 (不平衡處理)
*   **技術核心**: 針對 94:6 的不平衡分佈，加入 `class_weight='balanced'`。
*   **成效**: 雖然提升了對少數類別的關注度，但對純粹排名導向的 AUC 分數有些微負面影響。

### 🏆 Version 6: 數據驅動之交互特徵 (終極版本)
*   **技術核心**: 採用「全組合窮舉法」，透過 **互資訊 (Mutual Information)** 科學化篩選出 Top 10 強關聯之二階交互特徵（如 `RESOURCE` + `MGR_ID`）。
*   **實作**: 對此 10 組「超級特徵」套用 OOF 編碼與 CV 集成。
*   **成效**: **歷史最高分 (0.91)**。成功捕捉到了「特定主管對於特定資源」的細微存取模式。

---

## 🛠️ 關鍵技術點解析

1.  **OOF (Out-of-Fold) Encoding**: 解決了 Target Encoding 容易在訓練集「背答案」的問題。
2.  **MI (Mutual Information)**: 一種衡量欄位間非線性相關性的指標，能有效找出最具預測能力的特徵組合。
3.  **Interaction Features**: 跨欄位組合 ID 是此類匿名 ID 資料集中的獲勝關鍵，能挖掘出單一維度看不見的潛在關係。

---

## 📂 目錄結構
*   `v1_baseline/`: 初步嘗試版本
*   `v2_target_encoding/`: 基礎強化版本
*   `v3_final_optimized/`: 平滑化實驗版本
*   `v4_cv/`: 導入交叉驗證版本
*   `v5_imbalance/`: 不平衡處理版本
*   `v6_interactions/`: 數據驅動交互特徵最強版
*   `train.csv / test.csv`: 原始數據

---
*本專案紀錄由 Antigravity 輔助開發產出*
