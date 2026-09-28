# PLAN：建立可辨識蘭花花、莖、葉的 SAM 3 fine-tuning 模型

- 建立日期：2026-09-28
- 提出者：陳彥彣
- 狀態：草稿
- 審核者與日期：

---

## 0. 開始前的三個問題

| # | 問題 | 回答 |
|---|---|---|
| 1 | 已標註的訓練資料位於何處、使用何種格式？ | 位於 `data/processed/`，使用 COCO 格式。 |
| 2 | 模型最終需要辨識哪些類別？ | 僅辨識 `flower`、`stem`、`leaf`；不辨識 root。 |
| 3 | 使用何種模型與硬體進行微調？ | 使用既有 SAM 3 基礎權重，在 RTX 5090 上進行 fine-tuning。 |

## 1. 目標

以 `data/processed/` 中的 COCO 標註資料微調既有 SAM 3 權重，建立可對新蘭花影像分割 `flower`、`stem`、`leaf` 三類實例的可重跑模型與評估流程。

## 2. 輸入與輸出

**輸入**

- `data/processed/` 中的原始影像與 COCO annotation JSON
- `SPEC.md` 中的類別、影像座標與資料格式定義
- 既有 SAM 3 基礎權重
- RTX 5090 訓練環境

**輸出**

- `data/processed/splits/`：可重現的 train、validation、test 影像清單或 COCO split JSON
- `scripts/2026-09-28_validate_orchid_coco.py`：資料集驗證腳本
- `scripts/2026-09-28_train_sam3_orchid.py`：SAM 3 fine-tuning 腳本
- `scripts/2026-09-28_evaluate_sam3_orchid.py`：測試集推論與評估腳本
- `results/figures/2026-09-28_sam3_orchid_examples.png`：測試影像的預測遮罩預覽
- `results/tables/2026-09-28_sam3_orchid_metrics.csv`：flower、stem、leaf 的 IoU、Dice、Precision、Recall
- `results/models/`：訓練 checkpoint 與訓練設定副本

## 3. 步驟拆解

每個步驟必須有明確的驗收標準。做完一步停下來回報。

| # | 步驟 | 驗收標準 |
|---|---|---|
| 1 | 盤點 `data/processed/` 的 COCO JSON、影像數量、類別名稱與 annotation 數量；不修改任何資料檔。 | 確認類別恰為 `flower`、`stem`、`leaf`；每筆 annotation 都能對應到存在的影像。 |
| 2 | 驗證遮罩格式、影像尺寸、空遮罩、重複 image ID 與損毀影像，輸出資料品質報告。 | 產出驗證表；所有錯誤項目明確列出，未通過時停止並回報。 |
| 3 | 依植株／拍攝序列建立固定的 train、validation、test split，避免近似影像跨集合。 | 三份資料互斥、可重跑，且每個集合都有三類有效 annotation。 |
| 4 | 建立 SAM 3 訓練設定：RTX 5090、bfloat16 autocast、初始 batch size 1、可儲存與恢復 checkpoint。 | 以一小批資料完成訓練 dry run，GPU 可用且不發生 dtype、資料格式或顯存錯誤。 |
| 5 | 執行 fine-tuning，保存每個 epoch 的 checkpoint、loss、訓練設定與驗證結果。 | 訓練正常結束；至少保留最佳 validation checkpoint 與最後 checkpoint。 |
| 6 | 使用從未參與訓練的 test split 推論，計算每類 IoU、Dice、Precision、Recall 並輸出遮罩預覽。 | 產出指標 CSV 與預覽圖；預覽圖能清楚區分 flower、stem、leaf。 |
| 7 | 將實測指標、限制與可重跑方式寫入 README，並在 DEVLOG 追加完成紀錄。 | README 可使他人重跑；DEVLOG 記錄資料版本、權重與主要結果。 |

## 4. 驗證計畫

除了 `instructions.md` 的通用檢查外，這個任務要特別確認：

- [ ] COCO 的每個 `category_id` 都對應到 `flower`、`stem`、`leaf` 之一。
- [ ] 訓練、驗證、測試集合沒有相同影像，且近似連拍影像不跨集合。
- [ ] 每一類在 train、validation、test 都有足夠的有效遮罩；數量不足時停止並回報。
- [ ] 模型輸出遮罩與原圖尺寸一致，且不發生 CUDA 的 bfloat16/float dtype 衝突。
- [ ] 報告每類與 macro-average 的 IoU、Dice、Precision、Recall；不得只報單一總分。
- [ ] 人工抽查預測預覽，特別檢查支架或固定夾是否被誤判為 stem。

## 5. 不確定與風險

- 尚未確認的事項：COCO JSON 的確切檔名、影像與 annotation 的目錄結構、各類樣本數與拍攝群組資訊。
- 可能出錯的地方：原始影像被當成已處理資料、COCO 類別 ID 與名稱不一致、支架被誤標為 stem、資料切分發生影像洩漏、SAM 3 完整微調造成顯存不足。
- 如果 COCO 內的類別不是 `flower`、`stem`、`leaf`，或任一類缺少有效標註，就停止並詢問。
- 如果 RTX 5090 上仍發生顯存不足，就停止並回報記憶體用量，再由使用者決定是否降低解析度、採用凍結層或調整訓練策略。
- 最終的數值驗收門檻尚未由使用者指定；在結果宣告「達標」前，需先確認每類 IoU/Dice 的目標值。

## 6. 不做什麼

明確排除掉的範圍，避免 agent 自行擴張：

- 不修改、刪除或覆寫 `data/raw/` 的任何檔案。
- 不自行產生假影像、假遮罩或假標註。
- 不訓練 YOLO 或其他非 SAM 3 模型。
- 不加入 root、support、花盆或支架為模型辨識類別。
- 不在未確認 PLAN 前建立訓練程式、下載權重或開始 fine-tuning。
