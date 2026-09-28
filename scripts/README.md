# scripts/ — 一次性執行腳本

每個任務的實際執行入口。從專案根目錄執行：

```bash
python scripts/YYYY-MM-DD_<用途>.py
```

- 命名：`YYYY-MM-DD_<用途>.py`，日期是任務開始日
- 一個腳本對應 `plans/` 底下一個 PLAN
- 輸入依 PLAN 指定，可讀取 `data/raw/`（唯讀）或 `data/processed/` 中已完成所需驗證的資料；本任務後續使用 `data/processed/annotations_area_fixed/`，其 area 已驗證，其餘資料品質檢查仍依 PLAN 完成。
- 產物寫入 `data/processed/` 與 `results/` 的獨立輸出位置，不覆寫原始資料或作為輸入的修正副本。
- 共用邏輯抽到 `src/`，這裡只留流程與參數
- 隨機種子要固定，讓結果可重現

## 蘭花 COCO 資料檢查

在專案根目錄執行（Windows 專案虛擬環境）：

```powershell
# PLAN 第 1 步：標註盤點，僅使用標準函式庫
.venv/Scripts/python.exe scripts/2026-09-28_validate_orchid_coco.py

# PLAN 第 2 步：影像解碼、尺寸、遮罩及修正 area 檢查
.venv/Scripts/python.exe scripts/2026-09-28_validate_orchid_coco.py --quality

# 第 2 步通過後執行第 3 步：既有切分、跨集合影像比對
.venv/Scripts/python.exe scripts/2026-09-28_check_orchid_splits.py
```

資料檢查實測套件為 numpy 2.5.3、pycocotools 2.0.11、Pillow 12.3.0；這不代表訓練環境已驗證。安裝至所用 Python 環境的指令為：

```powershell
python -m pip install numpy==2.5.3 pycocotools==2.0.11 Pillow==12.3.0
```

第 3 步產出：

- `data/processed/splits/2026-09-28_orchid_split_manifest.csv`：301 張影像的原始切分、COCO ID、相對路徑、JSON／檔案／RGB 像素雜湊、尺寸及來源 ID。未知來源 ID 留空，不從檔名推定。
- `results/tables/2026-09-28_orchid_split_checks.csv`：檢查結果，包含 PASS／FAIL／PENDING 與理由。
- `results/tables/2026-09-28_orchid_split_pairs.csv`：相同／近似候選及待填覆核欄位。
- `results/figures/2026-09-28_orchid_split_review.html`：用瀏覽器開啟以並排檢視；引用專案內的真實影像，不可單獨搬移後期待連結仍有效。

近似候選使用整張匯出影像（包含補邊）的 dHash 64 位元及排除 DC 的 DCT pHash 63 位元；漢明距離分別 ≤8 或 ≤10 即列入候選。這是可重現的初始篩選門檻，不是 SPEC 驗收門檻，不保證找出所有裁切、旋轉或不同角度的同源影像。可用 `--dhash-distance`、`--phash-distance` 調整，實際設定會寫入報告；來源對照仍是必要條件。

完成第 3 步需準備兩份**獨立輸入檔案**，不可直接把產出檔案作為 `--groups`／`--reviews` 輸入：

1. 資料提供者的來源 CSV：欄位為 `split,image_id,file_name,plant_id,sequence_id`，涵蓋全部影像；split 使用 train／valid／test，ID 與檔名符合相應 COCO。植株與序列 ID 必須能跨集合一致識別同一來源。可複製 manifest 至另一個路徑後依真實紀錄填寫，不得自行替未知影像編造不同來源 ID。
2. 人工覆核 CSV：將 pairs CSV 複製至另一個路徑，保留 `pair_id`，填入 `decision`（different_source／same_source／unsure）、`reviewer`、`reviewed_at`（YYYY-MM-DD）、`note`。只有有完整紀錄的 different_source 才能解除該候選待確認狀態；完全重複影像無法用覆核豁免。

```powershell
# 路徑應改為實際取得並填妥的輸入 CSV
.venv/Scripts/python.exe scripts/2026-09-28_check_orchid_splits.py --groups <來源對照表.csv> --reviews <人工覆核表.csv>
```

腳本不重新切分資料。第 3 步遇到來源缺漏、未覆核候選或跨集合同源時會回傳退出碼 1；参数／I/O 錯誤回傳 2。若 pairs 產出已填有人工覆核資料，腳本拒絕覆寫；先將它移至獨立輸入位置再執行。整體為 PENDING／FAIL 時不得進入訓練或宣告資料洩漏檢查通過。

補充：使用者其後確認人工覆核無重複植株，並授權先完成可搬移程式；依 PLAN 新增的補充紀錄，程式準備與後續執行保留來源缺失限制。上述自動 PENDING 不會被人工改成 PASS，也不作為程式準備的阻擋。

## SAM 3 訓練與評估：搬至訓練機後執行

以下是待 RTX 5090 實測的候選安裝／執行流程。本機只完成 CPU 與設定驗證，沒有執行模型訓練、推論或顯存測試。原始碼固定在官方 commit `2345a4ad109ac29c569da749c91d84f10dc08c40`；換 commit 必須重新核對 API，而非直接更新主分支。

### 1. 搬移資料與建立環境

搬移專案程式與完整 `data/processed/annotations_area_fixed/`；這些資料不在 Git。最終評估還需 `data/processed/support_eval/test/`。不要搬移 Windows `.venv`，請在 Linux／WSL2 的專案根目錄新建環境。使用 RTX 5090 作為可見 CUDA device 0。

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 若已有 checkout，使用既有目錄並確認 commit；不需重複 clone。
git clone https://github.com/facebookresearch/sam3.git ../sam3
git -C ../sam3 checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
python -m pip install -r requirements-training.txt -e '../sam3[train]'
python -m pip check
```

此版本 SAM 3 要求 NumPy <2，不能把本機資料檢查用的 NumPy 2.5.3 當成訓練環境鎖定版本。`setuptools<81` 保留上游使用的 pkg_resources。安裝基於 [官方 SAM 3 說明](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40) 與 [PyTorch wheel 配對](https://pytorch.org/get-started/previous-versions/#v2100)，仍需目標機 smoke test 確認相容。

取得具有存取權的官方 SAM 3（非 SAM 3.1 video）基礎權重後，將下列路徑改為實際值；不要將 Hugging Face token 填入程式或提交 Git。程式只讀取指定權重，不自動下載。

```bash
export SAM3_REPO="$(realpath ../sam3)"
export SAM3_BASE="/absolute/path/to/sam3.pt"
python scripts/2026-09-28_check_sam3_environment.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE"
```

環境盤點腳本即使套件齊全也保留小批次測試 PENDING，退出碼為 1；看逐項報告，不將此退出碼誤當完成訓練測試。若 GPU、來源 commit 或權重不符，先處理對應問題。

### 2. 準備設定與測試儲存／恢復

```bash
# 沒有 GPU 或權重也可先解析設定；不提供 checkpoint 時，僅產生待補權重的設定。
python scripts/2026-09-28_train_sam3_orchid.py --sam3-repo "$SAM3_REPO" --run-name config_check --prepare-only

# 4 張真實 train 影像、一個 batch；valid 取涵蓋三類的最小前綴，只作程式測試。
python scripts/2026-09-28_train_sam3_orchid.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --run-name smoke --smoke-test --workers 0

# 同一 run 恢復 optimizer、epoch、scaler 與 RNG，總 epoch 延長到 2。
python scripts/2026-09-28_train_sam3_orchid.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --run-name smoke --smoke-test --workers 0 --epochs 2 --resume
```

smoke test 使用官方 loader 的固定子集抽樣，僅用 train 的 4 張影像；不產生假影像或標註。valid 按 image ID 順序取涵蓋三類標註的最小前綴，並保存 smoke_validation_images.json；避免只取第一張、缺少某類而使 macro IoU 為 N/A。為使一個 batch 確實更新參數，smoke 模式關閉 scheduler warmup／cooldown。smoke 的 validation 不代表完整 valid 結果，checkpoint 帶有 smoke 標記，正式 test 評估會拒絕使用它。

確認兩次命令均成功、`results/models/smoke/completion.json` 的 `resume_exercised=true`，且 checkpoint 與 validation CSV 存在後，才建立獨立正式 run。GPU 記憶體不足時程式保存 `oom.json` 並停止，不會自行調整 batch size、解析度或凍結層。

`--prepare-only` 產生的是當地絕對路徑設定；搬到另一台機器後重跑準備命令，不直接使用舊的 `config_resolved.yaml`。

### 3. 正式訓練

```bash
python scripts/2026-09-28_train_sam3_orchid.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --run-name orchid_v1

# 中斷後用同一個 run name 恢復；--epochs 是總目標，不是新增 epoch 數。
python scripts/2026-09-28_train_sam3_orchid.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --run-name orchid_v1 --resume
```

預設 10 epochs、batch size 4（最後不足 4 張仍保留）、gradient accumulation 1、bfloat16、1008×1008、全模型微調。train 使用全部 211 張，valid 使用全部 60 張。三類 prompt 分別為 flower／leaf／stem；未使用的 0=sam3 僅從 run 內衍生 JSON 的 categories 移除。

採官方 Roboflow optimizer／matcher／box／classification loss，加上 instance mask focal loss（權重 200）與 Dice loss（權重 10）；不使用額外 semantic head loss。學習率沿用官方倍率後 transformer=8e-5、vision=2.5e-5、language=5e-6，weight decay=0.1，gradient clip=0.1，scheduler timescale／warmup／cooldown 為 20 steps。固定 square resize，不隨機增強；全批次包含三類查詢及其負查詢。seed 設定為 123，上游有效種子為 seed×max_epochs；保存 RNG 以供恢復，但不宣稱跨 CUDA／套件版本 bitwise 一致。

每個 run 會保存：

- `config_resolved.yaml`、`input_signature.json`、`data_limitations.json`：設定、資料／基礎權重雜湊與来源限制。
- `environment.json`、`requirements.actual.txt`：實際執行環境；完成 smoke test 後才可作為鎖定版本依據。
- `checkpoints/checkpoint_N.pt`、`checkpoint.pt`、`best.pt`：每 epoch、最後、最佳 valid macro IoU checkpoint。
- `logs/train_stats.json` 與 TensorBoard：每 epoch loss、各 loss 分項與 optimizer 紀錄。
- `validation/epoch_NNN.csv`、`best.json`：完整 valid 指標與選模理由。
- `completion.json` 或 `failure.json`／`oom.json`：執行結果。

恢復僅限原 run，會檢查來源 commit、train／valid 影像及 JSON、基礎權重的雜湊與固定設定。官方預訓練權重的 `detector.` 前綴與專案 checkpoint 分別處理，兩者都用 strict state-dict 載入，漏載參數直接報錯。

### 4. 最終 test 評估

支架 manifest 欄位為 `split,image_id,image_path,mask_path,review_status,reviewer,reviewed_at`。`split` 必須為 test；路徑相對專案根目錄；人工覆核後 `review_status=approved`，日期 YYYY-MM-DD。全部 30 張都要有原尺寸、8-bit 單通道、像素值 0／255 的 PNG；沒有支架也須明確提供全零遮罩。

```bash
python scripts/2026-09-28_evaluate_sam3_orchid.py --sam3-repo "$SAM3_REPO" --checkpoint results/models/orchid_v1/checkpoints/best.pt
```

輸出 `results/tables/2026-09-28_sam3_orchid_metrics.csv`、同前綴 `_evaluation.json` 與 `results/figures/2026-09-28_sam3_orchid_examples.png`（300 dpi）。預覽預設前 6 張；可用 `--preview-count 30` 查看全部測試影像，不改變評估範圍。

依 SPEC 用原始 COCO 尺寸、分數 **≥0.35**、同類實例聯集，先彙總全部 test 的 TP／FP／FN 再算 IoU／Dice／Precision／Recall；macro 遇任何 N/A 不跳過該類。官方 processor 已將 1008×1008 輸出還原至輸入影像尺寸，本程式再確認形狀一致；不去除資料原先 512×512 的補邊。

缺少任何支架遮罩、覆核／對應不明、格式／尺寸不符或全資料集支架面積為零，該指標為 N/A。各類 IoU 或支架門檻未通過時退出碼 1；數值門檻全部通過時退出碼 0，仍不代表来源独立性或人工預覽已完成正式驗收，JSON 會分別記錄。

### 5. 本機 CPU 回歸檢查

```bash
python -m pytest tests/test_orchid_pipeline.py -q -p no:cacheprovider
```

測試使用現有真實 COCO 遮罩作一致性檢查，公式測試不產生模型指標；不把真實遮罩自我比對的 1.0 當成模型表現。未搬移資料或未提供固定版 `SAM3_REPO` 時，對應測試會 skip，須確認報告沒有把 skip 當作完整驗證。尚待 GPU 驗證：模型與套件載入、完整 batch 前向／反向、bf16、顯存、checkpoint 恢復及實際推論預覽。
