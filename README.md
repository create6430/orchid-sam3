# orchid-sam3：蘭花花、莖、葉分割

應用於農業機器人之視覺分割自動化：規劃微調 SAM 3，分割 flower、stem、leaf。

- 專案負責人：陳彥彣
- 提出者：廖柏任
- 狀態：過往真實GPU初始training／validation、A1及clean-context resume成功證據保留。本次搬機前兩程序functional smoke在第一個forward達900秒timeout，驗收未完成，判FAIL；這不是已證實的程式邏輯錯誤。RTX3080／WSL residency與速度波動另列Known Limitation。必要檔案與環境資訊已備齊，但依本次functional PASS門檻，Ready to transfer=NO；Target-machine smoke尚未執行。未開始正式訓練。詳見[搬機清單](LAB_DEPLOYMENT.md)與DEVLOG。
- 任務建立日期：2026-09-28；結束日期待確認。

## 如何開始

1. 閱讀 [工作規則](instructions.md)、[SPEC](SPEC.md) 與 [PLAN](plans/2026-09-28_sam3_orchid_flower_stem_leaf_finetune.md)。
2. 依 [資料說明](data/README.md) 準備影像與 COCO JSON。
3. 依 [訓練機執行說明](scripts/README.md#sam-3-訓練與評估搬至訓練機後執行) 建立 Linux／WSL2 專用環境，先做小批次測試與 checkpoint 恢復，再正式訓練。
4. 搬機依 [LAB_DEPLOYMENT.md](LAB_DEPLOYMENT.md) 複製實際working tree、完整資料與真實pretrained；在實驗室電腦另跑target-machine smoke，不沿用開發機的PASS作目標機驗收。

入口為 `scripts/2026-09-28_train_sam3_orchid.py` 與 `scripts/2026-09-28_evaluate_sam3_orchid.py`，共用邏輯在 `src/orchid_sam3/`。訓練使用官方 SAM 3 Trainer、instance mask loss、batch size 4、bfloat16、1008 px、10 epochs；每個 epoch 用 valid 的 macro IoU 選模型，test 不參與選模。可用 `--prepare-only` 在沒有 GPU／權重時檢查設定。

目前 301 張影像與 5,591 筆修正標註通過資料品質檢查。跨集合未發現完全重複影像；使用者人工覆核結論為無重複植株，但沒有拍攝序列來源與逐對紀錄。程式保留這項限制，不將自動檢查的 PENDING 改成正式無洩漏驗收。

- area 修復已完成：pycocotools 2.0.11 已複核並修正全部 5,591 筆，後續使用 `data/processed/annotations_area_fixed/{train,valid,test}/`。原始匯出保留不變；其他資料品質與環境驗證仍待完成。

## 環境狀態

| 項目 | 狀態 |
|---|---|
| 目前本機 Python | 3.13.12；尚未驗證為訓練環境 |
| 訓練作業系統／專用環境 | Ubuntu 26.04／WSL2、Python 3.12.14；`.venv-train`，與 Windows `.venv` 分開 |
| 預定 GPU | RTX 5090，32 GB；執行前驗證硬體與可用顯存 |
| GPU 驅動／CUDA 安裝方式 | 本機 RTX 3080 10 GB，驅動 616.92；使用 PyTorch cu128 wheel，CUDA 實測可用 |
| SAM 3 來源、commit、基礎權重位置與版本 | facebookresearch/sam3，固定 `2345a4ad109ac29c569da749c91d84f10dc08c40`；官方 `D:\my-project\sam3.pt`（WSL `/mnt/d/my-project/sam3.pt`），有效 image 分支 1134 項 strict load 通過 |
| PyTorch／torchvision 與其他依賴 | torch 2.10.0+cu128、torchvision 0.25.0+cu128、NumPy 1.26.4、Hydra 1.3.7；完整實測版本見 run 內 `requirements.actual.txt` |

[requirements.txt](requirements.txt) 已記錄本次 WSL2 實測環境的 pip freeze 版本；先依 scripts/README 準備固定 commit 的相鄰 `sam3-reference`，再於專案根目錄的 Linux 環境安裝。這是實測快照，不是跨平台或含套件雜湊的 lock；RTX 5090 正式 batch 4 仍須在目標機驗證。

[requirements-training.txt](requirements-training.txt) 提供附加依賴範圍；精確實測版本見 requirements.txt。每個 run 保存 `environment.json`、`requirements.actual.txt`、設定與資料／基礎權重雜湊。Windows `.venv` 不作 GPU 訓練；本機已建立 WSL Python 3.12 `.venv-train`。

## 資料與成果

- `data/raw/`：匯入影像，唯讀，不進 Git。
- `data/processed/annotations/`：COCO 匯出資料，train／valid／test 為 211／60／30 張，不進 Git。
- `data/processed/support_eval/test/`：待人工準備的支架評估遮罩與 manifest。
- `scripts/`、`src/`：已建立的執行入口與共用函式。
- `results/figures/`、`results/tables/`：預定可重現圖表。
- `results/models/`：預定 checkpoint 與設定副本；權重不進 Git，設定檔保留版本控制。

驗收依 SPEC 的每類 IoU 與支架像素誤判比例，目前尚未驗收。

## Agent 與變更紀錄

Claude Code、Copilot、Cursor 的指向檔均指向 [instructions.md](instructions.md)。其他工具使用前也須讀取該檔及 SPEC。

進度與決策記錄在 [DEVLOG.md](DEVLOG.md)，僅追加，不重寫歷史。



## area 修正資料的保存限制

依使用者決定，目前不保留修復腳本與專用依賴檔。修正資料位於 `data/processed/annotations_area_fixed/`，其中的 `area_changes.csv` 與 `repair_manifest.json` 記錄逐筆差異、來源雜湊及當時套件版本；這些紀錄不能代替可執行的修復程式。

資料目錄不進 Git，僅 clone 本專案無法取得或重建修正副本。固定備份位置尚待確認；須另行備份完整修正資料及紀錄，遺失時需重新建立修復流程。
