# SAM3 蘭花專案：移到實驗室電腦的部署清單

更新：2026-10-01。這份清單只包含部署與 target-machine smoke，不包含正式長時間訓練啟動。

## 本次狀態

- Development-machine Functional GPU Smoke Test：**FAIL（本次未完成驗收）**。`smoke_transfer_20261001`第一個forward超過900秒，未完成iteration；第二階段未執行。這不等於已證實training程式錯誤，也不撤銷過往A1／clean1的成功證據。
- Ready to transfer to lab machine：**NO**。必要檔案與環境資訊已盤點，但尚未符合使用者要求的本次兩階段functional PASS條件。
- Target-machine Smoke Test：尚未執行。
- 已知限制：RTX3080 10GB／WSL曾有memory residency與時間波動；`make_resident -12`也曾出現在成功案例。不要據此宣稱OOM，也不要用它取代functional結果。

本次報告：[functional](results/tables/2026-10-01_smoke_transfer_20261001_functional.json)、[完整驗收表](results/tables/2026-10-01_smoke_transfer_20261001_summary.csv)、[搬移manifest](results/tables/2026-10-01_smoke_transfer_20261001_transfer_manifest.csv)。部署指令已準備好，不代表已在實驗室執行或授權立即開始正式訓練。

## 必須搬移／取得的檔案

請搬移**目前工作目錄的實際檔案**，不能只clone尚未包含本次修改的Git版本。

| 項目 | 必要內容與用途 |
|---|---|
| Source | `src/orchid_sam3/` 全部Python檔；保留目前runtime相容性修正 |
| Scripts | `scripts/` 全部Python入口及README；必要入口為環境檢查、資料檢查、train、evaluate與`2026-10-01_functional_gpu_smoke.py` |
| Config | `src/orchid_sam3/config.py`、官方SAM3固定commit的`sam3/train/configs/`、BPE asset；本專案設定由make_config生成，並非搬舊run的絕對路徑YAML直接使用 |
| Environment | `requirements.txt`（實測freeze）、`requirements-training.txt`（依賴說明）；可另保留本次run的`environment.json`、`requirements.actual.txt`作比對 |
| 規格與紀錄 | `SPEC.md`、`plans/2026-09-28_sam3_orchid_flower_stem_leaf_finetune.md`、`instructions.md`、README、DEVLOG、本清單與tests |
| 官方SAM3 | `../sam3-reference/`含`.git`的乾淨checkout，或在目標機clone下方固定commit。不能僅複製無Git資訊的SAM3程式，check_source會核對commit及tracked修改 |
| Pretrained | 真正的`sam3.pt`；目前來源`D:\my-project\sam3.pt`，3,450,062,241 bytes，SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`；遵守既有模型存取授權 |
| Dataset | 完整`data/processed/annotations_area_fixed/`，含train/valid/test影像、COCO JSON、area_changes.csv及repair_manifest.json。資料不在Git，必須另行複製；不要修改原始資料 |
| Class mapping | `src/orchid_sam3/data.py`：花=flower/1、葉=leaf/2、莖=stem/3；原JSON多出的0=sam3只在衍生training JSON移除 |

正式test的支架誤判評估另外需要`data/processed/support_eval/test/`的真實遮罩與manifest；若尚未備妥，標記該正式評估未準備，不捏造資料。它不是本次train/validation functional smoke的前置條件。資料來源序列／植株獨立性的既有限制仍保留在PLAN與data_limitations中。

`data/raw/`應另行安全備份，但不是目前training CLI的直接輸入。搬機不能代替資料備份。

## 不必搬移的 smoke 暫存產物

- Windows `.venv/`、Linux `.venv-train/`、`__pycache__/`、pip／uv cache：在目標機重建環境。
- `results/models/smoke*/`的大型smoke checkpoints、TensorBoard與中間JSON：不要當正式訓練起點。
- `diagnostic_*.pt`、timeout stack與大量console logs：不影響部署；保留於開發機作evidence即可。
- 舊run的`config_resolved.yaml`／`config_target_*.yaml`含開發機絕對路徑，不作目標機輸入。

建議隨專案攜帶本次functional報告、搬移manifest與小型驗證表，便於核對；**不需要刪除任何開發機smoke evidence**。

## 實驗室目標與版本

| 項目 | 開發機已實測／目標機要求 |
|---|---|
| 系統 | Linux或WSL2；開發機Ubuntu26.04，目標機仍須實測 |
| GPU | 目標為RTX5090，作為可見CUDA device0；開發機RTX3080僅smoke |
| Python | 開發機3.12.14；使用Python3.12環境，優先相同patch版本 |
| PyTorch／torchvision | 2.10.0+cu128／0.25.0+cu128 |
| CUDA wheel runtime | 12.8；需相容的NVIDIA主機驅動。nvidia-smi顯示的CUDA上限不等於torch.version.cuda |
| SAM3 | 官方commit `2345a4ad109ac29c569da749c91d84f10dc08c40`，editable安裝；不可更新main替代 |
| NumPy／Hydra | 1.26.4／hydra-core1.3.7；其餘依requirements.txt |
| Precision／DDP | bf16、原始單GPU NCCL/DDP；正式設定batch4、1008×1008，不由smoke結果自動改動 |

## 路徑與環境設定

建議保持相鄰目錄`orchid-sam3/`、`sam3-reference/`、`sam3.pt`。修改目標機實際路徑，不沿用`D:\...`或`/mnt/d/...`。

```bash
cd /absolute/lab/path/orchid-sam3
# 已搬入含.git的固定checkout時，略過clone；不得覆寫既有checkout。
git clone https://github.com/facebookresearch/sam3.git ../sam3-reference
git -C ../sam3-reference checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
git -C ../sam3-reference status --short

python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
# requirements.txt含cu128 index及-e ../sam3-reference；必須從專案根目錄安裝。
python -m pip install -r requirements.txt
python -m pip check

export SAM3_REPO="$(realpath ../sam3-reference)"
export SAM3_BASE="$(realpath ../sam3.pt)"
export CUDA_VISIBLE_DEVICES=0  # 改成實際RTX5090的實體GPU索引
sha256sum "$SAM3_BASE"
python -c 'import torch,sam3; print(torch.__version__,torch.version.cuda); print(torch.cuda.is_available(),torch.cuda.get_device_name(0)); print(sam3.__file__)'
```

若SAM3放在不同目錄，先調整`requirements.txt`中的editable本地路徑及`SAM3_REPO`，不能讓Python仍import另一份SAM3。資料路徑由專案根目錄相對組合；完整資料應落在既有`data/processed/annotations_area_fixed`。`results/`需可寫且有足夠磁碟空間（單個完整checkpoint約10GB，測試會保留多份）。

WSL使用Windows NVIDIA主機驅動，不在WSL另裝Linux顯示驅動；依[NVIDIA官方WSL指引](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)設定。原生Linux則由管理者確認驅動相容。torch2.10.0／torchvision0.25.0／cu128配對亦列於[PyTorch官方版本安裝表](https://pytorch.org/get-started/previous-versions/#v2100)。上述版本是已觀測組合，不表示RTX5090已實測通過。

## Target-machine smoke 指令（不是正式訓練）

先確認目標機沒有不能中斷的工作，再使用乾淨process/context。**不要直接shutdown共用實驗室WSL或GPU工作。**

```bash
source .venv-train/bin/activate
python scripts/2026-09-28_check_sam3_environment.py --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE"
# 此盤點工具的training dry run本來就是PENDING；看各項報告，不把它當GPU訓練PASS。

export LAB_SMOKE_RUN="lab_smoke_$(date +%Y%m%d_%H%M%S)"
# batch4驗證目標機原訂batch；4張真實train影像=每階段1 iteration。
python scripts/2026-10-01_functional_gpu_smoke.py --phase initial --target-machine --batch-size 4 --run-name "$LAB_SMOKE_RUN" --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --timeout 900
# 僅在initial exit0且result.json=PASS後執行；這是新的training process。
python scripts/2026-10-01_functional_gpu_smoke.py --phase resume --target-machine --batch-size 4 --run-name "$LAB_SMOKE_RUN" --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE" --timeout 900
# CPU重新載入checkpoint並稽核實際落盤，不啟動訓練。
python scripts/2026-10-01_functional_gpu_smoke.py --phase audit --target-machine --batch-size 4 --run-name "$LAB_SMOKE_RUN" --sam3-repo "$SAM3_REPO" --checkpoint "$SAM3_BASE"
```

所有目標機run都在`results/models/$LAB_SMOKE_RUN/`，不覆蓋正式run；表格在`results/tables/`。日期前綴沿任務腳本2026-10-01，run name內另含實際執行時間。此流程僅執行1+1個batch4 iterations，並非正式訓練。

## Target-machine PASS 驗收條件

1. 實際PyTorch辨識RTX5090、CUDA/bf16可用、SAM3來源commit與版本正確，pip check通過。
2. 原始pretrained與真實train/valid可读；mapping、image/annotation pairing正確；未使用mock或smoke權重作base。
3. 兩個training worker正常exit0；各有真正forward、finite loss、backward、optimizer step與實際權重更新。
4. 第一階段最小validation及checkpoint存在；第二階段恢復model/optimizer/scaler/RNG/training state；batch4時step1→2，且有新的checkpoint。
5. audit成功CPU重新載入checkpoint，optimizer step／training step／RNG／scaler一致；validation CSV、train_stats、logs、TensorBoard、config與檔案清單實際存在規劃位置。
6. Target-machine報告PASS，沒有將timeout或未執行項目冒充PASS；若有環境限制，獨立記錄，不直接宣稱程式bug或OOM。

只有target smoke通過後才另行規劃正式run；本清單不授權立即開始正式訓練，也不以smoke指標評價最終花／莖／葉準確率。
