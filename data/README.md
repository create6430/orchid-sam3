# 資料存放說明

資料不進 Git。以下來源資訊轉錄自本機 `processed/annotations/README.dataset.txt` 與 `README.roboflow.txt`，未重新驗證線上下載狀態。

- 來源：[Roboflow sam3 資料集](https://universe.roboflow.com/rens-workspace-8qzys/sam3-yc3wz)。
- 匯出版本：v1，標題日期 2026-09-18 12:56am；匯出時間 2026-09-17 16:57 GMT。
- 附帶授權標示：CC BY 4.0。
- 前處理：自動校正方向、移除 EXIF 方向資訊、縮放補黑邊至 512 × 512；匯出說明記載無資料增強。
- 固定匯出包的 NAS／雲端備份位置與 checksum：待確認，來源頁面不能取代固定版本備份。

## 本機目錄與盤點

| 位置 | 用途 | 影像數 | annotation 數 |
|---|---|---|---|
| `data/raw/images/` | 匯入影像，唯讀；是否為相機未處理原檔待確認 | 301 | 不適用 |
| `data/processed/annotations/train/` | 訓練影像與 `_annotations.coco.json` | 211 | 3789 |
| `data/processed/annotations/valid/` | 驗證影像與 `_annotations.coco.json` | 60 | 1147 |
| `data/processed/annotations/test/` | 測試影像與 `_annotations.coco.json` | 30 | 655 |

2026-09-29 本機檔案大小：raw 7,084,666 bytes，processed 8,841,127 bytes（含現有說明檔與 JSON）；後續新增產物會改變大小。

## 還原與使用

目前只保留修正資料與 `area_changes.csv`、`repair_manifest.json`，不保留修復腳本及專用依賴檔，無法從原始匯出包重跑 area 修復。以下步驟僅還原原始匯出；修正副本須另外取得並備份。資料不進 Git，固定修正副本備份位置尚待確認。

- area 修復已完成：pycocotools 2.0.11 已複核並修正全部 5,591 筆，後續使用 `data/processed/annotations_area_fixed/{train,valid,test}/`。原始匯出保留不變；其他資料品質與環境驗證仍待完成。

1. 取得上述 v1 的 COCO 匯出包；固定備份尚待確認，不以其他版本默默替代。
2. 保留匯出檔名及 train／valid／test 目錄，放入 `data/processed/annotations/`，保留附帶 README。
3. 原始匯入副本放入 `data/raw/images/`；已有資料不可覆寫，raw 與匯出影像的來源對應須再確認。
4. JSON 的 `images[].file_name` 相對其所在目錄解析；依 SPEC 驗證類別與尺寸。
5. 植株／拍攝序列對照表待資料提供者提供；現有數量不代表已通過資料洩漏檢查。

支架評估遮罩需人工準備於 `data/processed/support_eval/test/`，格式與覆核方式依 PLAN。缺遮罩時不得宣告支架指標驗收通過。

完整定義見 [SPEC](../SPEC.md)；執行順序見 [PLAN](../plans/2026-09-28_sam3_orchid_flower_stem_leaf_finetune.md)。
