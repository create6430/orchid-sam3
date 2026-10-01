# DEVLOG

> 只往後加，不修改、不重整舊紀錄。
> 每個任務結束就補一筆。這份是寫給半年後的自己，以及接手的學弟妹看的。

格式：

```
## YYYY-MM-DD ｜ <標題>
- 做了什麼：
- 為什麼這樣做：
- 結果：
- 下一步：
```

---

## 2026-09-02 ｜ 建立 repo 結構

- 做了什麼：建立資料夾骨架，寫好 `instructions.md`、`SPEC.md`、`PLAN_template.md`
- 為什麼這樣做：導入 plan-first 的 agent 工作流程，先把規則與資料規格固定下來
- 結果：骨架完成，`SPEC.md` 中的感測器校正係數尚待補齊
- 下一步：補完校正係數，跑第一個實際任務驗證流程可行

---

## YYYY-MM-DD ｜ <標題>

- 做了什麼：
- 為什麼這樣做：
- 結果：
- 下一步：

## 2026-09-29 ｜ 修正 orchid-sam3 規格與規劃文件

- 做了什麼：對齊 COCO 類別、邊界框與像素評估規則；補齊支架遮罩準備、環境建置、輸出尺寸檢查，修復 SPEC 表格；更新專案與資料 README、環境待確認說明及工作規則。
- 為什麼這樣做：移除模板執行指令與錯誤資料路徑，讓 SPEC、PLAN、實際資料描述與驗收方式一致。
- 結果：文件記錄現有 301 張影像及 5,591 筆 annotation；既有 .gitignore 已含權重、環境機密與 logs 規則。requirements 尚未鎖定版本，不代表訓練環境已建立。先前 2026-09-02 紀錄為沿用模板內容，本專案不使用其中提及的力或加速度校正係數。
- 驗證與限制：本次檢查文件與資料清單；未執行訓練、遮罩品質或資料洩漏驗證，未產生模型結果，無既有模型評估需重跑。
- 下一步：確認訓練平台與 SAM 3 版本、取得植株／序列對照表及固定資料備份、人工準備支架遮罩；審核 PLAN 後執行資料與環境驗證。

## 2026-09-29 ｜ 釐清 PLAN 檢查與驗收範圍

- 做了什麼：將 image ID 唯一性檢查限定於各份 COCO JSON；將 N/A 阻擋整體驗收的規則限定於各類 IoU 與支架誤判比例，補充指標仍須報告數值或 N/A 原因。
- 為什麼這樣做：避免跨 JSON 的相同 ID 被誤認為重複影像，並使 PLAN 與 SPEC 的補充指標定位一致。
- 結果：文件規則已對齊；本次不改資料、評估公式或門檻，不產生模型驗收結果。
- 下一步：完成既有待確認事項並審核 PLAN，再進行資料與環境驗證。

## 2026-09-29 ｜ 文件與資料雙重檢查

- 做了什麼：交叉檢查文件，讀取全部 COCO 標註，解碼 301 張影像並比對尺寸及像素雜湊；依 COCO 官方 maskApi.c 的壓縮 RLE 邏輯檢查游程與遮罩面積。
- 結果：影像均可解碼且尺寸符合 JSON，未發現像素完全相同的重複影像；5,591 筆 RLE 游程皆非負、總長度符合影像大小且遮罩非空。但所有 area 均等於 bbox 面積，與遮罩面積不符（train id=1：10,908 對 3,885 px²）。
- 修正：在 SPEC、PLAN、專案與資料 README 記錄此阻擋；未修改任何影像或 annotation。
- 限制：最初僅檢查 polygon 的分支不適用此 RLE 資料，已改依實際格式核對；尚未以已安裝的官方 COCO 套件交叉驗證，未人工檢查標註語意、近似影像或群組洩漏，未驗證訓練環境。
- 參考：https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
- 下一步：使用正式 COCO 工具複核 area，再確認產生獨立修正副本的方案；其他訓練前待確認事項仍依 PLAN。

## 2026-09-29 ｜ 使用 COCO 工具修復 area

- 授權：使用者要求修復已確認的 area 問題。
- 做了什麼：在隔離 .venv 安裝 pycocotools 2.0.11／numpy 2.5.3，新增可重跑腳本與 requirements-data.txt，產生 annotations_area_fixed 獨立副本，更新 SPEC、PLAN 與資料使用說明。
- 結果：train 3,789、valid 1,147、test 655 筆，共 5,591 筆 area 已修正；每筆 COCO area 均等於 decode 遮罩的像素加總。
- 驗證：重新讀取輸出 JSON，除 area 外與原檔完全一致；所有來源檔案 SHA-256 未變，複製影像等其他檔案雜湊相同。輸出內保存逐筆差異 CSV 與 repair_manifest.json。
- 限制與下一步：只完成 area 修復，未完成標註語意、群組洩漏、支架遮罩或 SAM 3 訓練環境驗證；依 PLAN 完成剩餘檢查。

## 2026-09-29 ｜ 保留修正資料，停止提供修復重跑指令

- 決定：使用者選擇維持沒有專案程式碼，不恢復修復腳本與專用依賴檔。
- 做了什麼：更新 README、PLAN 與資料說明，移除失效指令，說明目前僅保留修正資料與差異／雜湊紀錄，無法重跑修復。
- 結果：歷史修復紀錄保留；本次未新增程式碼，也未修改或刪除任何資料與虛擬環境。
- 限制與下一步：修正資料不進 Git，須另外備份；固定备份位置待確認。若需重建副本，須重新建立修復流程，不得將現況宣告為完整可重現。

## 2026-09-29 ｜ 釐清原始匯出與修正副本的用途

- 做了什麼：SPEC 分別標明原始匯出與後續使用的修正副本路徑，限定 area 錯誤描述僅指原始匯出；scripts/README 補充 processed 輸入與獨立輸出規則。
- 為什麼這樣做：避免讀錯資料版本，或將已修正副本誤認為仍有原始 area 錯誤。
- 結果：文件與 PLAN 的輸入路徑一致；本次沒有新增程式碼或修改資料，修復流程仍無法重跑。
- 下一步：另行備份修正資料，依 PLAN 完成其餘資料品質與環境驗證。

## 2026-09-29 ｜ 實作 PLAN 第 1 步 COCO 盤點

- 授權：使用者要求按照 PLAN 在 scripts 撰寫程式；依逐步執行規則先完成第 1 步。
- 做了什麼：新增 scripts/2026-09-28_validate_orchid_coco.py，使用 Python 標準函式庫唯讀檢查修正副本的類別映射、各類標註數及影像引用；輸出 results/tables/2026-09-28_orchid_coco_inventory.csv，包含來源 JSON SHA-256 與錯誤明細。失敗回傳非零退出碼，報告限定寫入 results/tables。
- 重跑：在專案根目錄執行 `.venv/Scripts/python.exe scripts/2026-09-28_validate_orchid_coco.py`；亦可使用一般 Python 3.9 以上版本，不需額外套件。
- 結果：實際執行退出碼 0。train／valid／test 為 211／60／30 張、3789／1147／655 筆標註；flower 分別 1715／553／359，leaf 為 910／231／132，stem 為 1164／363／164。三個集合類別映射符合 SPEC，所有 annotation 均對應存在的影像檔案；0=sam3 未被引用。
- 合理性檢查：單位為影像與標註筆數，未轉換座標或面積；數量級與既有資料紀錄一致，分類加總為 5591；邊界條件符合 SPEC 類別與 JSON 相對路徑。能量／質量守恆及物理極端條件不適用；無解析解對照，既有數量紀錄僅供交叉核對，不代表標註品質正確。
- 限制：未修改資料；尚未驗證遮罩、影像解碼、近似影像、群組洩漏或樣本充分性，未建立 SAM 3 訓練環境及訓練／評估入口。README 既有使用者修改保留不動。既有 area 修復仍無可重跑程式。
- 下一步：依專案逐步確認規則，確認後進行 PLAN 第 2 步，擴充遮罩格式、尺寸、空遮罩與損毀影像的驗證。

## 2026-09-29 ｜ 完成 PLAN 第 2 步資料品質驗證

- 授權：使用者回覆「繼續」，接續第 2 步。
- 做了什麼：擴充 scripts/2026-09-28_validate_orchid_coco.py，新增 `--quality`；維持原本不需額外套件的第 1 步模式。逐張以 Pillow verify 與重新開啟後 load 檢查檔案結構、完整解碼及尺寸；同份 JSON 的 image ID 不得重複，不同集合各自處理。逐筆驗證 COCO polygon／RLE 格式、RLE 尺寸與游程長度、非空二元遮罩、修正 area 與解碼面積一致性；原始資料維持唯讀。
- 輸出：results/tables/2026-09-28_orchid_coco_quality.csv，包含每張影像、每筆遮罩的結果、面積（px²）、來源 JSON SHA-256、環境版本及各類有效遮罩統計；錯誤逐項列出且退出碼非零。
- 環境：Windows、Python 3.13.12；使用既有 numpy 2.5.3／pycocotools 2.0.11，於專案 .venv 新增 Pillow 12.3.0。這是資料驗證環境，不代表 SAM 3 或 GPU 訓練環境已驗證。資料驗證依賴安裝指令已放在腳本說明，requirements.txt 仍依第 4 步保留待訓練環境驗證狀態。
- 重跑：在專案根目錄執行 `.venv/Scripts/python.exe scripts/2026-09-28_validate_orchid_coco.py --quality`。
- 結果：退出碼 0；301 張影像全部可解碼、尺寸均符合 JSON，無同份 JSON 內重複 image ID；5591 筆遮罩全部為壓縮 RLE、非空，area 全部等於解碼像素加總。train／valid／test 有效遮罩分別為 3789／1147／655，各集合三類皆有有效遮罩。
- 驗證：CSV 的 301 筆影像、5591 筆遮罩及 3 筆集合通過紀錄一致；以實際原始匯出作反向檢查，正確抓出全部 5591 筆既有 area 錯誤及其衍生的 9 筆集合／類別無通過遮罩錯誤。每集合取一筆真實 RLE 轉成未壓縮游程，確認兩種格式解碼面積一致；缺少輸入會回報 FAIL，寫入 data/raw 的報告路徑會被拒絕（退出碼 2）。沒有產生假影像或假標註。
- 合理性檢查：尺寸以 px、面積以 px² 表示，未縮放、補邊或轉換座標；資料數量與第 1 步相符，遮罩面積大於 0 且不超過影像像素數。RLE 總長度與影像大小一致，前景游程加總、COCO area、解碼像素數互相吻合。物理能量／質量守恆及力學極端條件不適用；無解析解，使用正式 COCO 解碼及既有錯誤副本交叉核對。
- 限制：現有資料沒有 polygon，該分支尚無真實 polygon 資料驗證；標註語意仍需人工確認。有效遮罩非零不代表樣本數統計充分。近似影像、植株／序列洩漏、支架遮罩、訓練環境與模型驗收均不在本步通過範圍。
- 參考：COCO RLE 格式依 https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c；影像檢查依 https://pillow.readthedocs.io/en/stable/reference/Image.html。
- 下一步：依逐步確認規則，在使用者確認後執行第 3 步，輸出既有切分清單並檢查跨集合重複／近似影像；仍需資料提供者提供植株與拍攝序列對照表，缺少時不得宣告資料洩漏檢查通過。

## 2026-09-29 ｜ 實作 PLAN 第 3 步切分與洩漏檢查，等待來源與人工覆核

- 授權：使用者再次回覆「繼續」，接續第 3 步；已先說明本步產物超過三個檔案。
- 做了什麼：新增 scripts/2026-09-28_check_orchid_splits.py，保存修正副本既有 train／valid／test 清單至 data/processed/splits/2026-09-28_orchid_split_manifest.csv；記錄 JSON SHA-256、檔案 SHA-256、含尺寸的 RGB 像素 SHA-256、影像 ID 與路徑。新增 checks／pairs CSV 與 HTML 並排覆核頁，並於 scripts/README.md 記錄執行、門檻及人工輸入格式。未改動影像、標註、切分或使用者既有 README 修改。
- 比對方式：窮舉跨集合影像配對；檔案或 RGB 像素雜湊相同視為完全重複。近似候選使用 dHash 64 位元與 DCT pHash 63 位元（排除 DC），漢明距離 ≤8 或 ≤10 列入候選；使用包含補邊的完整匯出影像與 LANCZOS 縮圖，無隨機抽樣。門檻僅為輔助篩選，可由參數調整並記錄，不代表通過來源獨立性驗收。
- 結果：原始清單保持 211／60／30 張；比對 20790 組跨集合配對，完全重複為 0，近似候選為 102 組（train-valid 68、train-test 29、valid-test 5）。近似不是已證實洩漏，全部標記待人工覆核；植株／拍攝序列 ID 留空，未依檔名或外觀推定。
- 狀態：第 3 步尚未驗收，overall=PENDING、退出碼 1。未找到來源對照表，已向使用者詢問路徑；未開始第 4 步環境建置或訓練。
- 驗證：清單的 301 個 split/image_id 唯一鍵與全部來源影像／JSON 雜湊一致；102 組 pair_id 不重複且均跨集合；HTML 的 204 個影像連結均可解析至既有檔案。使用真實但來源 ID 空白的 manifest 測試來源檢查，確認回傳待確認；未提供人工覆核時 102 組均保持待確認。重跑四份產物的 SHA-256 完全一致；以產出作為覆核輸入會被拒絕（退出碼 2），且產物不變。未製造假影像、假遮罩或假群組資料。
- 合理性檢查：尺寸單位 px，雜湊距離單位 bit；配對數 211×60＋211×30＋60×30＝20790，與窮舉結果一致。每張影像保持原集合與 ID，未重新取樣或轉換 COCO 座標；只在記憶體縮圖計算感知雜湊。能量／質量守恆與力學極端情況不適用；沒有解析解或已標註的近似影像標準答案，不能估計篩選召回率。邊界檢查確認缺少來源／覆核不能通過。
- 限制與下一步：需資料提供者提供涵蓋全部影像、ID 跨集合一致的植株／序列紀錄，以及人工覆核 102 組候選。感知雜湊可能漏掉裁切、旋轉或不同視角的同源影像；無完全重複不等於沒有洩漏。取得資料後以 --groups／--reviews 重跑第 3 步；如確認同源跨集合，依 PLAN 先確認群組重切方案，不自行更動切分。

## 2026-09-29 ｜ 記錄使用者人工覆核結論：無重複植株

- 使用者回覆：先確認「無來源」，隨後表示「以人工覆核，無重複植株」。
- 做了什麼：在 PLAN 第 3 步補充中記錄採用此人工覆核結論作為植株重複檢查依據；覆核依據為本次對話，未推定覆核者姓名或製造來源 ID。
- 結果：人工覆核結論為無重複植株。先前自動檢查的 0 組完全重複及 102 組近似候選結果維持原樣；未將整體人工陳述改寫成逐對來源判定，也未手動修改自動 CSV 的 PENDING。
- 驗證與限制：本次只更新文字紀錄，不改動資料、切分、演算法或數值結果，因此物理量、守恆與極端條件檢查不適用。沒有來源對照表或逐對覆核紀錄，拍攝序列來源仍未驗證；後續成果須保留這項限制。

## 2026-09-29 ? ? 4 ????????????????

- ????????????????????????????????????????? 4 ????????
- ??????? scripts/2026-09-28_check_sam3_environment.py ? results/tables/2026-09-28_sam3_environment.csv??? Python?OS?GPU???????SAM 3 ??? commit ???????????????????? GPU ????????
- ??????????? `python scripts/2026-09-28_check_sam3_environment.py --sam3-repo <SAM3?????> --checkpoint <????>` ???
- ?????Windows 11?Python 3.13.12?GPU ? RTX 3080?10240 MiB??? 616.92?? SPEC ? RTX 5090 ????? .venv ??? torch?torchvision?sam3?data?results?src ??? pt?pth?ckpt?safetensors ?????? SAM 3 ????????? 1?overall=FAIL?? 4 ?????
- ??????? https://github.com/facebookresearch/sam3 ???????????????? Hugging Face ?????????????????? SAM 3 commit??????????????????????
- ??????????????????????????? CUDA ????????? toolkit??????? MiB??? batch size????? SPEC????????????????????????????????????bf16???? checkpoint ????????
- ?????????? 5090 ???????? SAM 3 ????????????????????????????????????????????????????????

## 2026-09-29 ｜ 完成可搬移的 SAM 3 訓練／評估程式，GPU 實測待執行

- 授權：使用者明確要求「先完成程式，之後才會搬到訓練機執行」；因此完成第 4–7 步的程式準備與操作文件，不啟動本機模型訓練或下載權重。
- 前筆編碼補正：前一筆環境檢查紀錄經 PowerShell 管線寫入時中文變成問號，保留歷史並在此補正：本機為 Windows 11、Python 3.13.12、RTX 3080 10240 MiB、驅動 616.92；.venv 未安裝 PyTorch／SAM 3，未取得模型權重，第 4 步 GPU 驗證未完成。
- 實作：新增 train／evaluate 腳本及 src/orchid_sam3 共用模組；使用官方 Trainer、完整模型微調、instance mask loss、batch size 4、gradient accumulation 1、bfloat16、1008 px、10 epochs。每 epoch 用完整 valid 的像素 macro IoU 選 best，保留每 epoch／最後 checkpoint、optimizer／scaler／RNG 狀態與 loss 紀錄；test 僅供最終評估。
- 來源：唯讀核對官方 SAM 3 commit 2345a4ad109ac29c569da749c91d84f10dc08c40（工作區 D:/my-project/sam3-reference，供核對使用，不是訓練安裝）。程式驗證固定 checkout，適配官方 detector 前綴與微調 checkpoint，strict 載入避免漏載；未修改第三方原始碼。上游完整環境變數日誌停用，改保存明確選定的硬體／套件資訊。
- 資料：訓練衍生 JSON 僅移除未使用的 0=sam3 類別列，image／annotation 完全保留；原資料唯讀。固定 square resize、不隨機增強、保留三類負查詢。輸出保存資料、基礎權重與共用程式 SHA-256，恢復時檢查一致性。
- 評估：按 SPEC 在原 COCO 尺寸上彙總同類遮罩聯集的 TP／FP／FN，分數門檻含等號 0.35；輸出各類與 macro IoU／Dice／Precision／Recall，零分母記 N/A。支架 manifest 逐張驗證路徑、人工覆核、尺寸、單通道 PNG 與 0／255 值，缺漏或零分母不通過。預覽程式輸出 300 dpi、像素座標及支架／stem 交集；尚未實際推論或產生預覽。
- 文件：更新 README、scripts/README、PLAN，提供 Linux／WSL2 Python 3.12 候選安裝、設定準備、4 張真實 train 影像 smoke test、resume、正式訓練與評估指令。新增 requirements-training.txt；既有 requirements.txt 保留未經 GPU 驗證的狀態。固定版 SAM 3 要求 numpy<2，勿搬移本機 NumPy 2.5.3 環境。
- 驗證：prepare-only 成功產生本機設定；7 項 CPU 測試全數通過（無 skip），包含真實 test 遮罩讀取／面積、公式與 N/A、支架缺漏、原始錯誤 area 拒絕、輸出路徑、官方目標函式存在與 train/valid 設定、衍生 JSON 一致性。11 個 Python 檔案 AST 語法通過，git diff --check 通過；結果表位於 results/tables/2026-09-28_sam3_code_validation.csv。本機 pycocotools 搭 NumPy 2.5.3 有既有 array-copy deprecation warnings，沒有將其忽略為訓練環境相容性證明。
- 合理性：尺寸 px、面積 px²、一般指標 0–1、支架比例百分比；資料計數與先前一致。零分母與缺失支架狀態有測試。守恆與力學極端條件不適用；沒有模型解析解，真實遮罩自我比對僅驗證程式一致性，不是模型表現。
- 限制：尚未在 RTX 5090 執行 GPU 前向／反向、bf16、顯存、checkpoint 恢復或最終推論；不宣告第 4–6 步實測通過。來源限制保留：使用者人工覆核无重複植株，但拍攝序列來源與逐對覆核表缺失。SPEC.md 的使用者現有編輯保留未改。
- 下一步：搬移完整修正資料，在訓練機依 scripts/README 執行 smoke 與 resume，成功後再跑正式訓練與 test 評估；人工提供支架遮罩並檢視預覽。目標機實測後才鎖定完整依賴與記錄模型指標。

## 2026-09-29 ｜ 搬移前修正 smoke validation 缺類問題

- 做了什麼：檢查訓練／恢復銜接與評估邏輯，發現原 smoke 模式只取 valid 第一張，而真實 image_id=0 僅有 flower／leaf、沒有 stem；若 stem 預測亦為空，macro IoU 會為 N/A，造成小批次測試非必要中止。
- 修正：smoke validation 改為按 image ID 順序選取涵蓋三類標註的最小前綴，實際為 image_id 0、1。執行時保存 smoke_validation_images.json；正式 valid 與 test 仍使用完整集合，不修改任何資料或 SPEC 指標。同步更新 CLI 與操作說明。
- 驗證：新增真實 valid 資料的類別覆蓋、最小前綴與重現性測試；共 8 項 CPU 測試通過、無 skip。更新 results/tables/2026-09-28_sam3_code_validation.csv。既有 NumPy／pycocotools deprecation warnings 仍存在。
- 合理性：只調整程式 smoke 測試取樣；三類皆有非空真實標註時，各類 IoU 聯集分母必大於零。不變更數值公式、面積或座標；物理守恆／力學極端條件不適用。沒有產生假資料或模型結果。
- 限制與下一步：GPU 前向／反向、checkpoint 儲存恢復與實際推論仍待訓練機驗證。共用程式雜湊已變更，舊設定準備結果應重跑；若已存在用舊程式建立的 checkpoint，不繞過恢復一致性保護。

## 2026-09-29 — Smoke test（本機 CPU）

- 對應 PLAN 第 4 步環境盤點與第 5 節 CPU 回歸檢查；未變更程式或訓練參數。
- `.venv/Scripts/python.exe -m pytest tests/test_orchid_pipeline.py -q -p no:cacheprovider`：8 passed，0 skipped，656 次 pycocotools/NumPy DeprecationWarning。
- 五個 scripts 入口的 `--help` 均 exit 0。
- `.venv/Scripts/python.exe scripts/2026-09-28_train_sam3_orchid.py --sam3-repo ../sam3-reference --run-name smoke_check_20260929 --prepare-only --smoke-test --workers 0`：exit 0；301 張真實影像、5,591 筆遮罩品質檢查通過，設定位於 results/models/smoke_check_20260929/。
- `.venv/Scripts/python.exe scripts/2026-09-28_check_sam3_environment.py --sam3-repo ../sam3-reference`：前置条件 FAIL；本機 RTX 3080 10 GB 非 SPEC 的 RTX 5090，torch/torchvision/sam3 未安裝，未提供 checkpoint。官方來源 commit 符合且工作樹乾淨。
- `python -m pip check` 無法執行：此 .venv 未安裝 pip。
- 合理性：影像尺寸與遮罩 area（px²）通過既有品質檢查；零分母 N/A、公式及真實遮罩自我比對測試通過。自我比對不代表模型品質；能量/質量守恆與物理初始條件不適用。模型輸出範圍、實際 IoU、bf16、顯存、前後向與 checkpoint 儲存/恢復尚未驗證。
- 結論：CPU smoke 通過，GPU smoke 尚未執行。後續需目標訓練環境及基礎權重，依 scripts/README.md 執行 smoke 與 resume。

## 2026-09-29 — RTX 3080 GPU smoke 前置檢查：BLOCKED

本次使用者授權 RTX 3080、小 batch、真實 SAM3 訓練/validation/save/resume，沿用 PLAN 輸出結構。已閱讀 PLAN、scripts/README、config/data/runtime/training/evaluation、train/evaluate 入口與既有 tests。尚未開始 GPU smoke；FAIL 表示未達實測驗收，不代表各階段已執行後出錯。

- NVIDIA 驅動辨識 RTX 3080 10240 MiB，driver 616.92；不能據此宣告 PyTorch CUDA 可用。
- 專案 .venv Python 3.13.12 缺 torch、torchvision、sam3、hydra、pip。SAM3_BASE 未設定，工作區未找到權重；預設 Hugging Face models--facebook--sam3 快取僅 refs/main，無權重。沒有假定其他磁碟也不存在權重。
- WSL 查詢回傳 Wsl/EnumerateDistros/Service/E_ACCESSDENIED；無法據此判定 WSL 未安裝。
- 真實 CocoSplit 重新讀取並解碼 train 211 張/3789 annotations、valid 60 張/1147 annotations，配對、尺寸、遮罩 area 檢查成功。mapping flower=1、leaf=2、stem=3；smoke valid 最小前綴 2 張。官方來源 commit 確認符合固定版本。
- 現有 runtime 只允許 Linux/WSL2 + RTX5090，config 與輸入 signature 固定 batch=4；本次 RTX3080 小 batch 需求需要限定 smoke 模式的後續調整。因必要權重/環境缺失，依使用者要求停止，未修改程式或宣告修復完成。
- PLAN 根目錄 results/；訓練根 results/models/。scripts/README 的 smoke run 為 results/models/smoke/，其 checkpoints/、logs/、tensorboard/、validation/ 及 metadata 應維持同一 run。resume 預期 checkpoints/checkpoint.pt。本次未建立該 run，未載入/儲存/resume 任何 checkpoint。
- PLAN 第 6 步 results/tables/2026-09-28_sam3_orchid_metrics.csv、同前綴 _evaluation.json 與 results/figures/2026-09-28_sam3_orchid_examples.png 為正式 test 產物；現有程式禁止 smoke checkpoint 用於正式 test，不能以此偽造 smoke 成果。
- 本次只新增 results/tables/2026-09-29_sam3_gpu_smoke_preflight.csv 並追加本紀錄，未覆蓋正式訓練結果，未產生 PLAN 以外的主要執行產物。
- 合理性：讀取檢查涵蓋尺寸 px、area px² 與二值遮罩。模型 loss 數量級/有限性、預測範圍及 GPU 邊界尚不可檢查；物理守恆不適用；無模型已知解對照或 GPU 極端測試。
- 需提供真實 SAM3 pretrained checkpoint 完整路徑，以及可用 Linux/WSL2 Python 訓練環境（或先完成環境建置）。目前尚未完成正式微調前的基本技術驗證。

## 2026-09-30 — RTX 3080 smoke 前置環境準備（未執行訓練）

- 使用者授權範圍：建立可執行真實 GPU smoke 的環境及最小 smoke mode 調整；明確禁止本階段執行 GPU smoke 或正式訓練。本次沒有建立模型、載入權重、forward/backward 或 optimizer step。
- WSL 原因：沙箱內 wsl 查詢 E_ACCESSDENIED；同一查詢在沙箱外成功，既有 Ubuntu 為 WSL2。不是證明 WSL 損壞，不需重裝。已啟動 Ubuntu 26.04，kernel 6.18.33.2-microsoft-standard-WSL2。
- 使用既有 uv/Python 3.12 建立 /mnt/d/my-project/orchid-sam3/.venv-train/，Python 3.12.14。未在 Windows .venv 安裝訓練套件。Linux 安裝命令已寫入 scripts/README.md。
- 安裝 torch 2.10.0+cu128、torchvision 0.25.0+cu128、SAM3 editable（固定來源 /mnt/d/my-project/sam3-reference）、Hydra 1.3.7 及 requirements-training 依賴，NumPy 1.26.4。
- 實際 PyTorch CUDA 初始化成功，cuda.is_available=True，GPU=NVIDIA GeForce RTX 3080，capability=(8,6)，bf16=True，wheel 支援 sm_86。SAM3、官方 Trainer、專案 Trainer imports 與設定內 27 個 callable imports 通過。
- decord 0.6.0 的 WHEEL 宣告 cp36，導致 Python 3.12 pip check FAIL。確認上游只有影片載入分支使用 decord，本專案使用靜態影像；從 requirements-training.txt 移除並從 Linux venv 解除安裝，沒有修改上游原始碼。之後 pip check 通過。
- Windows checkout 的 CRLF 在 WSL 被誤判為內容修改；SAM3 checkout local core.autocrlf=true 後兩邊 tracked source 皆乾淨，固定 commit 不變。
- src/orchid_sam3/runtime.py：新增 smoke 參數，僅 smoke 允許 RTX3080/RTX5090；實際 require_runtime(smoke=True) 通過，formal mode 拒絕 RTX3080。保留 Linux/CUDA/bf16/source 驗證。
- src/orchid_sam3/config.py：新增 smoke_batch_size（1..4、僅 smoke）；正式及未指定時維持 4。4 張 train、valid 前綴、1008px、loss 與輸出結構不變。實際解析 batch=1，checkpoint dir 仍為 results/models/smoke/checkpoints。
- scripts/2026-09-28_train_sam3_orchid.py：新增 --smoke-batch-size，未帶 --smoke-test 時拒絕；實際 batch 寫入 resume signature/OOM metadata；傳遞 smoke runtime 模式。
- scripts/2026-09-28_check_sam3_environment.py：新增只影響環境盤點的 --smoke-test，允許3080；不執行訓練，training_dry_run 仍 PENDING。
- tests/test_orchid_pipeline.py：加入 smoke batch 1/2/4、非法參數、正式 default=4 與 checkpoint 路徑回歸測試。Windows 9 passed（656 個既有 pycocotools warnings），Linux 9 passed（6.16s）。設定 target import 27/27 成功；git diff --check 通過。
- PLAN 和 scripts/README.md 已記錄本次授權、環境建立方式與 smoke 專用參數。正式訓練設定、原始 dataset、使用者既有 README 修改均保留。
- 結果：results/tables/2026-09-30_sam3_prerequisites.csv；實際環境及依賴版本：results/tables/2026-09-30_sam3_training_environment.json。未將 requirements.txt 宣告為 GPU smoke 驗證過的 lock。
- 唯一未完成的必要前置項目：官方 SAM3 pretrained sam3.pt 尚未取得，無實際 checkpoint 完整路徑。官方 https://huggingface.co/facebook/sam3 要求本人登入並接受存取條件；已請使用者下載 sam3.pt 並提供完整路徑，不收集 token、不使用假權重。建議儲存 D:\my-project\sam3.pt，但此建議不是已存在的檔案。
- 合理性：本次沒有模型數值結果，物理守恆/模型品質不適用；sm_86/bf16 是環境能力，不代表10GB能完成全模型微調。取得权重後仍需另行實測 GPU smoke，現在不宣告整條 pipeline 已跑通。

## 2026-10-01 — 真實 GPU smoke 執行中（承接 2026-09-30）

- 使用者已提供 D:\my-project\sam3.pt 並授權 RTX3080 最小 smoke、validation、save/resume 與實際輸出驗證；不執行正式訓練。
- WSL 實際讀取 /mnt/d/my-project/sam3.pt，3,450,062,241 bytes，SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e。weights_only=True 反序列化成功。
- run=results/models/smoke，4 張真實 train、batch1、workers0、1008px、bf16，預期 valid 為涵蓋三類的2張前綴。
- runtime.py 最小相容修正：(1) 官方 checkpoint 僅排除未啟用 SAM2 neck 的22個額外項目，其餘1134項 strict load 成功；(2) 上游 MLP fused 算子拒絕梯度，梯度啟用時改用原 fc1/act/drop/norm/fc2 運算，保持參數與架構；(3) PLAN 未使用 semantic loss，semantic head 輸出 detach，使 DDP 正確識別其兩個未用參數，沒有新增 loss 或凍結有效分支。
- 第3次實測完成第1個 iteration：loss 約205、forward/backward/scaler.step 返回，耗時613.38秒，PyTorch peak allocated 指標約15GB；第2個 batch 遇 DDP reduction 錯誤，已定位至 semantic_seg_head.weight/bias（1088/1089）。尚未以 checkpoint 差異驗證參數更新，不宣告完整 smoke 通過。
- 第4次曾試 smoke-only cudnn_benchmark=False，但首個iteration在程序16分18秒仍未完成，已中止並撤回此效能調整；不是 OOM 結果。SIGINT 未及時結束，核對 PID 為本任務後用 SIGTERM 結束。相關設定和堆疊留在 logs。
- 第5次沿用原 cuDNN 設定，只保留上述必要相容修正，正在執行。validation/checkpoint/resume 尚待實測。CPU 回歸仍9 passed。
- 詳細日誌：results/models/smoke/logs/smoke_initial_console.log、smoke_retry_2_console.log 至 smoke_retry_5_console.log；前次 failure 已各自保存到 logs/attempt_N_failure.json。根目錄 failure.json 在重試期間可能仍是前次失敗紀錄，應以當次 console 與最終完成紀錄判斷。

## 2026-10-01 — 初始 GPU smoke 完成，resume 實測中

- 第5次初始執行 exit 0；RTX3080、batch1、1008px、bf16，4 iterations 全數完成。epoch loss=470.9253807067871，4筆 iteration loss 均有限；未 OOM。
- 2張真實 valid（image IDs 0、1）完成 SAM3 推論與原始尺寸遮罩評估，validation/epoch_001.csv 實際存在。CSV 的類別 IoU 門檻 PASS/FAIL 是模型評估欄位，不能代替技術 smoke；不據此評價模型最終品質。
- checkpoints/best.pt、checkpoint.pt、checkpoint_1.pt 均實際落盤，各10,081,260,002 bytes。checkpoint_1 的 epoch=1、steps.train=4、optimizer state step=4。
- mmap唯讀比較 pretrained 與 epoch1：687個抽查的小型參數張量改變，涵蓋視覺／文字／transformer／instance head，證據存 results/tables/2026-10-01_sam3_gpu_smoke_initial_updates.json。
- resume 使用相同run、batch1、--epochs 2 --resume，日誌確認從 results/models/smoke/checkpoints/checkpoint.pt 恢復到 epoch1。官方 Trainer 先執行上一輪 validation；目前仍在此階段，不能宣告 resume 後訓練通過。
- resume console: results/models/smoke/logs/smoke_resume_console.log；唯讀堆疊 smoke_resume_stack*.txt 顯示真實SAM3影像／segmentation／文字分支。GPU持續運算、接近10GB，驗證明顯較慢；未把它誤判為完成或OOM。
- 新增 scripts/2026-10-01_audit_gpu_smoke.py，僅稽核現有產物與參數變化，待總計2epochs完成後執行；未變更src造成resume signature失配。Linux既有CPU測試9 passed（6.11s）。
- 已說明超過三檔的文件更新範圍：PLAN補記執行授權、README／scripts README記錄環境與實際命令、requirements.txt保存實測freeze（SAM3 editable改為已固定commit的相鄰checkout），DEVLOG追加進度。原始freeze保留run內。

## 2026-10-01 — GPU smoke 最終結果：初始流程 PASS，resume 未完成，整體 FAIL

- 本次僅執行 PLAN 第4步的最小 GPU 技術驗證，未開始正式訓練。初始4 iterations、forward、有限loss、backward、optimizer step、2張valid推論及checkpoint保存均已實際完成。
- `checkpoint_1.pt` 實際反序列化後 epoch=1、steps.train=4、所有已建立optimizer state的step=4；抽查702個小型浮點權重張量均有限，其中687個相較官方pretrained權重改變，涵蓋視覺、文字、transformer及instance分割分支。
- resume 確實讀取 `/mnt/d/my-project/orchid-sam3/results/models/smoke/checkpoints/checkpoint.pt`，恢復到epoch1，並完成官方Trainer預先重跑的validation與epoch1 checkpoint保存。進入resumed training後，forward與loss有限性檢查已返回，但backward長時間未返回，沒有任何完整resumed optimizer step。
- 已向使用者說明本次觀察上限60分鐘；核對PID與命令列後，在3655.66秒以SIGTERM中止本次resume程序。這是受限觀察後的人工中止，不是捕捉到Python例外或torch CUDA OOM；不能據此推論RTX3080永遠無法完成。WSL核心多次記錄 `dxgkio_make_resident: Ioctl failed: -12`；原生堆疊在autograd索引梯度的CUDA stream synchronize等待。資源壓力／同步問題的根因尚未確定。
- 本次 `failure.json` 已更新為 `SmokeResumeObservationTimeout`，原DDP失敗仍完整保留於 `logs/attempt_3_failure.json`。`completion.json` 保留初始結果：epochs=1、training_completed=true、resume_exercised=false；不可將它解讀為resume已完成。
- 實際稽核命令：`.venv-train/bin/python scripts/2026-10-01_audit_gpu_smoke.py`。退出碼1符合未完成狀態；34 PASS、13 FAIL、1 N/A。FAIL集中於缺少第二輪更新／loss／validation／checkpoint及最終completion；未把缺失項目跳過或改成PASS。
- 稽核CSV：`D:\my-project\orchid-sam3\results\tables\2026-10-01_sam3_gpu_smoke_audit.csv`；參數證據：同目錄 `2026-10-01_sam3_gpu_smoke_updates.json`；逐檔完整Windows／WSL路徑與大小：同目錄 `2026-10-01_sam3_gpu_smoke_files.csv`。
- PLAN輸出根與實際根均為 `D:\my-project\orchid-sam3\results`；本次run為其下 `models\smoke`。現存training設定、logs、TensorBoard、validation CSV、checkpoint與稽核表均實際位於既有規則指定的位置。未發現主要測試輸出寫到PLAN之外；`D:\my-project\sam3.pt` 是使用者提供的輸入，不是測試輸出。
- 三份checkpoint `best.pt`、`checkpoint.pt`、`checkpoint_1.pt` 各10,081,260,002 bytes；`checkpoint_2.pt` 與 `validation/epoch_002.csv` 不存在。初始validation與resume預先重跑的validation皆保存於 `validation/epoch_001.csv`，相同run無另建目錄。未建立正式模型run，也未覆蓋正式test成果。
- 輸入仍為 `D:\my-project\orchid-sam3\data\processed\annotations_area_fixed\train\_annotations.coco.json` 與 `valid\_annotations.coco.json`（影像在各自同一目錄）；花=flower/1、莖=stem/3、葉=leaf/2。原始dataset未修改。
- 最後診斷在run的 `logs/smoke_resume_timeout.json`、`smoke_resume_kernel_final.log`、`smoke_resume_final_stack.txt`、`smoke_resume_native_stack_2.txt`；初始成功日誌為 `smoke_retry_5_console.log`，resume日誌為 `smoke_resume_console.log`。
- 合理性：資料尺寸px、遮罩面積px²已經品質檢查；實際prediction還原到原圖座標並與bool真值mask相容，validation比值均在[0,1]且有限。初始loss=470.9253807067871，是加權訓練目標，不是辨識品質。初始條件為真實官方權重及真實蘭花標註；零分母／空遮罩等既有CPU回歸9 passed。能量／質量守恆與解析物理解不適用；沒有模型品質的已知正確解可用作本次驗收。
- PLAN第6步的正式test CSV／300 dpi預覽不屬於smoke，Visualization標N/A，未用smoke權重冒充正式成果。這次少量valid的類別IoU門檻結果不代表最終模型品質。
- 結論：正式SAM3微調前要求的整條pipeline基本技術驗證尚未全部完成；尚缺resume後至少一個完整更新及其後續保存驗證。需先釐清WSL/CUDA的長時間等待，再重測；本次不自行降低解析度、凍結層、替換optimizer或開始正式訓練。RTX5090／正式batch4亦未實測。

## 2026-10-01 — Resume 單 batch 分階段診斷

- 範圍：直接讀取 `results/models/smoke/checkpoints/checkpoint.pt`；真實 SAM3、RTX3080、batch1、1008px、bf16、單一 resumed iteration、無 validation。沒有重新執行原始四 iterations，也沒有正式訓練。
- 最小變更僅為 PLAN 記錄本次授權、新增 `scripts/2026-10-01_diagnose_resume.py` 及本 DEVLOG。診斷入口加入階段 timestamp/elapsed、CUDA allocated/reserved/peak、nvidia-smi、模型／optimizer／scaler／scheduler 狀態稽核、480秒階段 timeout 與獨立 checkpoint 名稱。`--original-restore` 直接呼叫原有 OrchidTrainer restore；觀測包裝仍呼叫真正的 torch/model/optimizer 實作，沒有 mock 模型。`--mode weights` 僅作 optimizer-state B 組，不能當作完整 resume 通過。
- production 的 `__init__.py/config.py/data.py/evaluation.py/runtime.py/training.py` SHA256 均與既有 run 的 input_signature 相符；正式 resume 語意未修改。尚未證實根因，因此未套用推測性修正。
- A1（完整 model+optimizer restore、DDP、單次 CPU checkpoint 讀取）PASS：checkpoint load 61.374秒、model restore 1.302秒、optimizer restore 2.281秒、forward 45.780秒、loss 13.360秒、backward 116.513秒、optimizer step 3.245秒、save 96.493秒。Loss=917.80126953125，687個抽查小型 tensor 更新。
- A1 新產物 `results/models/smoke/checkpoints/diagnostic_a1.pt`，10,081,260,130 bytes；以 CPU mmap 實際重新反序列化驗證 model keys、RNG、train step=5、optimizer steps=5、scaler growth tracker=5、687個小型 tensor 變動。來源仍為 train/optimizer step=4；驗證表 `results/tables/2026-10-01_resume_diag_checkpoint_verification.json`。新 checkpoint 明確標示 completed_batches=1、partial_epoch=true，沒有冒充完整 epoch2。
- NODDP1（完整 restore、單次讀取、沒有 DDP wrapper）在 `forward_start` 超過480秒自動中止；尚未到 loss/backward。最後 Torch allocated=10,400,802,304 bytes、reserved=10,708,058,112 bytes，兩者亦為當時 peak；最後 nvidia-smi=10000 MiB、100%。DataLoader/batch transfer 均完成。NCCL process group 仍由 Trainer 初始化，但沒有 DDP 包裝與梯度 collective；這是 wrapper 隔離，不是完全移除 distributed runtime。
- ORIGINAL1（DDP、原始未修改 restore，包含第二次 CPU checkpoint 讀取與 RNG 恢復）同樣在 `forward_start` 480秒 timeout。最後 Torch allocated=13,779,602,944 bytes、reserved=14,166,261,760 bytes，亦為當時 peak；model/optimizer/RNG restore、batch transfer 全部完成。原始 restore 與 A1 的已記錄恢復狀態完全一致。這不是重現先前的 backward hang，而是同環境另一個尚未返回的階段。
- 三組輸入 batch hash 相同：`4293b6a86e3a5d0ff7012f7dba4cdbd77b8af7952d176b6fd60a36e2af1084f1`。全部840,509,750個參數的 requires_grad、device、shape、optimizer membership 前後一致；無重複／錯誤 optimizer 引用。AdamW moments 位於 cuda:0，non-capturable/non-fused 的 step scalar 位於 CPU（符合此設定），沒有 shape mismatch。Optimizer state 共6,708,341,428 bytes，其中 CUDA 6,708,337,064 bytes。
- Fresh 對照使用先前已成功的四 iterations 與本次 restore 前狀態，未重新跑四 iterations。Fresh optimizer state 為0；完整 restore 為 step4，A1更新後為step5。AMP為bf16但GradScaler仍啟用：scale65536、growth tracker從0恢復4再更新5。Scheduler由設定重建，第一個 resumed step 依原始 epoch/loader 長度計算 where=0.5、step=4；restore後 LR 與初始 LR 的差異是既有排程狀態，未見錯誤引用或device。
- A1 restore 前 allocated/reserved=3.330/3.430 GiB，optimizer restore 後=9.581/9.879 GiB，DDP及batch後=12.833/13.193 GiB；forward後=16.265/17.240 GiB，backward前=18.792/23.234 GiB，backward後=13.164/13.705 GiB；全程peak allocated/reserved=21.455/23.234 GiB。Torch allocator 數值與 nvidia-smi 實體 GPU 使用量分開記錄，不能把超過10GiB的allocator數值直接當作實體VRAM或OOM證明。
- WSL kernel：NODDP1於單調時間160.466608出現新的 `dxgkio_make_resident: Ioctl failed: -12`；ORIGINAL1在852.511864與976.471874再次出現。未捕捉到 PyTorch OOM exception。A1期間的kernel記錄因當時自動審核使用額度阻擋讀取、後續WSL重啟而無法追溯，標示未知，不宣稱沒有錯誤。Timeout時嘗試py-spy但offline cache缺套件，故沒有新的Python/native stack；精確定位限於已同步的階段紀錄，不虛構底層阻塞函式。
- 各案例 console、stages.jsonl、gpu_monitor.jsonl、state_*.json、result.json、source_preserved.json 在 `results/models/smoke/logs/resume_diag_<case>/`；CSV在 `results/tables/2026-10-01_resume_diag_<case>.csv`。原始 checkpoint、初始成功 evidence、既有 failure/completion/稽核結果均保留，不以本次診斷覆寫歷史。
- Optimizer A/B：A=`original1 --mode full --original-restore`，B=`weights1 --mode weights --original-restore`；兩組都保留DDP、兩次CPU讀取、model/RNG/scaler restore與相同batch。B僅略過optimizer state，相關參數群維持fresh值；其餘已記錄狀態一致。B同樣在forward 480秒timeout，沒有loss/backward/optimizer step或新checkpoint；最後allocated/reserved=6.581/6.756 GiB（當時已記錄peak亦同）、nvidia-smi=9995 MiB/100%。B在1628.539503、1822.524084、1989.401989出現3筆新的make_resident -12；未捕捉到Torch OOM。
- ORIGINAL1最後nvidia-smi=9982 MiB/100%。三個timeout的最後成功GPU階段均為`transfer_to_gpu_complete`；最後進入的是`forward_start`，未到新增的post-forward CUDA sync。因此本次定位為SAM3 forward呼叫內未返回，無法據此辨認底層特定kernel／同步函式；也沒有重現新的backward hang。
- 證據支持優先調查WSL／Windows GPU residency與高記憶體壓力：失敗組GPU接近10GB且都有新的residency錯誤。但A1完整restore曾成功，而較低記憶體的B組仍timeout；不能宣稱「必然OOM」、optimizer restore是唯一原因、或DDP是唯一原因。順序執行未控制WSL／Windows GPU context與其他GPU負載，單次成功不證明穩定。
- 已排除本次案例的checkpoint不可讀、model/optimizer結構不相容、錯誤tensor device／shape／optimizer引用，以及DataLoader／CPU→GPU搬移卡住。Scheduler與optimizer.step尚未被呼叫，不能作為這幾次forward等待的直接觸發動作。未排除CUDA內部同步、WSL GPU residency、驅動／GPU context狀態、記憶體壓力；本次沒有validation仍出現等待，但不能倒推先前backward hang的根因。
- 下一個最小步驟是環境隔離，並非修改正式resume：先確認沒有其他WSL工作會被中斷，再於新啟動的WSL／GPU context重跑同一個full/original-restore單batch對照、保留480秒限制並記錄啟動前GPU使用量。不得為了此診斷逕行中止其他人的WSL工作；本次沒有重啟／關閉WSL或Windows驅動，也沒有提出缺乏證據的程式修正。
- 最終彙總：`results/tables/2026-10-01_resume_diag_summary.csv`、同名JSON與`2026-10-01_resume_diag_files.csv`。清單實際盤點72個已存在產物，均位於PLAN的`D:\my-project\orchid-sam3\results`下；包含各組日志／表格與唯一新增的diagnostic_a1.pt。各組source_preserved=true；來源大小與mtime皆未改變，重新讀取仍step4。未發現主要診斷輸出寫到PLAN外。
- 重現入口（必須使用新case名稱以避免覆寫）：`.venv-train/bin/python -u scripts/2026-10-01_diagnose_resume.py --case <new_case> --mode full --original-restore --timeout 480`；B改為`--mode weights`，非DDP另加`--no-ddp`。本次A1/NODDP1使用單次讀取模式，故沒有`--original-restore`；此差異明列於summary，不能當成完全相同的正式路徑。
- 驗證：四個實際GPU單batch案例、成功checkpoint重新載入／step／更新驗證、production signature核對、診斷入口py_compile與git diff --check通過。沒有重跑無關CPU測試或完整四iterations。
- 結論：A1的「完整model+optimizer恢復→forward→有限loss→backward→step→新checkpoint」單次診斷PASS，但原始restore路徑仍TIMEOUT，原因尚未證實；**整體checkpoint resume仍不可判定已修復／穩定PASS**。尚未完成正式微調前所需的穩定性驗證。沒有開始正式訓練，沒有以fresh optimizer取代正式resume，也不據此評價模型品質。

## 2026-10-01 — Clean WSL context 單一 resume：本案例 PASS，穩定性未定

- 使用者授權先安全檢查、保存baseline、重啟WSL，再只執行一個原始full/DDP resume單batch。未更改training/runtime/diagnostic入口、optimizer、AMP、解析度或DDP；未重裝環境，沒有fresh training、其他A/B案例或正式訓練。此次只修改PLAN授權紀錄及DEVLOG，新增的logs/tables/checkpoint皆在既有results結構。
- 安全檢查：唯一發行版Ubuntu在最初查詢時已Stopped；啟動檢查後只有系統服務與本次觀測程序。Python服務為networkd-dispatcher及unattended-upgrade-shutdown，沒有使用者training/notebook/GPU compute process，故無殘留GPU Python需要手動kill。Windows桌面GPU應用未終止。`wsl --shutdown` exit0；boot ID由`630e632b-c6df-4e5c-8dd5-c906f2f7a962`變為`21ea6048-5955-45c8-a2a7-7d0fb94b82ad`。
- baseline已保存完整nvidia-smi／GPU程序、ps／listeners、free -h、/proc/meminfo、dmesg。重啟前GPU=718 MiB、WSL RAM used約798 MiB；重啟後idle GPU=719 MiB、RAM used約830 MiB；兩邊swap used=0。最初Windows觀測為659 MiB，另一次Windows baseline649 MiB，屬不同時間採樣；比較表採相同WSL查詢的718/719 MiB。重啟前已是低GPU使用量，不能假設此次從先前10GB hang狀態直接reset。
- 重啟後原`.venv-train/bin/python`、torch2.10.0+cu128、torch.cuda.is_available=True、NVIDIA GeForce RTX3080、SAM3 import全部PASS。訓練前再確認boot ID一致、GPU727 MiB、無compute process、WSL RAM約772 MiB used／14GiB available、swap0；未先載入任何訓練模型。
- 唯一訓練命令：`.venv-train/bin/python -u scripts/2026-10-01_diagnose_resume.py --case clean1 --mode full --original-restore --timeout 480`。來源`/mnt/d/my-project/orchid-sam3/results/models/smoke/checkpoints/checkpoint.pt`；batch1、1008px、bf16、原始model/optimizer/RNG/scaler restore與DDP，包含原本第二次CPU checkpoint讀取。無validation、單一resumed iteration。
- 各步PASS：checkpoint load65.116秒、model restore1.031秒、optimizer restore2.031秒、RNG checkpoint reload62.870秒、batch load／GPU transfer成功、forward254.285秒、loss17.202秒、backward113.611秒、optimizer step3.116秒、checkpoint save94.679秒。所有階段均在480秒上限內；沒有timeout或Torch OOM exception。Loss=917.80126953125，train step4→5、687個抽查小型tensor更新。
- 新checkpoint：`D:\my-project\orchid-sam3\results\models\smoke\checkpoints\diagnostic_clean1.pt`，10,081,260,130 bytes。以CPU mmap重新載入並驗證source step4／new step5、optimizer steps4→5、scaler growth tracker5、RNG存在、model keys一致、687個小型tensor變化；診斷標記full/DDP/original_restore=True、completed_batches=1、partial_epoch=True。沒有將此檔案冒充完整epoch2。
- 記憶體：optimizer restore後allocated/reserved=9.581/9.879 GiB，其中optimizer CUDA state=6,708,337,064 bytes；DDP與batch搬移後=12.833/13.193 GiB。Forward期間peak allocated/reserved=16.935/17.529 GiB；forward完成時allocated/reserved=16.265/17.240 GiB。Loss完成時累積peak=21.455/23.234 GiB，backward完成時累積peak仍相同；**未在loss後reset peak counters，故21.455/23.234是截至backward完成的累積峰值，不是backward獨立峰值**。backward完成時allocated/reserved=13.164/13.705 GiB。nvidia-smi定期採樣最高9993 MiB，不能把Torch allocator GiB等同實體VRAM使用量。
- 同boot的kernel證據：optimizer restore期間253.318530與forward期間386.907975各捕捉一筆新的`dxgkio_make_resident: Ioctl failed: -12`；兩階段最終仍返回成功。Restore觀測WSL RAM約10GiB used／5.1GiB available，forward觀測約11GiB used／3.8GiB available，swap均0。重啟前baseline另有dxg wait-sync WARNING call trace（不是該-12錯誤）；重啟後idle baseline沒有make_resident -12。
- 完成後取得的`system_after.json`已是另一boot（`2cc3bf43-fd7f-43e6-bff7-9a1e2a6f47c0`）；不能以其沒有-12宣稱全程無錯誤。訓練相同boot的`system_during_restore.json`與`system_during_forward.json`已保存至少兩筆；完整run的kernel錯誤總數未知。這個限制不影響已保存的逐階段CUDA同步完成、步數更新及checkpoint重新載入證據。
- 與A1比較：兩者loss相同、已記錄restore state完全一致、batch SHA256相同（4293b6a86e3a5d0ff7012f7dba4cdbd77b8af7952d176b6fd60a36e2af1084f1）、更新tensor數相同、成功時allocator峰值相同。A1單次CPU讀取；clean1保留原始兩次讀取。Forward45.780→254.285秒明顯變慢，backward116.513→113.611秒相近。A1 kernel history未知，不能宣稱A1沒有residency錯誤。
- 與三個timeout比較：original1／noddp1／weights1都在forward480秒中止，clean1在254.285秒返回並完成後續步驟。original1與clean1的已記錄restore state一致、forward前CUDA allocated/reserved一致；都包含原始兩次讀取、full optimizer及DDP。noddp1移除DDP wrapper、weights1略過optimizer state均未避免先前timeout。這次保留原始流程成功，不支持把optimizer/DDP或原始restore視為必然失敗的邏輯。
- 判讀選項：**較支持D：WSL/CUDA GPU context或memory residency壓力相關的不穩定**。重啟後仍迅速接近GPU容量且出現-12，但最終可成功，故不能把-12或timeout直接等同OOM，也不能由一次成功證明重啟已修復根因。Windows圖形／driver context沒有被重設，其他主機狀態未完全控制，A/B/C不作無條件排除。
- 下一個最小診斷建議（本次不執行）：維持原始單batch流程，只補上與階段時間對齊的Windows dedicated/shared GPU memory、RAM/commit，以及WSL RAM/kernel telemetry，辨別主機記憶體／residency壓力；先不改optimizer、DDP或resume邏輯，也不再批次連跑案例。
- 產物：`results/models/smoke/logs/resume_diag_clean1/`；重啟baseline在同run的`logs/2026-10-01_clean_context_clean1_*`；逐階段CSV=`results/tables/2026-10-01_resume_diag_clean1.csv`；五案例比較、memory表、checkpoint驗證、狀態比較、檔案清單與完整report分別為`results/tables/2026-10-01_clean_context_clean1_{summary.csv,memory.csv,checkpoint_verification.json,state_comparison.json,files.csv,report.json}`。
- 保存驗證：原有162個results檔案大小與mtime全數不變、無缺檔，包含checkpoint.pt與diagnostic_a1.pt；production六個src檔案SHA256仍符合原run signature。所有主要新增產物實際存在PLAN的`D:\my-project\orchid-sam3\results`內；沒有覆寫前次summary、failure/completion或成功evidence。
- 合理性：時間採秒、Torch記憶體採bytes／GiB、nvidia-smi採MiB並分開比較；loss有限、step更新與checkpoint重新載入一致，初始條件為原始step4與同一真實batch。能量／質量守恆與解析物理解不適用；沒有藉此評估最終分割品質，亦未為此重跑無關CPU極端案例。
- 本案例驗收：Clean context resume／Forward／Loss／Backward／Optimizer step／Checkpoint save皆PASS。現在已實證原始restore可以完成一次真實resumed iteration與保存；**穩定性仍未證實，不能宣稱所有既有timeout已解決**。正式RTX5090／batch4訓練與模型品質不在本次驗證範圍。

## 2026-10-01 — 搬機前最終 functional smoke：未完成驗收；部署清單完成

- 使用者要求單次兩階段functional smoke：第一階段真pretrained、真資料、batch1、3～5iterations／validation／save並正常退出；第二階段新process恢復全部state並更新／save／reload。為保留原始CLI與完整epoch語意，規劃4+4 iterations（兩次4張真實資料的小型epoch），並已向使用者說明；未改正式training/resume邏輯、模型結構或進行效能調校。
- 修改範圍事先說明超過三檔：新增`2026-10-01_functional_gpu_smoke.py`，原因是原CLI缺少逐階段watchdog與更新觀測；它以runpy呼叫原CLI，只包裝真實DDP forward、loss、Tensor.backward、GradScaler.step、原restore／validation／save做時間與CUDA記憶體記錄，每step抽查參數變化，沒有替代模型／optimizer。品質檢查仍是真實原validator，僅將report參數改成本次獨立表名以保護历史evidence。新增`LAB_DEPLOYMENT.md`因使用者要求搬機清單；PLAN記錄run／範圍／timeout；README與scripts README更新狀態和操作入口；DEVLOG追加結果。正式六個src檔案SHA256仍與input signature一致。
- 觀測入口支援initial→resume→audit；每次initial/resume均為獨立worker，initial拒絕覆蓋既有run，resume要求同run initial PASS。原CLI的model／optimizer／scaler／RNG restore、scheduler、DDP、validation與checkpoint規則不變。audit不開始訓練；兩階段未完成時明確輸出FAIL/NOT_RUN，完整成功才做CPU checkpoint reload及內容驗收。target-machine旗標仍是smoke，但額外要求PLAN的RTX5090。
- 安全檢查：僅Ubuntu，初始Stopped；啟動檢查後沒有使用者training/notebook/GPU compute工作，只有系統服務與本次觀測。保存ps／nvidia-smi／free／dmesg後執行`wsl --shutdown`成功，核對boot ID變更才執行第一階段。原checkpoint與舊diagnostic產物未刪除。磁碟剩餘約1.75TB；没有重裝套件。原venv環境仍Python3.12.14、torch2.10.0+cu128、torchvision0.25.0+cu128、固定SAM3 commit，pip check=`No broken requirements found.`。
- 唯一GPU案例：run=`results/models/smoke_transfer_20261001`，命令`.venv-train/bin/python -u scripts/2026-10-01_functional_gpu_smoke.py --phase initial --run-name smoke_transfer_20261001 --sam3-repo /mnt/d/my-project/sam3-reference --checkpoint /mnt/d/my-project/sam3.pt --batch-size 1 --timeout 900`。900秒是單一步驟上限，不是整體run上限，用來容納既有較慢fresh iteration；未因單純較慢提前中止，也未將上限延長為無限制等待。
- 真實SAM3 pretrained嚴格載入／840M trainable參數初始化、RTX3080環境檢查、真實COCO quality／pairing、mapping、DDP與第一個DataLoader batch均已到達。Dataset仍train211/3789、valid60/1147、test30/655，class flower=1／leaf=2／stem=3，base SHA256仍`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
- 本次worker PID522在step0的第一個`forward_start`（2026-10-01T13:28:12.477257Z）進入真正SAM3後，900秒內未返回，watchdog以SIGTERM結束worker，exit=-15，result=TIMEOUT。沒有forward_complete、finite loss、backward、optimizer step或權重更新，因此完成iterations=0；沒有執行本次validation／checkpoint save，第二階段沒有符合啟動前提而未執行。未以過往A1／clean1或初始4iteration成果替代本次兩階段驗收。
- 本次GPU最後採樣10008 MiB／97%，forward前最後Torch allocated/reserved=6,983,255,552／7,172,259,840 bytes；這是呼叫前記錄，不是卡住期間的完整峰值。持續kernel log在同boot捕捉5筆`dxgkio_make_resident: Ioctl failed: -12`（128.542528、288.509183、434.575755、581.108359、931.400484）；未捕捉到PyTorch OOM exception。Kernel觀測helper不使用GPU，initial失敗後已自動結束。
- 判讀嚴格分開：**Development-machine Functional Smoke Test=FAIL（本次驗收未完成）**，不是已證明training邏輯錯誤；**Known RTX3080/WSL Limitation=記憶體residency／執行時間不穩定**，timeout與-12不直接等同OOM；**Ready to transfer=NO**（必要檔案已盤點，但使用者要求的functional PASS條件尚未滿足）；**Target-machine Smoke Test=尚未執行**。過往A1與clean-context原始resume成功證據維持有效，不被本次timeout撤銷。
- 新report與表格：`results/tables/2026-10-01_smoke_transfer_20261001_{functional.json,report.json,summary.csv,initial_stages.csv,initial_quality.csv,files.csv,preservation.json}`。Baseline／restart證據亦在同prefix表格；run包含實際config、input signature、環境freeze、console／stage／GPU／kernel／dependency logs。所有已產生主要輸出均在PLAN的results下；本次未產生的validation／metrics／checkpoint不冒充路徑驗收PASS，完整output acceptance仍FAIL。
- 保護驗證：既有194個results檔案的大小／mtime完全不變、無缺檔，包含checkpoint.pt、diagnostic_a1.pt、diagnostic_clean1.pt及原tables/logs。沒有覆寫正式訓練結果，沒有正式訓練，也沒有額外GPU重試／A/B／性能研究。
- 部署交付：`LAB_DEPLOYMENT.md`包含實際working tree source、動態config／官方YAML與BPE、requirements／venv、真pretrained、完整dataset、class mapping、PLAN、必要scripts、可省略的smoke暫存產物、lab路徑設定、固定版本、activate／安裝與target smoke命令、PASS條件。實驗室需另跑`--target-machine --batch-size 4`兩程序smoke（step1→2），不能用開發機證據直接當RTX5090驗收。正式支架評估資料若尚未備妥，維持其既有限制，不為PASS捏造；它不是training/validation functional smoke的輸入。
- 驗證：新增script py_compile／--help通過；實際initial timeout與worker清理完成；使用真實產物執行audit得到exit1、FAIL／resume NOT_RUN，與現況一致。未重新執行GPU來測未用到的target分支，也未宣稱新wrapper的backward／save成功路徑已在本次跑通。環境依賴檢查、必要檔案hash盤點、source signature與舊evidence保護檢查通過。
- 合理性：時間用秒，Torch bytes與nvidia-smi MiB分開；真實初始權重、同既有dataset與設定，沒有假資料。Loss／梯度／更新的數值合理性在本次NOT_REACHED；能量／質量守恆與解析物理解不適用；過往成功是參考，不能替代本次驗收。未評價模型最終辨識品質。
- 本輪停止於既定timeout，不繼續無限制研究RTX3080。搬機清單可供後續使用，但依使用者本次明確門檻，只有新的完整兩階段functional PASS後才可將Ready改為YES。
