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
