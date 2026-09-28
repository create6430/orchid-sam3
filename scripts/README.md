# scripts/ — 一次性執行腳本

每個任務的實際執行入口。從專案根目錄執行：

```bash
python scripts/YYYY-MM-DD_<用途>.py
```

- 命名：`YYYY-MM-DD_<用途>.py`，日期是任務開始日
- 一個腳本對應 `plans/` 底下一個 PLAN
- 讀 `data/raw/`（唯讀）、寫 `data/processed/` 與 `results/`
- 共用邏輯抽到 `src/`，這裡只留流程與參數
- 隨機種子要固定，讓結果可重現
