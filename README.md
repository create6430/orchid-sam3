# <專案名稱>

一句話說明這個 repo 在做什麼。

## 這個 repo 是什麼

- 研究題目：
- 負責人：
- 起訖時間：

## 怎麼跑一次

```bash
# 環境
pip install -r requirements.txt

# 執行主要分析，產生 results/figures/ 底下的圖
python scripts/2026-09-02_modal_analysis.py
```

## 資料在哪

原始量測資料**不放在 git**（檔案太大）。

- 位置：`"D:\my-project\Lab-AI-Template\repo-template\data\raw"` 或 OneDrive 連結
- 下載後放到 `data/raw/`
- 資料格式、單位、感測器設定見 [`SPEC.md`](SPEC.md)

## 資料夾說明

```
├─ CLAUDE.md           Claude Code 的規則指向檔
├─ instructions.md     給 AI agent 的常駐規則（開工前先讀）
├─ SPEC.md             單位、座標系、資料格式、感測器規格
├─ DEVLOG.md           開發流水帳
├─ requirements.txt    Python 套件
├─ plans/              每個任務的規劃檔（範本：PLAN_template.md）
├─ data/
│   ├─ raw/            原始資料（唯讀，不進 git）
│   └─ processed/      處理後資料（不進 git）
├─ src/                可重複使用的函式
├─ scripts/            一次性執行腳本
└─ results/
    ├─ figures/        圖檔，300 dpi，座標軸須標單位
    └─ tables/         表格 CSV
```

## 給 AI agent 使用者

本 repo 採用 plan-first 流程。使用任何 AI coding agent 前，請先讓它讀 [`instructions.md`](instructions.md)：

- **Claude Code**：`CLAUDE.md` 內已指向，直接開始即可
- **GitHub Copilot**：`.github/copilot-instructions.md` 內已指向
- **Cursor**：`.cursor/rules/main.mdc` 內已指向
- **網頁版 ChatGPT / Gemini**：把 `instructions.md` 整份貼在對話第一則訊息
