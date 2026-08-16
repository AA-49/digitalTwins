# 批判性自審報告

## 整體結構評估

- [x] 故事線由預測、解釋、模擬到證據整合，四個 Stage 各自對應一個實驗。
- [x] Introduction 中的三項貢獻均可在後文找到方法與驗證。
- [x] 所有已報告數字均可追溯至 notebook artifacts、測試或目前程式碼。
- [ ] 尚未加入外部 baseline，因此不能主張優於其他糖尿病預測方法。
- [ ] 尚未完成多次重跑、統計信賴區間與外部資料集驗證。

## 逐段檢查

### Abstract

- Topic sentence：模型除了預測之外，也需要可查驗的解釋、情境分析與限制溝通。
- 優點：包含問題、方法、具體結果與影響；明確揭露 Medium recall 0.11%。
- 問題：結果較多，若投稿頁數有限可刪減 Stage 4 節點與邊數。
- 建議：模型重訓後必須同步更新 Abstract、Experiment 1、Conclusion 的所有數字。

### Introduction and Related Work

- Topic sentence：研究目標是把四種能力整合為一個可稽核的研究原型。
- 優點：明確區分原始 proposal 與目前實作，避免把 ShanghaiT2DM、HbA1c 或自動介入生成誤寫為已完成成果。
- 問題：Related Work 目前只建立最低限度背景，距完整論文的 30-50 篇引用仍有差距。
- 建議：投稿前補充 BRFSS 糖尿病分類、類別不平衡、臨床 XAI、人類 Digital Twin 與 healthcare KG 的直接相關研究。

### Method

- Topic sentence：21-feature evidence contract 貫穿訓練、預測、SHAP、情境與 graph。
- 優點：資料分割、模型參數、permutation sample、SHAP class semantics 與 Ollama 安全邊界均可重現。
- 問題：沒有 formal causal model，也沒有 longitudinal update equation，因此應維持「Digital Twin prototype」而非成熟 clinical twin 的定位。

### Experiments

- Topic sentence：每個 Stage 回答不同層級的研究問題。
- 優點：Experiment 1 是 held-out quantitative evaluation；Experiment 2 區分 global 與 local explanation；Experiment 3 明確標示 manual re-prediction；Experiment 4 以完整性與安全測試為主。
- 問題：四個實驗的證據強度不對稱，Stage 3 只有一個 case，Stage 4 沒有 user study。
- 建議：新增 repeated stratified cross-validation、calibration、multiple-patient SHAP cases、scenario sensitivity，以及 graph comprehension study。

## 致命問題

1. Medium/prediabetes recall 僅 0.11%，現階段不能宣稱模型適用於三類 screening。
2. BRFSS 是 cross-sectional survey；Stage 3 的 19.01 percentage-point 變化不能解讀成 intervention effect。
3. 原 proposal 的自動臨床 intervention 與 mechanistic chain 未由目前資料或程式驗證，不能寫入成果聲明。

## 重大問題

1. 缺少 baseline model comparison 與外部 validation。
2. 缺少多次隨機種子、信賴區間與 statistical significance。
3. Stage 2 permutation importance 僅使用 1,000 筆及三次 repeats。
4. Stage 3 與 Stage 4 缺少人因、臨床正確性與 automation-bias 評估。

## 次要問題

1. `confusion_matrix.png` 的標題與 x-axis labels 有擁擠現象，正式 PDF 前應重新排版。
2. 報告尚未填入作者、學校、課程或 supervisor 資訊。
3. 參考文獻目前採最低可用集合，尚未針對指定學校格式調整。

# 審稿分數預測

| 維度 | 分數 | 說明 |
| --- | ---: | --- |
| 新穎性 | 5/10 | 四階段整合與安全 evidence contract 有價值，但各元件多為既有技術。 |
| 技術品質 | 5/10 | 實作邊界嚴謹；模型 minority-class performance 是主要弱點。 |
| 清晰度 | 8/10 | Stage-to-experiment 對應與非因果語義清楚。 |
| 實驗完整性 | 4/10 | 缺 baseline、重複實驗、外部驗證與 user study。 |
| 影響力 | 5/10 | 適合研究原型與 explainability interface study，尚未達臨床影響。 |
| 可重現性 | 7/10 | Notebook、artifacts、Docker、參數與測試齊全；大型 model artifact 需另行可信傳送。 |
| 呈現品質 | 6/10 | 報告結構完整，但部分既有圖表需重新排版。 |

- 整體分數：5/10
- 預測決定：Borderline（workshop / postgraduate research report level）
- 不宜定位：clinical AI validation paper 或頂會完整實驗論文

# 改進要點清單

## 必須修改

1. 改善或重新定義 Medium/prediabetes 任務，並以 repeated evaluation 驗證。
2. 加入至少兩個合理 baseline，使用相同 split、metrics 與 tuning budget。
3. 對 Stage 3 全文維持 manual, model-based, non-causal 用語。

## 強烈建議

1. 報告 calibration curve、Brier score、class-specific PR-AUC 與 confidence interval。
2. 對多個 true-positive、false-positive、false-negative 病例做 SHAP error analysis。
3. 對 Stage 4 進行 explanation comprehension 與 unsafe-advice detection study。

## 錦上添花

1. 重製 Figure 1，顯示四個 experiments 共用的 21-feature evidence flow。
2. 將完整 hyperparameters、feature decoding table 與 test inventory 放入 appendix。
