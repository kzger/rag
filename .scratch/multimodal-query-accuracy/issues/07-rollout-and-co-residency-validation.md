# 07 — 部署準確度流程並驗證 Full-Service Co-residency

**What to build:** 將校準後的 multimodal accuracy pipeline 套用至 Local RAG Deployment，讓 WebUI 的真實 Image + Text Query 使用新流程，同時證明單張 H100 上所有 generation、retrieval 與 ingestion GPU services 仍可持續共同運行。

**Blocked by:** 06 — 校準多模態準確度與延遲預設值

**Status:** ready-for-agent

- [ ] Active Compose env 與 live RAG service 使用 06 選定的 feature flag、候選、融合、信心、image budget 與 VLM generation 設定，且 source-of-truth 與 container env 一致。
- [ ] WebUI 選定 collection 後，以 `j7ef-plus.jpg` 和 `logo_max.png` 執行弱文字問題，實際結果符合 verified／ambiguous／no_match policy 並提供適當 citations。
- [ ] WebUI 未選 collection、連續不同圖片、無匹配圖片及下游服務錯誤的行為符合規格，且不會把 direct VLM guessing 偽裝成 RAG。
- [ ] 觀察資料可追蹤 query understanding、visual retrieval、text retrieval、fusion、verification、confidence decision 與 generation latency，但不包含 base64 圖片或 secrets。
- [ ] 在 ingestion、text query 與 Image + Text Query 流量期間，所有必要服務連續五分鐘 healthy 且 container restart count 不變。
- [ ] 驗證期間沒有 GPU OOM，且不會透過停止 ingestion 或其他必要服務取得 query capacity。
- [ ] Text-only query 與 ingestion correctness 執行回歸驗證，確認 multimodal accuracy rollout 沒有改變其既有行為。
- [ ] 保存非敏感設定摘要、服務健康、quality metrics、P50/P95 latency 與 Full-Service Co-residency evidence，並記錄 feature flag 回退程序。
