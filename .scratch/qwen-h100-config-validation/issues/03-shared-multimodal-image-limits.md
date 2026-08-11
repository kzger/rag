# 03 — 統一 RAG 與 Qwen 的多模態圖片預算

**What to build:** 讓使用者能從 Docker source-of-truth 設定 Qwen 每次 prompt 的多模態圖片上限與 RAG prompt budget，兩者會一致地傳到 resolved deployment，並在不相容時於啟動前阻止必然失敗的 request。

**Blocked by:** 01 — 建立一般驗證與 Certified Profile 的擴充介面

**Status:** ready-for-agent

- [ ] Qwen image-per-prompt limit 是具名、可配置的正整數，resolved model command 不再硬編碼圖片數量。
- [ ] RAG image budget 是獨立的正整數設定，且不得大於 Qwen image-per-prompt limit。
- [ ] 兩值相等或 RAG budget 較小時一般驗證通過；RAG budget 較大時錯誤同時指出兩個參數和值。
- [ ] 零、負數、非整數、缺少或無法解析的圖片限制會安全失敗，不延遲到 runtime HTTP 400。
- [ ] 合法設定會同時反映在 resolved Qwen command 與 RAG service environment。
- [ ] Live smoke test 使用支援的圖片數完成 generation，不再發生 image-count HTTP 400。
- [ ] 本 ticket 只處理部署參數、server limit 與 validator 關係，不修改 `multimodal-query-accuracy` 的候選排序、歷史圖片隔離、驗證或回答策略。
