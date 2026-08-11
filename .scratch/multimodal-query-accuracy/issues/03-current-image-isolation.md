# 03 — 隔離歷史圖片並保障本次圖片預算

**What to build:** 在多輪 Image + Text Query 中，系統只把本次上傳的圖片當作 query image；先前圖片以文字摘要保留語意，不會污染本次檢索、視覺判斷或占用 retrieved evidence 的 image budget。

**Blocked by:** 01 — 建立弱文字 Image + Text Query 品質基準

**Status:** ready-for-human

- [x] 第二次圖片詢問的 retrieval 與後續視覺處理只接收第二張圖片，第一張圖片的原始 data URI 不會再次送出。
- [x] 歷史圖片可由先前產生且可稽核的文字摘要保留必要對話語意；摘要失敗時安全省略原始圖片而不阻斷本次請求。
- [x] Generation image budget 依序保障本次 query image、最高相關 retrieved page image、次高相關 retrieved page image。
- [x] 歷史圖片數量不會降低本次 query image 或最高相關 retrieved page image 被送入 VLM 的機會。
- [x] 以 `j7ef-plus.jpg` 後接 `logo_max.png` 的多輪案例證明第二次回答不會沿用第一張產品圖的識別結論。
- [x] Text-only 多輪對話與單次圖片詢問的既有 API/streaming 行為不受影響。
- [x] 日誌能指出圖片來源角色與預算分配，但不記錄圖片內容。

## Evidence

- Public `/generate` and feature-off compatibility coverage:
  `tests/unit/test_rag_server/test_query_rewriting.py`.
- Ranked VLM image-budget and payload-safe logging contract:
  `tests/unit/test_rag_server/test_vlm.py`.
- Live multi-turn report and sanitized allocation observations:
  `docs/research/evidence/multimodal-current-image-isolation-2026-08-11.json`.
  The rerun after `0c6646d` completed with the live three-image budget
  (current query + two ranked retrieved pages) and no VLM image-limit error.
