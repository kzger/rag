# 01 — 建立弱文字 Image + Text Query 品質基準

**What to build:** 建立一套可重現的 Image + Text Query 評估流程，讓維運人員能以真實使用者可能輸入的弱文字問題量測現況準確度、拒答品質與延遲，並保存後續 tickets 可比較的 baseline。

**Blocked by:** None — can start immediately

**Status:** ready-for-human

- [x] 評估資料集至少涵蓋「這是什麼？」、「是這個嗎？」、「這又是什麼？」、含 OCR 型號、無文字產品照、相似產品、連續不同圖片及 knowledge base 無匹配資料。
- [x] 使用者提供的 `j7ef-plus.jpg` 與 `logo_max.png` 成為固定測試案例；測試前置檢查會在資產不可讀時清楚失敗，而不是略過案例。
- [x] 評估透過公開 multimodal `/generate` 行為執行，並能確認已選 collection 的請求確實使用 knowledge base。
- [x] 報告至少包含 candidate retrieval Hit@K、verified identification accuracy、unsupported-claim rate、low-confidence rejection precision/recall、TTFT 與完整回答時間。
- [x] Baseline 報告記錄模型、collection、候選設定與測試資料版本，且不保存圖片 base64 或敏感設定。
- [x] Unit tests 不使用網路；live evaluation 與 unit tests 有清楚分離的執行入口。

## Answer

Implemented the versioned weak-query dataset and standalone live evaluator in
`scripts/eval/`. The public Standard `/v1/generate` baseline ran against the
`jorjin_glasses` collection and shared `Qwen/Qwen3.6-27B-FP8` endpoint on
2026-08-10. All eight cases completed. Candidate retrieval Hit@K was 0.75 and
verified identification accuracy was 0.75 for the identifiable cases, while
the manually adjudicated atomic-claim unsupported rate was 0.4524 and
low-confidence rejection precision and recall were both 0.0. TTFT was 0.904 s
P50 / 2.312 s P95; full response latency was 3.792 s P50 / 5.112 s P95.

The answer-hash-bound manual adjudication and machine-readable report are
`docs/research/evidence/multimodal-accuracy-baseline-adjudications-2026-08-10.json`
and
`docs/research/evidence/multimodal-accuracy-baseline-2026-08-10.json`. It stores
asset paths and SHA-256 hashes but no image payloads or credentials. Offline
unit tests cover asset preflight, selected-collection request construction,
stream parsing, metrics, manifest coverage, and report redaction.
