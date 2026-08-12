# 04 — 自動理解弱圖片查詢並融合雙路檢索

**What to build:** 使用者即使只問「這是什麼？」，系統也會自動從本次圖片抽取 OCR、品牌、型號與可區分視覺特徵，形成 standalone query，並融合 visual retrieval 與 enriched-text retrieval 產生穩定的候選排序。

**Blocked by:** 02 — 讓圖片檢索回傳多個不同頁面候選

**Status:** ready-for-human

- [x] Query understanding 重用共享 Qwen endpoint，不新增常駐 generation model，並以受 token 上限約束的結構化結果表示 OCR、品牌、型號、物件類別、Logo 與其他可區分特徵。
- [x] Query understanding 結果只用於檢索，不會被視為可直接支持最終答案的 knowledge base fact。
- [x] 結構化結果解析失敗或資訊不足時，請求可安全回退至原始 multimodal query，並留下可觀察原因。
- [x] Visual retrieval 使用本次圖片與必要原始文字；enriched-text retrieval 使用結構化特徵、原始文字及必要的文字對話摘要。
- [x] 兩路候選依 source/page 去重，並以 deterministic weighted reciprocal rank fusion 排序；相同輸入產生穩定結果。
- [x] 候選保留 visual rank、text rank、各路 raw score 與 fusion score，但 log 不含 base64 圖片。
- [x] `/generate` 對 `j7ef-plus.jpg` 搭配弱文字問題可檢索到對應產品文件候選，且不要求使用者輸入產品名稱。
- [x] 記錄 query understanding 與雙路檢索的階段耗時，並與 01 baseline 比較 TTFT 影響。

## Answer

Implemented and live on the deployed rag-server (2026-08-12):

- `src/nvidia_rag/rag_server/multimodal_accuracy.py` — `build_query_understanding_prompt` / `parse_query_understanding` / `build_enriched_text_query` / `fuse_visual_text_candidates` (weighted RRF, source/page dedup, fused candidates keep visual_rank/text_rank/fusion_score). Logs contain no base64 image data.
- `vlm.py::understand_query_async` reuses the shared Qwen endpoint; parse failure or exception falls back to the raw multimodal query with an observable log warning.
- Wired in `main.py` (query understanding + dual retrieval + fusion stages with stage timings) and `utils/configuration.py` (`enable_query_understanding`, `enable_multimodal_accuracy`, `visual_candidates`, `max_candidates`).

Live evidence (deployed server, `jorjin_glasses`, j7ef-plus.jpg + 弱文字):
`Multimodal accuracy stages: query_understanding_ms=1300.8 dual_retrieval_ms=476.7 fusion_ms=0.9 visual_candidates=5 text_candidates=5 fused_candidates=5`; J7EF spec PDF page 3 is the top fused candidate. Full 8-case eval: candidate_retrieval_hit_at_k = 0.75.

Unit tests: 113 passed (test_multimodal_accuracy.py, test_multimodal_fusion.py, test_configuration.py) on `develop`-equivalent branch via `uv run pytest`.
