# 04 — 自動理解弱圖片查詢並融合雙路檢索

**What to build:** 使用者即使只問「這是什麼？」，系統也會自動從本次圖片抽取 OCR、品牌、型號與可區分視覺特徵，形成 standalone query，並融合 visual retrieval 與 enriched-text retrieval 產生穩定的候選排序。

**Blocked by:** 02 — 讓圖片檢索回傳多個不同頁面候選

**Status:** ready-for-agent

- [ ] Query understanding 重用共享 Qwen endpoint，不新增常駐 generation model，並以受 token 上限約束的結構化結果表示 OCR、品牌、型號、物件類別、Logo 與其他可區分特徵。
- [ ] Query understanding 結果只用於檢索，不會被視為可直接支持最終答案的 knowledge base fact。
- [ ] 結構化結果解析失敗或資訊不足時，請求可安全回退至原始 multimodal query，並留下可觀察原因。
- [ ] Visual retrieval 使用本次圖片與必要原始文字；enriched-text retrieval 使用結構化特徵、原始文字及必要的文字對話摘要。
- [ ] 兩路候選依 source/page 去重，並以 deterministic weighted reciprocal rank fusion 排序；相同輸入產生穩定結果。
- [ ] 候選保留 visual rank、text rank、各路 raw score 與 fusion score，但 log 不含 base64 圖片。
- [ ] `/generate` 對 `j7ef-plus.jpg` 搭配弱文字問題可檢索到對應產品文件候選，且不要求使用者輸入產品名稱。
- [ ] 記錄 query understanding 與雙路檢索的階段耗時，並與 01 baseline 比較 TTFT 影響。
