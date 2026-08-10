# 05 — 視覺驗證候選並執行信心政策

**What to build:** 系統會以本次圖片與融合後的候選頁面證據進行二次視覺驗證，並在生成答案前決定結果是 verified、ambiguous 或 no_match，避免把錯誤候選或模型猜測描述成確定事實。

**Blocked by:** 02 — 讓圖片檢索回傳多個不同頁面候選；03 — 隔離歷史圖片並保障本次圖片預算；04 — 自動理解弱圖片查詢並融合雙路檢索

**Status:** ready-for-agent

- [ ] 候選驗證重用共享 Qwen endpoint，比較本次 query image、standalone query、候選文字與可用候選頁面圖片。
- [ ] 每個驗證結果包含 candidate identity、match decision、confidence、supporting evidence 與 conflicts，且結構解析失敗不會被視為 verified。
- [ ] Deterministic confidence policy 至少考量候選存在性、Top-1 絕對分數、Top-1/Top-2 margin、verification confidence、evidence conflicts 與必要 metadata。
- [ ] `verified` 只使用通過驗證的 context 回答並提供正確 citation；caption 或 query understanding 的推測不得成為唯一證據。
- [ ] `ambiguous` 說明無法唯一判定並只列出有證據支持的候選與差異，不任選型號。
- [ ] `no_match` 明確說明 knowledge base 無法確認，不會無聲退回 direct VLM guessing。
- [ ] 未選 collection 時，使用者可辨識該回答沒有使用 knowledge base；下游服務失敗使用既有錯誤格式回報。
- [ ] 以 `logo_max.png` 及 knowledge base 無匹配案例證明系統不會因單一錯誤頁面候選而確定宣稱產品型號。
- [ ] Feature flag 關閉時可回退既有 multimodal pipeline；開啟時維持既有 streaming response 與 citations 相容性。
