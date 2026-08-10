# 06 — 校準多模態準確度與延遲預設值

**What to build:** 使用可重現的 baseline 資料集校準 multimodal accuracy pipeline，選出能顯著降低錯誤確定回答且維持可接受互動延遲的候選數、融合權重、信心門檻、圖片預算與 generation 設定。

**Blocked by:** 05 — 視覺驗證候選並執行信心政策

**Status:** ready-for-agent

- [ ] 以相同資料、collection 與模型對 feature flag 關閉／開啟進行 A/B evaluation。
- [ ] 評估包含使用者提供的兩張圖片、弱文字問題、相似產品、連續圖片、無匹配資料及故意歧義案例。
- [ ] 報告比較 Hit@K、verified identification accuracy、unsupported-claim rate、拒答 precision/recall、TTFT 與完整回答 P50/P95。
- [ ] 根據 evidence 選定 visual/text candidate counts、final page count、fusion weights、absolute confidence threshold、margin threshold、verification threshold 與 image budget 預設值。
- [ ] 將 Local RAG Deployment 的 VLM generation temperature 校準至 0.0 或 0.1 中有較佳 groundedness 的值。
- [ ] 額外 Qwen 呼叫具有有界 token 與候選數；若 latency 不可接受，優先縮短結構化輸出、平行 retrieval 或快取圖片摘要。
- [ ] 所有可調參數具設定說明、安全範圍與回退值，且不要求修改 API request schema。
- [ ] 報告明確列出尚未達標案例，不以少量成功示例取代整體指標。
