# 02 — 讓圖片檢索回傳多個不同頁面候選

**What to build:** 當使用者送出 Image + Text Query 時，knowledge base 不再只採用 similarity Top-1 所屬頁面，而會回傳具分數、來源與頁碼的多個不同候選頁面，讓後續流程有機會從相似產品中找到正確資料。

**Blocked by:** 01 — 建立弱文字 Image + Text Query 品質基準

**Status:** ready-for-agent

- [ ] Image retrieval 的 Top-K 代表實際候選搜尋範圍，而不是只讀取第一筆結果後捨棄其他候選。
- [ ] 相同 source identity 與 page number 的 chunks 被聚合，同一頁的多個 chunks 不會排擠其他文件或頁面。
- [ ] 每個候選保留 source、page、raw similarity score 與該頁可用 chunks，供後續排序、驗證與 citations 使用。
- [ ] 單筆候選缺少必要 metadata 時只略過該筆並留下可觀察原因，不會使所有有效候選消失。
- [ ] 啟用功能時，公開 multimodal `/generate` 行為可使用多頁候選；停用 feature flag 時維持既有行為。
- [ ] API streaming 與既有 request schema 保持相容，且日誌不輸出圖片 data URI 或 base64。
- [ ] 以 baseline 資料證明正確頁面可出現在 Hit@K 候選中，並記錄相對於現況的 retrieval latency。
