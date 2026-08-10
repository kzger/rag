# 02 — 讓圖片檢索回傳多個不同頁面候選

**What to build:** 當使用者送出 Image + Text Query 時，knowledge base 不再只採用 similarity Top-1 所屬頁面，而會回傳具分數、來源與頁碼的多個不同候選頁面，讓後續流程有機會從相似產品中找到正確資料。

**Blocked by:** 01 — 建立弱文字 Image + Text Query 品質基準

**Status:** ready-for-human

- [x] Image retrieval 的 Top-K 代表實際候選搜尋範圍，而不是只讀取第一筆結果後捨棄其他候選。
- [x] 相同 source identity 與 page number 的 chunks 被聚合，同一頁的多個 chunks 不會排擠其他文件或頁面。
- [x] 每個候選保留 source、page、raw similarity score 與該頁可用 chunks，供後續排序、驗證與 citations 使用。
- [x] 單筆候選缺少必要 metadata 時只略過該筆並留下可觀察原因，不會使所有有效候選消失。
- [x] 啟用功能時，公開 multimodal `/generate` 行為可使用多頁候選；停用 feature flag 時維持既有行為。
- [x] API streaming 與既有 request schema 保持相容，且日誌不輸出圖片 data URI 或 base64。
- [x] 以 baseline 資料證明正確頁面可出現在 Hit@K 候選中，並記錄相對於現況的 retrieval latency。

## Answer

Implemented opt-in diverse image-page retrieval behind
`ENABLE_MULTIMODAL_ACCURACY` for Elasticsearch, Milvus, and LanceDB. The
backends persist a canonical source/page identity and backfill existing
collections. Milvus groups on that identity in the vector query. Elasticsearch
uses exact vector scoring plus field collapse and LanceDB performs an exhaustive
ordered distance scan, because their KNN APIs otherwise select raw chunks before
grouping. The adapters skip malformed hits individually, aggregate every
available page chunk, and preserve each chunk's content and metadata alongside
the raw similarity score and original rank. The disabled path retains the prior
Top-1 page expansion, public request and streaming schemas are unchanged, and
image URLs and encoded payloads are omitted from logs.

The corrected 2026-08-10 live run on `jorjin_glasses` returned five distinct
page candidates through both `/v1/search` and `/v1/generate`, with candidate
Hit@5 0.75 and retrieval latency 0.541 s P50 / 0.683 s P95 versus 0.368 s P50 /
0.542 s P95 for the Top-1 comparison. With the same `VECTOR_DB_TOPK=100`, a K=20
run recovered the previous Top-1 miss `J10A Sales kit.pdf#page=9`, raising
candidate Hit@K to 1.0; its retrieval latency was 0.550 s P50 / 0.816 s P95.
Evidence is recorded in
`docs/research/evidence/multimodal-diverse-page-candidates-2026-08-10.json`,
`docs/research/evidence/multimodal-diverse-page-k20-recovery-2026-08-10.json`,
and `docs/research/evidence/multimodal-top1-page-comparison-2026-08-10.json`.

This ticket intentionally does not choose among similar candidates. The live
generation-quality diagnostic regressed when all five candidates reached the
current VLM prompt; enriched fusion and verified confidence handling remain the
responsibility of tickets 04 and 05.
