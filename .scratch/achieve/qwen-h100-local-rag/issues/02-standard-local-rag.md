# 打通標準 Local RAG Deployment

Type: task
Status: ready-for-human
Blocked by: 01

## 目標

在共享 Qwen endpoint 之上，建立可持久化的標準查詢與 ingestion 路徑：Qwen、VLM embedding、VLM reranker、Elasticsearch、ingestor server、RAG server 與 frontend 必須能一起啟動並完成文字文件的 ingest/query smoke test。

## 工作範圍

- 以 `deploy/compose/.env` 作為 self-hosted Docker 設定的 source of truth；若需要額外檔案，使用可合併的 Compose override，而非只保留 shell export。
- 將 `APP_LLM_*` 與 `APP_VLM_*` 指向 ticket 01 的同一 served model/service；`ENABLE_VLM_INFERENCE=true`、`VLM_TO_LLM_FALLBACK=false`、`APP_VLM_ENABLE_THINKING=false`、`APP_VLM_MAX_TOTAL_IMAGES=2`。
- 啟動 `nvidia/llama-nemotron-embed-vl-1b-v2` 與 `nvidia/llama-nemotron-rerank-vl-1b-v2`，兩者使用 GPU 0。RAG server 與 ingestor server 使用相同 embedding model、endpoint 和 2048 dimensions；reranker 開啟 image input。
- 使用 CPU Elasticsearch 與既有 named volume；RAG server 和 ingestor server 都指向同一 Elasticsearch service。啟動 Redis、SeaweedFS 或 blueprint 明確要求的其他 CPU dependency。
- 啟動 frontend、ingestor server 與 RAG server，並建立 dependency healthchecks/order，讓應用服務只在必要 dependency ready 後成為 healthy。
- 保持 generation concurrency 為 1；本票只驗證 standard pipeline，不啟用 ticket 04 的進階 pipeline 開關。

## 驗收條件

- [x] `docker compose config` 成功，且 `.env` 中 LLM/VLM model name 都等於 ticket 01 的 served model name，兩個 URL 都解析到同一 Qwen service。
- [x] RAG 與 ingestor 的 embedding 設定完全一致；VLM embedding、VLM reranker 與 Qwen 都只指派 GPU 0。
- [x] `docker compose ps` 顯示 Qwen、VLM embedding、VLM reranker、Elasticsearch、必要 CPU dependencies、ingestor、RAG server 與 frontend 均為 running/healthy。
- [x] `GET /v1/health?check_dependencies=true` 對 RAG server 與 ingestor server 都成功，health payload 顯示 Elasticsearch 及所選 local model endpoints。
- [x] 建立 collection、上傳一份小型文字文件、等待 ingestion 完成，再以 RAG query 取得含該文件內容的回答與 citation。
- [x] frontend 可載入、顯示 healthy backend，並能在 Standard pipeline 對同一 collection 完成一次 query。
- [x] `docker compose ps` 與 container inspect 證明沒有啟動預設 Nemotron LLM、預設 Nemotron VLM 或獨立 caption-generation container；唯一 generation process 是 ticket 01 的 Qwen。
- [x] container 內對 generation endpoint 的連線使用 service DNS，不使用 `localhost:8999` 或 host-published address。

## 證據

附上 resolved Compose service 清單、兩個 health responses、ingestion job 結果、query/citation 摘要，以及「實際啟動的 generation containers」檢查結果。測試資料須可安全刪除，且不得提交 secret。

## Comments

## Answer

Resolved by the committed Compose profile and the standard ingest/query row in
`docs/research/evidence/qwen-h100-runtime-results.md`.
