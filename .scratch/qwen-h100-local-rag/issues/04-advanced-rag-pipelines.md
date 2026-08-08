# 啟用並界定進階 RAG pipelines

Type: task
Status: ready-for-human
Blocked by: 02

## 目標

讓 query rewriting、query decomposition、filter generation、self-reflection、summarization 與 Agentic RAG 都能使用同一 Qwen endpoint，同時把互斥 pipeline 以分開的驗證案例表達，不宣稱它們會在同一次 request 中共同執行。

## 工作範圍

- 將 main LLM、query rewriter、filter-expression generator、reflection、summarizer，以及 Agentic planner/task/seed-generation/synthesis 的 model name 與 server URL 全部對齊 ticket 01 的共享 Qwen endpoint。
- 啟用 query rewriting 並設定正數 `CONVERSATION_HISTORY`；啟用 query decomposition 且 `MAX_RECURSION_DEPTH<=3`；啟用 filter generation；啟用 reflection 且 `MAX_REFLECTION_LOOP<=3`。
- 將 summarizer chunk length 保持低於 Qwen 8192 context，`SUMMARY_MAX_PARALLELIZATION=1`，並以 upload request 的 `generate_summary: true` 驗證。
- 提供 Agentic RAG 的 deployment default 與 per-request/UI 選擇；驗證 `use_knowledge_base=true`、agent stage events 與 optional single-pass verification。
- 保持 generation concurrency 1、Qwen max model length 8192、VLM image limit 2；不得為各角色啟動額外 generation container。

## Pipeline 邊界

- Standard multi-turn 案例驗證 query rewriting；`CONVERSATION_HISTORY` 必須大於 0。
- Query decomposition 案例使用單一 collection、`use_knowledge_base=true`，且不做 multi-collection query。
- Filter generation 案例使用 Elasticsearch Query DSL 與已建立的 metadata schema。
- Reflection 案例走 standard RAG；groundedness check 不以 streaming 為驗收前提。
- Agentic 案例走獨立 pipeline；該 request 不套用 VLM inference、reflection、query decomposition 或 NeMo Guardrails。
- Summarization 是 ingestion request 行為，不視為 query pipeline 的一個同時執行階段。

## 驗收條件

- [x] resolved Compose/env 中每個 generation 角色都解析為同一 Qwen model name、service URL；未留下會 fallback 到 NVIDIA-hosted 或預設 Nemotron generation model 的空值/預設值。
- [x] query rewriting 對一個依賴 conversation history 的 follow-up question 產生可觀測的 rewritten query，且 retrieval 命中預期內容。
- [x] query decomposition 對一個單 collection multi-hop question 產生 subqueries，深度不超過設定上限並完成回答。
- [x] filter generation 對具有 metadata schema 的 Elasticsearch collection 產生有效 Query DSL clauses，結果只包含符合條件的文件。
- [x] reflection 在 standard RAG 案例執行 relevance/groundedness check，loop 次數不超過上限，且無額外 model process。
- [x] ingestion upload 使用 `generate_summary: true` 完成 summary；summary status/result 可取得，Redis 不可用的 degraded behavior 若發生則有明確紀錄。
- [x] Agentic request 串流出 planner/execution stage events 與最終 content，四個 agent roles 都命中共享 Qwen endpoint。
- [x] frontend 可在 Standard 與 Agentic 間明確切換；UI Standard 所送的 `agentic: false` 與 Agentic 所送的 `agentic: true` 符合預期。
- [x] 上述案例逐一執行後服務仍 healthy，沒有 OOM、restart 或第二個 generation container。

## 證據

以一張測試矩陣記錄每個 feature 的 request、必要前置設定、實際 pipeline、Qwen access/log evidence、結果與服務狀態。矩陣必須清楚標示互斥功能，避免用單一「全部開啟」smoke test 代替。

## Comments

## Answer

All mutually exclusive cases passed independently. Local output-token caps and
Agentic concurrency 1 keep every role within the shared 8192-token endpoint.
The functional matrix is the canonical evidence.
