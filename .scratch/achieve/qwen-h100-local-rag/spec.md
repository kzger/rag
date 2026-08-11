# Qwen H100 Local RAG Deployment

## 目標

在 1×H100 80GB 上建立 Docker Compose 形式的 Local RAG Deployment。所有必要的 generation、retrieval 與 ingestion GPU services 必須同時常駐；文字生成、視覺生成與 image captioning 共用一個 pinned Qwen endpoint。

## 已決定的邊界

- generation baseline 是 `Qwen/Qwen3.6-27B-FP8`；H100-compatible NVFP4 只作容量 fallback，兩者不會同時常駐。
- Elasticsearch、VLM embedding/reranker、Nemotron OCR 與三個 detectors 組成必要的 retrieval/ingestion service set。
- whole-page-as-image、Nemotron Parse、Paddle OCR 與 audio 不屬於部署範圍。
- Agentic RAG 與 standard pipeline 的 query rewriting、decomposition、filter generation、reflection、summarization 分案例驗證，遵守各 pipeline 的互斥條件。
- host 只在 loopback 發布 frontend 8090、RAG 8081、ingestor 8082 與 Qwen 8999；其他服務只在 Compose network 可達。
- 部署成敗 gate 依 ADR-0001：全部必要服務連續五分鐘 healthy，且所有 container restart count 不變。功能正確性、品質、latency 與 throughput 是另行記錄的 follow-up evidence，不是這個 gate。

## 後續操作變更

- 2026-08-09 依操作需求將共享 Qwen endpoint 的 context 從 8192 提升為 32768。由於 LLM、VLM、Reflection 與 captioning 共用同一 endpoint，此值不是 VLM-only 設定。32K profile 已通過圖片 RAG 功能測試與 314 秒 Full-Service Co-residency gate；舊的 8K 結果只作 fallback 與歷史證據。
- Image query 必須跳過文字 reflection 與 reranking，避免把 base64 data URL token 化；圖片只進入 multimodal embedding/retrieval 與最終 VLM generation。

## 工作分解

依序執行 `issues/01-shared-qwen-endpoint.md` 至 `issues/06-validate-and-document-co-residency.md`，並遵守每張 ticket 的 `Blocked by` 關係。若 FP8 降載與 NVFP4 fallback 都無法通過 gate，結論必須是單卡 Full-Service Co-residency 不可行，不得改以 ingestion/query 分時宣稱成功。
