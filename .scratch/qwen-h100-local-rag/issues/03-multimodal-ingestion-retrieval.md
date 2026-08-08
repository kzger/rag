# 接通多模態 ingestion 與 retrieval

Type: task
Status: ready-for-agent
Blocked by: 02

## 目標

在標準 Local RAG Deployment 上接通圖片/版面處理鏈：Nemotron OCR、page elements、graphic elements、table structure、共享 Qwen captioning，以及 ticket 02 的 VLM embedding/reranker。

## 工作範圍

- 啟動 `nemotron-ocr`、`page-elements`、`graphic-elements` 與 `table-structure` 四個 local NIM，全部指派 GPU 0；使用 Compose service DNS 的內部 HTTP/gRPC endpoint。
- 開啟 text、table、chart、infographic/image extraction 與 image captioning；將 `APP_NVINGEST_CAPTIONMODELNAME`、`APP_NVINGEST_CAPTIONENDPOINTURL`、`VLM_CAPTION_MODEL_NAME`、`VLM_CAPTION_ENDPOINT` 對齊 ticket 01 的共享 Qwen model 與 `/v1/chat/completions`。
- 讓 structured/image element modality 產生 VLM embedding 可用的 image content，並以 ticket 02 的 2048-dimension VLM embedding 寫入 Elasticsearch。
- 將 ingestion batch/concurrency 限為 1 作為單卡 baseline；沿用 Qwen 的兩張圖片上限。
- 使用 Nemotron OCR + detector 路徑；whole-page-as-image、Nemotron Parse、Paddle OCR 與 audio segmentation/ASR 維持關閉，且不啟動其 profiles/services。
- 重新建立測試 collection，避免沿用 ticket 02 只有文字 embedding 的資料。

## 驗收條件

- [ ] resolved Compose config 明確列出 OCR 與三個 detector services，且四者都使用 GPU 0、具 readiness healthcheck，並只透過 `nvidia-rag` network 提供給 ingestion runtime。
- [ ] ingestor container 的 caption model name 等於共享 Qwen served model name；兩個 caption endpoint 變數解析到同一 Qwen service 的 `/v1/chat/completions`。
- [ ] multimodal extraction、image modality、VLM embedding 與 VLM reranker image input 的必要 env 均已設定，RAG/ingestor embedding model、endpoint、dimension 仍完全一致。
- [ ] `APP_NVINGEST_EXTRACTPAGEASIMAGE=False`、`APP_NVINGEST_PDFEXTRACTMETHOD` 未選 `nemotron_parse`、`APP_NVINGEST_SEGMENTAUDIO=False`，且 Paddle OCR、Nemotron Parse、audio 與獨立 caption model containers 都未啟動。
- [ ] 一份含連續文字、圖片/圖表及表格的 PDF 可成功 ingest；job/log evidence 顯示 OCR、適用的 detectors、Qwen captioning 與 VLM embedding 都實際被呼叫。
- [ ] 以文字問題可 retrieve 到圖片或表格衍生內容，VLM reranker 可處理 retrieved image passage，最終回答包含對應 citation。
- [ ] 對上述文件執行一次帶圖片的 query，並由共享 Qwen endpoint 成功回應。
- [ ] ingest/query 後所有必要服務仍 healthy，且沒有 GPU OOM 或 container restart。

## 證據

附上非敏感 env 摘要、service health、ingestion stage/log 摘要、retrieved modality/citation、query 結果與排除服務清單。測試 PDF 的來源或生成方式必須可重現。

## Comments
