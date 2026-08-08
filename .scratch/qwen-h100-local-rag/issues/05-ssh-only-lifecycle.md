# 收斂 SSH-only 網路與服務生命週期

Type: task
Status: ready-for-agent
Blocked by: 03, 04

## 目標

將完整 Local RAG Deployment 收斂成適合遠端 SSH 使用的可恢復服務：host 只在 loopback 暴露 frontend、RAG API、ingestor API 與 Qwen API，其餘服務只在 Compose network 內可達。

## 工作範圍

- host publish 僅保留 `127.0.0.1:8090`、`127.0.0.1:8081`、`127.0.0.1:8082`、`127.0.0.1:8999`；container ports 分別維持 frontend 3000、RAG 8081、ingestor 8082、Qwen 8000。
- Elasticsearch、Redis、SeaweedFS、NV-Ingest runtime、VLM embedding/reranker、OCR 與三個 detectors 只使用 Compose network/expose，不發布 host port。
- 為所有必要服務補齊能代表「可接受依賴流量」的 healthcheck、`depends_on` health ordering、合理 start period/retries 與 `restart: unless-stopped`（或有明確理由的等價 policy）。
- 持久化 Qwen/model NIM cache、Elasticsearch data、SeaweedFS objects、ingestor data、LanceDB/其他實際使用的 application data；暫存資料不得被誤列為持久化需求。
- 提供由遠端 client 使用 SSH local forwarding 存取 8090/8081/8082/8999 的命令，並記錄 start、status、logs、restart、stop（保留資料）流程。
- 驗證 Docker daemon 或主機重啟後服務可依健康順序自動恢復；本票不執行刪除 volumes 的 teardown。

## 驗收條件

- [ ] `docker compose config` 的全部 `ports` 經機器可讀檢查後，集合恰為 loopback 8090/8081/8082/8999；沒有 `0.0.0.0`、`::` 或未指定 host IP 的 publish。
- [ ] `ss -ltn` 證明四個允許 port 只綁定 `127.0.0.1`，且 9200、6379、9010/9011、9081、1979、7670/7671/8265、8000–8017 等內部服務 port 未在 host listen。
- [ ] 從同一 Compose network 的一次性 test container 可連到每個必要內部 service；從 host published address 無法直接連到內部-only services。
- [ ] 每個必要 service 都有 healthcheck 與 restart policy；RAG/ingestor 的 dependency-aware health 只在其必要 dependencies ready 後成功。
- [ ] 停止並重建 application containers 後，既有 collection、已 ingest 文件、object data 與 model cache 仍存在，且可再次查詢。
- [ ] 經核准的 daemon/host restart 測試後，必要 containers 自動恢復為 healthy，不需手動依序啟動。
- [ ] 從另一台機器使用文件中的 SSH forwarding 後可開啟 frontend，並可呼叫三個 loopback API；未建立 tunnel 時遠端無法直接連線。
- [ ] lifecycle 文件不包含 secret，也不建議將 `.env` 或 API keys 複製到 shell history。

## 證據

附上 resolved port mapping、host sockets、內外網路探測、health/dependency 圖、volume 清單、restart 前後查詢，以及 SSH tunnel smoke test。若 host reboot 需要人工核准，記錄核准與實際測試時間。

## Comments
