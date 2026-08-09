# 單張 H100 Qwen Local RAG：FAE 操作手冊

本文件提供給不熟悉 RAG、VLM、NVIDIA NIM 或 Docker Compose 的 FAE，目標是讓操作人員能夠安全地完成部署、參數調整、啟停、測試與第一線故障排除。

本文件適用於本 repository 的實驗性 `qwen-h100-local-rag-tickets` 分支，不適用於 NVIDIA RAG Blueprint 的預設三 GPU 部署。已驗證的基準是：

- 1 張 NVIDIA H100 80GB，GPU ID `0`。
- Qwen `Qwen/Qwen3.6-27B-FP8`。
- vLLM context window `8192`，GPU memory utilization `0.48`。
- 所有生成、檢索與 ingestion GPU 服務同時常駐，不採用分時切換。
- 只有 `127.0.0.1:8090`、`:8081`、`:8082`、`:8999` 對 host 發布。

相關英文文件：

- [`qwen-h100-local-rag.md`](qwen-h100-local-rag.md)：canonical runbook。
- [`qwen-h100-manual-deployment.md`](qwen-h100-manual-deployment.md)：手動部署程序。
- [`qwen-hardware-revalidation.md`](qwen-hardware-revalidation.md)：更換硬體後的重驗程序。
- [`qwen-h100-api-endpoints.md`](qwen-h100-api-endpoints.md)：API endpoint reference。

## 1. 先理解系統中有哪些元件

使用者從 Web UI 提問時，請求會經過以下路徑：

```text
Web UI :8090
    |
    v
RAG Server :8081
    |-- Elasticsearch：文件與向量索引
    |-- SeaweedFS：圖片與物件儲存
    |-- VLM Embedding NIM：文字／圖片 embedding
    |-- VLM Reranker NIM：檢索結果重新排序
    `-- Shared Qwen：回答、VLM、query rewrite、filter、reflection、agentic

Ingestor :8082
    |-- NV-Ingest + Redis
    |-- OCR + page/graphic/table detectors
    |-- VLM Embedding NIM
    `-- Shared Qwen：圖片 caption 與摘要
```

共有 14 個必要服務。`rag-server` 顯示 healthy 只表示 API process 存活；完整檢查必須呼叫帶有 `check_dependencies=true` 的 health endpoint。

## 2. 三種容易混淆的 token 限制

### 2.1 模型 context window

`QWEN_MAX_MODEL_LEN` 是 vLLM 可接受的「輸入 tokens + 輸出 tokens」總上限。目前是 `8192`：

```text
input prompt + retrieved context + chat history + requested output <= 8192
```

這是 Qwen process 啟動時的限制，不能只靠 Web UI 改變。

### 2.2 最大輸出 tokens

`LLM_MAX_TOKENS` 與 `APP_VLM_MAX_TOKENS` 只限制最多可生成多少 tokens，不會自動替 retrieval context 保留空間。降低 output token 數不一定能修復 context overflow，因為 pipeline 可能用更多文件填滿剩餘空間。

### 2.3 放入 prompt 的 retrieval context

`APP_RETRIEVER_TOPK` 決定 reranker 後有多少文件送進生成階段。這個 8K profile 已實測：

- `APP_RETRIEVER_TOPK=5`：目前驗證文件會超過 8192 tokens。
- `APP_RETRIEVER_TOPK=4`：成功。

因此目前 profile 必須保留 `APP_RETRIEVER_TOPK=4`，除非更換模型 context、文件特性並重新完成測試。

## 3. 重要檔案與參數

Docker 部署的主要設定來源是 `deploy/compose/.env`。不要只在 shell 執行 `export`，因為重開機或重新登入後會遺失。

| 目的 | 變數 | 基準值 | 需重建的服務 |
| --- | --- | ---: | --- |
| Qwen input + output 總 context | `QWEN_MAX_MODEL_LEN` | `8192` | Qwen、RAG、Ingestor、NV-Ingest |
| vLLM 可使用的 GPU 比例 | `QWEN_GPU_MEMORY_UTILIZATION` | `0.48` | Qwen |
| 標準 RAG 最大輸出 | `LLM_MAX_TOKENS` | `2048` | RAG |
| VLM 最大輸出 | `APP_VLM_MAX_TOKENS` | `2048` | RAG |
| VLM 每次最多圖片 | `APP_VLM_MAX_TOTAL_IMAGES` | `2` | RAG |
| Reranker 後送進 prompt 的文件數 | `APP_RETRIEVER_TOPK` | `4` | RAG |
| 從 vector DB 取出、送往 reranker 的候選數 | `VECTOR_DB_TOPK` | `100` | RAG |
| 保留的對話歷史輪數 | `CONVERSATION_HISTORY` | `5` | RAG |
| Query decomposition 深度 | `MAX_RECURSION_DEPTH` | `3` | RAG |
| Reflection 次數 | `MAX_REFLECTION_LOOP` | `3` | RAG |
| Agentic context budget | `AGENTIC_CONTEXT_MAX_TOKENS` | `4096` | RAG |
| Agentic 各角色最大輸出 | `AGENTIC_*_LLM_MAX_TOKENS` | `1024` | RAG |
| 摘要輸入 chunk | `SUMMARY_LLM_MAX_CHUNK_LENGTH` | `6144` | Ingestor |

Compose override 位於 `deploy/compose/docker-compose-qwen-h100.yaml`。其中包含 Qwen 的 `--max-model-len`、loopback ports、healthcheck、restart policy，以及 8K profile 的 `APP_RETRIEVER_TOPK=4` 預設值。

`scripts/qwen_h100_local_rag.py` 是安全檢查器。它會拒絕偏離已驗證 profile 的設定，例如 context 不是 `8192` 或 `APP_RETRIEVER_TOPK` 不是 `4`。若要建立新的正式 profile，不能繞過 validator；必須同步更新 validator、文件與測試證據。

## 4. 第一次部署

### 4.1 Host prerequisites

- H100 80GB 可被 Docker 看到。
- NVIDIA driver 560 或更新版本。
- Docker 與支援 Compose `!override` tag 的 Docker Compose。
- NVIDIA Container Toolkit。
- 至少 200GB 可用磁碟空間。
- NGC API key 可拉取 NVIDIA NIM images。
- Repository support matrix 指定 Ubuntu 22.04；其他版本應標記為未正式支援的實驗環境。

檢查：

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
docker --version
docker compose version
docker info | grep -i 'runtimes.*nvidia'
df -h .
cat /etc/os-release
```

### 4.2 Clone 與切換分支

```bash
git clone https://github.com/kzger/rag.git
cd rag
git switch qwen-h100-local-rag-tickets
```

### 4.3 設定 NGC key

`deploy/compose/.env` 必須包含本機 key，但 key 絕對不能 commit。若 key 已在 `~/.bashrc`：

```bash
bash -ic 'set +x; source ~/.bashrc >/dev/null 2>&1; test -n "${NGC_API_KEY:-}"; sed -i "/^export NGC_API_KEY=/d" deploy/compose/.env; sed -i "1iexport NGC_API_KEY=${NGC_API_KEY}" deploy/compose/.env' >/dev/null 2>&1
```

只驗證是否存在，不顯示內容：

```bash
bash -c 'set -a; source deploy/compose/.env; set +a; test -n "${NGC_API_KEY:-}"'
```

登入 NGC：

```bash
source deploy/compose/.env
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

安全規則：

- 不要執行 `git add deploy/compose/.env`。
- 不要把完整 `docker inspect` environment 或完整 resolved Compose config 貼到 ticket。
- 不要在 terminal 執行 `echo "$NGC_API_KEY"`。
- 若 key 曾出現在 logs、chat 或 CI artifact，立即 rotate。

### 4.4 Validate 與啟動

```bash
scripts/qwen_h100_local_rag.sh validate
scripts/qwen_h100_local_rag.sh config --services
scripts/start_qwen_h100_local_rag.sh
```

啟動 script 會執行 validate、pull、`up -d`，並等待四個 public endpoints。第一次下載 model 與 images 通常需要 15–30 分鐘。

已有完整 image/cache 時可跳過 pull：

```bash
QWEN_SKIP_PULL=1 scripts/start_qwen_h100_local_rag.sh
```

不要同時啟動 Blueprint 預設的 `nim-llm`、獨立 VLM 或 captioning model；本 profile 使用同一個 Qwen endpoint 承擔所有生成角色。

## 5. 日常啟動、停止與重啟

### 5.1 啟動

```bash
scripts/start_qwen_h100_local_rag.sh
```

### 5.2 停止但保留 containers、資料與 cache

```bash
scripts/stop_qwen_h100_local_rag.sh
```

### 5.3 移除 containers 與 Compose network，但保留 named volumes

```bash
scripts/stop_qwen_h100_local_rag.sh --down
```

### 5.4 重啟服務

```bash
scripts/qwen_h100_local_rag.sh restart
```

`down` 不會加上 `--volumes`。除非客戶已明確同意刪除 ingestion data、Elasticsearch indices、SeaweedFS objects 與 model cache，否則禁止使用 `down -v` 或手動刪除 `rag-vol-*` volumes。

## 6. 查看狀態與 logs

### 6.1 Container 狀態

```bash
scripts/qwen_h100_local_rag.sh ps
```

預期 14 個必要服務都是 `healthy`。

### 6.2 所有服務 logs

```bash
scripts/qwen_h100_local_rag.sh logs --follow --tail 100
```

### 6.3 指定服務

```bash
scripts/qwen_h100_local_rag.sh logs --follow --tail 200 rag-server
scripts/qwen_h100_local_rag.sh logs --since 10m rag-server qwen-vllm
scripts/qwen_h100_local_rag.sh logs --since 10m ingestor-server nv-ingest-ms-runtime
```

也可以直接使用 container name：

```bash
docker logs --follow --tail 200 rag-server
```

按 `Ctrl-C` 只會停止追蹤 logs，不會停止 container。

常用錯誤過濾：

```bash
scripts/qwen_h100_local_rag.sh logs --since 10m rag-server qwen-vllm 2>&1 \
  | grep -iE 'error|exception|traceback|timeout|context length|oom'
```

## 7. 參數調整標準程序

每次只改一類參數，並記錄修改前後值。不要同時改 context、GPU utilization、top-k 與 model variant，否則失敗時無法判斷原因。

### 7.1 修改 RAG／VLM output 或 retrieval context

在 `deploy/compose/.env` 調整，例如：

```bash
export LLM_MAX_TOKENS=2048
export APP_VLM_MAX_TOKENS=2048
export APP_VLM_MAX_TOTAL_IMAGES=2
export APP_RETRIEVER_TOPK=4
```

Validate 並只重建 RAG server：

```bash
scripts/qwen_h100_local_rag.sh validate
scripts/qwen_h100_local_rag.sh up --force-recreate rag-server
QWEN_SKIP_PULL=1 QWEN_START_TIMEOUT_SECONDS=180 \
  scripts/start_qwen_h100_local_rag.sh
```

### 7.2 修改 summary 參數

```bash
scripts/qwen_h100_local_rag.sh up --force-recreate ingestor-server
```

### 7.3 修改真正的 Qwen context window

在 `deploy/compose/.env` 修改：

```bash
export QWEN_MAX_MODEL_LEN=8192
```

這個值會傳給 vLLM `--max-model-len`。修改後必須重建 Qwen 與所有使用它的服務：

目前 validator 明確只接受 `8192`。要增加為 `12288`、`16384` 或其他值，必須先由工程人員建立新 profile，同步修改 `scripts/qwen_h100_local_rag.py` 的預期值並完成本文件第 8、9 節的測試。新 profile 通過 validator 後，才能執行以下重建指令：

```bash
scripts/qwen_h100_local_rag.sh up --force-recreate \
  qwen-vllm rag-server ingestor-server nv-ingest-ms-runtime
```

不要為了通過啟動而直接停用 validator。

較大的 context 會增加 KV cache 壓力。可能結果包括：

- vLLM 啟動時判定 KV cache 不足。
- Qwen container unhealthy 或 restart。
- 其他同卡 NIM 發生 OOM。
- Full-Service Co-residency 失敗。

### 7.4 修改後確認 server 實際值

```bash
curl -fsS http://127.0.0.1:8081/v1/configuration | jq .
docker inspect --format '{{json .Config.Cmd}}' compose-qwen-vllm-1 | jq .
```

不要用 `docker inspect ... .Config.Env` 分享輸出，因為 environment 可能包含 credentials。

## 8. 基本測試程序

### 8.1 Compose contract

```bash
scripts/qwen_h100_local_rag.sh validate
QWEN_VARIANT=nvfp4 scripts/qwen_h100_local_rag.sh validate
```

第二個指令只驗證 fallback config，不會同時啟動 NVFP4。

### 8.2 Unit tests

```bash
uvx --from pytest pytest \
  tests/unit/test_deploy/test_qwen_h100_local_rag.py -q
```

### 8.3 Health tests

```bash
curl -fsS http://127.0.0.1:8999/v1/models | jq .
curl -fsS 'http://127.0.0.1:8081/v1/health?check_dependencies=true' | jq .
curl -fsS 'http://127.0.0.1:8082/v1/health?check_dependencies=true' | jq .
curl -fsS http://127.0.0.1:8090/ >/dev/null
```

RAG 與 Ingestor health JSON 中的 database、object storage、NIM、processing 與 task management 項目都應為 healthy。

### 8.4 Shared Qwen smoke test

```bash
curl -fsS -X POST http://127.0.0.1:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"請用一句話說明 RAG。"}],"max_tokens":64,"temperature":0}' \
  | jq -r '.choices[0].message.content'
```

### 8.5 建立 collection 與 multimodal ingestion

```bash
collection=fae_validation_$(date +%Y%m%d_%H%M%S)

curl -fsS -X POST http://127.0.0.1:8082/v1/collection \
  -H 'Content-Type: application/json' \
  --data "{\"collection_name\":\"$collection\",\"embedding_dimension\":2048,\"metadata_schema\":[]}"

curl -fsS -X POST http://127.0.0.1:8082/v1/documents \
  -F 'documents=@data/multimodal/functional_validation.pdf;type=application/pdf' \
  -F "data={\"collection_name\":\"$collection\",\"blocking\":true,\"generate_summary\":true};type=application/json" \
  | jq .
```

預期 `failed_documents` 為空，並且結果包含 text、table 與 image elements。

### 8.6 Retrieval smoke test

```bash
curl -fsS -X POST \
  "http://127.0.0.1:8081/v2/vector_stores/$collection/search" \
  -H 'Content-Type: application/json' \
  --data '{"query":"文件中有哪些圖片與表格？","max_num_results":6}' \
  | jq .
```

### 8.7 Web UI

在 host 本機開啟 `http://127.0.0.1:8090`。若從另一台電腦使用，先建立 SSH tunnel：

```bash
ssh -N \
  -L 8090:127.0.0.1:8090 \
  -L 8081:127.0.0.1:8081 \
  -L 8082:127.0.0.1:8082 \
  -L 8999:127.0.0.1:8999 \
  USER@RAG_HOST
```

在 client 瀏覽 `http://127.0.0.1:8090`。選取剛建立的 collection 後提問。若瀏覽器曾保存舊設定，確認 Settings 中的 Reranker Top K 不大於 `4`。

## 9. 五分鐘 Full-Service Co-residency 驗收

這是 deployment acceptance gate。功能正確率、latency 與 throughput 需記錄，但不是這個 gate 的額外條件。

先在 shell A 啟動 monitor：

```bash
scripts/qwen_h100_local_rag.sh monitor \
  docs/research/evidence/qwen-fae-observations.jsonl
```

Monitor 開始後，在 shell B 持續送出 ingestion、Qwen generation 與 RAG search 流量。必須涵蓋 idle 與 active samples，不能先跑完流量再開始 monitor。

獨立重新判讀結果：

```bash
uv run python scripts/qwen_h100_local_rag.py evaluate-observations \
  docs/research/evidence/qwen-fae-observations.jsonl \
  --minimum-duration 300
```

通過條件：

- 觀測時間至少 300 秒。
- 14 個必要服務全程 healthy。
- 每個 container 的 restart count 前後不變。

若基準 FP8 無法通過，依序測試：

1. FP8、context 8192、GPU utilization 0.48。
2. FP8、GPU utilization 0.45。
3. FP8、utilization 0.45、context 4096。
4. Pinned NVFP4 fallback。

每次只啟動一個 Qwen variant，且每個 profile 都要重新跑完整五分鐘。禁止以分開的 ingestion/query 時段宣稱 Full-Service Co-residency 成功。

## 10. 常見問題與處理方式

### 10.1 Web UI 顯示 `Error from rag-server`

先查看：

```bash
scripts/qwen_h100_local_rag.sh logs --since 10m rag-server qwen-vllm
```

若看到：

```text
maximum context length is 8192 tokens
```

確認：

```bash
curl -fsS http://127.0.0.1:8081/v1/configuration \
  | jq '.rag_configuration | {max_tokens,reranker_top_k,vdb_top_k}'
```

本 profile 的 `reranker_top_k` 應為 `4`。若 Web UI 保存了 `10`，在 Settings 改成 `4`、重新整理並建立新對話。不要只降低 `max_tokens`；retrieval pipeline 可能用更多 context 填滿剩餘空間。

### 10.2 Qwen unhealthy 或 OOM

```bash
nvidia-smi
scripts/qwen_h100_local_rag.sh logs --tail 300 qwen-vllm
docker stats --no-stream
```

確認沒有額外 GPU process，也沒有同時啟動預設 LLM/VLM profiles。依 fallback ladder 調整，不要任意停止 OCR、detectors、embedding 或 reranker 來騰出空間。

### 10.3 RAG health 失敗

```bash
curl -sS 'http://127.0.0.1:8081/v1/health?check_dependencies=true' | jq .
```

從 JSON 找出第一個 unhealthy dependency，再查看該服務 logs。不要一開始就重建全部 containers，否則會失去故障現場。

### 10.4 Ingestion 失敗

```bash
scripts/qwen_h100_local_rag.sh logs --since 10m \
  ingestor-server nv-ingest-ms-runtime redis nemotron-ocr \
  page-elements graphic-elements table-structure
```

確認文件小於 400MB、Redis/NV-Ingest healthy、OCR 與 detectors 可用、磁碟未滿。

### 10.5 Port 衝突或無法遠端連線

```bash
ss -ltnp | grep -E ':(8090|8081|8082|8999) '
```

四個 ports 都必須只綁定 `127.0.0.1`。遠端 client 必須使用 SSH tunnel；不要把 Compose port 改成 `0.0.0.0`。

### 10.6 Image pull authentication 失敗

重新確認 NGC key 是否存在並執行 `docker login nvcr.io`。禁止把 key 貼到 ticket 或 logs。

## 11. 回復修改

若參數修改後失敗：

1. 將 `deploy/compose/.env` 的變數恢復為本文件基準值。
2. 執行 `scripts/qwen_h100_local_rag.sh validate`。
3. 只重建受影響服務。
4. 重新執行 health、API smoke tests 與五分鐘 gate。
5. 保存失敗 profile 的硬體、設定、logs 與 observations，不要覆寫先前成功證據。

如果 Compose 或 tracked 程式檔也被修改，先使用以下指令查看差異，不要直接使用 `git reset --hard`：

```bash
git status --short
git diff -- deploy/compose/docker-compose-qwen-h100.yaml \
  scripts/qwen_h100_local_rag.py
```

## 12. FAE 交付紀錄清單

每次部署或硬體重驗至少記錄：

- Git commit 與 branch。
- GPU model、數量、UUID、VRAM、driver 與 OS。
- Docker／Compose 版本與可用磁碟。
- Qwen model ID、immutable revision、variant 與 context。
- 14 個 images 的 pullable RepoDigests。
- `QWEN_GPU_MEMORY_UTILIZATION`、`APP_RETRIEVER_TOPK` 等 profile values。
- Health 結果與 before/after restart counts。
- 五分鐘 observation JSONL。
- Ingestion、generation、search 的 timestamped evidence。
- Peak used VRAM 與 minimum free VRAM。
- 功能結果、TTFT、tokens/s 與已知限制。
- 未完成的 daemon restart、host restart 或 off-host SSH checks。

證據不得包含 NGC key、API keys、password、authorization headers 或完整 container environment。
