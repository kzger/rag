# 單張 H100 Qwen Local RAG：FAE 操作手冊

本文件提供給不熟悉 RAG、VLM、NVIDIA NIM 或 Docker Compose 的 FAE，目標是讓操作人員能夠安全地完成部署、參數調整、啟停、測試與第一線故障排除。

本文件適用於本 repository 的實驗性 `qwen-h100-local-rag-tickets` 分支，不適用於 NVIDIA RAG Blueprint 的預設三 GPU 部署。已驗證的基準是：

- 1 張 NVIDIA H100 80GB，GPU ID `0`。
- Qwen `Qwen/Qwen3.6-27B-FP8`。
- vLLM context window `32768`，GPU memory utilization `0.48`。
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

### 1.1 Port 與服務內容對照

以下是 Qwen H100 profile 的實際 port contract。Host 只發布四個 loopback port；
`127.0.0.1` 表示只能從主機本身或 SSH tunnel 存取，不應改成 `0.0.0.0`。

| Host port | Container port | Service | 內容／用途 |
| ---: | ---: | --- | --- |
| `127.0.0.1:8090` | `rag-frontend:3000` | Web UI | 瀏覽器操作介面 |
| `127.0.0.1:8081` | `rag-server:8081` | RAG API | Health、configuration、search、generate、summary、metrics |
| `127.0.0.1:8082` | `ingestor-server:8082` | Ingestor API | Collection、document upload、ingestion status 與 data catalog |
| `127.0.0.1:8999` | `qwen-vllm:8000` | Shared Qwen OpenAI API | Model discovery、text/image chat completion；同時供 LLM、VLM、caption、summary、rewrite、reflection 與 agentic 使用 |

其餘 port 只存在於 Compose `nvidia-rag` 網路。表中的 HTTP、gRPC 與 metrics
port 即使同屬一個 NIM，也分列為個別項目，方便依錯誤訊息定位。

| Internal port／endpoint | Service | 內容／用途 | 主要使用者 |
| --- | --- | --- | --- |
| `nemotron-vlm-embedding-ms:8000` | VLM Embedding NIM | HTTP embedding API，處理文字與圖片向量 | RAG、Ingestor |
| `nemotron-ranking-vl-ms:8000` | VLM Reranker NIM | HTTP multimodal reranking API | RAG |
| `nv-ingest-ms-runtime:7670` | NV-Ingest | HTTP API、readiness 與 ingestion pipeline 入口 | Ingestor |
| `nv-ingest-ms-runtime:7671` | NV-Ingest | Simple Broker port；本 profile 使用 Redis broker，通常不直接呼叫 | NV-Ingest internal |
| `nv-ingest-ms-runtime:8265` | NV-Ingest | Ray dashboard／runtime diagnostics | 維運診斷 |
| `page-elements:8000` | Page Elements NIM | HTTP page-element inference | Ingestor、NV-Ingest |
| `page-elements:8001` | Page Elements NIM | gRPC page-element inference | NV-Ingest |
| `page-elements:8002` | Page Elements NIM | NIM metrics | 維運診斷 |
| `graphic-elements:8000` | Graphic Elements NIM | HTTP graphic-element inference | Ingestor、NV-Ingest |
| `graphic-elements:8001` | Graphic Elements NIM | gRPC graphic-element inference | NV-Ingest |
| `graphic-elements:8002` | Graphic Elements NIM | NIM metrics | 維運診斷 |
| `table-structure:8000` | Table Structure NIM | HTTP table-structure inference | Ingestor、NV-Ingest |
| `table-structure:8001` | Table Structure NIM | gRPC table-structure inference | NV-Ingest |
| `table-structure:8002` | Table Structure NIM | NIM metrics | 維運診斷 |
| `nemotron-ocr:8000` | Nemotron OCR NIM | HTTP OCR inference | Ingestor、NV-Ingest |
| `nemotron-ocr:8001` | Nemotron OCR NIM | gRPC OCR inference | NV-Ingest |
| `nemotron-ocr:8002` | Nemotron OCR NIM | NIM metrics | 維運診斷 |
| `elasticsearch:9200` | Elasticsearch | REST API、向量與文件索引 | RAG、Ingestor |
| `redis:6379` | Redis | Ingestion message broker、task 與 summary status | NV-Ingest、Ingestor、RAG |
| `seaweedfs:9010` | SeaweedFS | S3-compatible object API，保存圖片與 multimodal objects | RAG、Ingestor |
| `seaweedfs:9011` | SeaweedFS | SeaweedFS auxiliary internal port；application data path 使用 `9010` | SeaweedFS internal |

Internal-only port 不可從 host 直接 `curl localhost:<port>` 判定健康，也不要為了
除錯臨時加入 host mapping。應使用 `scripts/qwen_h100_local_rag.sh logs`、dependency
health endpoint，或從同一 Compose network 內的 container 呼叫。完整 API path
請參考 [`qwen-h100-api-endpoints.md`](qwen-h100-api-endpoints.md)。

## 2. 三種容易混淆的 token 限制

### 2.1 模型 context window

`QWEN_MAX_MODEL_LEN` 是 vLLM 可接受的「輸入 tokens + 輸出 tokens」總上限。目前是 `32768`：

```text
input prompt + retrieved context + chat history + requested output <= 32768
```

這是 Qwen process 啟動時的限制，不能只靠 Web UI 改變。本 profile 的
LLM、VLM、reflection、query rewrite、captioning 與 agentic 角色共用同一個
Qwen endpoint，因此無法只替 VLM 設定不同的 context window。

### 2.2 最大輸出 tokens

`LLM_MAX_TOKENS` 與 `APP_VLM_MAX_TOKENS` 只限制最多可生成多少 tokens，目前
兩者都是 `8192`；它們不會把模型 context window 從 32768 降成 8192，也不會
自動替 retrieval context 保留空間。降低 output token 數不一定能修復 context
overflow，因為 pipeline 可能用更多文件填滿剩餘空間。

### 2.3 放入 prompt 的 retrieval context

retrieval 分成兩層 Top-K：`VECTOR_DB_TOPK` 決定先從 vector DB 取出的候選數，
`APP_RETRIEVER_TOPK` 決定 reranker 後實際送進生成 prompt 的文件數。目前
`deploy/compose/.env` 設為 `5`，即 `100 -> 5`；Compose override 的 fallback 仍是
`4`，只在未載入 canonical `.env` 時生效。先前的 8K profile 曾實測：

- `APP_RETRIEVER_TOPK=5`：目前驗證文件會超過 8192 tokens。
- `APP_RETRIEVER_TOPK=4`：成功。

該限制是針對 8K profile；32K profile 的 context 較寬裕，因此 `.env` 現行值為 `5`，
而 Compose fallback 保留 `4` 作為未載入 `.env` 時的保守上限。要再往上調，仍須重新
測試文件特性、GPU 記憶體與完整驗收。

## 3. 重要檔案與參數

Docker 部署的主要設定來源是 `deploy/compose/.env`。不要只在 shell 執行 `export`，因為重開機或重新登入後會遺失。

| 目的 | 變數 | 目前 `.env` 值 | 需重建的服務 |
| --- | --- | ---: | --- |
| Qwen input + output 總 context | `QWEN_MAX_MODEL_LEN` | `32768` | Qwen、RAG、Ingestor、NV-Ingest |
| vLLM 可使用的 GPU 比例 | `QWEN_GPU_MEMORY_UTILIZATION` | `0.48` | Qwen |
| 標準 RAG 最大輸出 | `LLM_MAX_TOKENS` | `8192` | RAG |
| VLM 最大輸出 | `APP_VLM_MAX_TOKENS` | `8192` | RAG |
| VLM 每次最多圖片 | `APP_VLM_MAX_TOTAL_IMAGES` | `3` | RAG |
| Reranker 後送進 prompt 的文件數 | `APP_RETRIEVER_TOPK` | `5` | RAG |
| 從 vector DB 取出、送往 reranker 的候選數 | `VECTOR_DB_TOPK` | `100` | RAG |
| 保留的對話歷史輪數 | `CONVERSATION_HISTORY` | `5` | RAG |
| Query decomposition 深度 | `MAX_RECURSION_DEPTH` | `3` | RAG |
| Reflection 次數 | `MAX_REFLECTION_LOOP` | `3` | RAG |
| Agentic context budget | `AGENTIC_CONTEXT_MAX_TOKENS` | `4096` | RAG |
| Agentic 各角色最大輸出 | `AGENTIC_*_LLM_MAX_TOKENS` | `1024` | RAG |
| 摘要輸入 chunk | `SUMMARY_LLM_MAX_CHUNK_LENGTH` | `6144` | Ingestor |
| 動態 metadata 過濾器 | `ENABLE_FILTER_GENERATOR` | `false` | RAG |
| Agentic 多步推理（預設關閉，可 per-request 開啟） | `ENABLE_AGENTIC_RAG` | `false` | RAG |

`ENABLE_FILTER_GENERATOR` 與 `ENABLE_AGENTIC_RAG` 兩者的 repo 與 Helm 預設都是
`false`，且應維持關閉：前者會產生對不上實際欄位值的過濾條件而導致檢索回 0 筆，
後者在單張 H100 上會讓同一個問題從約 18 秒變成約 52 秒。詳見第 10.7 節的實測數據。

兩個 Top-K 都必須明確寫在 `deploy/compose/.env`，讓該檔保持唯一的本機設定
來源。Compose override 位於 `deploy/compose/docker-compose-qwen-h100.yaml`；
其中包含 Qwen 的 `--max-model-len`、loopback ports、healthcheck、restart policy，
並保留 `APP_RETRIEVER_TOPK=4` fallback，僅用來防止未載入 canonical `.env` 的
直接 Compose 呼叫失去保守上限。

`scripts/qwen_h100_local_rag.py` 會分開處理一般合法性與 certified conformity。
一般 `validate` 依型別、範圍、跨欄位關係及安全拓撲檢查設定；合法的 custom
context、Top-K 或 workload budget 可以啟動，但摘要會標記為 `custom`。執行
`validate --certified` 才會要求完全符合 FP8 或 NVFP4 實測基準，並把每個偏移
列為 profile drift。建立新的正式 profile 仍須同步更新文件、測試及 live evidence。

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
export LLM_MAX_TOKENS=8192
export APP_VLM_MAX_TOKENS=8192
export APP_VLM_MAX_TOTAL_IMAGES=2
export VECTOR_DB_TOPK=100
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
export QWEN_MAX_MODEL_LEN=32768
```

這個值會傳給 vLLM `--max-model-len`。修改後必須重建 Qwen 與所有使用它的服務：

一般 validator 接受 `1024` 到 `131072` 的整數，但所有 generation、agentic
與 summary token budgets 都不得超過它。偏離 `32768` 後仍可作為 custom
configuration 啟動；若要宣稱為 certified profile，必須完成本文件第 8、9 節
的測試並補上對應 evidence。修改後可執行以下重建指令：

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
scripts/qwen_h100_local_rag.sh validate --certified
QWEN_VARIANT=nvfp4 scripts/qwen_h100_local_rag.sh validate --certified
```

最後一個指令只驗證 NVFP4 fallback config，不會同時啟動 NVFP4。

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

### 8.8 使用圖片查詢知識庫

圖片查詢的目的不是單純「描述圖片」，而是以圖片與文字一起搜尋已選取的
collection，再由 VLM 根據檢索內容回答。操作前必須先完成第 8.5 節的
multimodal ingestion。

Web UI 操作：

1. 開啟 `http://127.0.0.1:8090`，在側欄選取已完成 ingestion 的 collection。
2. 建立新對話，確認 Knowledge Base 與 VLM 已啟用。
3. 按輸入框的附件按鈕，上傳一張 PNG 或 JPEG。
4. 在同一則訊息輸入問題，例如「這張圖片中的商品，知識庫有哪些相似資料？」。
5. 送出後查看答案與 citations；需要顯示知識庫圖片時，展開 citation 詳細資料。

此分支會自動讓圖片 query 略過文字 reflection 與文字 reranker，即使全域設定
開啟這兩項功能也不會把 base64 圖片誤當成文字 token。圖片仍會送往 VLM
embedding 做多模態檢索，並送往共享 Qwen 做 VLM generation。這個 routing
規則是修復 `Reflection LLM NIM unavailable` 假象的必要條件；加大 context
window 本身不能修復 base64 被送進文字模型的問題。

API 操作範例；請將 collection 與圖片路徑換成實際值：

```bash
collection=YOUR_COLLECTION
image_path=data/multimodal/Creme_clutch_purse1-small.jpg

uv run python - "$image_path" "$collection" >/tmp/image-query.json <<'PY'
import base64
import json
import sys
from pathlib import Path

image_path, collection = sys.argv[1:]
image = base64.b64encode(Path(image_path).read_bytes()).decode()
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "這張圖片中的物品，知識庫有哪些相似資料？"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    "detail": "auto",
                },
            },
        ],
    }],
    "use_knowledge_base": True,
    "collection_names": [collection],
    "enable_reranker": True,
    "enable_citations": True,
    "enable_vlm_inference": True,
    "enable_filter_generator": False,
    "agentic": False,
    "reranker_top_k": 4,
}
print(json.dumps(payload))
PY

curl -N -X POST http://127.0.0.1:8081/v1/generate \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/image-query.json
```

`/tmp/image-query.json` 含有原始圖片的 base64；測試後應刪除，且不可提交到 Git。
不要把完整 base64 放在 shell variable 後再傳給 `jq --arg`；一般圖片可能超過
作業系統的 command-line `ARG_MAX`，導致 `Argument list too long`。

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

1. FP8、context 32768、GPU utilization 0.48。
2. FP8、context 8192、GPU utilization 0.48。
3. FP8、context 8192、GPU utilization 0.45。
4. FP8、utilization 0.45、context 4096。
5. Pinned NVFP4 fallback。

每次只啟動一個 Qwen variant，且每個 profile 都要重新跑完整五分鐘。禁止以分開的 ingestion/query 時段宣稱 Full-Service Co-residency 成功。

## 10. 常見問題與處理方式

### 10.1 Web UI 顯示 `Error from rag-server`

先查看：

```bash
scripts/qwen_h100_local_rag.sh logs --since 10m rag-server qwen-vllm
```

若看到：

```text
maximum context length is ... tokens
```

確認：

```bash
curl -fsS http://127.0.0.1:8081/v1/configuration \
  | jq '.rag_configuration | {max_tokens,reranker_top_k,vdb_top_k}'
```

本 profile 的 Qwen 上限應為 `32768`、`vdb_top_k` 應為 `100`，且
`reranker_top_k` 應為 `4`。API/UI request 可以覆寫這兩個 Top-K；因此即使
container 的環境預設正確，Web UI 若保存了 `10`，仍須在 Settings 改成 `4`、
重新整理並建立新對話。不要只降低 `max_tokens`；retrieval pipeline 可能用更多
context 填滿剩餘空間。

若只有「上傳圖片 + query」失敗，而且訊息是 `Reflection LLM NIM unavailable`，
先確認服務已使用本分支最新的本機 build：

```bash
scripts/qwen_h100_local_rag.sh up --build --force-recreate rag-server
scripts/qwen_h100_local_rag.sh logs --since 5m rag-server qwen-vllm
```

修正版 log 應出現 `Skipping text reflection for image`，且不應再出現 reranker
的 `Input length ... exceeds maximum allowed token size`。若仍有舊錯誤，通常是
container 仍在使用先前的 prebuilt image，而不是 context window 太小。

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

### 10.7 查詢很慢（單題 30 秒以上）

先確認這兩個旗標。它們在 repo 與 Helm 的預設都是 `false`，若被 `.env` 開成 `true`，
單題會從約 18 秒變成約 52 秒。

```bash
docker exec rag-server printenv ENABLE_FILTER_GENERATOR ENABLE_AGENTIC_RAG
```

**`ENABLE_FILTER_GENERATOR`：預設 `false`，建議維持關閉。**

它會先叫一次 LLM，從使用者的問題「推導」出 metadata 過濾條件。問題是這個推導沒有
對照實際存在的欄位值。實測「提供 J7EF Plus 的詳細規格」時，它產生了：

```json
[{"term":{"metadata.content_metadata.filename.keyword":"J7EF Plus"}}]
```

但知識庫裡的檔名是 `JORJIN TECHNOLOGIES-J7EF PULS 產品規格_v3.pdf`，完全對不上，
於是**檢索回 0 筆**。接著 self-reflection 會判定 context 不相關、改寫查詢、再檢索——
但它改寫的是「查詢」，修不了「過濾器」，所以 `MAX_REFLECTION_LOOP` 三輪全部白跑。

可用以下方式確認是否踩到：

```bash
scripts/qwen_h100_local_rag.sh logs --since 5m rag-server | grep -E "Dynamic filter generated|Retrieved 0 documents"
```

看到 `Retrieved 0 documents` 緊接在 `Dynamic filter generated` 之後，就是這個原因。

**`ENABLE_AGENTIC_RAG`：預設 `false`，建議維持關閉，需要時改用 per-request 開啟。**

Agentic 會把問題拆成多個子任務，各自檢索與推理。實測同一個問題：

| 模式 | 耗時 | 答案 |
| --- | --- | --- |
| Agentic | 52.1 秒 | 完整 |
| 非 Agentic | 17.7 秒 | 一樣完整，另附 8 筆引用 |

在單張 H100 上所有子任務共用同一個 Qwen，且 `AGENTIC_CONCURRENCY_LIMIT=1` 讓它們
只能排隊執行，所以成本是線性疊加的。需要多步推理時，在單一請求裡指定即可：

```json
{"messages": [...], "agentic": true}
```

兩者都關閉後，同一題從 52.1 秒降到約 18 秒。

**`ENABLE_QUERY_DECOMPOSITION`：預設 `false`，建議維持關閉。**

它會把一個問題拆成多個子問題，每個子問題各跑一次 LLM 與一次完整檢索。實測追問
「介紹這個產品」：

| 設定 | 耗時 | 子問題 | 答案長度 |
| --- | --- | --- | --- |
| `true` | 30.5 秒 | 3 個 | 782 字 |
| `false` | 8.5 秒 | 0 | 937 字 |

拆解不但沒有換到更好的答案，還會**擴大提問範圍**。實際 log 中，使用者只問「介紹這個
產品」，拆解卻產生了 J9／J7M／J7EF Plus 三款比較、上市時間、售價與購買渠道、競品差異
等子問題，全都不是使用者問的，而每一個都要一次 LLM 呼叫加一次整庫檢索。

注意：**`ENABLE_QUERYREWRITER` 要保留開啟**。代名詞（「這個產品」）是由 query rewriter
解析的，實測它正確改寫成「請介紹 JReality J7EF Plus …」；真正造成延遲的是後面的拆解。
`CONVERSATION_HISTORY`（本部署為 `5`，預設 `0`）同樣必須保留，否則追問無從解析。

三者都關閉後，圖片查詢約 8 秒、文字追問約 8.5 秒；未關閉前分別是 8 秒與 64 秒。


### 10.8 不需要新增「提早返回」機制

若懷疑 self-reflection 一直在空轉，先確認它是不是拿不到文件。跳出機制本來就存在：
`ReflectionCounter` 在 relevance 分數達標時就會結束迴圈，實測一旦檢索拿到文件，
分數立刻是 2 並跳出。所以看到迴圈跑滿三輪時，要查的是**檢索為什麼回 0 筆**
（通常就是上面的假過濾器），而不是去加提早返回的邏輯。

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
