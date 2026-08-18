# Qwen H100 API endpoints

The deployment is SSH-only. The following host endpoints bind to loopback and
are reachable remotely only through the documented SSH tunnel.

## Published endpoints

| Service | Base URL | Method and path | Purpose |
| --- | --- | --- | --- |
| Frontend | `http://127.0.0.1:8090` | `GET /` | Browser UI |
| RAG server | `http://127.0.0.1:8081` | `GET /v1/health?check_dependencies=true` | Full dependency health |
| RAG server | `http://127.0.0.1:8081` | `GET /v1/configuration` | Effective public configuration |
| RAG server | `http://127.0.0.1:8081` | `POST /v1/generate` | Standard, advanced, multimodal, or Agentic generation |
| RAG server | `http://127.0.0.1:8081` | `POST /v1/search` | Blueprint retrieval API |
| RAG server | `http://127.0.0.1:8081` | `POST /v2/vector_stores/{collection}/search` | OpenAI-compatible vector-store search |
| RAG server | `http://127.0.0.1:8081` | `GET /v1/summary` | Collection/document summary retrieval |
| RAG server | `http://127.0.0.1:8081` | `GET /v1/metrics` | Application metrics |
| Ingestor | `http://127.0.0.1:8082` | `GET /v1/health?check_dependencies=true` | Full dependency health |
| Ingestor | `http://127.0.0.1:8082` | `POST /v1/collection` | Create one collection |
| Ingestor | `http://127.0.0.1:8082` | `GET/POST/DELETE /v1/collections` | List, create, or delete collections |
| Ingestor | `http://127.0.0.1:8082` | `GET/POST/PATCH/DELETE /v1/documents` | List, upload, update, or delete documents |
| Ingestor | `http://127.0.0.1:8082` | `GET /v1/status?task_id={task_id}` | Poll asynchronous ingestion |
| Shared Qwen | `http://127.0.0.1:8999` | `GET /v1/models` | Served-model discovery/readiness |
| Shared Qwen | `http://127.0.0.1:8999` | `POST /v1/chat/completions` | OpenAI-compatible text/image generation |

Exact RAG and Ingestor request/response schemas are committed in
`docs/api_reference/openapi_schema_rag_server.json` and
`docs/api_reference/openapi_schema_ingestor_server.json`. Streaming generation
may include `reasoning_content`; Agentic streams additionally include
`event_type` and `stage`, with final answer text in `content`.

## Common requests

Create a collection:

```bash
curl -fsS -X POST http://127.0.0.1:8082/v1/collection \
  -H 'Content-Type: application/json' \
  --data '{"collection_name":"manual_demo","embedding_dimension":2048,"metadata_schema":[]}'
```

Upload one PDF synchronously with summary generation:

```bash
curl -fsS -X POST http://127.0.0.1:8082/v1/documents \
  -F 'documents=@data/multimodal/functional_validation.pdf;type=application/pdf' \
  -F 'data={"collection_name":"manual_demo","blocking":true,"generate_summary":true};type=application/json'
```

For a non-blocking upload, set `blocking` to `false`, retain the returned
`task_id`, and poll:

```bash
curl -fsS 'http://127.0.0.1:8082/v1/status?task_id=TASK_ID'
```

Search the collection:

```bash
curl -fsS -X POST \
  http://127.0.0.1:8081/v2/vector_stores/manual_demo/search \
  -H 'Content-Type: application/json' \
  --data '{"query":"What visual and tabular content appears in the document?","max_num_results":6}'
```

Call the shared Qwen endpoint directly:

```bash
curl -fsS -X POST http://127.0.0.1:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Summarize the role of OCR in RAG."}],"max_tokens":128,"temperature":0}'
```

The RAG server's complete `/v1/generate` payload includes pipeline-specific
fields; generate clients from the committed OpenAPI schema rather than copying
a partial payload into production code.

## Internal-only endpoints

These services are intentionally not published on the host. Application
containers reach them by Compose DNS on the `nvidia-rag` network.

| Service | Internal endpoint |
| --- | --- |
| Qwen | `http://qwen-vllm:8000/v1` |
| VLM embedding | `http://nemotron-vlm-embedding-ms:8000/v1` |
| VLM reranking | `http://nemotron-ranking-vl-ms:8000` |
| Nemotron OCR | HTTP `http://nemotron-ocr:8000/v1/infer`, gRPC `nemotron-ocr:8001` |
| Page detector | gRPC `page-elements:8001` |
| Graphic detector | gRPC `graphic-elements:8001` |
| Table detector | gRPC `table-structure:8001` |
| Elasticsearch | `http://elasticsearch:9200` |
| Redis | `redis:6379` |
| SeaweedFS S3 | `seaweedfs:9010` |
| NV-Ingest | `nv-ingest-ms-runtime:7670` |

Do not add host port mappings for these dependencies. Diagnose them from an
application container or with `docker compose exec` on the internal network.
