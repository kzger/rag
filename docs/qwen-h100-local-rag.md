# Qwen H100 Local RAG runbook

This runbook operates the experimental single-GPU deployment described by
`.scratch/qwen-h100-local-rag/spec.md`. It keeps every required generation,
retrieval, and ingestion GPU service resident on one H100. It is not a
time-shared deployment.

Operator deliverables:

- [`qwen-h100-fae-guide.zh-TW.md`](qwen-h100-fae-guide.zh-TW.md) — Traditional
  Chinese deployment, tuning, testing, and troubleshooting guide for FAE users
  without prior RAG/VLM knowledge.
- [`qwen-h100-manual-deployment.md`](qwen-h100-manual-deployment.md) — complete
  manual installation, validation, SSH access, and lifecycle procedure.
- [`qwen-hardware-revalidation.md`](qwen-hardware-revalidation.md) — evaluation
  and acceptance process after changing hardware.
- [`qwen-h100-api-endpoints.md`](qwen-h100-api-endpoints.md) — published and
  internal endpoint reference with example requests.
- `scripts/start_qwen_h100_local_rag.sh` and
  `scripts/stop_qwen_h100_local_rag.sh` — guarded startup and persistent stop
  wrappers around the canonical lifecycle command.

## Prerequisites

- One NVIDIA H100 80GB visible as GPU 0.
- NVIDIA driver 560 or newer and working Docker GPU passthrough.
- Docker and Docker Compose with support for the Compose `!override` tag.
- At least 200GB free disk.
- An NGC API key with permission to pull and run the NVIDIA NIM images.
- A checkout on the Qwen H100 feature branch.

The repository support matrix specifies Ubuntu 22.04. Treat another host OS as
an unsupported experiment and record it with the results. A local CUDA toolkit
is not required when GPU passthrough works inside containers.

The lifecycle wrapper reads `deploy/compose/.env`, including the local
`NGC_API_KEY` value. Copy the key from the operator's existing credential
source without printing it, and never stage or commit the secret. Log in to
the registry through standard input:

```bash
source deploy/compose/.env
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

All deployment settings live in the local `deploy/compose/.env`; its committed
form must contain only non-secret defaults. Model caches and application data
use named volumes prefixed with `rag-vol-`.

The certified retrieval defaults are explicit in that file:
`VECTOR_DB_TOPK=100` fetches the vector-store candidate pool and
`APP_RETRIEVER_TOPK=4` caps the reranked chunks inserted into the shared Qwen
prompt. Request payloads and saved frontend settings can still override these
environment defaults.

The following deployment settings are intentionally tunable in
`deploy/compose/.env`. General validation checks their types, documented
ranges, and cross-field relationships; changing a certified default makes the
configuration custom but does not make it invalid:

| Setting group | Variables | General validation |
| --- | --- | --- |
| Qwen resources | `QWEN_MAX_MODEL_LEN`, `QWEN_GPU_MEMORY_UTILIZATION`, `QWEN_MAX_NUM_SEQS` | Context 1024–131072; GPU utilization greater than 0 and at most 0.95; sequences 1–8 |
| Images | `QWEN_MAX_IMAGES_PER_PROMPT`, `APP_VLM_MAX_TOTAL_IMAGES` | Positive integers up to 32; the RAG budget cannot exceed the Qwen limit |
| Retrieval and conversation | `VECTOR_DB_TOPK`, `APP_RETRIEVER_TOPK`, `CONVERSATION_HISTORY`, `MAX_RECURSION_DEPTH`, `MAX_REFLECTION_LOOP` | Candidate pool 1–400; final Top-K cannot exceed it; history 0–100; loop depths 1–20; query rewriting requires positive history |
| Token budgets | `LLM_MAX_TOKENS`, `APP_VLM_MAX_TOKENS`, `AGENTIC_CONTEXT_MAX_TOKENS`, `APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS`, `AGENTIC_*_LLM_MAX_TOKENS`, `SUMMARY_LLM_MAX_CHUNK_LENGTH` | Positive integers no greater than `QWEN_MAX_MODEL_LEN` |
| Workload | `AGENTIC_CONCURRENCY_LIMIT`, `NV_INGEST_FILES_PER_BATCH`, `NV_INGEST_CONCURRENT_BATCHES`, `SUMMARY_MAX_PARALLELIZATION` | Respectively 1–16, 1–250, 1–16, and 1–64 |

FP8 and NVFP4 model IDs, revisions, served names, and memory envelopes are also
explicit. A custom model is valid only when every Qwen-backed role uses its
resolved `--served-model-name` and all internal endpoints target an existing,
compatible resolved service.

## Validate the resolved deployment

The wrapper combines the four Blueprint Compose files with the H100 override.
It validates the resolved service graph rather than the source YAML layout.

```bash
scripts/qwen_h100_local_rag.sh validate
scripts/qwen_h100_local_rag.sh validate --certified
scripts/qwen_h100_local_rag.sh config --services
```

`validate` is the normal preflight used by `pull` and `up`. It checks safety,
topology, types, ranges, and compatibility, then reports whether the resolved
configuration is certified FP8, certified NVFP4, or custom.
`validate --certified` additionally requires exact conformity with the profile
selected by `QWEN_VARIANT` and reports each drift with its parameter, actual
value, and certified baseline. Use it when reproducing the published H100
evidence; a safe custom deployment should use normal validation.

The non-tunable safety contract remains strict: the Qwen image digest, required
and forbidden services, GPU 0 placement, the four unique loopback host ports,
internal role/endpoints, excluded high-resource ingestion modes, healthchecks,
and `restart: unless-stopped`. Configuration summaries contain only an
allowlisted set of non-secret values.

## Start and inspect

```bash
scripts/qwen_h100_local_rag.sh pull
scripts/qwen_h100_local_rag.sh up
scripts/qwen_h100_local_rag.sh ps
scripts/qwen_h100_local_rag.sh logs --tail 100
```

Follow all deployment logs, or narrow them to one or more services:

```bash
scripts/qwen_h100_local_rag.sh logs --follow --tail 100
scripts/qwen_h100_local_rag.sh logs --follow --tail 200 rag-server
scripts/qwen_h100_local_rag.sh logs --since 10m rag-server qwen-vllm
docker logs --follow --tail 200 rag-server
```

Press `Ctrl-C` to stop following logs; this does not stop any container.

First startup can take 15–30 minutes while model artifacts populate the named
caches. Do not start the default `nim-llm`, `vlm-ms`, or `vlm-captioning-ms`
profiles alongside this deployment.

Check the public endpoints from the host:

```bash
curl -fsS http://127.0.0.1:8999/v1/models
curl -fsS 'http://127.0.0.1:8081/v1/health?check_dependencies=true'
curl -fsS 'http://127.0.0.1:8082/v1/health?check_dependencies=true'
curl -fsS http://127.0.0.1:8090/ >/dev/null
ss -ltn | grep -E ':(8090|8081|8082|8999) '
```

Only ports 8090, 8081, 8082, and 8999 may be published, all on 127.0.0.1.
Elasticsearch, Redis, SeaweedFS, NV-Ingest, embedding, reranking, OCR, and
detector endpoints remain inside the `nvidia-rag` network.

## SSH access

From the client workstation, forward only the loopback services:

```bash
ssh -N \
  -L 8090:127.0.0.1:8090 \
  -L 8081:127.0.0.1:8081 \
  -L 8082:127.0.0.1:8082 \
  -L 8999:127.0.0.1:8999 \
  USER@RAG_HOST
```

The frontend is then available at `http://127.0.0.1:8090` on the client.

## Pipeline verification matrix

Run each row independently. Agentic requests do not combine with VLM,
reflection, decomposition, or guardrails.

| Case | Required request/config | Observable result |
| --- | --- | --- |
| Standard multimodal | Standard pipeline with an ingested PDF containing text, an image, and a table | OCR/detectors/captioning/embedding complete; answer cites retrieved content |
| Query rewrite | Follow-up question with conversation history | Rewritten retrieval query is logged and retrieves the expected document |
| Decomposition | Single collection, `use_knowledge_base=true` | Subqueries are emitted, depth is at most 3 |
| Filter generation | Elasticsearch collection with a metadata schema | Generated Elasticsearch Query DSL restricts results |
| Reflection | Standard RAG request | Relevance/groundedness loop runs at most 3 times |
| Summary | Upload with `generate_summary: true` | Summary status and result are available through the API |
| Agentic | `agentic=true`, `use_knowledge_base=true` | Planner/execution stage events and final content stream from Qwen |

Record correctness, quality, and latency as follow-up evidence. They are not
the Full-Service Co-residency deployment gate.

## Five-minute gate

After all required containers are healthy, start the monitor first:

```bash
scripts/qwen_h100_local_rag.sh monitor \
  docs/research/evidence/qwen-h100-fp8-048-observations.jsonl
```

While that command is running, use a second shell to submit representative
multimodal ingestion, generation, and retrieval traffic. The accepted run used
the following requests; the generation and search requests were repeated until
the monitor completed:

```bash
collection=qwen_active_20260808_2
curl -fsS -X POST http://127.0.0.1:8082/v1/collection \
  -H 'Content-Type: application/json' \
  --data "{\"collection_name\":\"$collection\",\"embedding_dimension\":2048,\"metadata_schema\":[]}"
curl -fsS -X POST http://127.0.0.1:8082/v1/documents \
  -F 'documents=@data/multimodal/functional_validation.pdf;type=application/pdf' \
  -F "data={\"collection_name\":\"$collection\",\"blocking\":true,\"generate_summary\":true};type=application/json"
curl -fsS -X POST http://127.0.0.1:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"In one short sentence, explain why OCR helps multimodal retrieval."}],"max_tokens":48,"temperature":0}'
curl -fsS -X POST \
  "http://127.0.0.1:8081/v2/vector_stores/$collection/search" \
  -H 'Content-Type: application/json' \
  --data '{"query":"What visual and tabular content appears in the validation document?","max_num_results":6}'
```

The command records health, restart counts, and GPU memory/utilization every 15
seconds for five minutes, then evaluates the ADR-0001 gate. Success requires all
required services to remain healthy and every restart count to remain stable.
The accepted 2026-08-08 result and functional matrix are recorded in
[`research/evidence/qwen-h100-runtime-results.md`](research/evidence/qwen-h100-runtime-results.md).

## Capacity fallback ladder

Run only one generation profile at a time and repeat the complete five-minute
gate after every change:

1. Current FP8 profile: context 32768, GPU memory utilization 0.48.
2. Set `QWEN_GPU_MEMORY_UTILIZATION=0.45` in `deploy/compose/.env` and recreate
   `qwen-vllm`.
3. If 32K cannot pass the gate, restore the previously accepted context 8192
   profile; use 4096 only as the final FP8 capacity fallback.
4. If FP8 still cannot pass, use the pinned NVFP4 override:

   ```bash
   export QWEN_VARIANT=nvfp4
   scripts/qwen_h100_local_rag.sh validate --certified
   scripts/qwen_h100_local_rag.sh pull
   scripts/qwen_h100_local_rag.sh up --force-recreate qwen-vllm rag-server ingestor-server nv-ingest-ms-runtime
   scripts/qwen_h100_local_rag.sh monitor \
     docs/research/evidence/qwen-h100-nvfp4-observations.jsonl
   ```

The NVFP4 profile uses the H100 Marlin compatibility path, not native
Blackwell FP4 kernels. Record text/image smoke results, TTFT, tokens/s, and peak
VRAM separately from the deployment gate. If every permitted profile fails,
record single-H100 Full-Service Co-residency as infeasible; do not switch to
ingestion/query windows and call it a success.

## Restart and stop without deleting data

```bash
scripts/qwen_h100_local_rag.sh restart
scripts/qwen_h100_local_rag.sh stop
scripts/qwen_h100_local_rag.sh down
docker volume ls --filter name=rag-vol-
```

`down` intentionally omits `--volumes`. Never add `-v` when retained
collections, objects, or model caches are required. After a Docker daemon or
approved host restart, use `scripts/qwen_h100_local_rag.sh ps` and the health
commands above to verify automatic recovery.

## Query latency

`ENABLE_FILTER_GENERATOR` and `ENABLE_AGENTIC_RAG` both default to `false` in this repo
and in the Helm chart, and should stay that way on a single H100. Measured on
`jorjin_glasses` with "提供J7EF Plus 的詳細規格":

| configuration | wall time | retrievals returning zero documents |
| --- | --- | --- |
| both enabled | 52.1 s | 3 |
| agentic only | 34.8 s | 0 |
| neither | ~18 s | 0 |

`ENABLE_FILTER_GENERATOR` asks an LLM to derive a metadata filter from the question
without checking it against the values that exist in the index. For the query above it
emitted `filename == "J7EF Plus"` while the indexed filename is
`JORJIN TECHNOLOGIES-J7EF PULS 產品規格_v3.pdf`, so retrieval returned nothing.
Self-reflection then spent every `MAX_REFLECTION_LOOP` round rewriting the *query*, which
cannot repair a *filter*. Confirm with:

```bash
scripts/qwen_h100_local_rag.sh logs --since 5m rag-server \
  | grep -E "Dynamic filter generated|Retrieved 0 documents"
```

`ENABLE_AGENTIC_RAG` splits the question into sub-tasks that each retrieve and reason.
With `AGENTIC_CONCURRENCY_LIMIT=1` they serialise onto the single shared Qwen endpoint,
so the cost adds up linearly for no measured quality gain — the non-agentic answer was
equally complete and carried eight citations. Enable it per request with
`{"agentic": true}` when multi-step reasoning is genuinely required.

`ENABLE_QUERY_DECOMPOSITION` also defaults to `false` and should stay there. It splits one
question into sub-questions, each costing an LLM call and a full retrieval. On the follow-up
"介紹這個產品" it took 30.5 s against 8.5 s disabled, and the longer run returned a *shorter*
answer (782 vs 937 characters) because it widened the question into sub-queries about other
product lines, launch dates, pricing and competitor comparisons that were never asked.

Keep `ENABLE_QUERYREWRITER` enabled: it is the step that resolves follow-up pronouns, and it
correctly rewrote that question to "請介紹 JReality J7EF Plus …" before decomposition widened
it. `CONVERSATION_HISTORY` (5 here, default 0) is likewise required for follow-ups.

Reflection already exits early: `ReflectionCounter` stops as soon as the relevance score
clears its threshold, and it did score 2 and stop immediately once retrieval returned
documents. A loop that runs to its limit means retrieval is returning nothing — look there
rather than adding an early-return path.

## Failure triage

- Authentication or manifest errors: confirm the shell key and `nvcr.io`
  login; never print the key.
- `unknown device` or no GPU: verify `docker run --rm --gpus all` with an
  NVIDIA CUDA image.
- Qwen initialization OOM: follow the capacity ladder in order.
- Web UI returns `Error from rag-server` with a context-length error: retain
  the single-H100 profile's `APP_RETRIEVER_TOPK=4` cap, select fewer
  collections, and inspect `rag-server` plus `qwen-vllm` logs.
- Image query reports that the Reflection LLM or reranker is unavailable:
  verify that the request is running code which skips text reflection and
  reranking for image queries. Base64 image data must be sent only to
  multimodal retrieval and VLM generation, never tokenized by those text
  pipeline stages.
- NIM startup timeout: inspect the affected service log and cache volume size;
  first downloads are slow.
- Dependency health failure: inspect the named dependency first; application
  health intentionally waits for dependencies.
- Port conflict: `ss -ltnp` must show only the four documented loopback binds.

The broader troubleshooting reference is `skills/rag-blueprint/references/troubleshoot.md`.
