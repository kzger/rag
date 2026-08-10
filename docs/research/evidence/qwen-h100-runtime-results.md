# Qwen H100 runtime results

Date: 2026-08-08 to 2026-08-09 UTC

This is the canonical mutable result record for the Qwen H100 experiment. The
runbook, research note, glossary, ADR, and tickets link here instead of copying
the observation series.

## Environment and accepted profile

- GPU: one NVIDIA H100 80GB HBM3, 81,559 MiB, GPU 0.
- Driver: 580.126.09; Docker 29.6.1; Docker Compose 5.3.0.
- Host: Ubuntu 24.04.4 LTS, outside the documented Ubuntu 22.04 matrix.
- Profile: `Qwen/Qwen3.6-27B-FP8` at immutable revision
  `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`.
- vLLM: `v0.19.0` image digest
  `sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90`.
- Limits: context 32768, GPU utilization 0.48, generation concurrency 1,
  vector DB candidates 100, reranked prompt chunks 4, ingestion batching 1,
  two images, no video, thinking disabled.

The current 32K baseline initialized successfully. The previously accepted 8K
profile remains the first fallback; FP8 0.45, FP8 4096-context, and NVFP4 were
not run and remain unverified fallback configurations.

## 32K image-query follow-up

On 2026-08-09, the shared Qwen endpoint was recreated with
`--max-model-len 32768`. vLLM reported 7.3 GiB available KV-cache memory and
the Qwen, RAG, and Ingestor containers remained healthy with restart count 0.
Idle memory after the rebuild was 63,718 MiB used and 17,362 MiB free.

An image-bearing query against `jorjin_glasses` originally revealed two
independent routing defects rather than a VLM context shortage:

- The base64 data URL reached text reflection and reranking. Image queries now
  bypass those text-only stages and retain the image for multimodal embedding,
  retrieval, and final VLM generation.
- Elasticsearch received `k=100` with its wrapper default
  `num_candidates=50`. Image retrieval now enforces
  `num_candidates >= k`.

The corrected query identified the supplied handbag, compared it with the
J7EF Plus VR-device collection, returned three citations, and completed with
HTTP 200. Seven additional active-window image queries each returned HTTP 200
and three citations; seven direct Qwen generations also returned HTTP 200.

## Endpoint and persistence evidence

- `/v1/models`, text chat (`TEXT_OK`), and data-URL image chat (`IMAGE_OK`)
  succeeded through `127.0.0.1:8999` with the configured model name.
- Initial weight download used 66 shards. Recreating Qwen reused the named
  Hugging Face, vLLM, and FlashInfer caches; the second load did not download
  the weights again and loaded weights in 6.45 seconds.
- Application container recreation retained both validation collections and
  their documents. A post-recreation query returned the expected table-reading
  answer.
- The only generation process was `qwen-vllm`; default Nemotron LLM/VLM,
  independent captioning, Parse, Paddle, and audio containers were absent.

## Resolved runtime images

The implementation commit is `f318a59`. The following pullable repository
digests were read from Docker `RepoDigests` for the 14 required running
containers. They are the resolved identities for this observed run and can be
used as `image@sha256:...` references on another host.

| Service | Image tag | Pullable repository digest |
| --- | --- | --- |
| `qwen-vllm` | `vllm/vllm-openai:v0.19.0` | `vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90` |
| `nemotron-vlm-embedding-ms` | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2@sha256:58c40b920840be6e2f4ad5d77c32c65d61e048070fe45d51fb4bdb6f84a71e21` |
| `nemotron-ranking-vl-ms` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2@sha256:14d3ecb180006d88f32a45e87d426db28abdc9d1c8c2295cf340e74836653439` |
| `page-elements` | `nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0` | `nvcr.io/nim/nvidia/nemotron-page-elements-v3@sha256:cf36bdf27260a0b7217dc449b33653a2c1f57620cfc2686656c2fef8c2e1884b` |
| `graphic-elements` | `nvcr.io/nim/nvidia/nemotron-graphic-elements-v1:1.8.0` | `nvcr.io/nim/nvidia/nemotron-graphic-elements-v1@sha256:80441d106ac31a7ed95560dcd33c763cdf7aa8687382b3a96539fc1e66d1f22d` |
| `table-structure` | `nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0` | `nvcr.io/nim/nvidia/nemotron-table-structure-v1@sha256:5b0e9e4cfb8aa33c1bb6c99a06b771533c445d3fc0c60f94e977973c9ea16c87` |
| `nemotron-ocr` | `nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0` | `nvcr.io/nim/nvidia/nemotron-ocr-v1@sha256:a26c233f7ad7f675008fcfdcb743dc6f26f4d6838a6e23f887370d3db9df8304` |
| `nv-ingest-ms-runtime` | `nvcr.io/nvidia/nemo-microservices/nv-ingest:26.3.0` | `nvcr.io/nvidia/nemo-microservices/nv-ingest@sha256:0a804b1d6ca7cbf8b3f386dcd76825ee25eb4056898f3a17c9f39f249afae9be` |
| `elasticsearch` | `docker.elastic.co/elasticsearch/elasticsearch:9.3.0` | `docker.elastic.co/elasticsearch/elasticsearch@sha256:4f6bdcb742e892539c6ac49b0dd3e4e182e90218546e8c6a22db378c344acb60` |
| `redis` | `redis/redis-stack:7.2.0-v18` | `redis/redis-stack@sha256:cfe04d0cf0184c1ff75cd666083d29a672b3714b5af1fe76db75b5989dca9684` |
| `seaweedfs` | `chrislusf/seaweedfs:3.73` | `chrislusf/seaweedfs@sha256:d372192d42d1d8d8dbfcb73fb6789152d7eeb6e27044f8c0c99202196cfd20fb` |
| `ingestor-server` | `nvcr.io/nvidia/blueprint/ingestor-server:2.6.0` | `nvcr.io/nvidia/blueprint/ingestor-server@sha256:5573a2ec107015daaeed6994d1f6779a1df57bd3e6f2daa904ac9ab8f0b7a557` |
| `rag-server` | `nvcr.io/nvidia/blueprint/rag-server:2.6.0` | `nvcr.io/nvidia/blueprint/rag-server@sha256:6042ad4d7e271fe1362ce8dbede6df7c659ff3df8878ae718550149c5075c972` |
| `rag-frontend` | `nvcr.io/nvidia/blueprint/rag-frontend:2.6.0` | `nvcr.io/nvidia/blueprint/rag-frontend@sha256:5cb3f47aa08887c7ca0b2bc8abfcbd1587b9f56712eb437c79c2b2b5887e8347` |

## Functional follow-up matrix

These results are evidence about wiring and behavior. Per ADR-0001 they do not
alter the health/restart deployment gate.

| Case | Isolated request or prerequisite | Observed Qwen/pipeline evidence | Result |
| --- | --- | --- | --- |
| Multimodal ingestion | `functional_validation.pdf`, blocking upload, summary enabled | OCR plus page/graphic/table detectors, Qwen captioning, VLM embedding | 1 document; 5 text, 2 table, 4 image elements; no failures |
| Multimodal retrieval | Text search over the new collection, then a separate image-bearing generation request | VLM embed/rerank returned six results; shared Qwen described the supplied handbag image | Results included four images, one table, and one text passage; image request returned 200 |
| Standard RAG | Standard `/v1/generate`, `agentic=false` | Qwen answer plus four citation results | Answer and citations returned; measured RAG TTFT 398.77 ms |
| Query rewriting | Multi-turn font-size follow-up | Logged rewritten query: “What font size does the Word template guide recommend for body text?” | Six results returned |
| Query decomposition | One collection, KB enabled, filter/VLM disabled for this case | Two subqueries, recursion depth 1/3, final synthesis | Final answer completed |
| Filter generation | Elasticsearch collection with `category` and `is_public` schema | Generated a validated `bool.must` Query DSL filter | One result; `category=poetry`, `is_public=true` |
| Reflection | Standard search/generation | Context relevance scores were produced by the shared Qwen endpoint | Completed within the three-loop bound |
| Summarization | Upload with `generate_summary=true` | Summary role used the shared Qwen endpoint and Redis tracking | `SUCCESS`, 974-character result |
| Agentic RAG | `agentic=true`, KB enabled, streaming | `initial_retrieval`, `plan`, `execute`, `synthesize`, `verify`; concurrency 1 | Final table-reading answer; verification passed |
| Frontend | Loopback frontend and existing selector tests | Standard maps to `agentic:false`; Agentic maps to `agentic:true` | Frontend healthy and selector behavior covered |

## Network, health, and diagnostic evidence

- Exactly `127.0.0.1:8090`, `:8081`, `:8082`, and `:8999` were published.
  Elasticsearch, Redis, SeaweedFS, NV-Ingest, embedding, reranking, OCR, and
  detector ports were not listening on the host.
- Dependency-aware RAG health reported Elasticsearch, object storage, Qwen,
  query rewriting, VLM embedding/ranking, and reflection healthy. Ingestor
  health reported embedding, summarization, and captioning healthy.
- All 14 required services were simultaneously healthy. Idle GPU memory use
  after startup was approximately 62,069 MiB, leaving approximately 19,011
  MiB free. Active ingestion peaked near 62,323 MiB in the spot observation.
- The first `/search` probe exposed a 32K reflection/filter output-token default
  incompatible with the 8K local endpoint. The implementation now derives
  reflection output tokens from `LLM_MAX_TOKENS` and caps local filter/agent
  roles. The corrected probes returned 200.
- OOM, fatal worker exit, and dependency-loss log searches are diagnostic only;
  no such event caused a health or restart transition during the accepted run.

## Full-Service Co-residency gate

The machine-readable observation series is
[`qwen-h100-fp8-048-observations.jsonl`](qwen-h100-fp8-048-observations.jsonl).
Timestamped workload evidence is
[`qwen-h100-active-traffic.jsonl`](qwen-h100-active-traffic.jsonl). During the
accepted window it records one completed multimodal ingest, 65 successful Qwen
generations, and 65 successful RAG searches. Observed GPU utilization ranged
from 0% to 100%, covering idle and active samples.
The accepted result requires at least 300 seconds with all 14 required
containers healthy and every restart count unchanged. Peak/minimum GPU values
are derived from that artifact by `qwen_h100_local_rag.py
evaluate-observations`.

The original 8K evaluator run accepted 305 seconds. Every health value was
`healthy`, restart counts were unchanged, peak memory used was 65,520 MiB, and
minimum free memory was 15,560 MiB.

The 32K follow-up evaluator accepted 314 seconds while the repeated generation
and image-RAG traffic above was active. All 14 required services remained
healthy, every restart count was unchanged, peak memory used was 64,412 MiB,
and minimum free memory was 16,668 MiB.

## Limitations

- A host reboot was not performed because it would disrupt the shared host.
  Container recreation, restart policies, dependency ordering, cache reuse,
  and retained application data were verified instead.
- The SSH forwarding command is provided in the runbook and the service binds
  were verified locally. A second workstation was not available for an actual
  off-host tunnel probe.
- The implementation branch was inherited as
  `qwen-h100-local-rag-tickets` from `main`; `origin/develop` is not an
  ancestor. The deployment work stayed on that feature branch, but this does
  not satisfy the ticket's “branched from develop” provenance criterion.
