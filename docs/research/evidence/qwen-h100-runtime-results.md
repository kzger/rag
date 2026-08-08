# Qwen H100 runtime results

Date: 2026-08-08 UTC

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
- Limits: context 8192, GPU utilization 0.48, generation concurrency 1,
  ingestion batching 1, two images, no video, thinking disabled.

The baseline initialized successfully, so FP8 0.45, FP8 4096-context, and
NVFP4 were not run. They remain fallback configurations, not verified runtime
profiles.

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
The accepted result requires at least 300 seconds with all 14 required
containers healthy and every restart count unchanged. Peak/minimum GPU values
are derived from that artifact by `qwen_h100_local_rag.py
evaluate-observations`.

The evaluator accepted 305 seconds. Every health value was `healthy`, restart
counts were unchanged, peak memory used was 65,520 MiB, and minimum free memory
was 15,560 MiB.

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
