# Qwen Local RAG hardware revalidation

Changing the GPU model, GPU count, VRAM, driver, container runtime, host OS, or
storage layout creates a new hardware profile. Never carry forward a previous
Full-Service Co-residency result without rerunning this procedure.

## 1. Freeze the candidate profile

Record the following before starting containers:

```bash
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
docker --version
docker compose version
docker info | grep -i 'runtimes.*nvidia'
df -h .
cat /etc/os-release
scripts/qwen_h100_local_rag.sh config > /tmp/qwen-hardware-candidate.yaml
```

Also retain the full pullable `RepoDigests` for all required images, the Qwen
model repository and immutable revision or local checksum, the active
`QWEN_VARIANT`, context length, GPU-memory utilization, and cache locations.
Never include credentials in the evidence bundle.

## 2. Check hard compatibility

- Confirm at least 200GB free disk and working NVIDIA Container Toolkit.
- Confirm driver and OS support against `docs/support-matrix.md` and every NIM
  image's support matrix.
- Confirm the selected Qwen quantization and vLLM kernels support the new GPU.
- Confirm all required GPU services can target the intended device IDs.
- Treat a different OS, unsupported GPU, or reduced VRAM as experimental even
  when containers happen to start.

Stop here and record the profile as incompatible if a required image or model
does not support the hardware. Do not hide incompatibility by removing required
services or alternating ingestion and query windows.

## 3. Validate configuration before loading models

```bash
scripts/qwen_h100_local_rag.sh validate
QWEN_VARIANT=nvfp4 scripts/qwen_h100_local_rag.sh validate
```

The second command validates the fallback configuration; it does not authorize
starting two Qwen variants simultaneously.

## 4. Run the FP8 baseline

Start with the committed FP8 profile:

```bash
scripts/start_qwen_h100_local_rag.sh
```

Verify all 14 services and run the complete functional matrix from
`docs/qwen-h100-local-rag.md`: multimodal ingestion/retrieval, standard RAG,
query rewriting, decomposition, filter generation, reflection, summarization,
and Agentic RAG. Record failures, response correctness, TTFT, and throughput as
diagnostic or comparison evidence.

## 5. Run the five-minute acceptance window

Start monitoring first, then send representative multimodal ingest, Qwen, and
RAG search traffic from another shell:

```bash
scripts/qwen_h100_local_rag.sh monitor \
  docs/research/evidence/qwen-hardware-candidate-observations.jsonl
```

Evaluate the artifact independently when needed:

```bash
uv run python scripts/qwen_h100_local_rag.py evaluate-observations \
  docs/research/evidence/qwen-hardware-candidate-observations.jsonl \
  --minimum-duration 300
```

The deployment gate passes only when every required service remains healthy
for at least five minutes and every restart count remains unchanged. Log OOM,
fatal worker, or dependency-loss signals as diagnostics; ADR-0001 does not make
them separate acceptance criteria.

## 6. Apply the capacity fallback ladder if required

Change only one parameter at a time and recreate `qwen-vllm`, `rag-server`,
`ingestor-server`, and `nv-ingest-ms-runtime`. Repeat functional checks and the
full five-minute window after every change:

1. Current FP8 baseline, context 32768, GPU memory utilization 0.48.
2. Previously accepted FP8 fallback, context 8192, utilization 0.48.
3. FP8 context 8192 with `QWEN_GPU_MEMORY_UTILIZATION=0.45`.
4. FP8 with utilization 0.45 and `QWEN_MAX_MODEL_LEN=4096`.
5. Pinned NVFP4 fallback using `QWEN_VARIANT=nvfp4`.

Never declare success from a mixture of different profiles or separate
ingestion/query windows. If no allowed profile passes, record Full-Service
Co-residency as infeasible on that hardware.

## 7. Compare and decide

Store one result row per immutable profile:

| Field | Required value |
| --- | --- |
| Hardware | GPU model/count/UUID, VRAM, driver, OS |
| Software | Git commit, Compose config checksum, Docker/Compose versions |
| Artifacts | Image RepoDigests, model revision/checksum |
| Capacity | Variant, context, memory utilization, peak used/minimum free VRAM |
| Gate | Duration, health result, before/after restart counts |
| Traffic | Ingest/generation/search timestamps and failure counts |
| Follow-up | Functional correctness, TTFT, tokens/s, known limitations |

Approve the replacement only after the exact candidate profile passes. Keep
the prior result for comparison; do not overwrite evidence from another host.
