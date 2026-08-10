# Require full-service GPU co-residency

The local RAG deployment must accept document ingestion and queries at any
time, so all required generation, retrieval, and ingestion GPU services will
remain available simultaneously on the single local GPU. This deliberately
accepts tighter memory limits and experimental capacity tuning instead of
stopping or reloading services between ingestion and query windows; if the
documented fallback ladder cannot make the complete service set healthy and
stable, the one-GPU deployment is considered infeasible rather than silently
changed to time sharing.

## Consequences

The initial deployment is accepted when every required service remains healthy
for five minutes and no container's restart count changes during that period.
Retrieval quality, ingestion correctness, latency, and throughput remain manual
follow-up tests and are not deployment acceptance gates.

## Accepted profile

On 2026-08-08 the initial FP8 baseline (8192 context, 0.48 vLLM GPU
utilization, concurrency 1) passed the gate on one H100 80GB. On 2026-08-09
the current 32768-context FP8 profile passed the same five-minute gate under
active generation and image-RAG traffic. The canonical results and observation
artifacts are linked from `docs/research/evidence/qwen-h100-runtime-results.md`.
The 8192-context profile remains the first capacity fallback; lower FP8
settings and NVFP4 remain unverified.
