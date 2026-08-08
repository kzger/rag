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

On 2026-08-08 the FP8 baseline (8192 context, 0.48 vLLM GPU utilization,
concurrency 1) passed the gate on one H100 80GB. The canonical result and
observation artifact are linked from
`docs/research/evidence/qwen-h100-runtime-results.md`. Lower FP8 settings and
NVFP4 were not attempted because the first permitted profile succeeded.
