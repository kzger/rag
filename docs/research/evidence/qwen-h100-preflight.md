# Qwen H100 preflight evidence

Date: 2026-08-08 UTC
Commit under test: pending implementation commit

## Environment

| Check | Observed result | Requirement assessment |
| --- | --- | --- |
| GPU | GPU 0, NVIDIA H100 80GB HBM3, 81,559 MiB | Meets the one-H100 target |
| GPU container access | `nvidia/cuda:12.9.1-base-ubuntu22.04` returned the H100 through `--gpus all` | Pass |
| Driver | 580.126.09 | Meets driver 560+ |
| Host CUDA toolkit | Not installed | Not required because container GPU access passed |
| Docker | 29.6.1 | Available |
| Docker Compose | 5.3.0 | Supports the required override syntax |
| Free disk | 342GB | Meets the 200GB requirement |
| Host OS | Ubuntu 24.04.4 LTS | Outside the documented Ubuntu 22.04 support matrix |
| NGC credential | Present in the operator's local `.env`; registry pulls authorized without exposing the value | Pass |
| Public port conflicts | None on 8081, 8082, 8090, 8999 | Pass |
| Existing RAG services | None | Clean startup state |

Docker did not list a legacy named `nvidia` runtime, but an actual CUDA
container successfully accessed the H100. This host is using a working modern
GPU injection path; the runtime-name heuristic alone is therefore not treated
as a blocker.

## Pinned artifacts

| Artifact | Immutable identity |
| --- | --- |
| FP8 vLLM amd64 image | `vllm/vllm-openai:v0.19.0@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90` |
| Qwen FP8 model revision | `Qwen/Qwen3.6-27B-FP8@e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| NVFP4 vLLM amd64 image | `vllm/vllm-openai:v0.24.0@sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8` |
| NVFP4 model revision | `nvidia/Qwen3.6-35B-A3B-NVFP4@491c2f1ea524c639598bf8fa787a93fed5a6fbce` |

## Static verification

Both FP8 and NVFP4 resolved Compose configurations pass
`scripts/qwen_h100_local_rag.sh validate`. The active service set contains 14
required services, assigns every GPU service to GPU 0, excludes default
generation/Parse/Paddle/audio services, and publishes exactly these sockets:

- `127.0.0.1:8090 -> rag-frontend:3000`
- `127.0.0.1:8081 -> rag-server:8081`
- `127.0.0.1:8082 -> ingestor-server:8082`
- `127.0.0.1:8999 -> qwen-vllm:8000`

Live health, smoke, cache-recreate, automatic-recovery, and five-minute
Full-Service Co-residency results are recorded separately as the deployment
attempt proceeds. This preflight alone does not establish success or
infeasibility.
