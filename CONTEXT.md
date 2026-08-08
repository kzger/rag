# Local RAG Deployment

This context defines the deployment language for running the NVIDIA RAG
Blueprint without sending document data or inference requests to hosted
services.

## Language

**Local RAG Deployment**:
A deployment in which document data and runtime inference remain on the local
machine. Setup-time downloads of models and container artifacts are allowed.
_Avoid_: Local install, offline deployment

**Full-Service Co-residency**:
An operating constraint in which all required generation, retrieval, and
ingestion GPU services remain available simultaneously on one local GPU.
_Avoid_: Time-shared deployment, ingestion/query windows

The accepted 2026-08-08 H100 profile and its evidence are recorded in
`docs/research/evidence/qwen-h100-runtime-results.md`.
