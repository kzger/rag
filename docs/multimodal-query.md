<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multimodal Query Support for NVIDIA RAG Blueprint

The multimodal query feature in the [NVIDIA RAG Blueprint](readme.md) enables you to query your knowledge base using both text and images. This is particularly useful for use cases where visual context enhances the query, such as:

- **Product identification**: "What is the price of this item?" + product image
- **Document lookup**: "Find documents related to this chart" + chart image
- **Visual Q&A**: "What material is this made of?" + product image

This feature combines:
- **VLM Embeddings**: `nvidia/llama-nemotron-embed-vl-1b-v2` for creating multimodal embeddings that understand both text and images
- **Vision-Language Model**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` for generating intelligent responses based on visual and textual context



## Prerequisites

Before enabling multimodal query support, ensure you have:

1. [Obtained an API Key](api-key.md)
2. [Deployed the NVIDIA RAG Blueprint](readme.md#deployment-options-for-rag-blueprint)
3. An NVIDIA H100 or A100 GPU for on-prem deployments



## Self-Hosted (On-Prem) Deployment

Use this section to deploy multimodal query support with locally hosted NVIDIA NIMs.

### 1. Start the Vector Database

Start the Milvus vector database service:

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

### 2. Deploy the Ingestion and VLM RAG NIMs

Set your NGC API key (replace with your actual key):

```bash
export NGC_API_KEY="nvapi-..."
```

Then run the deployment commands:

```bash
# Create the model cache directory
mkdir -p ~/.cache/model-cache
export MODEL_DIRECTORY=~/.cache/model-cache

# (Optional) Select a specific GPU for the VLM Microservice
# Use `nvidia-smi` to check available GPUs and set the desired GPU ID
export VLM_MS_GPU_ID=1  # Default is GPU 5; change to use a different GPU

# Deploy ingestion NIMs plus the VLM RAG NIMs.
USERID=$(id -u) docker compose --profile ingest --profile vlm-rag -f deploy/compose/nims.yaml up -d
```

:::{warning}
The first deployment may take 10-20 minutes as models download (~10GB+). Subsequent deployments will be faster as models are cached.
:::

Monitor the deployment status:

```bash
watch -n 5 'docker ps --format "table {{.Names}}\t{{.Status}}"'
```

Wait until the services show as `healthy`:

### 3. Configure Environment Variables

Set the model names and service URLs for the RAG pipeline:

```bash
# VLM (Vision-Language Model) configuration
export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_VLM_SERVERURL="http://vlm-ms:8000/v1"
export APP_LLM_SERVERURL=""

# Optional: use the same VLM for document summaries when no LLM NIM is running.
# You can also point SUMMARY_LLM* to a separate LLM or NVIDIA-hosted endpoint.
export SUMMARY_LLM="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export SUMMARY_LLM_SERVERURL="http://vlm-ms:8000/v1"

# Multimodal embedding model configuration
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-nemotron-embed-vl-1b-v2"
export APP_EMBEDDINGS_SERVERURL="nemotron-vlm-embedding-ms:8000/v1"
export ENABLE_VLM_INFERENCE="true"
# Optional staged accuracy pipeline. Image retrieval keeps up to
# APP_RETRIEVER_TOPK unique document pages in backend similarity order.
export ENABLE_MULTIMODAL_ACCURACY="true"
export VLM_TO_LLM_FALLBACK="false"
```

### 4. Configure Image Extraction for Ingestion

Enable image extraction and storage during document ingestion:

```bash
# Configure image extraction
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"
export APP_NVINGEST_EXTRACTIMAGES="True"

# Disable reranker for image-query requests. Image queries use the multimodal
# vector retrieval path directly and bypass reranking.
export ENABLE_RERANKER="false"
export APP_RANKING_SERVERURL=""
```

### 5. Start the Ingestor Server

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

Verify the service is healthy

### 6. Start the RAG Server

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
```

Verify the service is healthy


### 7. Verify All Services Are Running

Check the status of all deployed containers:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Confirm all the containers are running and healthy


## NVIDIA-Hosted (Cloud) Deployment

Use this section to deploy multimodal query support using NVIDIA-hosted API endpoints.

:::{note}
When using NVIDIA-hosted endpoints, you might encounter rate limiting with larger file ingestions (>10 files). For details, see [Troubleshoot](troubleshooting.md).
:::

### 1. Start the Vector Database

```bash
docker compose -f deploy/compose/vectordb.yaml up -d
```

### 2. Configure Environment Variables

#### a. Open `deploy/compose/.env` and uncomment the section `Endpoints for using cloud NIMs`. Then set the environment variables by running the following code.

```bash
source deploy/compose/.env
```

#### b. Set the environment variables to use NVIDIA-hosted endpoints for VLM models:

Set your NGC API key (replace with your actual key):

```bash
export NGC_API_KEY="nvapi-..."
```

Then set the VLM configuration:

```bash
# VLM (Vision-Language Model) configuration - cloud hosted
export APP_VLM_MODELNAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export APP_VLM_SERVERURL="https://integrate.api.nvidia.com/v1"
export APP_LLM_SERVERURL=""

# Optional: use the same NVIDIA-hosted VLM for document summaries.
# You can also leave SUMMARY_LLM* pointing at another supported summarizer.
export SUMMARY_LLM="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
export SUMMARY_LLM_SERVERURL="https://integrate.api.nvidia.com/v1"

# Multimodal embedding model configuration - cloud hosted
export APP_EMBEDDINGS_MODELNAME="nvidia/llama-nemotron-embed-vl-1b-v2"
export APP_EMBEDDINGS_SERVERURL="https://integrate.api.nvidia.com/v1"
export ENABLE_VLM_INFERENCE="true"
export ENABLE_MULTIMODAL_ACCURACY="true"  # Optional diverse document pages
export VLM_TO_LLM_FALLBACK="false"
```

### 3. Configure Image Extraction for Ingestion

```bash
# Configure image extraction
export APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
export APP_NVINGEST_IMAGE_ELEMENTS_MODALITY="image"
export APP_NVINGEST_EXTRACTIMAGES="True"

# Disable reranker (not supported with multimodal queries)
export ENABLE_RERANKER="false"
export APP_RANKING_SERVERURL=""
```

### 4. Start the Ingestor Server

```bash
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
```

Verify the ingestor server is healthy

### 5. Start the RAG Server

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
```

Verify the RAG server is healthy

### 6. Verify All Services Are Running

Check the status of all deployed containers

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

You should see output similar to the following:

```output
NAMES                                   STATUS
compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
ingestor-server                         Up 5 minutes
compose-redis-1                         Up 5 minutes
rag-frontend                            Up 9 minutes
rag-server                              Up 9 minutes
elasticsearch                           Up 36 minutes (healthy)
seaweedfs                               Up 35 minutes (healthy)
```



## Helm Chart Deployment

Use this section to deploy multimodal query support on Kubernetes using Helm charts.

:::{note}
This configuration disables the default LLM NIM and text embedding NIM, replacing them with VLM NIM and VLM embedding NIM. The GPU resources previously used by the disabled services will be available for the VLM services.
If MIG slicing is enabled on the cluster, ensure to assign a dedicated slice to the VLM. Check [mig-deployment.md](./mig-deployment.md) for more information.
:::

### 1. Modify values.yaml

Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable multimodal query support:

```yaml
# Multimodal Query Configuration
# This replaces the default LLM and text embedding NIMs with VLM variants

# Enable VLM NIM for multimodal generation
nim-vlm:
  enabled: true

# Enable VLM embedding NIM for multimodal embeddings
nvidia-nim-llama-nemotron-embed-vl-1b-v2:
  enabled: true
  image:
    repository: nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2
    tag: "1.12.0"

# Optional: disable the default text embedding NIM
nvidia-nim-llama-nemotron-embed-1b-v2:
  enabled: false

# Disable LLM NIM (VLM handles generation)
nim-llm:
  enabled: false

# Enable dedicated VLM captioning NIM (image-cap model changed after RC1)
nimOperator:
  nim-vlm-captioning:
    enabled: true

# Configure environment variables
envVars:
  # VLM inference settings
  ENABLE_VLM_INFERENCE: "true"
  ENABLE_MULTIMODAL_ACCURACY: "true"  # Optional diverse document pages
  VLM_TO_LLM_FALLBACK: "false"
  APP_VLM_MODELNAME: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
  APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

  # VLM embedding settings
  APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
  APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"

  # Disable reranker (not supported with multimodal queries)
  ENABLE_RERANKER: "False"
  APP_RANKING_SERVERURL: ""

ingestor-server:
  envVars:
    # Image extraction settings
    APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY: ""
    APP_NVINGEST_IMAGE_ELEMENTS_MODALITY: "image"
    APP_NVINGEST_EXTRACTIMAGES: "True"

    # Summary generation settings.
    # Required for generate_summary=true when nim-llm is disabled.
    SUMMARY_LLM: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    SUMMARY_LLM_SERVERURL: "nim-vlm:8000"

    # VLM embedding settings for ingestor
    APP_EMBEDDINGS_SERVERURL: "nemotron-vlm-embedding-ms:8000/v1"
    APP_EMBEDDINGS_MODELNAME: "nvidia/llama-nemotron-embed-vl-1b-v2"

nv-ingest:
  envVars:
    EMBEDDING_NIM_ENDPOINT: "http://nemotron-vlm-embedding-ms:8000/v1"
    EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-nemotron-embed-vl-1b-v2"
```

### 2. Deploy or Upgrade the Chart

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### 3. Verify the Deployment

Verify the VLM pods are running:

```bash
kubectl get pods -n rag | grep -E "(vlm|embedding)"
```

Expected output:
```
nim-vlm-f4c446cbf-ffzm7                              1/1     Running   0          22m
nemotron-vlm-embedding-ms-...                   1/1     Running   0          22m
```

:::{note}
It may take several minutes for the VLM pods to initialize and download the model weights.
:::



## Using Multimodal Queries

After deployment, you can start querying your knowledge base with both text and images.

:::{important}
**You must select a collection before querying.** Multimodal queries require a knowledge base to search against. Before performing any query (including visual Q&A, product identification, or document lookup), ensure you have:

1. **Created a collection**: Use the Web UI, Python client, or API to create a new collection
2. **Ingested documents**: Upload documents (PDFs, images, etc.) to your collection
3. **Selected the collection**: When querying, explicitly specify the collection name

Queries without a selected collection will not return relevant results from your knowledge base.
:::

### Web UI

Access the RAG frontend at `http://localhost:8090` to experiment with multimodal queries through the user interface.

1. In the sidebar, select your collection from the **Collection** dropdown
2. Upload an image and/or enter your text query
3. Click **Send** to get responses based on your knowledge base

For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).

### Python Client

When using the Python client, pass image input using the OpenAI vision content
format and always specify `collection_names` in your query:

```python
import base64
from pathlib import Path

image_b64 = base64.b64encode(Path("Creme_clutch_purse1-small.jpg").read_bytes()).decode()
image_query = [
    {"type": "text", "text": "What material is this made of?"},
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
            "detail": "auto",
        },
    },
]

await rag.generate(
    messages=[{"role": "user", "content": image_query}],
    use_knowledge_base=True,
    collection_names=["your_collection_name"],
    enable_reranker=False,
)
```

For details, see [NVIDIA RAG Blueprint Python Package](python-client.md).

### Interactive Notebook

For a step-by-step guide with code examples covering collection creation, document ingestion, and querying with images, see the [Multimodal Query Notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/image_input.ipynb).

## Limitations

- **Image-query reranking is bypassed**: When the user query includes an image,
  use `enable_reranker: False`. Image queries use the multimodal vector
  retrieval path directly.
- **Image-page candidate retrieval**: The default behavior expands only the
  highest-ranked image-search page. Set `ENABLE_MULTIMODAL_ACCURACY=true` to
  retain up to `APP_RETRIEVER_TOPK` distinct document/page candidates. Existing
  Elasticsearch, Milvus, and LanceDB collections are automatically backfilled
  with a canonical source/page identity on first use; subsequent ingestion
  maintains it. Milvus groups in the vector query. Elasticsearch uses exact
  vector scoring with field collapse, and LanceDB performs an exhaustive ordered
  scan, because their KNN APIs otherwise select raw chunks before grouping.
  Enable this mode on those backends only when that latency tradeoff is
  acceptable.
- **Multi-turn image isolation and budget**: The same accuracy flag treats only
  images in the latest user turn as query images. Raw images from earlier turns
  are omitted before retrieval and VLM inference; prior assistant text remains as
  the auditable conversation summary. If no summary was produced, the historical
  image is still omitted and the current request continues. The VLM image budget
  is allocated to current-query images first, followed by retrieved page images
  in relevance order. Logs report only role-based image counts and allocation,
  never image URIs or base64 payloads. With the flag disabled, the legacy
  conversation behavior remains unchanged.

This staged pipeline broadens candidates and isolates current-turn evidence. The
verification gate below then decides which of those candidates may support an
answer.

## Candidate Verification and Abstention

`ENABLE_MULTIMODAL_VERIFICATION_GATE=true` adds a non-streaming verification step
between retrieval and generation. It sends the query image and the top
`MULTIMODAL_VERIFICATION_MAX_CANDIDATES` fused candidate pages (page text plus one
page image each, within the `APP_VLM_MAX_TOTAL_IMAGES` budget) to the shared VLM
endpoint in one `temperature=0` call, and re-derives the outcome deterministically
in server code. Retrieval and fusion scores are never used as a threshold — RRF
values are too compressed to separate a match from a no-match — so they stay
telemetry only.

The outcome is one of three:

- **`verified`** — answers using only the selected candidate's page, with citations
  filtered to that page.
- **`ambiguous`** — abstains, listing only candidates backed by concrete identity
  evidence together with their conflicts.
- **`no_match`** — abstains, stating the knowledge base cannot confirm the item. It
  never silently falls back to direct VLM guessing.

Both abstention outcomes stream a standardized message through the normal SSE
schema with zero citations. A malformed verifier response or an unavailable
endpoint abstains or raises the existing service error; it never degrades into a
speculative answer.

### Model-token bridge

A product photo and a specification page differ in layout, so the VLM frequently
reports a visual `mismatch` even for the correct page. To recover these, a
deterministic bridge extracts strong model tokens (letter-and-digit, four or more
alphanumeric characters, whole-token) from the query image's OCR and model fields
and matches them against candidate page text or source filename. A bridge match is
positive identity evidence and can override that layout mismatch.

The bridge is deliberately narrow, and only these signals veto it:

- A candidate with concrete identity evidence naming a *different* model token.
- Conflicts **on the bridged candidate itself** that name a model token outside the
  bridge match. Conflicts raised by other candidates explaining why they do not
  match, and colour/shape/layout differences, are the expected photo-versus-spec
  noise and are ignored.
- A verifier verdict of `insufficient`, which asserts nothing to corroborate.

Brand or logo text cannot drive the bridge: brand names carry no digit and so never
form a strong model token, which is what keeps decorative-logo and unrelated-product
images on the abstention path.

`ENABLE_MULTIMODAL_ABSTENTION_PROMPT=true` additionally overrides the default
identification prompt rules so the model does not infer identity from retrieval
rank, generic visual similarity, category, or logo alone.

All of these flags default to `false`, including when `ENABLE_MULTIMODAL_ACCURACY`
is enabled, so the legacy pipeline remains the fallback. Verification decisions are
logged as `Verifier judgment`, `Bridge matches`, and `Verification outcome` lines
carrying structured fields only — never image URIs or base64 payloads.

### Which settings reach the container

`deploy/compose/docker-compose-rag-server.yaml` declares no `env_file`, so its explicit
`environment:` block is the whole contract — a variable absent from that list has no
effect on the container even when exported in `deploy/compose/.env`.

Tunable today in the Compose deployment:

| Variable | Purpose |
| --- | --- |
| `ENABLE_MULTIMODAL_ACCURACY` | Staged accuracy pipeline (diverse pages, query understanding, fusion) |
| `ENABLE_MULTIMODAL_VERIFICATION_GATE` | Candidate verification and abstention |
| `ENABLE_MULTIMODAL_ABSTENTION_PROMPT` | Conditional identity-abstention prompt rules |
| `APP_VLM_MAX_TOTAL_IMAGES` | Image budget the gate is sized against (default 3) |
| `APP_VLM_TEMPERATURE` | VLM generation temperature |

Read from the environment by `configuration.py` but **not currently forwarded by
Compose**, so they stay at their code defaults: `MULTIMODAL_VISUAL_CANDIDATES` (5),
`MULTIMODAL_TEXT_CANDIDATES` (5), `MULTIMODAL_MAX_CANDIDATES` (5),
`MULTIMODAL_VISUAL_WEIGHT` (0.5), `MULTIMODAL_TEXT_WEIGHT` (0.5), `MULTIMODAL_RRF_K`
(60), `MULTIMODAL_VERIFICATION_MAX_CANDIDATES` (2),
`MULTIMODAL_VERIFICATION_MIN_MATCH_CONFIDENCE` (0.80),
`MULTIMODAL_VERIFICATION_MIN_NO_MATCH_CONFIDENCE` (0.80),
`MULTIMODAL_VERIFICATION_MAX_TOKENS` (512) and `MULTIMODAL_VERIFICATION_TEMPERATURE`
(0.0). To tune any of these, add it to the compose `environment:` block first — and add
the matching key to `deploy/helm/nvidia-blueprint-rag/values.yaml`, which the
`test_compose_helm_parity` unit test enforces.

Verify what a running server actually resolved with:

```bash
docker exec rag-server env | grep MULTIMODAL
```

### Calibrated values

Measured on a 9-case multimodal set (`scripts/eval/evaluate_multimodal_accuracy.py`) against
a deployed Qwen server. Accuracy is held to identification 0.750, unsupported-claim 0.111 and
rejection precision/recall 1.0/1.0 with zero citation leak; anything that moved those was
rejected.

**The values below are the defaults — the sweep found nothing worth changing.** Every
alternative either damaged accuracy or made no measurable difference. Treat stage timings as
having roughly ±0.2 s of run-to-run noise and TTFT p50 about ±0.5 s, and confirm any apparent
win with a second run before adopting it.

| Variable | Value | Safe range | Why |
| --- | --- | --- | --- |
| `MULTIMODAL_VERIFICATION_MAX_CANDIDATES` | `2` | 2 | 1 drops identification to 0.250 and rejection precision to 0.714 — the gate then only inspects the top-ranked page. Raising it needs a matching `APP_VLM_MAX_TOTAL_IMAGES`. |
| `MULTIMODAL_VERIFICATION_MAX_TOKENS` | `512` | ≥512 | 256 collapses identification to 0.000. The verifier emits free-text `supporting_evidence` and `conflicts` per candidate and needs the room. |
| `MULTIMODAL_QUERY_UNDERSTANDING_MAX_TOKENS` | `512` | 256-512 | 256 matches 512 on accuracy but is **not** faster: repeated runs gave 1420 and 1787 ms at 256 against 1627-1783 ms at 512, so the ranges overlap and a single-run 0.21 s "saving" did not reproduce. |
| `APP_VLM_MAX_TOTAL_IMAGES` | `3` | ≥3 with the gate on | One query image plus two candidate pages. Below 3 starves the gate. Not compared against 5. |
| `MULTIMODAL_VISUAL/TEXT/MAX_CANDIDATES` | `5` | 3-5 | 3 and 5 were indistinguishable in both accuracy and latency. |
| `MULTIMODAL_VISUAL_WEIGHT` / `TEXT_WEIGHT` / `RRF_K` | `0.5` / `0.5` / `60` | — | Not varied; no evidence either way. |

The verification stage costs ~4.4 s of a ~7 s TTFT and that cost is **decode-bound, not
image-bound**: halving the output budget moved it to 3.56 s, while downscaling every image
moved it not at all. Since the output budget cannot be reduced without losing accuracy, the
gate's latency is the price of the rejection guarantee. Turning
`ENABLE_MULTIMODAL_VERIFICATION_GATE` off is the only large saving available, and it forfeits
that guarantee.

## Quality and latency calibration

The public evaluation seam records request fields separately from deployment
settings and compares reports only when dataset version, manifest filename and
manifest SHA-256, asset hashes, model, and collection are identical. It reports
candidate-minus-baseline quality
and TTFT/total P50/P95 deltas and fails closed for mismatches. Missing latency or
failed cases remain explicit unmet cases.

Remote deployment settings must be supplied separately as a sanitized JSON
artifact following `scripts/eval/multimodal_server_settings.example.json`.
Evaluator-host environment variables are not remote provenance: without that
file, the report marks server settings unavailable. Local environment capture is
an explicit, separately labeled diagnostic mode only.

The corrected nine-case strict-provenance comparison leaves VLM temperature
calibration unresolved: lexical quality is tied at `0.50`, manual grounded
identification is `2/4` for both, and rejection is `5/5` for both. Neither shows
superior groundedness. The
query-understanding temperature validator
permits `0.0–2.0`; verification temperature and confidence validators permit
`0.0–1.0`; positive candidate/token limits have no configured upper bound. The
VLM generation temperature has no local configuration validator, so its safe
operating range is deployment/model-specific and must be established by the
live trials. Reranker score filtering is validated as `0.0–1.0`; its selected
default remains the prior evidence-backed `0.0` value. The live A/B isolated
request temperature only, so top-p, max tokens, thinking, and thinking budget
remain observed deployment values rather than ticket-06 calibrations.

The manifest now includes a deliberate ambiguity case reusing `logo_max.png`.
The company logo cannot uniquely support either model, so the safe expected
outcome is abstention/rejection with zero expected sources. Its manifest hash
changed and the nine-case rescored metrics are current. Feature-on is selected
over feature-off for rejection behavior, but its latency cost is substantial and
acceptance remains unresolved. Unmet cases remain explicit: the pronoun case
fails both temperatures, J10A expected evidence is pages 1/2 but output cites
page 4 in both, unsupported positive details remain, and candidate/fusion/
threshold defaults were not isolated by the temperature run. Older pre-strict
feature reports are legacy context only; the final checked-in strict nine-case
feature off/on A/B selects feature-on for rejection behavior.

The lexical negation heuristic is harness correctness for the calibration metric
false positive, not a product or deployment behavior claim.

The scoped Compose fallback change is only the image budget 5→3; VLM temperature
remains the pre-ticket fallback `0.6` because the corrected comparison is tied on
manual groundedness. Active environment overrides may differ.

Raw-score, RRF-absolute, and margin thresholds are telemetry-only. Ticket 08
evidence found no separation between match and no-match cases, so these values
must not become ineffective quality gates. The manifest records deliberate
ambiguity as present, reusing `logo_max.png`, and it passes manual adjudication
as a safe abstention/rejection case. See
`docs/research/evidence/ticket-06-calibration-2026-08-12.json` and the evaluator
README for the bounded call budget and comparison command.

The final evidence still has unmet cases: J10A expects evidence from pages 1/2,
but the output cites page 4. This is not an all-cases-pass claim.


## Related Topics

- [Vision-Language Model (VLM) for Generation](vlm.md)
- [Multimodal Retriever (VLM Embedding & VLM Reranker)](multimodal-retriever.md)
- [Image Captioning Support](image_captioning.md)
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)
