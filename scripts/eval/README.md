# RAG evaluation scripts

Scripts in this folder benchmark a deployed RAG stack. They load a local dataset into the ingestor, send each question to the RAG server’s generate API, and evaluate the responses using RAGAS with [NVIDIA metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/).

## Prerequisites

- The RAG server and ingestor server must be accessible on the network (for example, after completing the [Quickstart: self-hosted Docker](../../docs/deploy-docker-self-hosted.md)).
- Set `NVIDIA_API_KEY` in the environment; it is required for `langchain_nvidia_ai_endpoints` to run the RAGAS judge.
- `RAG_EVAL_JUDGE_MODEL` (optional) — LLM id passed to `ChatNVIDIA` for RAGAS scoring; defaults to `mistralai/mixtral-8x22b-instruct-v0.1` when unset or empty.

## Install (this repository)

From the repository root, sync and run with uv’s `--project` flag pointing at this folder:

```bash
uv sync --project scripts/eval
uv run --project scripts/eval python scripts/eval/evaluate_rag.py --help
```

Or work inside `scripts/eval` (creates `.venv` next to `pyproject.toml`):

```bash
cd scripts/eval
uv sync
uv run python evaluate_rag.py --help
```

Use the same `--project scripts/eval` / `cd scripts/eval` pattern for every command below when invoking `evaluate_rag.py`.

## Accepted dataset format

Use `evaluate_rag.py` as the driver. Invoke it with one or more dataset root directories via `--dataset-paths`. It does not support dataset names, only filesystem paths.

`evaluate_rag.py` validates each dataset root with `validate_dataset_roots():`

- The path must be a directory.
- A `corpus\` directory must exist (documents to ingest are discovered recursively under it).
- A `train.json` file must exist, be a regular file, and contain UTF-8 JSON (see shapes below).

If ingestion is not skipped, every file under corpus/ that is not already marked as ingested for the target collection will be uploaded. Use the same collection name that the eval run will query (by default, the dataset directory’s basename, unless you pass `--collection`).

### Converting external benchmarks into this layout

When importing a dataset from elsewhere (hosted catalogs, JSONL, CSV, APIs, etc.), materialize `corpus/` as PDF whenever possible—export or print sources to PDF so the bundle matches common document RAG and the default `--file-type pdf` (including PDF page metrics during ingest).

### Directory layout (summary)

```text
my_dataset/                 ← dataset root (pass this to --dataset-paths)
  corpus/                   ← required; source documents (nested dirs allowed)
    doc_a.pdf
    notes.pdf
  train.json                ← required; eval questions and answers
```

### `train.json`

Use a JSON array of objects. The script reads these fields from each object:

| Field              | Required                  | Used by evaluator                                                      |
|--------------------|---------------------------|-------------------------------------------------------------------------|
| `question`         | Yes                       | Sent to the RAG server as the user message for each item.              |
| `answer`           | Yes (for meaningful scores) | Used as the reference for RAGAS judge metrics (`evaluate_result`).   |
| `id` or `query_id` | No                        | If present, stored on the saved eval row for traceability.             |
| `contexts`         | No                        | Optional; format below. |
| Other keys (for example, `is_impossible`) | No | Ignored by the current driver unless you extend it.                    |

Context relevance and response groundedness compare the model answer to the contexts retrieved from the RAG server. E2E accuracy uses `question`, `answer`, and the model’s answer.

Each item in `contexts` should include a `filename` and a `text` field:

- `filename` — the same file name the document has under `corpus/` (its basename, not a subdirectory path), matching the file on disk exactly.
- `text` — the reference span (page, snippet, or chunk text) used for answering the query.

```json
"contexts": [
  {
    "filename": "COMPANY_2020_10K",
    "text": "…"
  }
]
```

Multiple objects are allowed when several files apply. A legacy shape of plain strings (`["…", "…"]`) is also valid for simple bundles where per-file tagging is not needed.

Minimal example:

```json
[
  {
    "id": "q1",
    "question": "What is the corporate tax rate in the United States?",
    "answer": "21%"
  }
]
```

Corpus files should be the documents you want indexed; prefer PDF when building the bundle from external data, especially when upstream references do not pin down a concrete on-disk name. Naming must stay consistent with how your ingestor stores `document_name` (citation parsing matches streamed citation results using basenames).

### Ingestion: `--file-type`

The default value is `pdf`. If the converted corpus is mostly PDFs (recommended when preparing benchmarks, including when you materialize sources from links that do not spell out a concrete file name), leave defaults or pass `--file-type pdf`; the substring `pdf` enables PDF page counts in ingestion metrics. For non-PDF corpora, use values such as `txt` or `txt,html` so they match what is under `corpus/`.

### Checklist

- The dataset path is a directory containing `corpus/` and `train.json`.
- `train.json` is valid JSON: array of objects (top-level dict / multi-turn bundles are rejected).
- Every turn / row has `question` and `answer` where you need judge scores.
- `corpus/` holds the files you intend to retrieve against.
- `NVIDIA_API_KEY` is set for cloud judge models.


## What gets measured

- Ingestion: time, file counts, and (for PDFs) pages per second — appended to `rag_<label>_evaluation_metrics.json`.
- Quality (RAGAS): [NVIDIA metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/) from `ragas.metrics` — `AnswerAccuracy` (column `nv_accuracy`), `ContextRelevance` (`nv_context_relevance`), `ResponseGroundedness` (`nv_response_groundedness`); the latter two require non-empty retrieved contexts from the RAG response.
- Token usage: aggregated from streaming generate responses when the server sends `usage` chunks.

The Judge model for RAGAS is read from `RAG_EVAL_JUDGE_MODEL` (default `mistralai/mixtral-8x22b-instruct-v0.1`); see `evaluate_rag.py`.

## Main entrypoint

`evaluate_rag.py` — orchestrates collection check/create, blocking bulk upload to the ingestor (no async task polling in this client), parallel queries to `/v1/generate`, then RAGAS evaluation and JSON exports.

### Example: single dataset, local services

Adjust host, ports, and ingestor URL to match your stack.

```bash
export NVIDIA_API_KEY=your_key_here

uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths /path/to/my_dataset \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --output_dir results
```

### Example: skip ingestion (collection already populated)

```bash
uv run --project scripts/eval python scripts/eval/evaluate_rag.py \
  --dataset-paths /path/to/my_dataset \
  --host localhost \
  --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --skip_ingestion \
  --output_dir results
```

### Useful flags

| Flag | Notes |
|------|--------|
| `--collection` | Override collection name (default: dataset folder name). |
| `--skip_ingestion` / `--skip_evaluation` | Partial runs. |
| `--force_ingestion` | Deletes the collection first, then re-ingests. |
| `--delete_collection` | Deletes the collection after the run. |
| `--top_k` | Sent as `reranker_top_k` when set. |
| `--vdb_top_k` | Vector DB candidate pool when set. |
| `--model` / `--llm_endpoint` | Optional overrides passed to generate (omit to use server defaults). |
| `--batch_size` | Ingestion batch size (server max applies). |
| `--thread` | Parallelism for queries (and related work). |
| `--timeout` | RAG HTTP timeout seconds (default 180). |

Document uploads use `blocking: true`. The client does not send `split_options`; chunk size and overlap follow the ingestor server configuration. The ingestor URL you pass is normalized to include the `v1` API prefix internally.

## Outputs

Under `--output_dir` (default `results`), each dataset gets a subfolder named after the dataset directory:

| File | Content |
|------|---------|
| `rag_<label>_evaluation_data.json` | Per-query model outputs and contexts (written before scoring). |
| `rag_<label>_evaluation_summary.json` | Mean metrics (+ token usage summary when available). |
| `rag_<label>_evaluation_results.json` | Full RAGAS vectors and per-sample usage when present. |
| `rag_<label>_evaluation_metrics.json` | Structured ingestion + evaluation + token KPIs (`RagEvaluationMetrics`). |

## Weak-text multimodal accuracy baseline

`evaluate_multimodal_accuracy.py` exercises the public streaming `/v1/generate`
endpoint with versioned image-and-text cases. It is intentionally separate from
unit tests: the CLI requires a running RAG deployment and an already-ingested
multimodal collection, while its unit tests use no network services.

The default manifest includes the two supplied images, an OCR fixture with a
visible `MODEL J7EF PLUS` label, and the repository's text-free purse image. It
covers weak demonstrative questions, OCR/model identification, similar-product
confusion, consecutive images, and knowledge-base no-match cases. Run it from
the repository root:

```bash
uv run --project scripts/eval python scripts/eval/evaluate_multimodal_accuracy.py \
  --endpoint http://127.0.0.1:8081 \
  --collection jorjin_glasses \
  --model Qwen/Qwen3.6-27B-FP8 \
  --vdb-top-k 100 \
  --reranker-top-k 5 \
  --output results/multimodal-accuracy-baseline.json
```

The evaluator first probes the public multimodal `/v1/search` endpoint and uses
its ordered results for candidate Hit@K, then measures the public Standard
`/v1/generate` stream. The report records the dataset version, asset SHA-256
hashes, non-sensitive candidate settings, citations, candidate Hit@K, verified
identification, case-level annotated unsupported-claim and rejection metrics,
plus TTFT/full-response P50 and P95. Its `metric_definitions` field records the
exact deterministic scoring rules. Images are encoded only in live requests;
reports contain paths and hashes, and any data URI echoed by a service is
redacted.

### Calibration and A/B comparison

Use the checked-in manifest for reproducible same-dataset runs. Reports are
compatible only when dataset version, manifest filename and SHA-256 content
hash, every asset hash, model, and collection match; the comparison command
fails closed otherwise. Selecting baseline/candidate additionally requires
complete explicit server provenance and at least one changed calibration key;
`neither` is valid for partial provenance. Deltas are
candidate minus baseline, so negative latency is faster:

```bash
uv run --project scripts/eval python scripts/eval/evaluate_multimodal_accuracy.py \
  --dataset scripts/eval/multimodal_accuracy_cases.json --endpoint http://127.0.0.1:8081 \
  --collection jorjin_glasses --model Qwen/Qwen3.6-27B-FP8 --vdb-top-k 100 \
  --reranker-top-k 5 --request-vlm-temperature 0.0 --variant baseline \
  --server-settings results/remote-multimodal-settings.json --output results/baseline.json
uv run --project scripts/eval python scripts/eval/evaluate_multimodal_accuracy.py \
  --dataset scripts/eval/multimodal_accuracy_cases.json --endpoint http://127.0.0.1:8081 \
  --collection jorjin_glasses --model Qwen/Qwen3.6-27B-FP8 --vdb-top-k 100 \
  --reranker-top-k 5 --request-vlm-temperature 0.1 --variant candidate \
  --server-settings results/remote-multimodal-settings.json --output results/candidate.json
uv run --project scripts/eval python scripts/eval/compare_multimodal_reports.py \
  --baseline results/baseline.json --candidate results/candidate.json \
  --selected neither --selection-rationale "Comparison only; selection requires complete provenance." \
  --output results/comparison.json
```

Remote server settings are never inferred from the evaluator host. To bind
non-sensitive deployment provenance, provide a sanitized JSON file following
`multimodal_server_settings.example.json`:

```bash
  --server-settings results/remote-multimodal-settings.json
```

The file is whitelisted to known calibration keys and validates expected JSON
types. Without it, report deployment settings are `null` with provenance
`unavailable`; they must not be described as Compose defaults. Capturing the
evaluator process environment is available only explicitly with
`--capture-local-environment` and is labeled `local-process-environment`, not
remote provenance. Never put credentials or endpoint secrets in the artifact.
The Compose fallback is only a fallback: an active `deploy/compose/.env` or
explicit environment can override it. That user-owned source-of-truth file is
not changed by ticket 06.

The comparison report includes quality and TTFT/total P50/P95 deltas, both
configurations, selection rationale, and unmet cases. Missing metrics or failed
cases are reported rather than treated as passing evidence. The report metadata
distinguishes request fields from server/deployment settings; secrets are never
copied. The measured request budget is one retrieval probe and one streamed
generation per manifest case. Internal deployment calls remain bounded
separately: query understanding is at most one bounded call per request when
enabled, and verification is at most one bounded call per request when enabled.
These internal calls are not counted as extra evaluator requests.

The corrected nine-case strict-provenance comparison leaves temperature
calibration unresolved: lexical quality is tied at `0.50` for both, manual
grounded identification is `2/4` for both, and rejection is `5/5` for both.
Neither shows superior groundedness. The
metadata records the exact
`MultimodalAccuracyConfig` defaults, including
query-understanding, candidate/fusion, verification, abstention, and image
budget knobs. Raw/RRF
absolute and margin thresholds are telemetry-only per ticket 08: existing
evidence showed no separation, so no ineffective gate is invented. See the
non-sensitive evidence artifact under
`docs/research/evidence/ticket-06-calibration-2026-08-12.json`.

The scoped Compose fallback change is only `APP_VLM_MAX_TOTAL_IMAGES` 5→3.
`APP_VLM_TEMPERATURE` remains at its pre-ticket fallback `0.6` because the
temperature comparison is tied on manual groundedness. Active environment
overrides may differ; the user-owned `.env` is not changed.

Feature-on is selected over feature-off for rejection behavior, but its latency
cost is substantial and acceptance remains unresolved. The nine-case ambiguity
case passes manual review and rescored lexical rejection is `1.0` at both
temperatures. The manifest includes a deliberate ambiguity case reusing
`logo_max.png`.
Because the company logo cannot uniquely support either model, its safe expected
outcome is abstention/rejection with zero expected sources. The manifest hash
changed and the nine-case rescored metrics are current. Remaining failures
include pronoun false denial, J10A expected evidence pages 1/2 versus output
page 4, unsupported positive details, unresolved latency acceptance, and
unisolated candidate/fusion/threshold values. Older pre-strict feature reports
are legacy context only. The final checked-in
strict nine-case feature off/on A/B is compatible with the current manifest and
selects feature-on for rejection behavior.

The lexical negation heuristic is harness correctness: it prevents the known
identity-denial false positive from being counted as identification. It is not
a product or deployment behavior claim.

### Annotating cases

Three conventions keep the lexical scoring honest; each exists because violating
it produced a false failure against a correct answer:

- **`accepted_answer_terms` must include the spelling the corpus uses.** The J7EF
  specification PDF is titled `JORJIN TECHNOLOGIES-J7EF PULS 產品規格_v3`, so an
  answer echoing the document's own `J7EF Puls` is correct and is annotated as
  accepted alongside `J7EF Plus`.
- **`forbidden_answer_terms` are for wrong-product claims, not negation.** Use them
  for competing model names (`J10A` in a J7EF case). Do not add bare negators such
  as `不是` — they match innocuous prose inside correct answers. Identity denial is
  detected separately by `_denies_identity`, which requires a negator bound to
  `同一`/`the same product` or sitting immediately before an accepted model term,
  so 「這不是 J7EF Plus」and 「這不太可能是同一個產品」still score as denials.
- **`expected_sources` must be pages that actually identify the product**, and may
  list several. Cover pages and bare brandmarks do not qualify: `J10A Sales kit.pdf`
  page 9 is a logo on an orange background carrying no identifying content, so the
  J10A case expects pages 1 (text `J10ASales Kit`) and 2 (product image and spec
  table) instead.

Lexical scoring in the live report is diagnostic only; it is not manual
adjudication and must not be treated as authoritative. To publish verified
identification, unsupported-claim, and rejection metrics, review each exact
answer against its query image and cited page, map every material factual claim
to a literal answer quote, attest exhaustive coverage, record support and
identification verdicts, then bind that review to the answer hashes:

```bash
uv run --project scripts/eval python scripts/eval/adjudicate_multimodal_accuracy.py \
  --report results/multimodal-accuracy-baseline.json \
  --adjudications results/multimodal-accuracy-adjudications.json \
  --output results/multimodal-accuracy-adjudicated.json
```

Stale reviews fail closed when any answer SHA-256 changes.
