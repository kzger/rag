# Qwen H100 manual deployment

This procedure deploys the single-GPU Qwen Local RAG Deployment without an
external orchestrator. Run every command from the repository root. The source
of truth is `deploy/compose/.env`; shell-only exports do not survive a restart.

## 1. Check out the implementation

```bash
git clone https://github.com/kzger/rag.git
cd rag
git switch qwen-h100-local-rag-tickets
```

Keep the tested commit, resolved Compose configuration, image digests, model
revision, and hardware inventory together in the deployment record.

## 2. Verify the host

The validated host used one H100 80GB. The repository support matrix requires
Ubuntu 22.04, NVIDIA driver 560 or newer, CUDA-compatible container runtime,
Docker Compose with `!override` support, and at least 200GB free disk.

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
docker --version
docker compose version
docker info | grep -i 'runtimes.*nvidia'
df -h .
cat /etc/os-release
```

Any other GPU or OS is a new experimental profile and must complete the
hardware revalidation procedure before being accepted.

## 3. Configure credentials and persistent paths

Copy the NGC key into the uncommitted Compose environment without printing it.
If the operator already keeps it in `~/.bashrc`, use:

```bash
bash -ic 'set +x; source ~/.bashrc >/dev/null 2>&1; test -n "${NGC_API_KEY:-}"; sed -i "/^export NGC_API_KEY=/d" deploy/compose/.env; sed -i "1iexport NGC_API_KEY=${NGC_API_KEY}" deploy/compose/.env' >/dev/null 2>&1
```

Verify presence without displaying the value, then authenticate to NGC:

```bash
bash -c 'set -a; source deploy/compose/.env; set +a; test -n "${NGC_API_KEY:-}"'
source deploy/compose/.env
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Never stage `deploy/compose/.env` after inserting the key. Model caches,
Elasticsearch, SeaweedFS, Redis, and ingestion state use named `rag-vol-*`
volumes and survive `stop`, `restart`, and `down` without `--volumes`.

## 4. Resolve and validate configuration

```bash
scripts/qwen_h100_local_rag.sh validate
scripts/qwen_h100_local_rag.sh config --services
scripts/qwen_h100_local_rag.sh config > /tmp/qwen-h100-compose.yaml
```

Validation must confirm the pinned Qwen image and model revision, shared Qwen
roles, one-GPU assignment, required multimodal services, lifecycle policies,
and exactly four loopback publications.

## 5. Start the deployment

The startup wrapper validates, pulls images, creates services in dependency
order, and waits up to 30 minutes for all public endpoints:

```bash
scripts/start_qwen_h100_local_rag.sh
```

For a warm cache, skip the pull with `QWEN_SKIP_PULL=1`. Select the NVFP4
fallback only when the documented FP8 ladder has failed:

```bash
QWEN_SKIP_PULL=1 scripts/start_qwen_h100_local_rag.sh
QWEN_VARIANT=nvfp4 scripts/start_qwen_h100_local_rag.sh
```

First startup commonly takes 15–30 minutes. Do not start the default
`nim-llm`, `vlm-ms`, or `vlm-captioning-ms` profiles beside this stack.

## 6. Verify exposure and health

```bash
scripts/qwen_h100_local_rag.sh ps
curl -fsS http://127.0.0.1:8999/v1/models
curl -fsS 'http://127.0.0.1:8081/v1/health?check_dependencies=true'
curl -fsS 'http://127.0.0.1:8082/v1/health?check_dependencies=true'
curl -fsS http://127.0.0.1:8090/ >/dev/null
ss -ltn | grep -E '127\.0\.0\.1:(8090|8081|8082|8999) '
```

All 14 required containers must be healthy. Only `127.0.0.1:8090`, `:8081`,
`:8082`, and `:8999` may be published; every other dependency remains on the
internal `nvidia-rag` network.

## 7. Run functional and residency checks

Use the exact ingest, generation, search, and five-minute monitor procedure in
`docs/qwen-h100-local-rag.md`. Acceptance requires at least 300 seconds with
all required services healthy and unchanged restart counts. Correctness,
quality, latency, and throughput are recorded follow-up evidence rather than
additional deployment gates.

## 8. Connect through SSH

From the client workstation:

```bash
ssh -N \
  -L 8090:127.0.0.1:8090 \
  -L 8081:127.0.0.1:8081 \
  -L 8082:127.0.0.1:8082 \
  -L 8999:127.0.0.1:8999 \
  USER@RAG_HOST
```

Use `http://127.0.0.1:8090` on the client. Do not bind the services directly to
the host's public interface.

## 9. Stop or restart safely

```bash
scripts/qwen_h100_local_rag.sh restart
scripts/stop_qwen_h100_local_rag.sh
scripts/stop_qwen_h100_local_rag.sh --down
```

The default `--stop` retains containers, volumes, and caches. `--down` removes
containers and the Compose network but still retains named volumes and caches.
Do not add `--volumes` unless permanent deletion was explicitly approved.
