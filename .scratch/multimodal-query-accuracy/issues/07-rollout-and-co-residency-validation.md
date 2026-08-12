# 07 — 部署準確度流程並驗證 Full-Service Co-residency

**What to build:** 將校準後的 multimodal accuracy pipeline 套用至 Local RAG Deployment，讓 WebUI 的真實 Image + Text Query 使用新流程，同時證明單張 H100 上所有 generation、retrieval 與 ingestion GPU services 仍可持續共同運行。

**Blocked by:** 06 — 校準多模態準確度與延遲預設值

**Status:** ready-for-human

- [ ] Active Compose env 與 live RAG service 使用 06 選定的 feature flag、候選、融合、信心、image budget 與 VLM generation 設定，且 source-of-truth 與 container env 一致。
- [ ] WebUI 選定 collection 後，以 `j7ef-plus.jpg` 和 `logo_max.png` 執行弱文字問題，實際結果符合 verified／ambiguous／no_match policy 並提供適當 citations。
- [ ] WebUI 未選 collection、連續不同圖片、無匹配圖片及下游服務錯誤的行為符合規格，且不會把 direct VLM guessing 偽裝成 RAG。
- [ ] 觀察資料可追蹤 query understanding、visual retrieval、text retrieval、fusion、verification、confidence decision 與 generation latency，但不包含 base64 圖片或 secrets。
- [ ] 在 ingestion、text query 與 Image + Text Query 流量期間，所有必要服務連續五分鐘 healthy 且 container restart count 不變。
- [x] 驗證期間沒有 GPU OOM，且未停止 ingestion 或其他必要服務取得 query capacity。
- [ ] Text-only query 與 ingestion correctness 執行回歸驗證，確認 multimodal accuracy rollout 沒有改變其既有行為。
- [x] 保存非敏感設定摘要、服務健康、quality metrics、P50/P95 latency 與 Full-Service Co-residency evidence，並記錄 feature flag 回退程序。

## Comments

- 2026-08-12 (pre-flight inventory, from the ticket-06 review): state of each item before
  anyone starts. Nothing below has been executed as ticket 07 work; it is context so the
  executing agent does not re-derive it.

### Read this first: item 06 selected no new values

Item 1 says to apply "06 選定的" settings. **Ticket 06 selected nothing to change.** Its sweep
concluded the shipped defaults are already the best known values, and two of them
(`MULTIMODAL_VERIFICATION_MAX_CANDIDATES=2`, `MULTIMODAL_VERIFICATION_MAX_TOKENS=512`) moved
from inherited to evidence-backed. There is no separate "calibrated profile" to install — the
current `deploy/compose/.env` plus the compose defaults *are* it. See issue 06's calibration
sections for the grid and for the values deliberately left alone.

Note also that **issue 06 is still `ready-for-agent` with every box unchecked**, and its items
4, 5 and 6 remain genuinely open (temperature unresolved, latency unresolved and unmitigated).
Ticket 07 is nominally blocked by it. Treat 07 as validating the *current* configuration
rather than a finished calibration.

### Already satisfied

- **Item 1, partial current-profile validation only.** All 14 multimodal tunables are now forwarded by
  `docker-compose-rag-server.yaml` (they previously were not, which is why 06 could not vary
  them) and the matching keys exist in `values.yaml`. Verified on the running container: every
  knob resolves to its configuration.py default. Re-verify at rollout with
  `docker exec rag-server env | grep -E "MULTIMODAL|APP_VLM"`.
- **Item 4, partially.** Final raw per-stage samples now establish query, visual, text,
  fusion, verification, and completed-generation timing counts/medians. Raw-score and
  global-secret requirements remain partial, so the acceptance checkbox stays unchecked.
  The new timing line is intended to exclude image/base64/response content, while existing
  logs may still contain query, evidence, endpoint text, or secrets; no blanket redaction
  claim is made.

### Open, with specifics

- **Item 1 conflicts with issue 06 item 5 on temperature.** The container runs
  `APP_VLM_TEMPERATURE=0.6` (not set in `.env`, so it takes the compose default). Issue 06
  requires the local deployment be calibrated to 0.0 or 0.1, and its own A/B was inconclusive.
  Critically, **every accuracy figure now on record was produced at 0.6** — the eval harness
  reports `vlm_temperature_sent: False`, so it exercised the server default. The WebUI will do
  the same. Pinning temperature to 0.0 or 0.1 for item 1 would therefore invalidate the
  identification 0.750 / rejection 1.0-1.0 baseline and needs its own A/B first.
- **Item 4 remaining gap.** Raw-score observability and global-secret sanitization remain
  unestablished; do not treat timing evidence as satisfying those requirements.
- **Items 5 and 6, GPU headroom is the risk.** The H100 is at **71.6 / 81.5 GB used with ~10 GB
  free at idle**. The co-residency test drives ingestion, text query and image query
  concurrently, and the verification gate adds a second VLM call per image query. The raw
  artifact preserves sampled GPU/container state and the final seven-service scan reports
  zero matches for the recorded OOM/fatal terms. Item 5 remains unchecked because
  `rag-server` has no Docker health probe, despite running status and dependency health.
- **Items 2 and 3, WebUI.** `rag-frontend` answers 200 on `http://127.0.0.1:8090/`. Assets for
  the required cases are checked in: `image/j7ef-plus.jpg`, `image/logo_max.png`, and
  `data/multimodal/Creme_clutch_purse1-small.jpg` for the no-match case. The
  `jorjin_glasses` collection holds 47 chunks. Expected behaviour per the gate is in
  `docs/multimodal-query.md`, "Candidate Verification and Abstention".
- **Item 8, evidence.** `scripts/eval/evaluate_multimodal_accuracy.py` accepts a server-settings
  file (`scripts/eval/multimodal_server_settings.example.json`) that stamps a non-sensitive
  config summary into the report. The final durable rollout artifact records complete
  explicit-file provenance, and the compact evidence links all durable raw artifacts by
  path and SHA-256.
  `scripts/eval/compare_multimodal_reports.py` produces the before/after deltas. Standards
  review and Spec review are both PASS; orchestrator owns final tests and commit review.
  For the rollback procedure item 8 asks for: setting `ENABLE_MULTIMODAL_VERIFICATION_GATE=false`
  disables verification only while staged retrieval remains enabled, and
  `ENABLE_MULTIMODAL_ACCURACY=false` restores the complete legacy path; both are read at
  container start, so a restart is required.

### Expected baseline for regression comparison

Latest 9-case run on this deployment, all at the current configuration: retrieval Hit@K 1.000,
identification 0.750, unsupported-claim 0.1111111111, rejection precision and recall 1.000,
citation leak 0.000. Retrieval P50/P95 is 0.5203830269/1.3735369775 s; TTFT is
7.0797253258/8.8739999841 s; total is 7.0798874989/11.1864175330 s. Final stage medians
are query 1783.8 ms, visual 530.7 ms, text 79.7 ms, fusion 0.9 ms, verification 4437.9 ms
over nine pipeline cases, and completed generation 2886.8 ms over four cases.

The lexical evaluator's one miss is `j10a-similar-model-disambiguation`: the answer identifies
J10A, but its citation is page 4 while the annotated expected sources are pages 1/2. This is
separate from manual groundedness comments and is a known open retrieval-ranking issue, not
a gate failure — see issue 05.

### Ticket-07 rollout evidence (2026-08-12)

- Sanitized evidence: `docs/research/evidence/ticket-07-rollout-2026-08-12.json`.
- Quality: Hit@K 1.0, identification 0.75, unsupported 0.1111111111111111, rejection
  precision/recall 1.0/1.0, citation leak 0. Retrieval P50/P95 is
  0.6469595612/1.2901993137 s; TTFT is 6.9289864402/9.2033017920 s; total is
  7.0798874989/11.1864175330 s. The lexical evaluator metric identifies 4 cases, passes
  3, and therefore reports 0.75.
  Its sole lexical miss is `j10a-similar-model-disambiguation`: expected source pages 1/2
  are absent from citations, which cite page 4. The pronoun case passes the lexical denial
  predicate; its opening denial remains a separate manual-review comment.
- Frontend HTTP was 200. No-collection returned HTTP 400 with normal SSE and zero
  citations. The public endpoint eval is WebUI-equivalent, not literal browser automation.
  A per-request bad `vlm_endpoint` injection was ineffective; live downstream-failure
  injection remains unestablished.
- Co-residency used a configured 300-second load-launch window; the overall drained run
  lasted 408.4 seconds. It ran from `2026-08-12T19:23:16Z` through
  `2026-08-12T19:30:04Z` with 21 samples. Traffic fields mean HTTP/stream success only:
  text 2/2, image 2/2, and blocking ingestion plus cleanup 4/4.
  Peak used was 72728 MiB, minimum free 8352 MiB, peak utilization 100%, dependency
  health 200, every required container status running, and restart values 0. The final
  seven-service OOM/fatal scan found zero matches for every recorded term.
- Command-result observations only (raw logs unavailable): full CI-aligned unit suite 2167
  passed, 1 xfailed. The initial unfiltered suite had
  three missing proprietary `nemo_retriever` collection errors; CI excludes those paths.
  Pre-commit ultimately passed after one frontend timeout flake was reproduced as 742/742
  on rerun. Targeted reported results remain multimodal accuracy 17 passed, eval 40 passed,
  and Compose/parity 33 passed, with quiet resolved Compose config. Orchestrator owns final
  review.
- Durable raw evidence is preserved in `docs/research/evidence/` with SHA-256 paths in the
  compact evidence JSON; the raw co-residency artifact retains every sample and traffic
  record for audit. Verification availability and metric predicate checks are recorded in
  `ticket-07-verification-raw-2026-08-12.json`.

### Final reconciliation

- Final raw rollout and co-residency artifacts replaced the prior versions. Durable paths
  and hashes, not temporary `/tmp/opencode` paths, are the evidence sources.
- Verification raw review status is Standards PASS and Spec PASS. Ticket-06 temperature /
  latency acceptance, literal WebUI automation, and live downstream-failure injection
  remain open. Item 4 remains unchecked for raw-score/global-secret requirements; item 5
  remains unchecked for the missing RAG Docker health probe; item 6 and item 8 are checked;
  item 7 remains unchecked because HTTP/stream success does not establish semantic text or
  ingestion correctness.
