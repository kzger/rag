# 07 — 部署準確度流程並驗證 Full-Service Co-residency

**What to build:** 將校準後的 multimodal accuracy pipeline 套用至 Local RAG Deployment，讓 WebUI 的真實 Image + Text Query 使用新流程，同時證明單張 H100 上所有 generation、retrieval 與 ingestion GPU services 仍可持續共同運行。

**Blocked by:** 06 — 校準多模態準確度與延遲預設值

**Status:** ready-for-agent

- [ ] Active Compose env 與 live RAG service 使用 06 選定的 feature flag、候選、融合、信心、image budget 與 VLM generation 設定，且 source-of-truth 與 container env 一致。
- [ ] WebUI 選定 collection 後，以 `j7ef-plus.jpg` 和 `logo_max.png` 執行弱文字問題，實際結果符合 verified／ambiguous／no_match policy 並提供適當 citations。
- [ ] WebUI 未選 collection、連續不同圖片、無匹配圖片及下游服務錯誤的行為符合規格，且不會把 direct VLM guessing 偽裝成 RAG。
- [ ] 觀察資料可追蹤 query understanding、visual retrieval、text retrieval、fusion、verification、confidence decision 與 generation latency，但不包含 base64 圖片或 secrets。
- [ ] 在 ingestion、text query 與 Image + Text Query 流量期間，所有必要服務連續五分鐘 healthy 且 container restart count 不變。
- [ ] 驗證期間沒有 GPU OOM，且不會透過停止 ingestion 或其他必要服務取得 query capacity。
- [ ] Text-only query 與 ingestion correctness 執行回歸驗證，確認 multimodal accuracy rollout 沒有改變其既有行為。
- [ ] 保存非敏感設定摘要、服務健康、quality metrics、P50/P95 latency 與 Full-Service Co-residency evidence，並記錄 feature flag 回退程序。

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

- **Item 1, source-of-truth consistency.** All 14 multimodal tunables are now forwarded by
  `docker-compose-rag-server.yaml` (they previously were not, which is why 06 could not vary
  them) and the matching keys exist in `values.yaml`. Verified on the running container: every
  knob resolves to its configuration.py default. Re-verify at rollout with
  `docker exec rag-server env | grep -E "MULTIMODAL|APP_VLM"`.
- **Item 4, partially.** Per-stage logging now exists and carries no base64 or secrets:
  `Multimodal accuracy stages` (query understanding, dual retrieval, fusion, candidate counts),
  `Multimodal verification stage` (verification_ms, candidates, images attached), plus
  `Verifier judgment`, `Bridge matches` and `Verification outcome` for the confidence decision.
  Image payloads are redacted via `_safe_log_text`.

### Open, with specifics

- **Item 1 conflicts with issue 06 item 5 on temperature.** The container runs
  `APP_VLM_TEMPERATURE=0.6` (not set in `.env`, so it takes the compose default). Issue 06
  requires the local deployment be calibrated to 0.0 or 0.1, and its own A/B was inconclusive.
  Critically, **every accuracy figure now on record was produced at 0.6** — the eval harness
  reports `vlm_temperature_sent: False`, so it exercised the server default. The WebUI will do
  the same. Pinning temperature to 0.0 or 0.1 for item 1 would therefore invalidate the
  identification 0.750 / rejection 1.0-1.0 baseline and needs its own A/B first.
- **Item 4 gaps.** Visual and text retrieval are timed together as one `dual_retrieval_ms`;
  the item asks for them separately. Generation latency is not in the stage log at all — it
  exists only as OpenTelemetry histograms (`rag_ttft_ms`, `llm_generation_time_ms`), and
  **`APP_TRACING_ENABLED="False"`** in this deployment, so nothing is exported. Either enable
  tracing or add the two timings to the stage log.
- **Items 5 and 6, GPU headroom is the risk.** The H100 is at **71.6 / 81.5 GB used with ~10 GB
  free at idle**. The co-residency test drives ingestion, text query and image query
  concurrently, and the verification gate adds a second VLM call per image query. Capture
  `nvidia-smi` throughout. Baseline restart counts are 0 for rag-server, ingestor-server,
  qwen-vllm, both nemotron NIMs, elasticsearch, nv-ingest and rag-frontend; re-read them after.
- **Items 2 and 3, WebUI.** `rag-frontend` answers 200 on `http://127.0.0.1:8090/`. Assets for
  the required cases are checked in: `image/j7ef-plus.jpg`, `image/logo_max.png`, and
  `data/multimodal/Creme_clutch_purse1-small.jpg` for the no-match case. The
  `jorjin_glasses` collection holds 47 chunks. Expected behaviour per the gate is in
  `docs/multimodal-query.md`, "Candidate Verification and Abstention".
- **Item 8, evidence.** `scripts/eval/evaluate_multimodal_accuracy.py` accepts a server-settings
  file (`scripts/eval/multimodal_server_settings.example.json`) that stamps a non-sensitive
  config summary into the report; recent runs did not pass it and recorded
  `server provenance: unavailable`. Pass it so item 8's summary is complete.
  `scripts/eval/compare_multimodal_reports.py` produces the before/after deltas.
  For the rollback procedure item 8 asks for: setting `ENABLE_MULTIMODAL_VERIFICATION_GATE=false`
  restores the legacy path, and `ENABLE_MULTIMODAL_ACCURACY=false` disables the whole staged
  pipeline; both are read at container start, so a restart is required.

### Expected baseline for regression comparison

Latest 9-case run on this deployment, all at the current configuration: retrieval Hit@K 1.000,
identification 0.750, unsupported-claim 0.111, rejection precision and recall 1.000, citation
leak 0.000, TTFT p50 ~7.2 s. Stage medians: query understanding ~1.7 s, dual retrieval ~0.5 s,
fusion ~0.001 s, verification ~4.4 s. Run-to-run noise is roughly ±0.2 s on stage medians and
±0.5 s on TTFT p50, so do not treat a sub-second difference as a regression without a repeat.

The one identification miss is `j10a-similar-model-disambiguation`, which answers correctly and
leaks no wrong citation but cites a page that does not state the model. It is a known open
retrieval-ranking issue, not a gate failure — see issue 05.
