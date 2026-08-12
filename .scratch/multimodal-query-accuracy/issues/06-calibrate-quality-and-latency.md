# 06 — 校準多模態準確度與延遲預設值

**What to build:** 使用可重現的 baseline 資料集校準 multimodal accuracy pipeline，選出能顯著降低錯誤確定回答且維持可接受互動延遲的候選數、融合權重、信心門檻、圖片預算與 generation 設定。

**Blocked by:** 05 — 視覺驗證候選並執行信心政策

**Status:** ready-for-agent

- [ ] 以相同資料、collection 與模型對 feature flag 關閉／開啟進行 A/B evaluation。
- [ ] 評估包含使用者提供的兩張圖片、弱文字問題、相似產品、連續圖片、無匹配資料及故意歧義案例。
- [ ] 報告比較 Hit@K、verified identification accuracy、unsupported-claim rate、拒答 precision/recall、TTFT 與完整回答 P50/P95。
- [ ] 根據 evidence 選定 visual/text candidate counts、final page count、fusion weights、absolute confidence threshold、margin threshold、verification threshold 與 image budget 預設值。
- [ ] 將 Local RAG Deployment 的 VLM generation temperature 校準至 0.0 或 0.1 中有較佳 groundedness 的值。
- [ ] 額外 Qwen 呼叫具有有界 token 與候選數；若 latency 不可接受，優先縮短結構化輸出、平行 retrieval 或快取圖片摘要。
- [ ] 所有可調參數具設定說明、安全範圍與回退值，且不要求修改 API request schema。
- [ ] 報告明確列出尚未達標案例，不以少量成功示例取代整體指標。

## Comments

- 2026-08-12 (code review): **Item 4 is blocked by a plumbing gap, not by missing analysis.**
  `deploy/compose/docker-compose-rag-server.yaml` passes only three multimodal variables into the
  container (`ENABLE_MULTIMODAL_ACCURACY`, `ENABLE_MULTIMODAL_VERIFICATION_GATE`,
  `ENABLE_MULTIMODAL_ABSTENTION_PROMPT`) plus `APP_VLM_MAX_TOTAL_IMAGES` and `APP_VLM_TEMPERATURE`.
  There is no `env_file`, so the compose file's explicit `environment:` list is the whole contract:
  setting any other knob in `deploy/compose/.env` has no effect on the container.

  Verified against the running deployment:

  ```
  $ docker exec rag-server env | grep MULTIMODAL
  ENABLE_MULTIMODAL_ABSTENTION_PROMPT=true
  ENABLE_MULTIMODAL_ACCURACY=true
  ENABLE_MULTIMODAL_VERIFICATION_GATE=true

  $ docker exec rag-server python -c "...NvidiaRAGConfig().multimodal_accuracy..."
  visual_candidates=5  text_candidates=5  max_candidates=5
  visual_weight=0.5    text_weight=0.5    rrf_k=60
  verification_max_candidates=2  verification_min_match_confidence=0.8
  ```

  So `MULTIMODAL_VISUAL_CANDIDATES`, `MULTIMODAL_TEXT_CANDIDATES`, `MULTIMODAL_MAX_CANDIDATES`,
  `MULTIMODAL_VISUAL_WEIGHT`, `MULTIMODAL_TEXT_WEIGHT`, `MULTIMODAL_RRF_K` and the
  `MULTIMODAL_VERIFICATION_*` thresholds are **currently unreachable at runtime** — they sit at the
  `configuration.py` defaults regardless of configuration. This is why the calibration run selected
  no values for them: they could not be varied. **Adding them to the compose `environment:` block is
  a prerequisite for item 4**; only temperature and the image budget are tunable today.

- 2026-08-12: **`APP_VLM_MAX_TOTAL_IMAGES` 5 -> 3 is unsupported by the committed evidence.** All
  four calibration runs executed at 3; no 3-vs-5 comparison exists. The value was aligned across
  Compose, Helm and `docs/vlm.md` for consistency, but the choice itself is still unvalidated.

- 2026-08-12: **Latency is the largest open risk.** Feature-on adds roughly +7.9 s TTFT p50 and
  +10.2 s p95 over feature-off, on top of a ~6.5 s baseline. Item 6's mitigations (shorter
  structured output, parallel retrieval, cached image summaries) were not implemented. Deciding
  whether this is acceptable is a product call and should precede further accuracy tuning, since
  rejection is already 1.0/1.0 and identification 0.75.

- 2026-08-12: Helm parity for the ticket-08 verification flags and `prompt.yaml` was repaired while
  reviewing this ticket (it was the cause of two long-failing `test_compose_helm_parity` tests).
  Kubernetes is not a current deployment target, so no further Helm work is planned; any new tunable
  added to compose will need the same entry in `deploy/helm/nvidia-blueprint-rag/values.yaml` to keep
  that test green.

## Calibration evidence — verification stage (2026-08-12)

Tunables were forwarded through `docker-compose-rag-server.yaml` (and the matching
`values.yaml` keys) so they could finally be varied; defaults were set to the
`configuration.py` values, and a control run confirmed forwarding alone changed nothing.
A `Multimodal verification stage: verification_ms=…` log line was added because the gate
had no timing at all — logged stages accounted for only ~2.1 s of a ~7 s TTFT.

Stage breakdown at the shipped configuration (medians, 9 cases):

| stage | median |
| --- | --- |
| query understanding | 1.64 s |
| dual retrieval | 0.45 s |
| fusion | 0.001 s |
| **candidate verification** | **4.40 s** |

Grid over the two verification knobs, same dataset/collection/model:

| config | verification median | TTFT p50 | identification | unsupported claim | rejection precision |
| --- | --- | --- | --- | --- | --- |
| **candidates=2, max_tokens=512 (shipped)** | 4.40 s | 7.27 / 6.75 s | **0.750** | **0.111** | **1.000** |
| candidates=2, max_tokens=256 | 3.56 s | 5.89 s | 0.000 | 0.444 | 0.556 |
| candidates=1, max_tokens=256 | 1.88 s | 3.62 s | 0.250 | 0.333 | 0.714 |
| candidates=1, max_tokens=512 | 1.88 s | 4.16 s | 0.250 | 0.333 | 0.714 |

**Both knobs are load-bearing; neither can be reduced.** Item 4 can therefore record
`MULTIMODAL_VERIFICATION_MAX_CANDIDATES=2` and `MULTIMODAL_VERIFICATION_MAX_TOKENS=512`
as evidence-backed rather than inherited:

- Dropping to one candidate costs identification 0.750 -> 0.250 and rejection precision
  1.000 -> 0.714 regardless of token budget (the 256 and 512 rows are identical), because
  the gate only ever inspects the top-ranked page.
- Dropping to 256 tokens with two candidates is worse still — identification collapses to
  0.000, and rejection recall stays 1.0 only because it abstains on nearly everything while
  precision falls to 0.556. **The mechanism was not established.** Each experiment recreated
  the container, so that run's logs no longer existed when they were inspected. Truncated
  JSON is the plausible cause — two candidates' `supporting_evidence` and `conflicts` free-text
  fields ran 150-200 tokens in observed responses, and `parse_candidate_verification`
  fails closed to `ambiguous` — but it is unverified. Thinking mode is off
  (`APP_VLM_ENABLE_THINKING=false`), so "less room to reason" is *not* the explanation.
  To confirm, re-run at 256 and grep the same container for `malformed verifier output`.
- The shipped configuration reproduced across two independent runs (identification 0.750
  both times), and TTFT p50 varied 6.75-7.27 s between identical runs, so treat ~0.5 s as
  measurement noise.

**Latency conclusion for item 6:** the verification stage cannot be made cheaper through
these knobs without breaking the accuracy the gate exists to provide. Remaining options are
architectural, not configuration: overlapping query understanding with visual retrieval
(bounded by dual retrieval's 0.45 s, so ~0.45 s at best), caching image summaries across
turns, or sending smaller page images. Whether ~7 s TTFT is acceptable is still the
product decision this ticket flagged as unresolved — but it is now clear that the cost buys
the rejection guarantee, and that cutting it cheaply is not available.
