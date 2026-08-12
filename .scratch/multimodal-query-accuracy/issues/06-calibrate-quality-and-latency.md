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
