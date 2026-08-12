# 05 — 視覺驗證候選並執行信心政策

**What to build:** 系統會以本次圖片與融合後的候選頁面證據進行二次視覺驗證，並在生成答案前決定結果是 verified、ambiguous 或 no_match，避免把錯誤候選或模型猜測描述成確定事實。

**Blocked by:** 02 — 讓圖片檢索回傳多個不同頁面候選；03 — 隔離歷史圖片並保障本次圖片預算；04 — 自動理解弱圖片查詢並融合雙路檢索

**Status:** resolved

- [x] 候選驗證重用共享 Qwen endpoint，比較本次 query image、standalone query、候選文字與可用候選頁面圖片。
- [x] 每個驗證結果包含 candidate identity、match decision、confidence、supporting evidence 與 conflicts，且結構解析失敗不會被視為 verified。
- [x] Deterministic confidence policy 使用候選存在性、verification confidence、evidence conflicts 與必要 metadata；RRF/raw score/margin 僅 telemetry。
- [x] `verified` 只使用通過驗證的 context 回答並提供正確 citation；caption 或 query understanding 的推測不得成為唯一證據。
- [x] `ambiguous` 說明無法唯一判定並只列出有具體身份證據支持的候選與衝突，不任選型號。
- [x] `no_match` 明確說明 knowledge base 無法確認，不會無聲退回 direct VLM guessing。
- [x] 未選 collection 時，使用者可辨識該回答沒有使用 knowledge base；下游服務失敗使用既有錯誤格式回報。
- [x] 以 `logo_max.png` 及 knowledge base 無匹配案例證明系統不會因單一錯誤頁面候選而確定宣稱產品型號。
- [x] Feature flag 關閉時可回退既有 multimodal pipeline；開啟時維持既有 streaming response 與 citations 相容性。

## Comments

- 2026-08-12: Ticket 08 supersedes RRF/raw VDB score/Top-1 margin rejection gating; these values remain telemetry-only. Ordinary high-confidence `mismatch` is now fail-closed and cannot qualify as `verified`; the intentional model-token/product-line bridge remains available for photo-vs-spec layout mismatch.
- 2026-08-12: Tests cover strict candidate schema parsing, ordinary mismatch rejection, identity-only ambiguous evidence, and the existing public `NvidiaRAG.generate` retrieval/verification seam. Final verification owner: orchestrator.
- 2026-08-12: Ticket 08 bridge-v3 live evidence is the completion evidence: identification accuracy 0.75, unsupported-claim rate 0.125, rejection precision/recall 1.0, and rejection citation leak rate 0.0.

- 2026-08-12 (code review fix): The bridge path in `decide_multimodal_outcome` accepted only
  `decision == "mismatch"`, so a verifier that *agreed* (`match`) on the same bridge evidence was
  rejected while a disagreeing one was verified — an inversion, since agreement carries strictly
  more evidence than an override. Widened to `decision in {"match", "mismatch"}`; `insufficient`
  stays excluded (it asserts nothing to corroborate) and remains pinned by
  `test_bridge_does_not_override_conflicts_or_insufficient_verdict`. New regression test:
  `test_bridge_verifies_whether_or_not_the_verifier_agrees`.
  **Live re-validation still owed:** this only fires when a model-token/alias bridge match exists,
  so the four no-match rejection cases are structurally unaffected (verified locally: without a
  bridge every decision still yields `no_match`), but the 8-case live eval has not been re-run
  against the widened gate. Re-run before treating the metrics above as current.

- 2026-08-12 (live A/B, deployed `jorjin_glasses`, 8 cases): **ticket-05 as merged regressed the
  ticket-08 model-token bridge.** Measured on the real server (rebuilt container per run):

  | metric | ticket-08 (`de7db4f`) | ticket-05 as merged | after bridge repair |
  |---|---|---|---|
  | `verified_identification_accuracy` | 0.50 | **0.25** | 0.50 |
  | `unsupported_claim_rate` | 0.25 | **0.375** | 0.25 |
  | rejection precision / recall | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
  | `rejection_citation_leak_rate` | 0.0 | 0.0 | 0.0 |
  | `candidate_retrieval_hit_at_k` | 0.75 | 0.75 | 0.75 |

  Root cause (from new `Verifier judgment` / `Bridge matches` logging in `main.py`): ticket-05 added
  three independent narrowings that each disable the bridge on real verifier output.
  1. `any(item.conflicts for item in candidates)` — Qwen populates `conflicts` on essentially every
     `mismatch`, and it fired even when a *non-bridged* candidate was merely explaining why it did
     not match (`j10a`: C1 was `match`/1.00/`exact_visible_text`/no conflicts, vetoed by C2).
     Now scoped to the bridged candidate and only when its conflicts name a model token outside
     `bridge_tokens`.
  2. Bridge selection required `evidence_type == "generic_similarity"`, but the verifier reports
     `visual_identity` for these photo-vs-spec disagreements. Restriction removed; genuine identity
     contradictions are still caught by the guard above.
  3. `_contradicts_bridge` treated a bare `decision == "mismatch"` as a contradiction — that is
     precisely the layout disagreement the bridge exists to override.

  Rejection behaviour was never affected in any run: the four no-match cases produce
  `Bridge matches: {}`, so this whole block is skipped and they resolve via `all candidates
  mismatched`. `test_bridge_does_not_override_conflicts_or_insufficient_verdict` was updated to
  encode the evidence-based rule (a conflict vetoes only when it names a different model).

  **Remaining 0.50 is measurement artifact, not model error** — all four identification cases now
  answer correctly with correct citations:
  - `j7ef-pronoun-confirmation`: answers「是的，這是 JORJIN TECHNOLOGIES 的 J7EF Plus 產品」and cites
    p3 correctly, but the fixture's `forbidden_answer_terms` contains the bare substring `不是`,
    which matches innocuous prose elsewhere in the answer (「…也不是紅黑相間的耳機」). The fixture
    needs a phrase-level check, not a substring.
  - `j10a-similar-model-disambiguation`: answers `J10A` correctly and cites a single J10A page, but
    retrieval returns `page=4` while the fixture expects `page=9` — the pre-existing
    retrieval/citation difference already recorded under ticket 08.

- 2026-08-12 (eval fixture repair + final run, variant `final-fixed-fixture`): three measurement
  bugs fixed in the harness, none of them server behaviour.
  1. `IDENTIFICATION_NEGATION_TERMS` substring-matched bare `不是`/`並非`/`not the`, so a correct
     answer that used a negator in passing (「…也不是紅黑相間的耳機」, itself a correct remark about
     mis-associated retrieved text) scored as a denial. Replaced with `_denies_identity`:
     phrase patterns for identity denial plus a negator anchored immediately before an accepted
     model term, so 「這不是 J7EF Plus」/「這不太可能是同一個產品」/`This is not the same product`
     still register. Covered by the existing `test_compute_metrics_*` case.
  2. `j7ef-*` accepted only `J7EF Plus`/`J7EF+`, but the source PDF is titled
     「JORJIN TECHNOLOGIES-J7EF **PULS** 產品規格_v3」, so an answer echoing the document's own
     spelling was marked wrong. `J7EF Puls` added to the accepted terms.
  3. `j10a` expected `J10A Sales kit.pdf#page=9`, which is a bare logo/brandmark on an orange
     background carrying no identifying content (verified against the indexed page text). Corrected
     to the pages that actually identify the product: `#page=1` (text "J10ASales Kit") and
     `#page=2` (product image + spec table).

  | metric | ticket-08 | ticket-05 as merged | bridge repaired | **final** |
  |---|---|---|---|---|
  | `candidate_retrieval_hit_at_k` | 0.75 | 0.75 | 0.75 | **1.00** |
  | `verified_identification_accuracy` | 0.50 | 0.25 | 0.50 | **0.75** |
  | `unsupported_claim_rate` | 0.25 | 0.375 | 0.25 | **0.125** |
  | `low_confidence_rejection_precision` | 1.0 | 1.0 | 1.0 | **1.0** |
  | `low_confidence_rejection_recall` | 1.0 | 1.0 | 1.0 | **1.0** |
  | `rejection_citation_leak_rate` | 0.0 | 0.0 | 0.0 | **0.0** |

  All four no-match cases abstain with zero citations in every run. The one remaining
  identification miss is **a genuine open finding, deliberately left failing**:
  `j10a-similar-model-disambiguation` answers `J10A` correctly and leaks no J7EF citation, but
  cites `page=4` (a product rendering that does not state the model) rather than the identifying
  page 1 or 2. The verification gate is behaving correctly — it restricted citations to one J10A
  page — so this is a retrieval/citation-ranking question, not a gate question.
