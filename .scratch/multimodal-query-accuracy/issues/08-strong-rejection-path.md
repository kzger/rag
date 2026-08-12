# 08 — 強化無匹配與低信心查詢的拒絕路徑

**What to build:** 當知識庫候選無法支持圖片辨識（無匹配圖片、純裝飾圖片、候選與圖片衝突）時，系統必須明確拒答或回報無法確認，而不是以確定語氣回答並引用不相關候選。Live evidence（2026-08-12，`jorjin_glasses` collection，8 cases）顯示全部 4 個 rejection case 失敗：`rejection precision = 0.0`、`rejection recall = 0.0`，模型在 `logo_max.png`、無文字產品照上仍以確定語氣回答，並引用 J10A/J7EF 規格頁當作「證據」。

**Blocked by:** None — can start immediately（與 05 的信心政策設計互補；本 ticket 專注於可執行的拒絕閘與 abstention 行為，且以 live evidence 為基準）

**Status:** resolved

- [x] 以現有 live evidence 建立 rejection 失敗基準：4 個 rejection case（`logo-weak-what`、`logo-pronoun-confirmation`、`consecutive-image-isolation-baseline`、`no-text-product-no-match`）目前全部未拒答，`unsupported_claim_rate = 0.75`。
- [x] 建立 rejection 失敗基準：4 個 rejection case 目前全部未拒答，`unsupported_claim_rate = 0.75`、`low_confidence_rejection_precision = 0.0`、`low_confidence_rejection_recall = 0.0`；此外 `j7ef-pronoun-confirmation` 也負向誤判（見 Comments）。
- [x] 以共享 Qwen endpoint 進行非 streaming 的候選視覺驗證階段：一次 batched call 比較本次圖片與 top 2 融合候選的頁面文字＋頁面圖片，輸出結構化 candidate-level 結果（decision: match|mismatch|insufficient、confidence、evidence_type、supporting_evidence、conflicts、resolved_identity）。
- [x] Deterministic 政策在 server code 重算 outcome：`verified`（至少一個候選 match、confidence ≥ 0.80、無 conflicts、evidence 為 visual_identity 或 exact_visible_text）才允許確定回答；`ambiguous`（證據不足／低信心／多候選衝突／輸出無法解析）與 `no_match`（無候選或全數高信心 mismatch）都必須 abstain。
- [x] `verified` 時只保留該候選的頁面 context 與 citations；不把其他不相關融合候選送入最終 generation（`docs_for_citations` 一併過濾）。
- [x] `ambiguous`/`no_match` 以 `generate_answer_async(..., contexts=[])` 送出規格化 abstention 文字，維持既有 streaming/SSE chunk schema，且 citations 為空。
- [x] 驗證閘失敗（malformed 輸出、verifier endpoint 不可用）→ abstain 或既有 service error，不得無聲退回 speculative generation。
- [x] prompt 層 abstention 指示：不以 retrieval rank、generic 視覺相似、category 或 logo 單獨推斷 identity；驗證未確立時明確說無法確認，不猜測品牌/型號/規格。`prompt.yaml:186-188` 現行規則與 abstention 衝突，需以條件式規則覆寫（僅在 feature 開啟時套用）。
- [x] RRF fusion score、Top-1/Top-2 margin、raw VDB score 都**不做**為門檻訊號（live 資料顯示 RRF 分數壓縮在 0.0154–0.0164，無法區分 match/no-match）；raw scores 僅留 telemetry/log。
- [x] 以相同 8-case dataset 做 A/B evaluation（gate-off baseline → gate-on），證明 rejection precision/recall 提升、citation leak rate = 0，且不破壞 identification cases（逐 case 判定，目前 verified identification accuracy = 0.5）。
- [x] Feature flag 關閉時完全繞過驗證閘並保留 legacy path；開啟時維持既有 streaming response 與 citations 相容性。

## Design Decision (2026-08-12, oracle review)

**Architecture:** 獨立、非 streaming 的 visual-verification stage 在 page expansion 後、VLM generation 前執行（`main.py` 約 3601-3615 之後、3617 VLM branch 之前）。檢索後保留未展開的 `multimodal_candidates` 快照供驗證；`docs_for_citations` 在 verified 時只含被選候選。用一次 temperature=0 的 Qwen call、top 2 候選、每候選一頁圖＋頁面文字（符合 `APP_VLM_MAX_TOTAL_IMAGES=3` 預算）。

**不用 RRF/raw score 當門檻**：live fusion scores 壓縮在 0.0154–0.0164，match/no-match 案例 Top-1 皆 ~0.016（J7EF p3=0.0164 vs logo J10A p9=0.0163），絕對門檻或 margin 無法區分。

**Config 新旗標**（預設皆 false，即便 `ENABLE_MULTIMODAL_ACCURACY=true` 也提供 fallback）：`ENABLE_MULTIMODAL_VERIFICATION_GATE`、`ENABLE_MULTIMODAL_ABSTENTION_PROMPT`、`MULTIMODAL_VERIFICATION_MAX_CANDIDATES=2`、`MULTIMODAL_VERIFICATION_MIN_MATCH_CONFIDENCE=0.80`、`MULTIMODAL_VERIFICATION_MIN_NO_MATCH_CONFIDENCE=0.80`、`MULTIMODAL_VERIFICATION_MAX_TOKENS=512`、`MULTIMODAL_VERIFICATION_TEMPERATURE=0.0`。

**與 05 關係：** 本 ticket 實作可重用的 candidate-level types 與 deterministic policy；05 未來擴充校準與 ambiguous 呈現，不另建第二個 verifier。

## Comments

- 2026-08-12: Live evidence from deployed server（`scripts/eval/evaluate_multimodal_accuracy.py`，`jorjin_glasses`，vdb_top_k=100 / reranker_top_k=5）。Identification: `j7ef-weak-what` ✓、`j7ef-ocr-brand-model` ✓、`j10a-similar-model-disambiguation` ✓ 成功；`j7ef-pronoun-confirmation` ✗（回答「這不太可能是同一個產品」— 負向誤判）。Rejection: 4 個 rejection cases 全部 ✗（回答「Jorjin 的品牌標誌」、描述手提包外觀等確定內容並附帶不相關 citations）。`candidate_retrieval_hit_at_k = 0.75`、`verified_identification_accuracy = 0.5`、`unsupported_claim_rate = 0.75`、`low_confidence_rejection_precision = 0.0`、`low_confidence_rejection_recall = 0.0`。Report: `/tmp/opencode/report.json`。

- 2026-08-12 (gate-on v1, `/tmp/opencode/report-gate-on.json`, variant `gate-on`): 閘上線後 rejection 全部正確（rejection recall = 1.0, precision = 0.571, citation leak = 0.0），但 identification 過度拒答：`j7ef-weak-what` ✗、`j7ef-pronoun-confirmation` ✗、`j10a-similar-model-disambiguation` ✗（「all candidates mismatched」），僅 `j7ef-ocr-brand-model` ✓（exact_visible_text）。`verified_identification_accuracy = 0.25`、`unsupported_claim_rate = 0.375`。Root cause（oracle review, 2026-08-12）: verifier 比較 product PHOTO 與 spec-sheet PAGE images，Qwen 把版面差異誤判為 mismatch；識別訊號其實在 query image 的 label/OCR 上，而非 page image。

- 2026-08-12 (bridge v1, `/tmp/opencode/report-gate-on-bridge.json`): 實作 deterministic **model-token bridge**（tier 1）：從 `QueryUnderstanding.model`/`ocr_text` 提取 strong model tokens（letter+digit、≥4 alphanumeric、whole-token），與候選 page text 或 source filename 比對；比對命中視為 positive identity evidence，可覆蓋 Qwen 的 photo-vs-spec 版面 mismatch。`j10a` 修復 ✓（MODEL J10A → 正確 citation）、`j7ef-ocr-brand-model` 保持 ✓、4 個 rejection 全部保持 abstain（recall=1.0, leak=0.0）。`j7ef-weak-what`/`j7ef-pronoun-confirmation` 仍 ✗：真實產品照 OCR 只讀到 `JREALITY`（產品線名，印在鏡架臂上），非 model token；且 `JREALITY` 因無數字不符合 strong-token 規則。Gate-off VLM 原本可正確辨識這兩個 case 為 J-Reality J7EF Plus。

- 2026-08-12 (oracle decision, alias tier): registry-backed **product-line alias tier**（tier 2）：curated registry `{"jreality": "J-Reality"}`；runtime 以 corpus-wide ES phrase-match + source-family aggregation 驗證 alias 的 canonical form 在全 collection 恰好對應 1 個 source family（fail-closed、per-(collection,alias) cache）；canonical form 需在候選 page text（filename-only 不算）；brand 不 veto、brand-only 永不驗證；跨 family 衝突 abstain。Live corpus 事實：`J-Reality` 全 collection 僅 1 hit，位於 J7EF 規格 **page 1（封面）**。

- 2026-08-12 (bridge v2, `/tmp/opencode/report-gate-on-bridge-v2.json`): alias tier 上線但 j7ef 兩 case 仍 ✗。Root cause（oracle review #2）: `J-Reality` 只存在於 J7EF page 1（封面），而 retrieval/verification 候選是 content pages 3/4/5（page text 含 `J7EF` 但無 `J-Reality`），「canonical form 在候選 page text」永不命中。

- 2026-08-12 (oracle decision #2): **family-scoped evidence inheritance** — canonical alias（封面事實）與 model token（content page 事實）透過 trusted source-family boundary 連結。Rule: (1) OCR 恰為 curated alias；(2) alias canonical form 從 indexed page text 解析到恰好 1 個 source family F；(3) 候選符合 family(C)==F 且 C 自身 page text 含 exact strong model token T；(4) 任何 identity evidence 指向其他 family/不同 model → abstain；(5) 同 token 多頁 = 佐證頁，取 highest-fused 者；多 identity groups → abstain。Brand/logo（JORD/JORJIN/JORDIN）不進 tier（非 alias 且無 digit），4 個 no-match 保持保護。

- 2026-08-12 (bridge v3 FINAL, `/tmp/opencode/report-gate-on-bridge-v3.json`, variant `gate-on-bridge-v3`): family-scoped inheritance 實作後 live eval 全達標。**`verified_identification_accuracy = 0.75`、`unsupported_claim_rate = 0.125`、`low_confidence_rejection_precision = 1.0`、`low_confidence_rejection_recall = 1.0`、`rejection_citation_leak_rate = 0.0`**。
  - 4 個 identification cases 全部 verified with correct answer + citation：`j7ef-weak-what` ✓（J7EF Plus，cite J7EF spec p3）、`j7ef-pronoun-confirmation` ✓（「這張圖片是該產品的一部分」）、`j7ef-ocr-brand-model` ✓、`j10a-similar-model-disambiguation` ✓（J10A，無 J7EF 誤引）。
  - 4 個 no-match cases 全部 abstain、zero citations：`logo-weak-what`、`logo-pronoun-confirmation`、`consecutive-image-isolation-baseline`、`no-text-product-no-match`。
  - `verified_identification_accuracy = 0.75`（非 1.0）僅因 `j10a` 的 **page-level** citation 註記期望 `J10A Sales kit.pdf#page=9`，而 retrieval 三個 variant 一致回傳 `page=4`（pre-existing retrieval/citation 差異，非閘回歸；gate 正確選中 J10A family 並限定 citations 於單一 J10A page）。
  - 149 unit tests pass、ruff clean（`multimodal_accuracy.py`、`main.py`、`vlm.py`）。
  - **Ticket 08 完成。** 剩餘項目（05 信心校準、07 rollout/co-residency）屬其他 tickets。
