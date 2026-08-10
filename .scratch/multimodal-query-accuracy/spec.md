# 提升 Image + Text Query 的 RAG 準確度

Status: ready-for-agent

## Problem Statement

使用者透過 WebUI 上傳圖片並搭配文字詢問時，無法被要求使用特定句型、提供品牌型號，或主動說明檢索意圖。常見輸入可能只有「這是什麼？」、「是這個嗎？」或「這又是什麼？」。目前 Image + Text Query 雖然有進入 knowledge base，但圖片路徑會跳過 query rewriting、query decomposition、text reflection 與 reranking，並在 image similarity search 後只採用最高分結果所屬的單一文件頁面。當最高分候選錯誤時，系統缺少其他候選、信心判定與二次視覺驗證，最終 VLM 容易根據錯誤 context 產生看似肯定、實際半對半錯的回答。

多輪對話還可能同時把歷史圖片、本次圖片與 retrieved page images 納入有限的 image budget。當歷史圖片用完 image budget 時，真正能驗證答案的 retrieved page images 可能無法送進 VLM，且「這」等指示詞可能被錯誤地對應到先前圖片。

問題不在於使用者沒有寫出足夠好的 prompt，而在於系統沒有把弱文字查詢與圖片轉換成可靠的 standalone multimodal query，也沒有在資料不足或候選歧義時安全拒答。

## Solution

Image + Text Query 將採用一條對使用者提問方式具韌性的 multimodal retrieval and verification pipeline。系統先從本次圖片自動取得 caption、OCR、品牌、型號、物件類別與可辨識視覺特徵，並結合當前文字與必要的對話上下文形成 standalone query。接著同時進行 visual retrieval 與 enriched-text retrieval，保留多個不同文件／頁面候選並融合排序，而不是在第一次 similarity search 後立即鎖定單一頁面。

融合後的候選將由共用 Qwen vision-language generation endpoint 進行二次視覺驗證；驗證只使用本次圖片、候選文件內容與有限數量的候選頁面圖片。歷史圖片不會原樣帶入本次識別流程，而是以先前已產生的文字摘要保留必要語意。只有通過信心門檻且具足夠候選區分度的 context 才能支持確定性回答。若沒有可信候選、候選間差距不足或證據互相衝突，系統必須說明無法確認、列出有證據支持的可能候選，並避免捏造品牌、型號或規格。

這套方案沿用 Local RAG Deployment 的 Full-Service Co-residency，重用既有 VLM embedding、VLM reranker 與共享 Qwen endpoint，不要求新增常駐 generation model。目標是在準確度優先的前提下，讓 Image + Text Query 的首次可見回應維持可接受的互動延遲。

## User Stories

1. As a WebUI 使用者, I want to 上傳圖片並只問「這是什麼？」, so that 我不需要知道產品名稱也能取得有資料依據的回答
2. As a WebUI 使用者, I want to 使用「這個呢？」等自然指示詞, so that 系統能理解我指的是本次上傳的圖片
3. As a WebUI 使用者, I want to 連續上傳不同圖片, so that 前一張圖片不會干擾本次產品辨識
4. As a WebUI 使用者, I want to 在圖片含有品牌或型號文字時自動利用 OCR 結果, so that 我不必手動抄寫圖片上的文字
5. As a WebUI 使用者, I want to 在圖片沒有清楚文字時仍利用外觀與物件特徵檢索, so that 純視覺圖片也能找到相關文件
6. As a WebUI 使用者, I want to 讓系統同時考量圖片與我附帶的文字, so that 任一模態資訊不足時可由另一模態補足
7. As a WebUI 使用者, I want to 讓模糊問題被自動改寫成 standalone query, so that 我不需要學習特定 prompt 格式
8. As a WebUI 使用者, I want to 讓系統比較多個可能的文件頁面, so that 第一個向量搜尋結果錯誤時仍有機會找到正確資料
9. As a WebUI 使用者, I want to 讓系統跨不同文件與頁面比較候選, so that 相似外觀的產品不會被過早判成同一型號
10. As a WebUI 使用者, I want to 讓候選頁面的圖片與文字一起參與驗證, so that 表格、Logo、產品照片與規格文字可以交叉確認
11. As a WebUI 使用者, I want to 在證據足夠時收到明確的品牌與型號, so that 我能快速辨識圖片內容
12. As a WebUI 使用者, I want to 在證據不足時收到誠實的不確定回答, so that 我不會把模型猜測誤認為文件事實
13. As a WebUI 使用者, I want to 在兩個候選非常接近時看到可能候選與差異, so that 我能理解系統為何無法唯一判定
14. As a WebUI 使用者, I want to 讓回答中的產品規格只來自通過驗證的 knowledge base context, so that 圖片辨識正確但規格內容不會被臆測
15. As a WebUI 使用者, I want to 看到引用的文件與頁面, so that 我可以自行核對辨識與回答來源
16. As a WebUI 使用者, I want to 在圖片與文件文字互相衝突時被告知衝突, so that 系統不會悄悄選擇其中一方並裝作確定
17. As a WebUI 使用者, I want to 在未選 collection 時清楚知道未使用 knowledge base, so that 我不會誤以為直接 VLM 回答是 RAG 結果
18. As a WebUI 使用者, I want to 在選定 collection 後保證 Image + Text Query 使用該 collection, so that 檢索範圍符合我的選擇
19. As a WebUI 使用者, I want to 讓過去對話的文字語意可被保留, so that 多輪追問仍然連貫
20. As a WebUI 使用者, I want to 讓過去圖片以摘要而非原始圖片保留, so that 對話連貫性不會耗盡本次視覺驗證的 image budget
21. As a WebUI 使用者, I want to 讓本次圖片在 image budget 中具有最高優先權, so that 系統永遠實際查看我剛上傳的圖片
22. As a WebUI 使用者, I want to 讓最相關的候選頁面圖片保留足夠 image budget, so that VLM 能比較 query image 與 knowledge base evidence
23. As a WebUI 使用者, I want to 在沒有檢索結果時收到「知識庫沒有相關資料」, so that 系統不會退回無依據的產品猜測
24. As a WebUI 使用者, I want to 在檢索或驗證服務失敗時收到可理解的錯誤, so that 服務故障不會被偽裝成低品質答案
25. As a 系統維運人員, I want to 觀察 query understanding、visual retrieval、text retrieval、fusion、verification 與 generation 各階段耗時, so that 我能定位準確度與延遲問題
26. As a 系統維運人員, I want to 記錄候選文件、頁面、原始分數、融合分數與驗證結果, so that 我能追查錯誤答案的來源
27. As a 系統維運人員, I want to 在 log 中隱藏 base64 圖片內容, so that 診斷資訊不會洩漏使用者圖片或造成大量日誌
28. As a 系統維運人員, I want to 用設定控制候選頁數、融合權重與信心門檻, so that 可以不改程式就進行品質調校
29. As a 系統維運人員, I want to 分別量測 retrieval hit rate、verification accuracy、answer groundedness 與 latency, so that 不會用單一主觀結果判斷改善是否有效
30. As a 系統維運人員, I want to 保持所有必要服務在單張 H100 上同時可用, so that 準確度改善不破壞 Full-Service Co-residency
31. As a 系統維運人員, I want to 重用共享 Qwen endpoint 完成 query understanding 與視覺驗證, so that 不需要新增常駐 generation model
32. As a 系統維運人員, I want to 限制額外 VLM 呼叫的輸入與輸出 token, so that 準確度改善仍具有可預測的延遲與 GPU 使用量
33. As a 系統維運人員, I want to 在功能停用時保留既有 multimodal pipeline 行為, so that 可以漸進部署並快速回退
34. As a 開發者, I want to 以同一個高層 generate seam 驗證完整多模態行為, so that 測試不會綁定易變的內部流程
35. As a 開發者, I want to 對圖片查詢使用多頁候選而非單頁鎖定, so that vector database 的 Top-K 參數具備實際語意
36. As a 開發者, I want to 對重複 chunks 依文件與頁面去重, so that 單一頁面的多個 chunks 不會排擠其他候選頁面
37. As a 開發者, I want to 將 visual 與 enriched-text 結果以可測試的 deterministic 規則融合, so that 相同輸入與候選分數產生穩定排序
38. As a 開發者, I want to 將 confidence policy 與自然語言回答分離, so that 是否可以確定回答不由生成 prompt 臨時決定
39. As a 開發者, I want to 在沒有候選分數或必要 metadata 時安全降級, so that 不完整資料不會造成例外或錯誤確定性回答
40. As a 品質驗證人員, I want to 使用包含「這是什麼？」等弱文字的測試案例, so that 測試能代表真實使用者行為
41. As a 品質驗證人員, I want to 測試外觀高度相似但型號不同的產品, so that 系統的候選驗證能力可以被量化
42. As a 品質驗證人員, I want to 測試圖片中的 Logo、型號、表格與無文字產品照, so that OCR 與 visual retrieval 路徑都受到覆蓋
43. As a 品質驗證人員, I want to 測試多輪對話中連續兩張不同圖片, so that history image isolation 不會回歸
44. As a 品質驗證人員, I want to 測試 knowledge base 沒有正確產品的圖片, so that low-confidence rejection 能阻止 hallucination
45. As a 品質驗證人員, I want to 比較改善前後的 P50/P95 首 token 與完整回答延遲, so that 準確度提升的成本可以被接受或調整

## Implementation Decisions

- 維持既有 `/generate` request schema；使用者不需要新增欄位或採用特殊提問格式。已選 collection 時仍以 `use_knowledge_base` 與 `collection_names` 決定 RAG 範圍。
- 新增可由部署設定控制的 multimodal accuracy pipeline。設定至少涵蓋功能開關、每一路候選數、最終不同頁面候選數、visual/text fusion 權重、絕對信心門檻、Top-1/Top-2 margin、驗證通過門檻，以及 generation 可使用的 retrieved page image 數量。
- 本次使用者圖片是 query image。歷史訊息中的原始圖片不得直接參與本次 retrieval、verification 或耗用 retrieved-context image budget；若後續對話需要其語意，使用先前產生且可稽核的文字摘要取代。
- Query understanding 使用現有共享 Qwen endpoint，從本次圖片與文字產生短而結構化的描述，至少包含 OCR text、品牌、型號、物件類別、顏色、形狀、Logo 與其他可區分特徵。輸出必須有固定 schema、受 token 上限約束，且解析失敗時可回退到原始 multimodal query。
- Query understanding 產生的內容用於檢索，不直接視為事實，也不得單獨支持最終品牌、型號或規格回答。
- Image + Text Query 同時執行 visual retrieval 與 enriched-text retrieval。Visual retrieval 使用本次圖片與必要的原始短文字；text retrieval 使用 OCR、caption、結構化特徵、原始文字及必要的對話摘要。
- 兩路檢索結果依 source identity 與 page number 聚合及去重，保留多個不同文件／頁面。不得因同一頁含多個 chunks 而排擠其他候選頁面。
- 融合排序採 deterministic rank fusion；預設使用 weighted reciprocal rank fusion，避免直接比較不同檢索路徑未校準的 raw scores。融合結果仍保留各路 rank、raw score 與融合分數供觀察與後續校準。
- Image retrieval 的 Top-K 代表候選搜尋範圍，不再只取第一個 similarity result。檢索介面必須回傳具 source、page、score 與 chunks 的候選頁面集合。
- 候選驗證使用共享 Qwen endpoint，比較本次 query image、standalone query、候選頁面文字及候選頁面圖片。驗證結果使用結構化 schema，至少包含 candidate identity、match decision、confidence、supporting evidence 與 conflicts。
- 現有 VLM reranker 可繼續用於 passage ranking，但不得把只有「這是什麼？」的 query text 當成充分的 reranker query。必須使用 enriched standalone query；若 query-side image 尚未被 reranker contract 支援，候選視覺驗證由共享 Qwen endpoint 完成。
- Final context 僅包含通過驗證的候選。若多個候選都通過但無法唯一區分，回答必須表達歧義並列出有證據支持的可能候選，不得任選一個。
- Confidence policy 是 generation 之前的 deterministic decision。至少考量候選是否存在、Top-1 絕對分數、Top-1/Top-2 margin、verification confidence、證據衝突與必要 metadata 完整性。
- Confidence policy 輸出三種結果：`verified` 可做確定性回答、`ambiguous` 只可列可能候選與差異、`no_match` 必須說明 knowledge base 無法確認。Final VLM prompt 必須遵守此結果。
- Generation image budget 依固定優先序分配：本次 query image、最高可信候選頁面圖片、次高可信候選頁面圖片。歷史圖片不得優先於任何上述圖片。
- VLM generation temperature 的 Local RAG Deployment 基準值調整為 0.0 或 0.1，以降低 context 已確定後的生成變異；最終值由評估結果決定。
- 所有 query understanding、retrieval、fusion、verification 與 confidence decision 必須產生結構化 observation。日誌只記錄文字摘要、來源、頁面與分數，不記錄完整 data URI 或 base64 圖片。
- API streaming contract 維持相容；在前置檢索與驗證完成後仍以既有格式串流答案與 citations。若前置階段失敗，使用既有錯誤格式回報，不得無聲退回 direct VLM guessing。
- 新功能以 feature flag 漸進啟用。停用時保留既有 image retrieval 行為，便於 A/B comparison 與安全回退。
- 遵守 ADR-0001：所有必要 generation、retrieval 與 ingestion GPU services 必須保持 Full-Service Co-residency；驗收期間不得用停止 ingestion service 換取 query capacity。
- 遵守 ADR-0002：text generation、vision-language generation、image query understanding、candidate verification 與 ingestion captioning 繼續共用已部署的 Qwen endpoint。不得把新增獨立常駐 VLM 當成必要條件。
- 若額外 Qwen 呼叫導致不可接受的延遲或資源競爭，優先縮短結構化輸出、限制候選頁數、平行執行 retrieval，以及快取圖片摘要；不得違反 Full-Service Co-residency。
- 不以單純增加 VDB Top-K、修改 system prompt 或要求使用者改寫問題作為完成條件，因為這些措施沒有解決單頁鎖定、歷史圖片污染及缺少信心判定的根因。

## Testing Decisions

- 主要且最高層的測試 seam 是公開的 multimodal `/generate` 行為。測試以固定的 in-process fake embedding、vector store、共享 Qwen response 與候選頁面資料驅動完整 pipeline，驗證最終回答類型、citations、候選來源與是否安全拒答，不斷言私有 helper 的呼叫順序。
- `/generate` 行為測試覆蓋：只有「這是什麼？」的 query、含 OCR 型號的 query、無文字產品照、相似產品、多輪不同圖片、無匹配候選、分數接近、驗證衝突、query understanding 解析失敗，以及下游服務錯誤。
- `/generate` 行為測試必須證明：已選 collection 時使用 knowledge base；未選 collection 時不把 direct VLM 回答標示為 RAG；通過驗證時引用正確文件頁；歧義或無匹配時不輸出未被證據支持的確定型號。
- `/generate` 行為測試必須證明 history image isolation：第二次 query 的 retrieval 與 verification 只收到第二張圖片；第一張圖片的原始 data URI 不存在於本次 VLM payload，但允許其文字摘要留在 conversation context。
- `/generate` 行為測試必須證明 image budget priority：即使歷史中已有圖片，本次 query image 與最高可信 retrieved page image 仍被送進 final VLM。
- 針對 vector store adapter 保留一組窄的 contract tests，驗證 image retrieval 回傳多個不同 source/page candidates、保留 scores、正確聚合同頁 chunks、跳過 metadata 不完整結果且不因單筆壞資料丟棄全部候選。這是支援最高層 seam 所需的唯一低層補充 seam。
- 針對 fusion 與 confidence policy 使用輸入輸出導向的 parameterized tests，驗證 visual/text rank 組合、去重、tie handling、Top-1/Top-2 margin、`verified`、`ambiguous` 與 `no_match`。測試只驗證 policy 結果，不依賴內部資料結構或演算法步驟。
- 沿用既有 RAG server generate endpoint tests、multimodal query rewriting tests、RAG main integration tests、Elasticsearch image retrieval tests、page context organization tests 及 multimodal reranker tests 作為 prior art；擴充現有 fixture 與 fake service seam，避免建立第二套測試框架。
- Unit tests 禁止網路呼叫。所有 Qwen、embedding、reranker、Elasticsearch 與 object storage interaction 都以現有 fake/mock seam 模擬。
- Integration validation 使用可重現的小型 multimodal collection，至少包含同品牌相似型號、帶 Logo/型號的圖片、無文字產品圖片、文件頁面圖片與 knowledge base 不存在的圖片。
- Quality evaluation 必須涵蓋至少四類指標：candidate retrieval Hit@K、verified identification accuracy、unsupported-claim rate、low-confidence rejection precision/recall。不得只以回答看起來合理作為驗收。
- Latency evaluation 記錄 query understanding、visual retrieval、text retrieval、fusion、verification、generation TTFT 與完整回答時間，並比較現況 baseline 的 P50/P95。最終門檻由可重現 evidence 決定，但實作必須避免無界限的序列 VLM 呼叫。
- Full-Service Co-residency 驗證沿用既有五分鐘 health/restart gate，並在測試期間持續執行 text query、Image + Text Query 與 ingestion traffic，確認準確度改動不造成 GPU OOM、container restart 或服務不可用。

## Out of Scope

- 不要求使用者採用固定 prompt、手動輸入品牌／型號或學習 RAG 專用句型。
- 不新增獨立常駐 LLM、VLM 或 captioning model；模型替換與新增 GPU 不屬於本規格。
- 不改變 Local RAG Deployment 的單張 H100 與 Full-Service Co-residency 約束。
- 不重新設計文件 ingestion、OCR 或 image captioning pipeline；只有在缺少必要 candidate page image／metadata 時才修補與 query accuracy 直接相關的輸出契約。
- 不改變 text-only query 的 query rewriting、query decomposition、reflection 或 reranking 行為。
- 不在本規格中加入 video、audio、即時攝影機串流或多張圖片的跨圖比較產品功能。
- 不把 WebUI 改造成要求使用者人工挑選候選的流程；歧義回答可以呈現候選，但不依賴人工選擇才能安全完成請求。
- 不承諾 Image + Text Query 與 text-only query 具有相同延遲；目標是可量測且受控的準確度／延遲取捨。
- 不以 prompt-only 改動宣告完成。

## Further Notes

- 2026-08-10 的實際服務紀錄證明 Image + Text Query 已使用 `jorjin_glasses` collection。一次 query 在約 464 ms 內取回 J7EF Plus 文件第 3 頁，另一次在約 419 ms 內只取回 J10A 文件第 9 頁；兩者皆在 retrieval 後進入 VLM generation。問題因此是 retrieval/verification quality，而不是完全繞過 RAG。
- 現況 text-only query 會執行 query rewriting、query decomposition、重複 retrieval 與 reranking，因此 latency 明顯較高；圖片 query 則跳過這些階段。新流程應只增加對準確度必要的 bounded stages，不直接複製完整 text-only pipeline。
- 現行 image retrieval 即使搜索多筆 similarity results，也只讀取最高分結果的 source/page，再回傳該頁 chunks。這是本規格最優先修正的行為。
- 現行 image history 會先占用 total image budget；當上限為兩張且對話已有兩張圖片時，retrieved page images 沒有剩餘 budget。歷史圖片隔離與明確 budget priority 必須一起完成，不能只提高上限。
- 現行 Local RAG Deployment 使用 `Qwen/Qwen3.6-27B-FP8` 作為共享 generation endpoint，並已限制每次 prompt 最多兩張圖片。實作可透過 feature flag 與評估結果調整 query pipeline 的有效圖片數，但若改動 vLLM 的全域 multimodal limit，必須重新通過 ADR-0001 的 co-residency gate。
- 規格中尚未定義最終數值門檻；agent 應先建立 baseline evaluation set，再根據 Hit@K、錯誤確定回答率、拒答品質與 P95 latency 提出可重現的預設值。
