# 06 — 提供獨立的 Certified H100 Profile Drift 驗證

**What to build:** 讓操作者能明確選擇檢查 FP8 或 NVFP4 是否完全符合已實測的單張 H100 co-residency baseline；合法 custom configuration 仍可部署，但會被清楚標示為非 certified，而不是被誤判成非法。

**Blocked by:** 02 — 讓 Retrieval 與對話參數成為真正可配置設定；03 — 統一 RAG 與 Qwen 的多模態圖片預算；04 — 開放 Token、Context、Concurrency 與 Ingestion 調校；05 — 以 Resolved Topology 驗證模型角色、Endpoints 與 Ports

**Status:** ready-for-agent

- [ ] 未修改的 FP8 baseline 通過一般驗證與 FP8 certified validation。
- [ ] 未修改的 NVFP4 baseline 通過一般驗證與 NVFP4 certified validation。
- [ ] 合法但偏離基準的 custom configuration 通過一般驗證，certified validation 則回報 profile drift。
- [ ] Drift report 列出 parameter、baseline value 與 actual value，不把 drift 混同為安全、型別或相容性錯誤。
- [ ] Profile baseline 只負責 conformity，不會重新成為一般 validator 的隱藏固定值來源。
- [ ] Configuration summary 能指出目前是 certified FP8、certified NVFP4 或 custom，且不包含 secrets。
- [ ] Certified evidence 能重現使用的 model revision、resource envelope 與關鍵 tuning values。
