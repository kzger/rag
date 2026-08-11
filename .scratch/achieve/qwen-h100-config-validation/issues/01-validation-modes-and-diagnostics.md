# 01 — 建立一般驗證與 Certified Profile 的擴充介面

**What to build:** 建立可逐步取代固定值比對的驗證介面，讓使用者能清楚區分安全或相容性錯誤、一般設定關聯錯誤，以及僅代表偏離已驗證 H100 基準的 profile drift；此階段維持既有部署行為，為後續遷移建立安全的 expand seam。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] 驗證結果能區分安全／拓撲違規、設定值或跨欄位關聯錯誤，以及 certified profile drift。
- [ ] 每個錯誤至少指出 service、parameter、actual value 與 violated rule，不輸出任何 credential。
- [ ] 一般合法性驗證與 certified conformity 驗證具備獨立、可測試的入口，不再共用模糊的固定值失敗語意。
- [ ] 既有 FP8、NVFP4 與安全拓撲測試在 expand 階段維持原有結果。
- [ ] 後續 tickets 能逐批遷移規則而不要求一次改完所有 callers，且每批完成後測試仍可保持綠燈。
- [ ] 此 effort 不修改 `multimodal-query-accuracy` 的 ticket、狀態或檢索／生成品質演算法。
