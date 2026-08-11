# 07 — 切換 Lifecycle 至一般驗證並移除舊固定值契約

**What to build:** 完成 validator 的 expand–contract 遷移：標準 lifecycle 使用一般合法性驗證啟動合法 custom deployment，操作者可另外要求 certified conformity；舊的重複固定值檢查被移除，文件與 live evidence 能說明實際行為。

**Blocked by:** 06 — 提供獨立的 Certified H100 Profile Drift 驗證

**Status:** ready-for-agent

- [ ] Validate 與 up 的預設模式接受合法 custom configuration，並在輸出中清楚標示 custom 或 certified 狀態。
- [ ] Lifecycle 提供明確的 certified validation 選項，help 說明一般合法性與 profile conformity 的差異。
- [ ] 已遷移設定不再由舊的固定 expected dictionaries 或 command argument 清單重複限制。
- [ ] Shell syntax、resolved Compose、一般 validator、FP8/NVFP4 certified profiles 與 lifecycle tests 全部通過。
- [ ] 至少一組偏離舊固定值但符合新規則的 custom configuration 完成 end-to-end smoke test。
- [ ] GPU、ports、必要／禁止服務、digest、healthcheck、restart 與跨服務相容性等安全回歸測試仍會拒絕非法配置。
- [ ] 部署文件與 source-of-truth 註解列出可調設定、安全不變條件及失去 certified 狀態的影響。
- [ ] Evidence、logs 與 git diff 不包含 credentials；使用者無關的未提交變更不會被納入。
- [ ] 最終 Standards 與 Spec review 無阻擋 finding，提交具備 GPG signature 與 DCO sign-off。
