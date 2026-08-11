# 02 — 讓 Retrieval 與對話參數成為真正可配置設定

**What to build:** 讓使用者可以透過 Docker source-of-truth 設定調整 retrieval 與對話控制值，合法的 custom configuration 能通過一般 preflight 並進入部署，而不是因為不等於舊基準數字而被拒絕。

**Blocked by:** 01 — 建立一般驗證與 Certified Profile 的擴充介面

**Status:** ready-for-agent

- [ ] Retrieval candidate pool、final Top-K、conversation history、recursion depth 與 reflection loop 等值改以型別和合理範圍驗證。
- [ ] Candidate pool 與 final Top-K 的次序關係不合法時，preflight 在部署前以具體參數和值拒絕。
- [ ] 合法修改上述任一設定後，resolved deployment configuration 與執行中的 service 都能看到使用者設定值。
- [ ] 合法 custom 值不會只因偏離舊的固定數字而被一般驗證拒絕。
- [ ] 非整數、負數、零值或超出語意範圍的設定會安全失敗。
- [ ] 既有 text-only、multimodal 與 agentic retrieval 行為不因 validator 重構而改變。
