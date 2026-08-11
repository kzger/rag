# 04 — 開放 Token、Context、Concurrency 與 Ingestion 調校

**What to build:** 讓單張 H100 custom deployment 可以在安全範圍內調整模型 context、token budgets、GPU memory、並行度與 ingestion workload，preflight 依型別、範圍和資源關係判斷是否合法，而不是要求完全等於基準設定。

**Blocked by:** 01 — 建立一般驗證與 Certified Profile 的擴充介面

**Status:** ready-for-agent

- [ ] Model context length、generation／agentic／summary token budgets 以正確型別、非負或正值及 context 關係驗證。
- [ ] GPU memory utilization 以合理浮點範圍驗證，一般模式不要求等於 FP8 或 NVFP4 的基準數值。
- [ ] Maximum sequences、agentic concurrency、ingestion batches、files per batch 與 summary parallelization 以正整數和明確安全範圍驗證。
- [ ] 合法修改任一 workload setting 後可通過一般 preflight，且 resolved configuration 保留該值。
- [ ] Token 或 context 關係超限、GPU utilization 越界及非法 concurrency／batch 值會在部署前提供可操作錯誤。
- [ ] 單卡 GPU placement 與不支援的高資源服務組合仍維持嚴格限制。
- [ ] FP8 與 NVFP4 的建議資源數值仍可作為 certified baseline，但不再是一般 custom configuration 的唯一合法值。
