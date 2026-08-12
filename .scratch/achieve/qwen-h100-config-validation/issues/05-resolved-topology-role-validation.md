# 05 — 以 Resolved Topology 驗證模型角色、Endpoints 與 Ports

**What to build:** 從實際 resolved services、served model、endpoints 與 port mappings 驗證各模型角色是否能互相連線且符合 Qwen H100 topology，移除與 deployment source-of-truth 重複的模型或 endpoint 固定常數，同時維持安全邊界。

**Blocked by:** 01 — 建立一般驗證與 Certified Profile 的擴充介面

**Status:** resolved

- [x] LLM、VLM、query rewriter、filter generator、reflection、summary 與 agentic roles 的共享模型關係由 resolved Qwen service 推導。
- [x] Embedding、ranking、caption 與 ingestion runtime endpoints 會對照 resolved services 驗證，不接受不存在或不相容的目標。
- [x] 一般驗證不再因模型名稱偏離 Python 內的重複常數而失敗，但會拒絕角色與實際 served model 不一致的配置。
- [x] 所有公開 host ports 仍只允許 loopback、合法且不重複；可配置與固定 contract 的 ports 具有明確規則和錯誤訊息。
- [x] 必要服務、禁止服務、GPU 0 placement、image digest、healthcheck 與 restart policy 的既有嚴格檢查保持有效。
- [x] Model revision 與 variant 關係在一般合法性和 certified conformity 間有清楚分工。
- [x] 錯誤與 configuration summary 不包含 API keys 或其他 secrets。
