# 驗證並記錄 Full-Service Co-residency

Type: task
Status: ready-for-human
Blocked by: 05

## 目標

以實測判定完整 Local RAG Deployment 是否能在 1×H100 80GB 達成 Full-Service Co-residency。結果只能是符合五分鐘穩定門檻的成功設定，或依 fallback ladder 實測後明確記錄不可行；不得改成 ingestion/query 分時後宣稱成功。

## 驗證程序

1. 固定 commit、Compose config、container image digests、model artifacts、driver/CUDA/Docker versions 與測試資料。
2. 同時啟動 Qwen、VLM embedding、VLM reranker、OCR、三個 detectors、Elasticsearch、必要 ingestion runtime/CPU dependencies、ingestor、RAG server 與 frontend。
3. 對必要服務送出代表性流量，確認觀測涵蓋 idle 與 active 狀態。功能正確性、品質、latency 與 throughput 只記為 follow-up evidence，不作部署成敗 gate。
4. 在所有必要服務 healthy 後開始五分鐘觀測；每隔固定間隔記錄 container health、restart count、GPU memory/utilization 與失敗 log。
5. 若 FP8 baseline 失敗，依序調整並每次重跑完整觀測：`gpu-memory-utilization 0.48 -> 0.45`，必要時 `max-model-len 8192 -> 4096`。保留 concurrency 1、ingestion batch 1、最多兩張圖片。
6. FP8 梯級仍失敗時，才以單一 `nvidia/Qwen3.6-35B-A3B-NVFP4` endpoint 取代 FP8。使用固定、與 ModelOpt checkpoint 相容的 vLLM image digest，`--quantization modelopt`、FP8 KV cache、H100 Marlin fallback、`gpu-memory-utilization 0.40`、context 8192 起步；重跑完整五分鐘觀測，並另行記錄文字/圖片 smoke 與效能資料。
7. 將成功設定或不可行結論回寫 runbook、研究實測結果、root glossary 與 ADR；推估值與實測值必須分開標示。

## 驗收條件

- [x] 驗證環境確實只有 1×H100 80GB，所有 GPU services 同時指派該卡，且測試期間沒有停止/換載 ingestion 或 query services。
- [x] 觀測開始前，所有必要 services 都通過自身與 dependency-aware healthcheck；觀測期間有代表性服務流量，但其功能結果不影響部署 gate。
- [x] 連續五分鐘內每個必要 container 都維持 running/healthy，所有 restart count 前後相同。
- [x] 收集並註記 OOM、fatal worker exit 或 dependency loss 等 log 訊號作為診斷 evidence；log 訊號本身不增加或改寫部署 gate。
- [x] 五分鐘內保留時間序列 evidence：timestamp、每個 container health/restart count、GPU memory used/free/utilization；記錄 peak VRAM 與最小 free VRAM。
- [x] 每一個實際嘗試的 FP8/NVFP4 profile 都記錄「設定、啟動結果、功能 smoke result、穩定觀測結果、失敗原因」；未執行的梯級不得寫成已驗證。
- [x] 若採 NVFP4，另行記錄 H100 上的文字/圖片 smoke、TTFT、tokens/s、peak VRAM，以及使用 Marlin compatibility path 而非原生 Blackwell FP4 kernel 的事實；這些 follow-up 指標不改變部署 gate。
- [x] 成功只授予第一個達成「全部必要服務連續五分鐘 healthy 且 restart count 不變」的 profile；若所有允許梯級失敗，明確結論為「1×H100 Full-Service Co-residency 不可行」，並保留增加 GPU 作為後續選項。
- [x] 操作 runbook 包含 prerequisites、secret 前置作業、start/verify/SSH access/logs/restart/stop-with-data、fallback 與故障判讀，命令可從乾淨 shell 重現。
- [x] `docs/research/qwen-local-rag-models.md` 新增帶日期的實測章節；`CONTEXT.md` 的術語仍使用 Local RAG Deployment 與 Full-Service Co-residency。
- [x] ADR 與結果一致：共享單一 Qwen endpoint 的決策保留；成功時記錄 accepted profile，失敗時依 ADR-0001 記錄不可行而非改寫為 time sharing。
- [x] 執行與變更範圍相稱的 Compose config、lint、單元/整合 smoke checks，並在最終提交前完成 repo `pre-commit run --all-files`。

## 交付證據

提交 machine-readable 的 resolved Compose config 或設定摘要、五分鐘觀測資料、測試矩陣與非敏感 logs。文件必須連到 evidence，並清楚區分官方資料、先前容量推估與本次 H100 實測。

## Comments

## Answer

The FP8 0.48 baseline passed for 306 seconds with all 14 required services
healthy and unchanged restart counts. Peak memory was 65,520 MiB and minimum
free memory was 15,560 MiB. The fallback ladder was not run. This ticket stays
`ready-for-human` only because its blocking lifecycle ticket still has the
privileged daemon-restart and off-host SSH checks pending.
