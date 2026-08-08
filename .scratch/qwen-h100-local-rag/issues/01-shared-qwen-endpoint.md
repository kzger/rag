# 建立共享 Qwen endpoint

Type: task
Status: ready-for-agent
Blocked by: None

## 目標

建立這次 Local RAG Deployment 的本機實作 branch，並在 1×H100 80GB 上以固定 artifact 啟動一個 OpenAI-compatible Qwen generation endpoint。文字生成、視覺生成與後續 ingestion captioning 都必須重用這個 process。

## 工作範圍

- 從 `develop` 建立並記錄本機 feature branch；所有後續 tickets 都在同一 branch 上完成。
- 為 `Qwen/Qwen3.6-27B-FP8` 新增可重現的 vLLM Compose service 或 Compose override。固定 container image digest，並以 Hugging Face commit SHA、等價 immutable revision 或本機 artifact checksum 固定模型權重；不得使用 floating `latest`、未記錄 digest 的 nightly 或可漂移的 model revision。
- 以單 GPU、`--max-model-len 8192`、`--gpu-memory-utilization 0.48`、`--max-num-seqs 1`、最多兩張圖片且 video 關閉作為 FP8 起始 profile；server-wide 關閉 Qwen thinking，避免短 context 被 reasoning 佔滿。
- 將模型 cache 放在持久化 named volume 或明確的持久化 host path，並加入 repo 的 `nvidia-rag` network。
- container 內使用 port 8000；host 只以 `127.0.0.1:8999` 發布。後續 Compose service 以 service DNS 和 `/v1` 連線，不以 container 內的 `localhost` 連線。
- 在後續 ticket 06 前只建立 FP8 baseline；NVFP4 是容量 fallback，不在本票同時常駐。

## 驗收條件

- [ ] `git branch --show-current` 顯示從 `develop` 建立的 feature branch，且 branch 名稱已記錄在操作文件或變更說明。
- [ ] Qwen service 的 model repository、immutable model revision/checksum、vLLM image digest、served model name、啟動參數與 GPU 0 assignment 都由已提交的 Compose/env 設定決定。
- [ ] `docker compose config` 成功，解析後的 host publish 是 `127.0.0.1:8999:8000`，service 同時位於 `nvidia-rag` network。
- [ ] `nvidia-smi` 證明目標是單張 H100 80GB，且 Qwen process 只使用 GPU 0；不符合時停止並記錄 blocker，不偷偷改成多 GPU。
- [ ] 第一次啟動完成後，`curl http://127.0.0.1:8999/v1/models` 與文字 `/v1/chat/completions` smoke test 成功，回傳的 model name 與設定一致。
- [ ] 使用 data URL 圖片的 `/v1/chat/completions` smoke test 成功，證明同一 endpoint 可處理 vision input。
- [ ] readiness healthcheck 能區分「process 已啟動」與「模型可接受請求」，Qwen 未 ready 時依賴服務不會被判定 ready。
- [ ] 重建 container 後不重新下載完整模型；cache volume/path 仍存在且模型 endpoint 再次 ready。
- [ ] `ss -ltn` 顯示 8999 只綁定 loopback，且沒有第二個 generation model container。

## 證據

在 ticket 完成紀錄中附上 branch、resolved Compose config、container image digest、model revision/checksum、`nvidia-smi`、兩個 smoke tests、cache 重建結果及 listening socket 摘要。不得記錄 API key 或其他 secret。

## Comments
