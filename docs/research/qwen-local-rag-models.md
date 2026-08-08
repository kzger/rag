# 本機 NVIDIA RAG Blueprint 的 Qwen3.6 LLM / VLM 選型研究

## 2026-08-08 H100 實測結果

在單張 81,559 MiB H100 上，原始建議 profile
`Qwen/Qwen3.6-27B-FP8`（context 8192、vLLM utilization 0.48、concurrency
1）成功讓 Qwen、VLM embedding/reranker、OCR、三個 detectors 與全部 CPU/
application services 同時常駐。完整結果與互斥 pipeline 測試矩陣以
[`evidence/qwen-h100-runtime-results.md`](evidence/qwen-h100-runtime-results.md)
為準；時間序列由同目錄的 JSONL 保存。

這次 baseline 成功，因此 0.45、4096-context 與 NVFP4 梯級都**未執行**，
不能寫成 H100 實測結果。先前 73–77 GiB 數字仍是保守推估；本次 idle
spot observation 約使用 62,069 MiB，active-ingest spot peak 約 62,323
MiB，兩種資料不可混用。部署 acceptance 只依五分鐘 health/restart gate，
功能正確性與 TTFT 留作 follow-up evidence。

> 資料截點：2026-08-08。模型狀態以 Qwen 官方 Hugging Face 組織、Qwen 官方模型卡、vLLM 官方文件與本 repository 原始碼為準。本文不把社群量化模型列為首選。

## 結論先行

在 LLM/VLM 必須採 Qwen3.6 的前提下，本機 1×H100 80GB 的推薦是 **`Qwen/Qwen3.6-27B-FP8` 作單一 generation endpoint**，同時填入 `APP_LLM_*` 與 `APP_VLM_*`。為了跟多模態 embedding、VLM reranker、OCR 及三個 object-detection NIM 共用同卡，起始值應保守設為 `--max-model-len 8192 --gpu-memory-utilization 0.48 --max-num-seqs 1`；這是容量規劃**推估**，不是 Qwen、vLLM 或 NVIDIA 驗證過的單卡整套拓撲。

本專案可以接自架的 vLLM，不要求 generation model 一定是 NVIDIA NIM：

- LLM 路徑接受 `APP_LLM_MODELNAME` 與 `APP_LLM_SERVERURL`；實作把 endpoint 傳給 `ChatNVIDIA(base_url=...)`，對自架 Qwen 不會附加 Nemotron 專用 generation 參數（[configuration.py](../../src/nvidia_rag/utils/configuration.py)、[llm.py](../../src/nvidia_rag/utils/llm.py)）。
- VLM 路徑接受 `APP_VLM_MODELNAME` 與 `APP_VLM_SERVERURL`，而且直接使用 OpenAI Python SDK 的 `AsyncOpenAI(base_url=...)` 呼叫 `client.chat.completions.create(...)`，圖片採 OpenAI `image_url` content part（[vlm.py](../../src/nvidia_rag/rag_server/vlm.py)）。
- vLLM 官方 server 提供 `/v1/chat/completions` 與 `/v1/completions`；其中 Completions 只適用文字模型，這個 RAG 專案的 generation 實際需要 **Chat Completions**。vLLM 的 OpenAI-compatible Chat API 也支援 vision input（[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)、[multimodal inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)）。
- Qwen3.6 本身是帶 vision encoder 的 post-trained causal language model；同一個 vLLM process 可以同時接受文字及 OpenAI `image_url` input，因此不需要另外常駐一個大型 generation VLM。
- NVIDIA Blueprint 官方文件對「完整 VLM generation + VLM embedding + VLM reranker」的規劃是至少 3 GPUs，而不是單卡；以下單卡方案是資源受限的實驗性共置，需要限制 batch/context，最好把 ingestion 與 query 分時執行（[repo VLM hardware note](../vlm.md)、[repo multimodal reranker hardware note](../multimodal-retriever.md)）。

最穩妥的實務設定是先把 vLLM context 限在 **8K–32K**，確認整套 RAG 的 embeddings、reranker、ingestion services 仍有足夠 VRAM，再逐步增加。模型卡的 256K/1M 是模型能力上限，不是單卡本機服務的免費容量。

## Qwen3.6 官方 post-trained multimodal variants

截至 2026-08-08，Qwen 官方 Hugging Face organization 以 `Qwen3.6` 搜尋得到四個適合 generation 的 post-trained repositories，全部是 `image-text-to-text`、`private=false`、`gated=false`；沒有 Qwen 官方 GPTQ、AWQ 或 GGUF Qwen3.6 checkpoint（[Qwen3.6 官方 collection](https://huggingface.co/collections/Qwen/qwen36)、[Hugging Face Qwen API 搜尋](https://huggingface.co/api/models?author=Qwen&search=Qwen3.6&limit=100&full=true)）。

| 官方 model | 架構 / 官方量化 | 原生 / 延伸 context | `.safetensors` 檔案合計 | 1×H100 80GB 與整套 NIM 共置 |
|---|---|---:|---:|---|
| [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | dense 27B, BF16 | 262,144 / YaRN 1,010,000 | 51.75 GiB | 不建議；光權重已吃掉大部分單卡 |
| [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | dense 27B, official fine-grained block-128 FP8 | 262,144 / YaRN 1,010,000 | 28.75 GiB | **推薦**；仍需 8K 起步與低 concurrency |
| [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | MoE 35B total / 3B active, BF16 | 262,144 / YaRN 1,010,000 | 66.97 GiB | 不可與所列 NIMs 安全常駐；active 3B 不代表只載 3B 權重 |
| [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | MoE 35B total / 3B active, official fine-grained block-128 FP8 | 262,144 / YaRN 1,010,000 | 34.89 GiB | generation 單服務可優先評估；整套單卡共置不如 27B-FP8 保守 |

檔案合計由各官方 Hugging Face model API 的 `siblings[].size` 相加，例如 [`Qwen3.6-27B-FP8?blobs=true`](https://huggingface.co/api/models/Qwen/Qwen3.6-27B-FP8?blobs=true)。它只表示 serialized weight payload，不包含 vision preprocessing、activation、CUDA graph、KV/state cache 或 allocator overhead，**不是實測 VRAM**。

HF metadata 的精度計數可進一步核對檔案尺度：27B-FP8 為 24,699,207,680 個 F8_E4M3 與 3,083,727,792 個 BF16 parameters；35B-A3B-FP8 為 34,453,061,632 個 F8_E4M3 與 1,500,863,920 個 BF16 parameters。兩個官方 FP8 checkpoint 都是 dynamic activation、E4M3、`weight_block_size=[128,128]` 的 fine-grained FP8；vision encoder、embedding、LM head 等部分保留 BF16。以上仍是權重格式/數量，不是 runtime VRAM。

四張官方 model card 都標示 Apache-2.0、Causal Language Model with Vision Encoder、文字與圖片 OpenAI Chat Completions 範例；FP8 卡說明其 block size 為 128，且官方稱 benchmark 接近原模型。Qwen3.6 預設 thinking，不支援 Qwen3 的 `/think`、`/nothink` 軟切換；vLLM/SGLang 必須以 `extra_body.chat_template_kwargs.enable_thinking=false` 關閉 thinking。

### 官方 vLLM 條件

Qwen3.6 model card 建議 **`vllm>=0.19.0`**，並以 8 GPUs、262K context 示範：

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

這是官方多卡 full-context 範例，不可解讀為單張 H100 也能用 262K。官方卡明示 OOM 時應降低 `--max-model-len`，並說 Qwen3.6 若要保留完整 thinking 能力建議至少 128K；因此本文的 8K/16K 是為了單卡共置所作的**功能性妥協**（[27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)、[35B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)）。

vLLM 官方 recipes 另列 27B-FP8 可用單張 40GB 級 H100/H200/L40S、35B-A3B-FP8 可用單張 H100/H200；這只證明 **generation model 本身**的單卡路線，不代表還能同時容納本 repo 全部 NIM，也不是各 context 長度的 peak-VRAM benchmark（[vLLM Qwen3.6-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)、[vLLM Qwen3.6-35B-A3B recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)）。

Qwen3.6 的 image request 使用 OpenAI content list 中的 `{"type":"image_url", ...}`，與本 repo `vlm.py` 完全同形；若加 `--language-model-only` 雖能省 vision encoder，但將失去本需求所需的圖片輸入能力。

## NVIDIA NVFP4 35B 是否更適合 1×H100？

### 明確結論

**不把 `nvidia/Qwen3.6-35B-A3B-NVFP4` 列為 H100 的正式首選；H100 仍優先使用 `Qwen/Qwen3.6-27B-FP8`。** NVFP4 checkpoint 的 serialized weights 確實更小，而且 NVIDIA 模型卡明列 Hopper 相容；vLLM 的 ModelOpt mixed-precision path 也能把 `W4A16_NVFP4` MoE 交給 FP4 Marlin fallback，因此 H100 可以載入/執行。但 NVFP4 的原生 W4A4 tensor-core 加速最低需要 Blackwell SM100。vLLM 官方 Qwen3.6 recipe 也只把 NVFP4 硬體列為 Blackwell，H100 則列為 FP8。H100 得到的是省權重記憶體的相容路徑，不是原生 NVFP4 加速，而且 NVIDIA 模型卡的測試硬體只有 GB300，沒有 H100 效能或 peak-VRAM 數據（[NVIDIA model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)、[vLLM ModelOpt implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/modelopt/)、[vLLM Qwen3.6 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)、[vLLM compression hardware guidance](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/)）。

因此有兩個不同答案：

- **若硬體是 H100：** 27B-FP8 是原生 Hopper FP8、官方明列單 H100 的較保守方案；35B-NVFP4 可以載入，但 Marlin fallback 的吞吐與 peak VRAM 需實測。
- **若硬體改為 Blackwell：** 35B-NVFP4 才是更有吸引力的單一 LLM/VLM endpoint，因為能使用原生 FP4 kernels；TensorRT-LLM 官方同樣把 optimized FP4 kernels 列為 B200 能力，而 H100 的原生低精度路徑是 FP8（[TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html)）。

### Artifact、量化格式與 H100 執行含義

| 項目 | 官方資料 / 結論 |
|---|---|
| Publisher / license / gate | NVIDIA 發布、源自 Qwen3.6-35B-A3B；Apache-2.0、`gated=false` |
| 架構與 modalities | MoE 35B total / 3B active；模型卡列 Text、Image、Video input、262K context |
| `.safetensors` 合計 | 23,424,338,320 bytes = **21.816 GiB**；是 artifact size，不是 runtime VRAM（[HF API with blobs](https://huggingface.co/api/models/nvidia/Qwen3.6-35B-A3B-NVFP4?blobs=true)） |
| 相較 Qwen 27B-FP8 | 小約 6.93 GiB；兩者 raw weights 同時載入仍約 **50.56 GiB**，尚未包含兩個 vLLM runtime/KV/vision/CUDA graph |
| ModelOpt version | 模型卡列 `nvidia-modelopt v0.44.0`，NVFP4 1.0 |
| 量化內容 | 模型卡稱 MoE transformer blocks 的 linear weights/activations 由 16-bit 降至 4-bit、disk/GPU memory 約縮小 3.06×；checkpoint [`hf_quant_config.json`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/blob/main/hf_quant_config.json) 實際是 mixed precision：MLP/MoE 為 group-size 16 的 `W4A16_NVFP4`，attention/linear-attention 為 FP8，KV cache 設 FP8 |
| 原生硬體加速 | vLLM：NVFP4 W4A4 最低 CC 10.0 / Blackwell；Hopper 建議 W8A8-FP8 或 W4AFP8。TensorRT-LLM：optimized FP4 kernels 為 B200，H100 原生支援 FP8 |
| H100 載入 | **可以**：NVIDIA 卡標示 Hopper compatible，vLLM mixed ModelOpt path 對 `W4A16_NVFP4` 使用 FP4 Marlin；但 recipe 的 NVFP4 prerequisite 只列 Blackwell，因此不可宣稱是 H100 官方 benchmark/原生 NVFP4 serving |

NVIDIA ModelOpt checkpoint 可由 vLLM 以 `--quantization modelopt` 提供 OpenAI-compatible server（[vLLM ModelOpt serving](https://docs.vllm.ai/en/v0.17.1/features/quantization/modelopt/)）。NVIDIA 卡明列 Image/Video input，底層仍是 Qwen3.6 conditional-generation 架構；因此沿用 vLLM `/v1/chat/completions` 的 OpenAI `image_url` schema 是合理的介面推論。不過 NVIDIA 卡本身只附文字 serving command，未提供這個量化 checkpoint 的 H100 image request 驗證，故部署前必須用真實圖片做 smoke test，不能只測文字 `/v1/chat/completions`。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":[{"type":"text","text":"描述這張圖"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}],"max_tokens":256}'
```

### Serving 版本與 H100 實驗命令

NVIDIA 模型卡唯一推薦的 engine 是 **vLLM**，並要求 `vllm/vllm-openai:nightly`；較新的 vLLM Qwen3.6 recipe 對 ModelOpt W4A16/NVFP4 要求 nightly/source build，並對 DGX Spark / RTX Pro 6000 範例給出 `v0.24.0`。正式測試應優先用**固定 image digest 的 nightly**；`v0.24.0` 可作可重現的最低候選，但它在 recipe 中驗證的是 Blackwell，不是 H100。不要沿用本文 Qwen 官方 FP8 的最低 `0.19.0`。TensorRT-LLM 雖支援 H100 本身，但其官方 quantization matrix 不把 Hopper 列為 NVFP4 原生路徑，且此 NVIDIA model card 沒列 TensorRT-LLM，故本 checkpoint 不建議改用 TensorRT-LLM（[TensorRT-LLM quantization matrix](https://nvidia.github.io/TensorRT-LLM/1.3.0rc14/features/quantization.html)）。

以下是把官方 ModelOpt/Marlin 參數縮成單 H100 共卡 smoke test 的**推估命令**，不是 NVIDIA 或 vLLM 驗證過的 H100 profile：

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --moe-backend marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.40 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

若特定 nightly/build 的 loader 或 backend 拒絕 H100，不應關閉安全檢查或假裝它是原生 FP4；應換用與 ModelOpt 0.44 相容的 pinned nightly，否則回到 Qwen 27B-FP8。啟動後仍要比較 H100 上的 TTFT、tokens/s、peak VRAM 與 vision correctness，因為官方 NVFP4 accuracy 表與 inference 測試使用 GB300。

若上述 H100 fallback 能以 `gpu-memory-utilization=0.40` 初始化，generation cap 是 32 GiB；套用後文相同的輔助 NIM 保守預算 27 GiB 與 8–12 GiB headroom，整卡約 **67–71 GiB（推估）**，容量上比 27B-FP8 的 73–77 GiB 寬鬆。這是它的明顯優勢，但 H100 吞吐與 peak VRAM 尚無官方 benchmark，且沒有原生 FP4 加速。

### 能否與 27B-FP8 LLM 同時常駐？

**不建議。** 兩份 serialized weights 已約 50.56 GiB；再加兩個 vLLM processes 的 allocator、KV/state cache、vision encoder workspace、CUDA graphs，以及本題要求的 embedding/reranker/OCR/detectors，遠超本文單卡保守包絡。就算把 27B-FP8 只作文字 LLM、35B-NVFP4 只作 VLM，也重複載入兩套完整 generation model，且 H100 無法原生加速後者。

若願意承擔 H100 Marlin fallback 缺少官方效能/顯存 benchmark 的風險，**單一 35B-NVFP4 同時作 LLM/VLM** 比「35B-NVFP4 VLM + 27B-FP8 LLM」合理：只有一份 21.816 GiB artifact，兩個 repo endpoint 都指向同一 OpenAI server。但對需要可預期吞吐與 Hopper 原生低精度 kernel 的部署，單一 27B-FP8 同時作 LLM/VLM 仍是本文最終推薦。

## 1×H100 80GB：Qwen3.6 與全部多模態 NIM 共置

### 官方數字與保守預算

repo 的 `nims.yaml` 把 generation 之外的服務都允許指向 GPU 0，但「能指定同一 GPU」不等於官方驗證它們可以同時滿載。下表先列 repo 對應版本/官方支援資料，再給本文保守預算：

| 服務 | 官方/設定依據 | 官方數字 | 本文保守預算（推估） |
|---|---|---:|---:|
| Qwen3.6-27B-FP8 vLLM | vLLM per-instance memory fraction | `0.48 × 80 = 38.4 GiB` requested cap | 38.4 GiB |
| `llama-nemotron-embed-vl-1b-v2` | NVIDIA current embedding NIM matrix，H100 SM9 FP8 | 約 8.87 GiB | 10 GiB |
| `llama-nemotron-rerank-vl-1b-v2:1.11.0` | NVIDIA 1.11.0 matrix，H100 SM9 FP8 | 約 3.88 GiB | 5 GiB |
| `nemotron-ocr-v1:1.3.0` | NVIDIA 1.3.0 generic non-optimized minimum | 1 GiB；H100 optimized 未公布實測 | 3 GiB |
| Page / Graphic / Table detection 1.8.0 | NVIDIA 1.8.0 generic minimum | 各 1 GiB；repo 另為 graphic/table 各設 2048 MB CUDA pool | 三者合計 9 GiB |
| 未歸屬 headroom | CUDA contexts、短暫輸入 buffers、allocator fragmentation | 無跨服務官方數字 | 8–12 GiB |

來源：[embedding NIM support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html)、[reranker 1.11.0 support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/1.11.0/support-matrix.html)、[OCR 1.3.0 support matrix](https://docs.nvidia.com/nim/ingestion/image-ocr/1.3.0/support-matrix.html)、[object detection 1.8.0 support matrix](https://docs.nvidia.com/nim/ingestion/object-detection/1.8.0/support-matrix.html)、[repo nims.yaml](../../deploy/compose/nims.yaml)。

上述保守欄合計已約 **65.4 GiB，另需 8–12 GiB headroom，總計約 73–77 GiB（推估）**。這表示理論上只剩很窄的餘裕；任何 NIM 自動選到不同 profile、較大 batch、更多圖片、vLLM CUDA graph/profile peak，均可能 OOM。NVIDIA repo 自己建議完整 multimodal pipeline 至少 3 GPUs，因此「所有服務永遠同時常駐一張 H100」不能列為支援保證。

repo 的 H100 MIG values 更能說明保守資源包絡：OCR 配 40GB slice，page/graphic/table detectors 各配 10GB slice，VLM embedding 要求完整 GPU；VLM reranker 的標準 Helm values 同樣 request 一張完整 GPU。這些是 scheduler resource envelopes，**不是實測佔用或可相加的 VRAM 數字**，但它們與官方 2–3 GPU 拓撲一致，故長期服務不應以「全部同時常駐 1×H100」作容量承諾（[H100 MIG values](../../deploy/helm/mig-slicing/values-mig-h100.yaml)、[Helm values](../../deploy/helm/nvidia-blueprint-rag/values.yaml)）。

vLLM 0.19 的 `--gpu-memory-utilization` 預設為 0.9，且是 per-instance limit；它不會替其他 NIM 做全域資源協調，所以絕不能保留預設值（[vLLM 0.19 cache config](https://docs.vllm.ai/en/v0.19.1/api/vllm/config/cache/)）。

### 推薦啟動基線（推估）

先啟動較小的 NIMs，確認 `nvidia-smi` 常駐用量，再啟動 vLLM：

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.48 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

- 若 vLLM 初始化 OOM：先降 `--gpu-memory-utilization` 至 `0.45`；若權重 + profile 本身已超過該 cap，再降 context 不一定有用，應停止非查詢期 NIM 或改分時。
- 若顯示 KV cache 不足：先把 `--max-model-len` 降到 4096；穩定後依序測 8192、16384，不要直接跳 262K。
- `APP_VLM_MAX_TOTAL_IMAGES=2`、RAG concurrency=1、ingestion batch=1 是本文建議的保守起點（推估）。repo 預設的 detector/OCR batch 32 不適合單卡共置壓測，應降低後逐項量測。
- 若要求穩定服務，最佳方案是**分時共用**：ingestion window 啟動 embed + OCR + detectors、停止 vLLM/reranker；query window 停止 OCR/detectors，啟動 Qwen3.6 + embed + reranker。這仍是單張 H100，但避免七個 GPU process 同時爭用。
- 若必須全部常駐且 8K 不夠，硬體結論是增加 GPU；不能藉由把 `gpu-memory-utilization` 調高來創造不存在的 VRAM。

### 本 repo 對 Qwen3.6 thinking 的重要限制

repo 的 VLM 路徑會傳 `chat_template_kwargs.enable_thinking`，但一般 LLM 路徑的 reasoning binding 目前只識別 Nemotron model name。因此：

```dotenv
ENABLE_VLM_INFERENCE=true
VLM_TO_LLM_FALLBACK=false
APP_VLM_MODELNAME=Qwen/Qwen3.6-27B-FP8
APP_VLM_SERVERURL=http://qwen-vllm:8000/v1
APP_VLM_ENABLE_THINKING=false
APP_VLM_MAX_TOTAL_IMAGES=2

APP_LLM_MODELNAME=Qwen/Qwen3.6-27B-FP8
APP_LLM_SERVERURL=http://qwen-vllm:8000/v1
```

在單卡短 context 方案中，建議讓主要 generation 走 VLM 路徑（`VLM_TO_LLM_FALLBACK=false`），如此 `APP_VLM_ENABLE_THINKING=false` 會明確送到 Qwen3.6。單獨設定 `LLM_ENABLE_THINKING=false` 不會使 repo 的 Qwen LLM path 自動傳 Qwen3.6 所需的 `chat_template_kwargs`；因此前述 vLLM 啟動命令另以 server-wide `--default-chat-template-kwargs '{"enable_thinking":false}'` 覆蓋 LLM-only 支線。若移除此 server default，query rewriter、agentic RAG、summarizer 等支線需另做端到端驗證或修改 client binding。

## 與本 repo 的介面核對

### LLM

`LLMConfig` 預設 engine 雖名為 `nvidia-ai-endpoints`，但目前 implementation 的自訂 URL 分支會建立 `ChatNVIDIA(base_url=url, model=...)`。對 `Qwen/...`，`_supports_nvidia_generation_params()` 為 false，因此不會傳 `min_tokens`、`ignore_eos` 等 NVIDIA-only body；`_bind_reasoning_config()` 也只對 Nemotron model name 生效。換言之，對標準 vLLM Chat Completions schema 是可接的（[llm.py](../../src/nvidia_rag/utils/llm.py)）。

應設定完整 URL（已帶 scheme 與 `/v1`），例如：

```dotenv
APP_LLM_MODELNAME=Qwen/Qwen3.6-27B-FP8
APP_LLM_SERVERURL=http://qwen-vllm:8000/v1
APP_LLM_APIKEY=
```

若未帶 `http(s)://`，repo 的 `sanitize_nim_url()` 會自動補 `http://` 與 `/v1`；為免 double `/v1` 或容器 DNS 誤判，建議明寫完整 URL（[common.py](../../src/nvidia_rag/utils/common.py)）。

### VLM 與圖片

VLM 實作把 query/retrieved images 整理成 OpenAI vision content：

```json
{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
```

之後送到同一個 `/v1/chat/completions`。這正是 Qwen3.6 模型卡和 vLLM 所示的介面，因此 image input schema 相容。repo 預設最多送五張圖，可用 `APP_VLM_MAX_TOTAL_IMAGES` 調低來節省顯存（[vlm.py](../../src/nvidia_rag/rag_server/vlm.py)、[docker-compose-rag-server.yaml](../../deploy/compose/docker-compose-rag-server.yaml)）。

```dotenv
ENABLE_VLM_INFERENCE=true
APP_VLM_MODELNAME=Qwen/Qwen3.6-27B-FP8
APP_VLM_SERVERURL=http://qwen-vllm:8000/v1
APP_VLM_APIKEY=
APP_VLM_MAX_TOTAL_IMAGES=2
APP_VLM_ENABLE_THINKING=false
VLM_TO_LLM_FALLBACK=false
```

Qwen3.6 官方 vLLM 用法接受 `extra_body={"chat_template_kwargs":{"enable_thinking": ...}}`；repo VLM 恰好傳同一形式。短 context 共置方案應設 `APP_VLM_ENABLE_THINKING=false`。若需要 reasoning stream，需另驗證 Qwen/vLLM 回傳的 `reasoning_content` 是否完全符合 repo 目前針對 Nemotron 做的 parser，不能只因一般答案可用便視為 reasoning UI 已相容。

### 容器網路與 API key 陷阱

- `rag-server` 在 Docker 內時，`http://localhost:8000/v1` 指向 **rag-server container 自己**，不是 host 上的 vLLM。最乾淨的方式是把 vLLM container 加入 repo 的 `nvidia-rag` network，endpoint 用 `http://qwen-vllm:8000/v1`。若 vLLM 跑在 host，需明確加入 Linux host-gateway mapping 後才使用 `host.docker.internal`。
- compose 目前仍以 `${NGC_API_KEY:?...}` 啟動 `rag-server`，且預設 embeddings/reranker/ingestion 是 NVIDIA services。因此把 LLM/VLM 換成 Qwen **不代表整套 compose 已不需要 NGC key**（[docker-compose-rag-server.yaml](../../deploy/compose/docker-compose-rag-server.yaml)）。
- 若同一 Qwen3.6 endpoint 同時做 LLM/VLM，兩組 `MODELNAME` 必須完全等於 vLLM served model name，或啟動 vLLM 時用 `--served-model-name` 統一別名。
- vLLM multimodal endpoint 若允許遠端 URL 圖片，官方建議用 `--allowed-media-domains` 限制來源以降低 SSRF；本 repo 主要傳 data URL，但仍應按實際暴露面設定（[vLLM multimodal security note](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)）。

## 啟動後驗證

安裝 `vllm>=0.19.0` 並用前述保守命令啟動後，至少驗證：

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"只回答 OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'
```

再從 **rag-server container** 內測同一 URL，才足以證明 Docker network 可達；最後才開 `ENABLE_VLM_INFERENCE` 做 base64 image query。

## License / gated 狀態

本文表列的四個 Qwen3.6 官方 repositories 在 Hugging Face model API 截點均為 `private=false`、`gated=false`、`license=apache-2.0`。例：[Qwen3.6-27B-FP8 API](https://huggingface.co/api/models/Qwen/Qwen3.6-27B-FP8)、[Qwen3.6-35B-A3B-FP8 API](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8)。也就是下載權重本身不需接受 gated agreement；但仍須遵守 Apache-2.0、模型卡使用說明，以及部署環境中其他 NVIDIA 元件各自的授權/API key 要求。

## 最終推薦

1. **1×H100 80GB、整套多模態 RAG 共用同卡：`Qwen/Qwen3.6-27B-FP8`。** 同一 vLLM endpoint 同時填入 LLM/VLM；以 8K、`gpu-memory-utilization=0.48`、concurrency 1 起步，數值均屬容量推估。
2. **VRAM 優先、可接受 H100 非原生 FP4 與 benchmark 工作：`nvidia/Qwen3.6-35B-A3B-NVFP4` 作單一 LLM/VLM。** Artifact 少約 6.93 GiB；不可再同駐 27B-FP8，正式採用前需量測文字/圖片 correctness、TTFT、吞吐與 peak VRAM。
3. **穩定優先：同卡但 ingestion/query 分時。** Qwen3.6 可在 query window 常駐；OCR 與 detectors 只在 ingestion window 啟動。若所有服務必須同時常駐，需接受實驗性部署與 OOM 風險。
4. **generation 品質優先且可增加 GPU：`Qwen/Qwen3.6-35B-A3B-FP8`。** 不建議把它與全部輔助 NIM 擠在單張 H100；BF16 兩款亦不適合本題的單卡全服務限制。
5. **不能改用較小的官方 Qwen3.6。** 截至資料截點，Qwen 官方沒有公開更小的 Qwen3.6 post-trained checkpoint，也沒有官方 AWQ/GGUF；若 27B-FP8 仍無法共置，應增加 GPU、分時，或評估上述 NVIDIA NVFP4，而不是使用未經本文一手來源驗證的第三方量化。
