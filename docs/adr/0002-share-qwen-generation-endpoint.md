# Share one Qwen generation endpoint

The local multimodal RAG deployment will serve
`Qwen/Qwen3.6-27B-FP8` through one OpenAI-compatible endpoint and use it for
text generation, vision-language generation, and ingestion image captioning.
This avoids keeping separate Nemotron LLM, VLM, and captioning models on the
single GPU; the H100-compatible NVFP4 model remains only a capacity fallback
if the FP8 deployment cannot keep the complete required service set healthy.
