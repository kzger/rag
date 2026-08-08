from nvidia_rag.utils.health import generation_health_url


def test_generation_health_url_uses_models_for_openai_base_url() -> None:
    assert generation_health_url("http://qwen-vllm:8000/v1") == (
        "http://qwen-vllm:8000/v1/models"
    )


def test_generation_health_url_uses_models_for_chat_completions_url() -> None:
    assert (
        generation_health_url("http://qwen-vllm:8000/v1/chat/completions")
        == "http://qwen-vllm:8000/v1/models"
    )


def test_generation_health_url_keeps_nim_readiness_for_service_base_url() -> None:
    assert generation_health_url("nim-llm:8000") == (
        "http://nim-llm:8000/v1/health/ready"
    )
