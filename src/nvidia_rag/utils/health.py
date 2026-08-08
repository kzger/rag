"""Shared health-endpoint URL helpers."""


def generation_health_url(endpoint: str) -> str:
    """Choose a readiness endpoint for NIM or OpenAI-compatible servers."""
    normalized = endpoint.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    if normalized.endswith("/v1/chat/completions"):
        return normalized.removesuffix("/chat/completions") + "/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/health/ready"
