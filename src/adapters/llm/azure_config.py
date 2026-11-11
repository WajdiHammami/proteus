import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from dotenv import dotenv_values

from src.ports.llm_port import LLMConfigurationError, ModelPricing


ENV_PREFIX = "AZURE_OPENAI_"


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str                                  # resource root, e.g. https://<resource>.services.ai.azure.com/
    api_key: str = field(repr=False)               # never shown in logs or tracebacks
    deployment: str = ""
    model: str | None = None                       # underlying model (tokenizer choice, logs)
    pricing: ModelPricing | None = None
    timeout: float = 30.0
    max_retries: int = 2

    @property
    def base_url(self) -> str:
        """v1 API base URL (no api-version needed)."""
        return self.endpoint.rstrip("/") + "/openai/v1/"


def load_azure_config(env_file: str | None = "azure.env") -> AzureOpenAIConfig:
    """
    Reads AZURE_OPENAI_* settings from `env_file`; real environment variables win.
    Raises LLMConfigurationError naming missing/invalid variables (never their values).
    """
    values = {k: v for k, v in (dotenv_values(env_file) if env_file and os.path.isfile(env_file) else {}).items() if v}
    values.update({k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX) and v})

    def get(name: str) -> str | None:
        value = values.get(ENV_PREFIX + name)
        return value.strip() if value and value.strip() else None

    missing = [ENV_PREFIX + name for name in ("ENDPOINT", "API_KEY", "DEPLOYMENT") if not get(name)]
    if missing:
        raise LLMConfigurationError(f"Missing Azure OpenAI settings: {', '.join(missing)} (in {env_file} or the environment)")

    return AzureOpenAIConfig(
        endpoint=_resource_root(get("ENDPOINT")),
        api_key=get("API_KEY"),
        deployment=get("DEPLOYMENT"),
        model=get("MODEL"),
        pricing=_pricing(get),
        timeout=_number(get, "TIMEOUT", 30.0),
        max_retries=int(_number(get, "MAX_RETRIES", 2)),
    )


def _resource_root(endpoint: str) -> str:
    """'https://x.services.ai.azure.com/openai/v1/responses' -> 'https://x.services.ai.azure.com/'"""
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise LLMConfigurationError(f"{ENV_PREFIX}ENDPOINT must be an https:// URL")
    return f"https://{parsed.netloc}/"


def _pricing(get) -> ModelPricing | None:
    input_price = _number(get, "PRICE_INPUT_PER_1M", None)
    output_price = _number(get, "PRICE_OUTPUT_PER_1M", None)
    if input_price is None and output_price is None:
        return None
    if input_price is None or output_price is None:
        raise LLMConfigurationError(f"Set both {ENV_PREFIX}PRICE_INPUT_PER_1M and {ENV_PREFIX}PRICE_OUTPUT_PER_1M")
    return ModelPricing(
        input_per_million=input_price,
        output_per_million=output_price,
        cached_input_per_million=_number(get, "PRICE_CACHED_INPUT_PER_1M", None),
        currency=get("PRICE_CURRENCY") or "USD",
    )


def _number(get, name: str, default):
    value = get(name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        raise LLMConfigurationError(f"{ENV_PREFIX}{name} must be a number") from None
    if number < 0:
        raise LLMConfigurationError(f"{ENV_PREFIX}{name} must not be negative")
    return number
