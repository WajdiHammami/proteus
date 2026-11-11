import pytest

from src.adapters.llm.azure_config import load_azure_config
from src.ports.llm_port import LLMConfigurationError


BASE = (
    "AZURE_OPENAI_ENDPOINT=https://res.services.ai.azure.com/openai/v1/responses\n"
    "AZURE_OPENAI_API_KEY=secret-key-123\n"
    "AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini\n"
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for key in [k for k in os.environ if k.startswith("AZURE_OPENAI_")]:
        monkeypatch.delenv(key)


@pytest.fixture
def env_file(tmp_path):
    def _write(text):
        path = tmp_path / "azure.env"
        path.write_text(text)
        return str(path)
    return _write


def test_reads_file_and_normalizes_endpoint(env_file):
    config = load_azure_config(env_file(BASE))
    assert config.endpoint == "https://res.services.ai.azure.com/"
    assert config.base_url == "https://res.services.ai.azure.com/openai/v1/"
    assert config.deployment == "gpt-4.1-mini"
    assert config.pricing is None


def test_key_never_in_repr(env_file):
    config = load_azure_config(env_file(BASE))
    assert "secret-key-123" not in repr(config)


def test_environment_wins(env_file, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "other")
    assert load_azure_config(env_file(BASE)).deployment == "other"


def test_works_from_environment_only(monkeypatch, tmp_path):
    for line in BASE.splitlines():
        key, value = line.split("=", 1)
        monkeypatch.setenv(key, value)
    assert load_azure_config(str(tmp_path / "missing.env")).deployment == "gpt-4.1-mini"


def test_missing_values_named_without_values(env_file):
    with pytest.raises(LLMConfigurationError) as info:
        load_azure_config(env_file("AZURE_OPENAI_API_KEY=secret-key-123\n"))
    assert "AZURE_OPENAI_ENDPOINT" in str(info.value) and "AZURE_OPENAI_DEPLOYMENT" in str(info.value)
    assert "secret-key-123" not in str(info.value)


def test_pricing(env_file):
    config = load_azure_config(env_file(BASE + (
        "AZURE_OPENAI_PRICE_INPUT_PER_1M=0.562\nAZURE_OPENAI_PRICE_CACHED_INPUT_PER_1M=0.141\n"
        "AZURE_OPENAI_PRICE_OUTPUT_PER_1M=2.25\n")))
    assert config.pricing.input_per_million == 0.562
    assert config.pricing.cached_input_per_million == 0.141
    assert config.pricing.output_per_million == 2.25
    assert config.pricing.currency == "USD"


@pytest.mark.parametrize("extra, match", [
    ("AZURE_OPENAI_PRICE_INPUT_PER_1M=0.5\n", "both"),
    ("AZURE_OPENAI_PRICE_INPUT_PER_1M=abc\nAZURE_OPENAI_PRICE_OUTPUT_PER_1M=1\n", "number"),
    ("AZURE_OPENAI_PRICE_INPUT_PER_1M=-1\nAZURE_OPENAI_PRICE_OUTPUT_PER_1M=1\n", "negative"),
])
def test_invalid_pricing(env_file, extra, match):
    with pytest.raises(LLMConfigurationError, match=match):
        load_azure_config(env_file(BASE + extra))


def test_endpoint_must_be_https(env_file):
    with pytest.raises(LLMConfigurationError, match="https"):
        load_azure_config(env_file(BASE.replace("https://", "http://")))


def test_real_project_file_loads_if_present():
    import os
    if not os.path.isfile("azure.env"):
        pytest.skip("no azure.env")
    config = load_azure_config("azure.env")
    assert config.endpoint.endswith(".azure.com/")
    assert config.pricing is not None
