import os
from types import SimpleNamespace

import openai

try:  # openai >= 3 ships on httpx2; older releases on httpx
    import httpx2 as httpx
except ImportError:
    import httpx
import pytest

from src.adapters.llm.azure_config import AzureOpenAIConfig, load_azure_config
from src.adapters.llm.azure_openai_adapter import AzureOpenAIAdapter
from src.ports.llm_port import (
    LLMConfigurationError, LLMOutputError, LLMRequest, LLMUnavailableError, ModelPricing, TokenUsage,
)


CONFIG = AzureOpenAIConfig(endpoint="https://res.services.ai.azure.com/", api_key="k", deployment="gpt-4.1-mini",
                           model="gpt-4.1-mini", pricing=ModelPricing(0.562, 2.25, 0.141))
REQUEST = LLMRequest(system="sys", user="usr", json_schema={"type": "object"}, schema_name="snippet_fields",
                     max_output_tokens=50)


def completion(content='{"description": "d"}', finish_reason="stop", refusal=None, cached=0):
    return SimpleNamespace(
        model="gpt-4.1-mini-2025",
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)),
    )


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def status_error(cls, status, body=None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://res.services.ai.azure.com/"))
    return cls("error", response=response, body=body)


def adapter(result):
    client = FakeClient(result)
    return AzureOpenAIAdapter(CONFIG, client=client), client


class TestSuccess:
    def test_request_mapping(self):
        llm, client = adapter(completion())
        llm.generate_json(REQUEST)
        [call] = client.calls
        assert call["model"] == "gpt-4.1-mini"
        assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
        assert call["response_format"]["json_schema"] == {"name": "snippet_fields", "strict": True, "schema": {"type": "object"}}
        assert call["max_completion_tokens"] == 50
        assert call["temperature"] == 0.0

    def test_response_mapping(self):
        llm, _ = adapter(completion(cached=100))
        response = llm.generate_json(REQUEST)
        assert response.data == {"description": "d"}
        assert response.usage == TokenUsage(120, 15, 100)
        assert response.model == "gpt-4.1-mini-2025"

    def test_name_and_pricing_exposed_through_port(self):
        llm, _ = adapter(completion())
        assert llm.name == "gpt-4.1-mini"
        assert llm.pricing == CONFIG.pricing

    def test_counts_tokens_with_tokenizer(self):
        llm, _ = adapter(completion())
        assert 0 < llm.count_tokens("class Buffer { int* data_; };") < 15


class TestErrors:
    @pytest.mark.parametrize("error", [
        status_error(openai.AuthenticationError, 401),
        status_error(openai.PermissionDeniedError, 403),
        status_error(openai.NotFoundError, 404),
        status_error(openai.BadRequestError, 400, {"message": "bad schema"}),
    ])
    def test_configuration_errors(self, error):
        llm, _ = adapter(error)
        with pytest.raises(LLMConfigurationError):
            llm.generate_json(REQUEST)

    @pytest.mark.parametrize("error", [
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 500),
        status_error(openai.APIStatusError, 503),
        openai.APITimeoutError(request=httpx.Request("POST", "https://x")),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
    ])
    def test_unavailable_errors(self, error):
        llm, _ = adapter(error)
        with pytest.raises(LLMUnavailableError):
            llm.generate_json(REQUEST)

    def test_content_filter_on_request(self):
        llm, _ = adapter(status_error(openai.BadRequestError, 400, {"code": "content_filter", "message": "filtered"}))
        with pytest.raises(LLMOutputError, match="content filter"):
            llm.generate_json(REQUEST)

    @pytest.mark.parametrize("result, match", [
        (completion(refusal="I can't"), "refused"),
        (completion(finish_reason="content_filter"), "content filter"),
        (completion(finish_reason="length", content='{"descr'), "max_output_tokens"),
        (completion(content="not json"), "invalid JSON"),
        (completion(content="[1, 2]"), "not an object"),
    ])
    def test_bad_answers(self, result, match):
        llm, _ = adapter(result)
        with pytest.raises(LLMOutputError, match=match):
            llm.generate_json(REQUEST)

    def test_provider_exceptions_never_leak(self):
        llm, _ = adapter(status_error(openai.AuthenticationError, 401))
        with pytest.raises(LLMConfigurationError) as info:
            llm.generate_json(REQUEST)
        assert not isinstance(info.value, openai.OpenAIError)


@pytest.mark.skipif(os.environ.get("PROTEUS_LIVE_AZURE") != "1", reason="set PROTEUS_LIVE_AZURE=1 to call Azure")
def test_live_azure_call():
    """Real call (a few hundred tokens, a fraction of a cent)."""
    llm = AzureOpenAIAdapter(load_azure_config("azure.env"))
    schema = {"type": "object", "properties": {"description": {"type": ["string", "null"]}},
              "required": ["description"], "additionalProperties": False}
    response = llm.generate_json(LLMRequest(
        system="Describe the C++ snippet in one short sentence.",
        user="```cpp\nstd::unique_ptr<int> p = std::make_unique<int>(3);\n```",
        json_schema=schema, schema_name="live_test", max_output_tokens=60,
    ))
    assert isinstance(response.data["description"], str)
    assert response.usage.input_tokens > 0
    print(f"\nlive: {response.data} usage={response.usage} cost=${llm.pricing.cost(response.usage):.6f}")
