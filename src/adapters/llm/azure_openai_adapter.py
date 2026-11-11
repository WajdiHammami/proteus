import json
import logging
from functools import cached_property

import openai

from src.adapters.llm.azure_config import AzureOpenAIConfig
from src.ports.llm_port import (
    LLMConfigurationError, LLMOutputError, LLMPort, LLMRequest, LLMResponse, LLMUnavailableError,
    ModelPricing, TokenUsage,
)


logger = logging.getLogger(__name__)


class AzureOpenAIAdapter(LLMPort):
    """
    LLMPort on Azure OpenAI (v1 API, chat completions with strict JSON-schema output).
    The only module that imports `openai`; every SDK error is translated into an LLMError.
    """

    def __init__(self, config: AzureOpenAIConfig, client=None):
        self.config = config
        self._client = client

    @property
    def name(self) -> str:
        return self.config.model or self.config.deployment

    @property
    def pricing(self) -> ModelPricing | None:
        return self.config.pricing

    @property
    def client(self):
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def count_tokens(self, text: str) -> int:
        if self._encoding is None:
            return super().count_tokens(text)
        return len(self._encoding.encode(text, disallowed_special=()))

    @cached_property
    def _encoding(self):
        try:
            import tiktoken
        except ImportError:
            logger.info("tiktoken not installed: token counts are approximate")
            return None
        try:
            return tiktoken.encoding_for_model(self.config.model or self.config.deployment)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")  # tokenizer of the GPT-4o / 4.1 / o-series families
        except Exception:  # encoding files are downloaded once; offline -> approximate
            logger.info("tiktoken encoding unavailable: token counts are approximate", exc_info=True)
            return None

    def generate_json(self, request: LLMRequest) -> LLMResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.config.deployment,
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": request.schema_name, "strict": True, "schema": request.json_schema},
                },
                max_completion_tokens=request.max_output_tokens,
                temperature=request.temperature,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as error:
            raise LLMConfigurationError(f"Azure OpenAI rejected the request ({error.status_code}): check key, endpoint and deployment") from error
        except openai.BadRequestError as error:
            if _is_content_filter(error):
                raise LLMOutputError("Azure OpenAI content filter blocked the request") from error
            raise LLMConfigurationError(f"Azure OpenAI rejected the request as invalid: {_message(error)}") from error
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as error:
            raise LLMUnavailableError(f"Azure OpenAI unavailable: {type(error).__name__}") from error
        except openai.APIStatusError as error:
            if error.status_code >= 500:
                raise LLMUnavailableError(f"Azure OpenAI server error {error.status_code}") from error
            raise LLMConfigurationError(f"Azure OpenAI error {error.status_code}: {_message(error)}") from error

        choice = completion.choices[0]
        message = choice.message
        if getattr(message, "refusal", None):
            raise LLMOutputError(f"Model refused: {message.refusal}")
        if choice.finish_reason == "content_filter":
            raise LLMOutputError("Azure OpenAI content filter blocked the answer")
        if choice.finish_reason == "length":
            raise LLMOutputError(f"Answer cut at max_output_tokens={request.max_output_tokens}")
        try:
            data = json.loads(message.content or "")
        except json.JSONDecodeError as error:
            raise LLMOutputError("Model returned invalid JSON") from error
        if not isinstance(data, dict):
            raise LLMOutputError("Model returned JSON that is not an object")

        return LLMResponse(data=data, usage=_usage(completion), model=getattr(completion, "model", None))


def _usage(completion) -> TokenUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return TokenUsage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        cached_input_tokens=cached or 0,
    )


def _is_content_filter(error: Exception) -> bool:
    body = getattr(error, "body", None)
    code = body.get("code") if isinstance(body, dict) else getattr(error, "code", None)
    return code == "content_filter" or "content_filter" in str(error)


def _message(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])[:200]
    return str(error)[:200]
