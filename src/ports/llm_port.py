"""
Provider-neutral access to a language model.

Nothing outside src/adapters/llm/ knows which provider is behind this port: the
filler builds an LLMRequest, gets an LLMResponse back, and only ever sees the
errors defined here. Changing provider = writing another LLMPort adapter.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMRequest:
    system: str                     # instructions
    user: str                       # the data to work on
    json_schema: dict[str, Any]     # standard JSON Schema the answer must match
    schema_name: str = "result"     # letters, digits, '_' and '-' only
    max_output_tokens: int = 200
    temperature: float = 0.0


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0    # part of input_tokens billed at the cached rate

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True)
class ModelPricing:
    """Prices per 1M tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None   # None = same as input
    currency: str = "USD"

    def cost(self, usage: TokenUsage) -> float:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        cached_rate = self.input_per_million if self.cached_input_per_million is None else self.cached_input_per_million
        return (
            (usage.input_tokens - cached) * self.input_per_million
            + cached * cached_rate
            + usage.output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]            # the parsed JSON object
    usage: TokenUsage
    model: str | None = None        # what actually answered, for logs


class LLMError(Exception):
    """Base class of every error an LLMPort may raise."""


class LLMConfigurationError(LLMError):
    """Bad credentials, unknown model/deployment, invalid request: every call would fail. Stop the run."""


class LLMUnavailableError(LLMError):
    """Timeout, rate limit, server error (after the adapter's own retries). Worth skipping, not stopping."""


class LLMOutputError(LLMError):
    """The model refused, was filtered, or returned something that isn't valid JSON for the schema."""


class LLMPort(ABC):
    @property
    def name(self) -> str:
        """Human-readable model name, for estimates and logs."""
        return type(self).__name__

    @property
    def pricing(self) -> ModelPricing | None:
        """Prices of the model behind the port; None when unknown (costs can't be estimated)."""
        return None

    @abstractmethod
    def generate_json(self, request: LLMRequest) -> LLMResponse:
        """
        Returns a JSON object matching `request.json_schema`.

        Raises:
            LLMConfigurationError, LLMUnavailableError, LLMOutputError (never provider exceptions).
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Tokens `text` costs as input. Adapters should use the model's tokenizer; this is a rough fallback."""
        return math.ceil(len(text) / 4)
