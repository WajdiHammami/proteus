import json
import logging
from dataclasses import dataclass
from typing import Any

from src.adapters.extractors.llm_prompt import (
    SCHEMA_NAME, SYSTEM_PROMPT, build_user_prompt, expected_output_tokens, response_schema,
)
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.llm_fill import CostEstimate, FillJob
from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.value_objects import SchemaField
from src.ports.extraction_port import FieldFillerPort
from src.ports.llm_port import (
    LLMConfigurationError, LLMOutputError, LLMPort, LLMRequest, LLMUnavailableError, TokenUsage,
)


logger = logging.getLogger(__name__)

# Per-call tokens the provider adds around the messages (roles, separators); small and approximate.
_MESSAGE_OVERHEAD_TOKENS = 12


@dataclass
class UsageStats:
    calls: int = 0
    failures: int = 0
    usage: TokenUsage = TokenUsage()
    cost: float | None = 0.0        # None once a call happened without known prices


class LLMFieldFiller(FieldFillerPort):
    """
    Fills the schema's LLM fields with one model call per snippet, through LLMPort.
    Knows nothing about the provider behind it.

      - LLMConfigurationError is re-raised (every call would fail).
      - LLMUnavailableError / LLMOutputError are logged and counted; the fields stay empty.
    """

    def __init__(self, llm: LLMPort, max_output_tokens: int = 200, max_code_chars: int = 6000):
        self.llm = llm
        self.max_output_tokens = max_output_tokens
        self.max_code_chars = max_code_chars
        self.stats = UsageStats()

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.llm.name})"

    @property
    def spent(self) -> float | None:
        return self.stats.cost

    # ------------------------------------------------------------------ #
    # Estimate (no calls)
    # ------------------------------------------------------------------ #

    def estimate(self, jobs: list[FillJob]) -> CostEstimate:
        estimate = CostEstimate(filler=self.name)
        names: dict[str, None] = {}
        for job in jobs:
            if not job.fields:
                continue
            request = self._request(job.snippet, job.fields, job.document)
            estimate.calls += 1
            estimate.input_tokens += self._input_tokens(request)
            estimate.expected_output_tokens += min(expected_output_tokens(job.fields), self.max_output_tokens)
            estimate.max_output_tokens += self.max_output_tokens
            names.update(dict.fromkeys(f.name for f in job.fields))
        estimate.fields = list(names)

        pricing = self.llm.pricing
        if pricing is not None:
            estimate.currency = pricing.currency
            # Prompts are short and differ per snippet: assume no cached input (conservative).
            estimate.expected_cost = pricing.cost(TokenUsage(estimate.input_tokens, estimate.expected_output_tokens))
            estimate.max_cost = pricing.cost(TokenUsage(estimate.input_tokens, estimate.max_output_tokens))
        return estimate

    def _input_tokens(self, request: LLMRequest) -> int:
        # The response schema is sent with the request and billed as input too.
        return (
            self.llm.count_tokens(request.system)
            + self.llm.count_tokens(request.user)
            + self.llm.count_tokens(json.dumps(request.json_schema))
            + _MESSAGE_OVERHEAD_TOKENS
        )

    # ------------------------------------------------------------------ #
    # Fill
    # ------------------------------------------------------------------ #

    def fill(
        self,
        snippet: CodeSnippet,
        fields: list[SchemaField],
        known: dict[str, Any],
        document: NormalizedDocument,
    ) -> dict[str, Any]:
        if not fields:
            return {}
        request = self._request(snippet, fields, document)
        self.stats.calls += 1
        try:
            response = self.llm.generate_json(request)
        except LLMConfigurationError:
            self.stats.failures += 1
            raise
        except (LLMUnavailableError, LLMOutputError) as error:
            self.stats.failures += 1
            logger.warning("LLM fill failed for snippet %d of %s: %s", snippet.index, document.id, error)
            return {}

        self.stats.usage = self.stats.usage + response.usage
        pricing = self.llm.pricing
        if pricing is None or self.stats.cost is None:
            self.stats.cost = None
        else:
            self.stats.cost += pricing.cost(response.usage)

        wanted = {f.name for f in fields}
        return {key: value for key, value in response.data.items() if key in wanted and value is not None}

    def _request(self, snippet: CodeSnippet, fields: list[SchemaField], document: NormalizedDocument) -> LLMRequest:
        return LLMRequest(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(snippet, fields, document, self.max_code_chars),
            json_schema=response_schema(fields),
            schema_name=SCHEMA_NAME,
            max_output_tokens=self.max_output_tokens,
        )
