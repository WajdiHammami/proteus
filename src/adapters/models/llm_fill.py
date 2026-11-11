from dataclasses import dataclass, field
from typing import Any

from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.value_objects import SchemaField


@dataclass
class FillJob:
    """The fields still missing for one snippet, with what the filler may look at."""

    snippet: CodeSnippet
    fields: list[SchemaField]
    known: dict[str, Any]
    document: NormalizedDocument


@dataclass
class CostEstimate:
    """What filling would cost, computed before any call is made."""

    filler: str                          # e.g. "LLMFieldFiller(gpt-4.1-mini)"
    calls: int = 0
    input_tokens: int = 0                # counted with the model's tokenizer
    expected_output_tokens: int = 0      # typical answer size
    max_output_tokens: int = 0           # hard upper bound (max tokens per call x calls)
    expected_cost: float | None = None   # None when the model's prices are unknown
    max_cost: float | None = None
    currency: str = "USD"
    fields: list[str] = field(default_factory=list)   # field names that would be filled

    @property
    def priced(self) -> bool:
        return self.expected_cost is not None
