from abc import ABC, abstractmethod
from typing import Any

from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.llm_fill import CostEstimate, FillJob
from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.value_objects import SchemaField


class SnippetExtractorPort(ABC):
    """
    Finds the C/C++ snippets in a normalized document. Deterministic and free:
    no network, no LLM. Each snippet becomes one record later.
    """

    @abstractmethod
    def extract(self, document: NormalizedDocument) -> list[CodeSnippet]:
        """Returns the snippets in document order (an empty list is valid)."""
        raise NotImplementedError


class FieldFillerPort(ABC):
    """
    Fills schema fields that can't be computed from the snippet itself
    (description, concepts, ...), typically with a language model.

    Only the snippet and its small context are sent, never the whole document,
    to keep the token count low. Filling runs last, once every snippet has been
    extracted, and only after `estimate()` has been shown and confirmed.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    def estimate(self, jobs: list[FillJob]) -> CostEstimate:
        """What filling these jobs would cost. Default: free (fillers that don't call a paid service)."""
        return CostEstimate(filler=self.name, expected_cost=0.0, max_cost=0.0)

    @property
    def spent(self) -> float | None:
        """What this filler actually cost so far (None if unknown). Default: free."""
        return 0.0

    @abstractmethod
    def fill(
        self,
        snippet: CodeSnippet,
        fields: list[SchemaField],
        known: dict[str, Any],
        document: NormalizedDocument,
    ) -> dict[str, Any]:
        """
        Returns values for (some of) `fields`, keyed by field name. Fields it can't
        fill are simply left out; the next filler gets a chance at them.

        Args:
            snippet: the code and its context.
            fields: the fields still missing.
            known: values already filled (may help the model, e.g. the heading).
            document: the source document, for metadata only.
        """
        raise NotImplementedError
