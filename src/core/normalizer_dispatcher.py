from collections.abc import Sequence

from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.raw_document import RawDocument
from src.ports.normalization_port import NormalizationPort, UnsupportedDocumentError


class NormalizerDispatcher(NormalizationPort):
    """
    Routes each RawDocument to the first registered normalizer whose `supports()`
    accepts it (registration order = priority). Raises UnsupportedDocumentError
    when none does.
    """

    # TODO: unknown mime types fail for now. Consider falling back to the plain-text
    # normalizer for other text/* types (text/csv, text/xml, ...) once we see them in practice.

    def __init__(self, normalizers: Sequence[NormalizationPort]):
        if not normalizers:
            raise ValueError("NormalizerDispatcher needs at least one normalizer.")
        self.normalizers = list(normalizers)

    def supports(self, raw: RawDocument) -> bool:
        return any(normalizer.supports(raw) for normalizer in self.normalizers)

    def normalize(self, raw: RawDocument) -> NormalizedDocument:
        for normalizer in self.normalizers:
            if normalizer.supports(raw):
                normalized = normalizer.normalize(raw)
                normalized.metadata["normalizer"] = type(normalizer).__name__
                return normalized
        raise UnsupportedDocumentError(
            f"No normalizer supports mime type '{raw.mime_type}' (source {raw.source_id})."
        )
