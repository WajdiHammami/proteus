from abc import ABC, abstractmethod

from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.raw_document import RawDocument


class UnsupportedDocumentError(Exception):
    """Raised when a normalizer is given a RawDocument it cannot handle."""


class NormalizationPort(ABC):
    """
    Abstract Port for turning a RawDocument (raw HTML, PDF text, markdown, ...)
    into clean text ready for extraction.

    Contract for implementations:
      - Pure transformation: no network or file I/O; everything needed is in the RawDocument.
      - Must not mutate the given RawDocument; always return a new NormalizedDocument.
      - Must carry forward metadata relevant to later stages (e.g. PDF `text_layer`).
      - An empty result is valid (e.g. a fully scanned PDF); it is not an error.
    """

    @abstractmethod
    def supports(self, raw: RawDocument) -> bool:
        """
        Returns True if this normalizer can handle the document,
        typically decided by `raw.mime_type`.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw: RawDocument) -> NormalizedDocument:
        """
        Cleans the raw content and returns a NormalizedDocument.

        Raises:
            UnsupportedDocumentError: if `supports(raw)` is False.
        """
        raise NotImplementedError
