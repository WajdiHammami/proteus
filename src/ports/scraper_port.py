from abc import ABC, abstractmethod

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source


class UnsupportedSourceError(Exception):
    """Raised when no scraper can handle a Source. The Source is left untouched (PENDING)."""


class ContentMismatchError(Exception):
    """
    Raised by a scraper that downloaded a Source and found content of another kind
    (e.g. a URL without a .pdf suffix that serves a PDF).

    The scraper must call `source.release()` before raising, so the Source is back to
    PENDING and can be handed to the right scraper. `payload` holds the bytes already
    downloaded, so that scraper doesn't have to download them again.
    """

    def __init__(self, detected_mime: str, payload: bytes | None = None, message: str | None = None):
        super().__init__(message or f"Content is '{detected_mime}'")
        self.detected_mime = detected_mime
        self.payload = payload


class ScraperPort(ABC):
    """
    Abstract Port for scraping or expanding sources.
    Operates on the Source domain model, handling state transitions.
    """

    @abstractmethod
    def supports(self, source: Source) -> bool:
        """
        Returns True if this scraper can fetch the source, typically decided by
        `source.type` (and the path suffix for FILE sources). Must not do any I/O.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch(self, source: Source) -> RawDocument:
        """
        Fetches the raw content for the given source and wraps it in a RawDocument.

        It should handle calling source.mark_crawling(),
        and subsequently source.mark_done() or source.mark_failed().

        Raises:
            ContentMismatchError: if the downloaded content belongs to another scraper.
        """
        raise NotImplementedError

    def fetch_payload(self, source: Source, payload: bytes) -> RawDocument:
        """
        Like `fetch`, but builds the document from bytes another scraper already
        downloaded (see ContentMismatchError). Optional: scrapers that don't override
        it are simply called through `fetch`.
        """
        raise NotImplementedError
