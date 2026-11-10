from abc import ABC, abstractmethod

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source


class ScraperPort(ABC):
    """
    Abstract Port for scraping or expanding sources.
    Operates on the Source domain model, handling state transitions.
    """

    @abstractmethod
    def fetch(self, source: Source) -> RawDocument:
        """
        Fetches the raw content for the given source and wraps it in a RawDocument.

        It should handle calling source.mark_crawling(),
        and subsequently source.mark_done() or source.mark_failed().
        """
        raise NotImplementedError
