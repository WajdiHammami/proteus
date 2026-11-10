from abc import ABC, abstractmethod

from src.adapters.models.source import Source


class SitemapPort(ABC):
    """
    Abstract Port for expanding a sitemap Source into the page Sources it lists.
    Unlike ScraperPort, it does not fetch page content: it only discovers URLs.
    """

    @abstractmethod
    def expand(self, source: Source) -> list[Source]:
        """
        Expands a SITEMAP source into new, PENDING URL sources.

        It should handle calling source.mark_crawling(),
        and subsequently source.mark_done() or source.mark_failed().
        """
        raise NotImplementedError
