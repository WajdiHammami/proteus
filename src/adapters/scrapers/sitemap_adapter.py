import re
import uuid
from urllib.parse import urlparse

from trafilatura.sitemaps import sitemap_search

from src.adapters.models.source import Source, SourceType
from src.ports.sitemap_port import SitemapPort


class TrafilaturaSitemapAdapter(SitemapPort):
    """
    Expands a sitemap into child URL Sources using trafilatura's sitemap module.

    The source URL can be:
      - a sitemap file (".xml", ".xml.gz", or ending in "sitemap"): parsed directly,
        nested sitemap indexes are followed;
      - a homepage: sitemaps are discovered via robots.txt or common locations;
      - a subpage: same discovery, then results are restricted to that path.
    """

    def __init__(
        self,
        target_lang: str | None = None,
        include_external: bool = False,
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
        max_urls: int | None = None,
        max_sitemaps: int = 10_000,
        sleep_time: float = 2.0,
    ):
        """
        Args:
            target_lang: ISO 639-1 code; keeps URLs matching this language (hreflang/heuristics).
            include_external: Keep URLs pointing to other hosts.
            include_pattern: Regex; only URLs matching it are kept.
            exclude_pattern: Regex; URLs matching it are dropped.
            max_urls: Cap on the number of child sources returned.
            max_sitemaps: Cap on the number of (nested) sitemap files processed.
            sleep_time: Delay in seconds between requests to the same host.
        """
        self.target_lang = target_lang
        self.include_external = include_external
        self.include_re = re.compile(include_pattern) if include_pattern else None
        self.exclude_re = re.compile(exclude_pattern) if exclude_pattern else None
        self.max_urls = max_urls
        self.max_sitemaps = max_sitemaps
        self.sleep_time = sleep_time

    def expand(self, source: Source) -> list[Source]:
        if source.type != SourceType.SITEMAP:
            raise NotImplementedError(f"Source type '{source.type}' not supported by sitemap adapter.")

        source.mark_crawling()

        try:
            urls = sitemap_search(
                source.url,
                target_lang=self.target_lang,
                external=self.include_external,
                sleep_time=self.sleep_time,
                max_sitemaps=self.max_sitemaps,
            )
            # trafilatura returns [] for unreachable hosts and missing sitemaps alike.
            if not urls:
                raise ValueError(f"No URLs found in sitemap: {source.url}")

            children = [
                Source(
                    id=str(uuid.uuid4()),
                    url=url,
                    type=self._infer_type(url),
                    parent_id=source.id,
                )
                for url in self._filter(urls)
            ]
            source.mark_done()
            return children

        except Exception:
            source.mark_failed()
            raise

    @staticmethod
    def _infer_type(url: str) -> SourceType:
        """PDF links get their own type so they can be routed to the PDF adapter."""
        if urlparse(url).path.lower().endswith(".pdf"):
            return SourceType.PDF
        return SourceType.URL

    def _filter(self, urls: list[str]) -> list[str]:
        kept: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            if self.include_re and not self.include_re.search(url):
                continue
            if self.exclude_re and self.exclude_re.search(url):
                continue
            seen.add(url)
            kept.append(url)
            if self.max_urls is not None and len(kept) >= self.max_urls:
                break
        return kept
