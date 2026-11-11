import logging
from collections.abc import Sequence

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceStatus, SourceType
from src.ports.scraper_port import ContentMismatchError, ScraperPort, UnsupportedSourceError


logger = logging.getLogger(__name__)

# Source type to switch to when a scraper reports content of another kind.
_TYPE_FOR_MIME = {"application/pdf": SourceType.PDF}


class ScraperDispatcher(ScraperPort):
    """
    Routes each Source to the first registered scraper whose `supports()` accepts it.

    The dispatcher never changes a Source's status itself; the chosen scraper does.
    If no scraper matches, UnsupportedSourceError is raised and the Source stays PENDING
    (a missing scraper is a configuration problem, not a failed fetch).

    If the chosen scraper raises ContentMismatchError (e.g. a URL that serves a PDF),
    the Source's type is corrected and it is re-routed once, reusing the downloaded
    bytes when the new scraper implements `fetch_payload`.

    SITEMAP sources are not scraped: they are expanded through SitemapPort by the pipeline.
    """

    def __init__(self, scrapers: Sequence[ScraperPort]):
        if not scrapers:
            raise ValueError("ScraperDispatcher needs at least one scraper.")
        self.scrapers = list(scrapers)

    def supports(self, source: Source) -> bool:
        return any(scraper.supports(source) for scraper in self.scrapers)

    def fetch(self, source: Source) -> RawDocument:
        scraper = self._select(source)
        try:
            raw = scraper.fetch(source)
        except ContentMismatchError as mismatch:
            return self._reroute(source, scraper, mismatch)
        return _tag(raw, scraper)

    def _reroute(self, source: Source, previous: ScraperPort, mismatch: ContentMismatchError) -> RawDocument:
        new_type = _TYPE_FOR_MIME.get(mismatch.detected_mime)
        if new_type is None:
            raise UnsupportedSourceError(
                f"Source '{source.url}' serves '{mismatch.detected_mime}', which no scraper handles."
            ) from mismatch

        original_type = source.type
        source.type = new_type  # the source really is of this type; keep it for later runs
        candidates = [s for s in self.scrapers if s is not previous and s.supports(source)]
        if not candidates:
            source.type = original_type
            raise UnsupportedSourceError(
                f"Source '{source.url}' serves '{mismatch.detected_mime}', which no scraper handles."
            ) from mismatch

        scraper = candidates[0]
        logger.info(
            "Re-routing %s from %s to %s (content is %s)",
            source.url, type(previous).__name__, type(scraper).__name__, mismatch.detected_mime,
        )
        raw = None
        if mismatch.payload is not None:
            try:
                raw = scraper.fetch_payload(source, mismatch.payload)
            except NotImplementedError:
                if source.status != SourceStatus.PENDING:
                    raise  # the scraper started and failed: a real error
        if raw is None:
            raw = scraper.fetch(source)  # downloads again
        raw.metadata["rerouted_from"] = original_type.value
        return _tag(raw, scraper)

    def _select(self, source: Source) -> ScraperPort:
        for scraper in self.scrapers:
            if scraper.supports(source):
                return scraper
        hint = " (expand sitemaps through SitemapPort)" if source.type == SourceType.SITEMAP else ""
        raise UnsupportedSourceError(
            f"No scraper supports source '{source.url}' of type '{source.type.value}'{hint}."
        )


def _tag(raw: RawDocument, scraper: ScraperPort) -> RawDocument:
    raw.metadata["scraper"] = type(scraper).__name__
    return raw
