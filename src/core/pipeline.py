import logging

from src.adapters.models.source import Source, SourceType
from src.core.router import ExtractionRouter
from src.core.schema_loader import load_schema_as_pydantic
from src.ports.normalization_port import NormalizationPort
from src.ports.scraper_port import ScraperPort
from src.ports.sitemap_port import SitemapPort


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    scrape -> normalize -> extract.

    `scraper` and `normalizer` are usually a ScraperDispatcher and a NormalizerDispatcher.
    SITEMAP sources are expanded through `sitemap` and each child Source is processed in turn.
    """

    def __init__(
        self,
        scraper: ScraperPort,
        normalizer: NormalizationPort,
        router: ExtractionRouter,
        schema_path: str,
        sitemap: SitemapPort | None = None,
    ):
        self.scraper = scraper
        self.normalizer = normalizer
        self.sitemap = sitemap
        self.router = router
        self.dynamic_model = load_schema_as_pydantic(schema_path)

    def run(self, source: Source) -> list:
        """Returns the extracted data of every document the source leads to."""
        if source.type == SourceType.SITEMAP:
            if self.sitemap is None:
                raise ValueError("A SitemapPort is required to process SITEMAP sources.")
            results = []
            for child in self.sitemap.expand(source):
                try:
                    results.extend(self.run(child))
                except Exception:
                    # One broken page must not stop the whole sitemap; the child is marked FAILED.
                    logger.exception("Failed to process %s (from sitemap %s)", child.url, source.url)
            return results

        return [self._process(source)]

    def _process(self, source: Source):
        raw = self.scraper.fetch(source)
        normalized = self.normalizer.normalize(raw)

        # Router handles strategy selection and cascading fallbacks automatically
        extracted_data, used_tier = self.router.route_and_extract(normalized.content, self.dynamic_model)

        logger.info("Extracted %s using tier: %s", source.url, used_tier.value)
        return extracted_data
