import logging
from collections.abc import Sequence

from src.adapters.cli.console_confirm import console_confirm
from src.adapters.infrastructure.yaml_loader import load_schema_from_yaml
from src.adapters.models.source import Source, SourceType
from src.core.filling import ConfirmCallback, FillingStage
from src.core.record_builder import ExtractionResult, PendingRecord, RecordBuilder
from src.ports.extraction_port import FieldFillerPort, SnippetExtractorPort
from src.ports.normalization_port import NormalizationPort
from src.ports.scraper_port import ScraperPort
from src.ports.sitemap_port import SitemapPort


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Two phases:

      1. collect (free):  for every source (sitemaps expanded):
                          scrape -> normalize -> find snippets -> free fields
      2. fill (paid):     estimate the cost of the LLM fields for *all* snippets,
                          ask for confirmation, then fill; finally validate the records.

    `confirm` is shown the estimates and returns True to go ahead (default: ask on the
    console). `debug=True` skips the confirmation.
    """

    def __init__(
        self,
        scraper: ScraperPort,
        normalizer: NormalizationPort,
        snippet_extractor: SnippetExtractorPort,
        schema_path: str,
        fillers: Sequence[FieldFillerPort] = (),
        sitemap: SitemapPort | None = None,
        confirm: ConfirmCallback | None = console_confirm,
        debug: bool = False,
    ):
        self.scraper = scraper
        self.normalizer = normalizer
        self.snippet_extractor = snippet_extractor
        self.sitemap = sitemap
        self.builder = RecordBuilder(load_schema_from_yaml(schema_path))
        self.filling = FillingStage(fillers, confirm=confirm, debug=debug)

    def run(self, sources: Source | Sequence[Source]) -> ExtractionResult:
        """Processes one or more sources and returns all records (and rejected snippets)."""
        if isinstance(sources, Source):
            sources = [sources]

        pending: list[PendingRecord] = []
        for source in sources:
            pending.extend(self._collect(source))

        report = self.filling.run(pending)
        result = self.builder.finalize(pending)
        result.fill = report
        logger.info(
            "Done: %d record(s), %d rejected; filling %s",
            len(result.records), len(result.rejected),
            "ran" if report.ran else f"skipped ({report.skipped_reason})",
        )
        return result

    def _collect(self, source: Source) -> list[PendingRecord]:
        if source.type == SourceType.SITEMAP:
            if self.sitemap is None:
                raise ValueError("A SitemapPort is required to process SITEMAP sources.")
            pending = []
            for child in self.sitemap.expand(source):
                try:
                    pending.extend(self._collect(child))
                except Exception:
                    # One broken page must not stop the whole sitemap; the child is marked FAILED.
                    logger.exception("Failed to process %s (from sitemap %s)", child.url, source.url)
            return pending

        raw = self.scraper.fetch(source)
        normalized = self.normalizer.normalize(raw)
        snippets = self.snippet_extractor.extract(normalized)
        logger.info("%s: %d snippet(s)", source.url, len(snippets))
        return self.builder.prepare(normalized, snippets)
