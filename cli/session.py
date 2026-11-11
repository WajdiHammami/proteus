"""
CLI session: settings, the wired-up components (composition root) and the state
of the last run (so /show, /record, /save work on what you just ran).
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.adapters.extractors.snippet_extractor import HeuristicSnippetExtractor
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument
from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceType
from src.adapters.normalizers.html_normalizer import TrafilaturaHtmlNormalizer
from src.adapters.normalizers.pdf_normalizer import PdfNormalizer
from src.adapters.normalizers.text_normalizer import TextNormalizer
from src.adapters.scrapers.pdf_adapter import PdfScraperAdapter
from src.adapters.scrapers.sitemap_adapter import TrafilaturaSitemapAdapter
from src.adapters.scrapers.trafilatura_adapter import TrafilaturaScraperAdapter
from src.core.normalizer_dispatcher import NormalizerDispatcher
from src.core.record_builder import ExtractionResult
from src.core.scraper_dispatcher import ScraperDispatcher
from src.ports.llm_port import LLMPort


class CLIError(Exception):
    """A user-facing error: printed in red, no traceback."""


SOURCE_TYPES = {"url": SourceType.URL, "file": SourceType.FILE, "pdf": SourceType.PDF, "sitemap": SourceType.SITEMAP}


@dataclass
class Settings:
    schema: str = "schemas/cpp_snippet.yaml"
    env_file: str = "azure.env"
    debug: bool = False          # skip the cost confirmation before LLM filling
    max_urls: int = 20           # cap on pages taken from a sitemap
    lines: int = 40              # preview length


@dataclass
class State:
    source: Source | None = None
    raw: RawDocument | None = None
    normalized: NormalizedDocument | None = None
    snippets: list[CodeSnippet] = field(default_factory=list)
    result: ExtractionResult | None = None


class Session:
    def __init__(
        self,
        settings: Settings | None = None,
        out: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
        interactive: bool | None = None,
    ):
        self.settings = settings or Settings()
        self.out = out
        self.input_fn = input_fn
        self.interactive = interactive
        self.state = State()
        self.scraper = ScraperDispatcher([TrafilaturaScraperAdapter(), PdfScraperAdapter()])
        self.normalizer = NormalizerDispatcher([TrafilaturaHtmlNormalizer(), PdfNormalizer(), TextNormalizer()])
        self.extractor = HeuristicSnippetExtractor()
        self._llm: LLMPort | None = None

    # ------------------------------------------------------------------ #
    # Components built on demand
    # ------------------------------------------------------------------ #

    @property
    def llm(self) -> LLMPort:
        """The only place that knows which provider is used."""
        if self._llm is None:
            from src.adapters.llm.azure_config import load_azure_config
            from src.adapters.llm.azure_openai_adapter import AzureOpenAIAdapter
            self._llm = AzureOpenAIAdapter(load_azure_config(self.settings.env_file))
        return self._llm

    def reset_llm(self) -> None:
        self._llm = None

    def sitemap(self) -> TrafilaturaSitemapAdapter:
        return TrafilaturaSitemapAdapter(max_urls=self.settings.max_urls)

    # ------------------------------------------------------------------ #
    # Sources
    # ------------------------------------------------------------------ #

    def make_source(self, target: str, kind: str | None = None) -> Source:
        """
        URL or local path -> Source. Without --type:
          http(s) ending in .pdf -> pdf; .xml/.xml.gz or a 'sitemap' name -> sitemap; else url.
          local path -> file (the dispatcher sends *.pdf to the PDF scraper).
        """
        if kind is not None:
            if kind not in SOURCE_TYPES:
                raise CLIError(f"Unknown type '{kind}'. Use one of: {', '.join(SOURCE_TYPES)}")
            source_type = SOURCE_TYPES[kind]
            if source_type in (SourceType.FILE,) or (source_type == SourceType.PDF and not _is_url(target)):
                target = str(self._local_path(target))
            return Source(id=str(uuid.uuid4()), url=target, type=source_type)

        if _is_url(target):
            path = urlparse(target).path.lower()
            name = path.rstrip("/").rsplit("/", 1)[-1]
            if path.endswith(".pdf"):
                source_type = SourceType.PDF
            elif path.endswith((".xml", ".xml.gz")) or "sitemap" in name:
                source_type = SourceType.SITEMAP
            else:
                source_type = SourceType.URL
            return Source(id=str(uuid.uuid4()), url=target, type=source_type)

        return Source(id=str(uuid.uuid4()), url=str(self._local_path(target)), type=SourceType.FILE)

    @staticmethod
    def _local_path(target: str) -> Path:
        path = Path(target).expanduser()
        if not path.exists():
            raise CLIError(f"No such file: {target}")
        if path.is_dir():
            raise CLIError(f"{target} is a directory; give a file")
        return path.resolve()

    def remember(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self.state, key, value)


def _is_url(target: str) -> bool:
    return urlparse(target).scheme in ("http", "https")
