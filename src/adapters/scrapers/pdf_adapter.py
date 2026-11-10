import io
import logging
import unicodedata
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
from pypdf import PageObject, PdfReader

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceType
from src.ports.scraper_port import ScraperPort


PDF_MIME_TYPE = "application/pdf"
PAGE_SEPARATOR = "\f"  # form feed: lets the normalization layer split pages back out

logger = logging.getLogger(__name__)


class PdfScraperAdapter(ScraperPort):
    """
    Reads PDF sources (remote URL or local file) and returns the raw extracted
    text as a RawDocument. No cleaning/normalization is applied here — pages are
    joined with a form-feed separator and passed through as-is.

    Each page's text layer is also inspected. Pages that look scanned (little or
    no text but an embedded image) or whose text is garbled (broken font
    encoding) are flagged in metadata["text_layer"] for a later OCR pass. Flagged
    documents are still a successful fetch.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        password: str | None = None,
        min_chars_per_page: int = 20,
        max_garbled_ratio: float = 0.3,
        layout: bool = True,
    ):
        """
        Args:
            timeout: HTTP timeout in seconds for remote PDFs.
            password: Password for encrypted PDFs (an empty password is always tried).
            min_chars_per_page: Pages with fewer non-whitespace chars are treated as having no text.
            max_garbled_ratio: Pages where more than this share of chars is junk are flagged for OCR.
            layout: Use pypdf's layout mode, which keeps indentation built from text positioning
                (essential for code). Pages where layout mode fails fall back to plain mode.
                Layout output is padded with spaces/blank lines; the normalizer compacts it.
        """
        self.timeout = timeout
        self.password = password
        self.min_chars_per_page = min_chars_per_page
        self.max_garbled_ratio = max_garbled_ratio
        self.layout = layout

    def fetch(self, source: Source) -> RawDocument:
        source.mark_crawling()

        try:
            is_remote = urlparse(source.url).scheme in ("http", "https")
            if source.type == SourceType.URL or (source.type == SourceType.PDF and is_remote):
                pdf_bytes = self._download_pdf(source.url)
                location_meta = {"url": source.url}
            elif source.type in (SourceType.FILE, SourceType.PDF):
                pdf_bytes = self._read_pdf_file(source.url)
                location_meta = {"path": source.url}
            else:
                raise NotImplementedError(f"Source type '{source.type}' not supported.")

            pages, page_has_images, pdf_meta = self._extract_text(pdf_bytes)
            text_layer = self._analyze_text_layer(pages, page_has_images)
            if text_layer["needs_ocr"]:
                logger.warning(
                    "PDF needs OCR on %d/%d page(s) %s: %s",
                    len(text_layer["ocr_pages"]), len(pages), text_layer["ocr_pages"], source.url,
                )

            source.mark_done()
            return RawDocument(
                id=str(uuid.uuid4()),
                source_id=source.id,
                content=PAGE_SEPARATOR.join(pages),
                mime_type=PDF_MIME_TYPE,
                metadata={
                    **location_meta,
                    "page_count": len(pages),
                    "page_separator": PAGE_SEPARATOR,
                    "pdf_metadata": pdf_meta,
                    "text_layer": text_layer,
                },
            )

        except Exception:
            source.mark_failed()
            raise

    def _download_pdf(self, url: str) -> bytes:
        """Downloads a PDF over HTTP(S)."""
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        content = response.content
        if not content:
            raise ValueError(f"Empty response when downloading PDF from URL: {url}")
        self._ensure_pdf(content, url)
        return content

    def _read_pdf_file(self, file_path: str) -> bytes:
        """Reads a local PDF file as bytes."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")
        content = path.read_bytes()
        self._ensure_pdf(content, file_path)
        return content

    def _extract_text(self, pdf_bytes: bytes) -> tuple[list[str], list[bool], dict[str, str]]:
        """Extracts raw per-page text, per-page image presence, and document info metadata."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            # Many PDFs are encrypted with an empty user password.
            if not reader.decrypt(self.password or ""):
                raise PermissionError("PDF is encrypted and could not be decrypted.")

        pages = [self._page_text(page) for page in reader.pages]
        page_has_images = [self._page_has_images(page) for page in reader.pages]

        info = reader.metadata or {}
        pdf_meta = {
            key.lstrip("/"): str(value)
            for key, value in info.items()
            if value is not None
        }
        return pages, page_has_images, pdf_meta

    def _page_text(self, page: PageObject) -> str:
        if self.layout:
            try:
                return page.extract_text(extraction_mode="layout") or ""
            except Exception:
                logger.debug("Layout extraction failed, falling back to plain mode", exc_info=True)
        return page.extract_text() or ""

    def _analyze_text_layer(self, pages: list[str], page_has_images: list[bool]) -> dict:
        """
        Classifies each page (0-based indices):
          - ocr_pages:   little/no text but an image (likely scanned), or garbled text
          - empty_pages: no text at all and no image (genuinely blank, nothing to recover)
        Short pages without images (e.g. a title page) are fine and not flagged.
        """
        ocr_pages: list[int] = []
        empty_pages: list[int] = []
        chars_per_page: list[int] = []

        for index, (text, has_images) in enumerate(zip(pages, page_has_images)):
            chars = [c for c in text if not c.isspace()]
            chars_per_page.append(len(chars))

            if len(chars) < self.min_chars_per_page and has_images:
                ocr_pages.append(index)
            elif not chars:
                empty_pages.append(index)
            elif self._garbled_ratio(chars) > self.max_garbled_ratio:
                ocr_pages.append(index)

        return {
            "needs_ocr": bool(ocr_pages),
            "ocr_pages": ocr_pages,
            "empty_pages": empty_pages,
            "chars_per_page": chars_per_page,
        }

    @staticmethod
    def _garbled_ratio(chars: list[str]) -> float:
        """Share of chars that are replacement, control, private-use or surrogate code points."""
        junk = sum(
            1 for c in chars
            if c == "\ufffd" or unicodedata.category(c) in ("Cc", "Co", "Cs")
        )
        return junk / len(chars)

    @classmethod
    def _page_has_images(cls, page: PageObject) -> bool:
        """
        Checks the page's resources for image XObjects, including images nested
        inside form XObjects. Malformed resources count as "has images" so that a
        textless page is flagged for OCR rather than silently treated as blank.
        """
        try:
            return cls._resources_have_images(page.get("/Resources"), depth=0)
        except Exception:
            logger.debug("Could not inspect page resources for images", exc_info=True)
            return True

    @classmethod
    def _resources_have_images(cls, resources, depth: int) -> bool:
        if resources is None or depth > 5:
            return False
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return False
        for ref in xobjects.get_object().values():
            xobject = ref.get_object()
            subtype = xobject.get("/Subtype")
            if subtype == "/Image":
                return True
            if subtype == "/Form" and cls._resources_have_images(xobject.get("/Resources"), depth + 1):
                return True
        return False

    @staticmethod
    def _ensure_pdf(content: bytes, location: str) -> None:
        # The PDF header may be preceded by a little junk; the spec allows it within the first 1KB.
        if b"%PDF-" not in content[:1024]:
            raise ValueError(f"Content at '{location}' does not look like a PDF.")
