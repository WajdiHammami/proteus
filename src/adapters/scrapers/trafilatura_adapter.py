import uuid
from pathlib import Path
from trafilatura.downloads import fetch_response
from trafilatura.utils import decode_file

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceType
from src.ports.scraper_port import ContentMismatchError, ScraperPort


PDF_MIME_TYPE = "application/pdf"


class TrafilaturaScraperAdapter(ScraperPort):
    """
    Fetches web pages (URL sources) and local text files (FILE sources).
    PDFs are left to the PDF scraper: local *.pdf files are not supported, and a URL
    that turns out to serve a PDF raises ContentMismatchError with the downloaded bytes.
    """

    def supports(self, source: Source) -> bool:
        if source.type == SourceType.URL:
            return True
        return source.type == SourceType.FILE and not _is_pdf_path(source.url)

    def fetch(self, source: Source) -> RawDocument:
        source.mark_crawling()

        try:
            if source.type == SourceType.URL:
                raw_html = self._fetch_raw_url(source)
                source.mark_done()
                return RawDocument(
                    id=str(uuid.uuid4()),
                    source_id=source.id,
                    content=raw_html,
                    mime_type="text/html",
                    metadata={"url": source.url}
                )

            elif source.type == SourceType.FILE:
                raw_content, mime_type, encoding = self._read_raw_file(source.url)
                source.mark_done()
                return RawDocument(
                    id=str(uuid.uuid4()),
                    source_id=source.id,
                    content=raw_content,
                    mime_type=mime_type,
                    metadata={"path": source.url, "encoding": encoding}
                )

            else:
                raise NotImplementedError(f"Source type '{source.type}' not supported.")

        except ContentMismatchError:
            raise  # source already released for another scraper
        except Exception as e:
            source.mark_failed()
            raise e

    def _fetch_raw_url(self, source: Source) -> str:
        """Downloads raw HTML content using Trafilatura; hands PDFs back to the dispatcher."""
        url = source.url
        response = fetch_response(url, decode=False)
        if not response or not response.data:
            raise ValueError(f"Failed to download raw content from URL: {url}")
        if response.status != 200:
            raise ValueError(f"Failed to download raw content from URL: {url} (HTTP {response.status})")

        # Links like /download?id=42 can serve a PDF without a .pdf suffix.
        content_type = (response.headers or {}).get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() == PDF_MIME_TYPE or _looks_like_pdf(response.data):
            source.release()
            raise ContentMismatchError(PDF_MIME_TYPE, payload=response.data, message=f"URL serves a PDF: {url}")

        downloaded_html = decode_file(response.data)
        if not downloaded_html:
            raise ValueError(f"Failed to download raw content from URL: {url}")
        return downloaded_html

    def _read_raw_file(self, file_path: str) -> tuple[str, str, str]:
        """Reads a local file; returns (text, mime type from the suffix, encoding used)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        raw_content, encoding = _decode_text(path.read_bytes(), file_path)

        # Simple mime-type distinction for local files
        suffix = path.suffix.lower()
        if path.suffix in _CASE_SENSITIVE_CODE_SUFFIXES:
            mime_type = _CASE_SENSITIVE_CODE_SUFFIXES[path.suffix]
        elif suffix in [".html", ".htm"]:
            mime_type = "text/html"
        elif suffix in [".md", ".markdown"]:
            mime_type = "text/markdown"
        elif suffix in _CODE_SUFFIXES:
            mime_type = _CODE_SUFFIXES[suffix]
        else:
            mime_type = "text/plain"

        return raw_content, mime_type, encoding


def _decode_text(data: bytes, location: str) -> tuple[str, str]:
    """
    Decodes a local text file. Older C/C++ sources are often not UTF-8, so:
      1. a BOM decides (UTF-8 / UTF-16 / UTF-32)
      2. strict UTF-8
      3. Windows-1252 (Latin-1 superset, the usual legacy encoding for Western code)
      4. Latin-1, which cannot fail
    Files containing NUL bytes (without a UTF-16/32 BOM) are binary and rejected.
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data[len(bom):].decode(encoding), encoding
            except UnicodeDecodeError as error:
                raise ValueError(f"File '{location}' has a {encoding} BOM but is not valid {encoding}.") from error

    if b"\x00" in data:
        raise ValueError(f"File '{location}' looks binary, not text.")

    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


# Longest first: the UTF-32-LE BOM starts with the UTF-16-LE one.
_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


def _is_pdf_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _looks_like_pdf(data: bytes) -> bool:
    # The spec allows a little junk before the header, within the first 1KB.
    return b"%PDF-" in data[:1024]


# C / C++ source files (freedesktop shared-mime-info types).
_CODE_SUFFIXES = {
    ".c": "text/x-csrc",
    ".h": "text/x-chdr",
    **dict.fromkeys([".cpp", ".cc", ".cxx", ".c++", ".cp", ".ipp", ".tpp", ".inl", ".ixx", ".cppm"], "text/x-c++src"),
    **dict.fromkeys([".hpp", ".hh", ".hxx", ".h++"], "text/x-c++hdr"),
}
# Unix convention: uppercase ".C" / ".H" are C++, not C.
_CASE_SENSITIVE_CODE_SUFFIXES = {".C": "text/x-c++src", ".H": "text/x-c++hdr"}