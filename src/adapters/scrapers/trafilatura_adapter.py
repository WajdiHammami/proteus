import uuid
from pathlib import Path
import trafilatura

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceType
from src.ports.scraper_port import ScraperPort


class TrafilaturaScraperAdapter(ScraperPort):
    def fetch(self, source: Source) -> RawDocument:
        source.mark_crawling()

        try:
            if source.type == SourceType.URL:
                raw_html = self._fetch_raw_url(source.url)
                source.mark_done()
                return RawDocument(
                    id=str(uuid.uuid4()),
                    source_id=source.id,
                    content=raw_html,
                    mime_type="text/html",
                    metadata={"url": source.url}
                )

            elif source.type == SourceType.FILE:
                raw_content, mime_type = self._read_raw_file(source.url)
                source.mark_done()
                return RawDocument(
                    id=str(uuid.uuid4()),
                    source_id=source.id,
                    content=raw_content,
                    mime_type=mime_type,
                    metadata={"path": source.url}
                )

            else:
                raise NotImplementedError(f"Source type '{source.type}' not supported.")

        except Exception as e:
            source.mark_failed()
            raise e

    def _fetch_raw_url(self, url: str) -> str:
        """Downloads raw HTML content using Trafilatura."""
        downloaded_html = trafilatura.fetch_url(url)
        if not downloaded_html:
            raise ValueError(f"Failed to download raw content from URL: {url}")
        return downloaded_html

    def _read_raw_file(self, file_path: str) -> tuple[str, str]:
        """Reads a local file and determines its basic mime type based on suffix."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")
        
        raw_content = path.read_text(encoding="utf-8")
        
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

        return raw_content, mime_type


# C / C++ source files (freedesktop shared-mime-info types).
_CODE_SUFFIXES = {
    ".c": "text/x-csrc",
    ".h": "text/x-chdr",
    **dict.fromkeys([".cpp", ".cc", ".cxx", ".c++", ".cp", ".ipp", ".tpp", ".inl", ".ixx", ".cppm"], "text/x-c++src"),
    **dict.fromkeys([".hpp", ".hh", ".hxx", ".h++"], "text/x-c++hdr"),
}
# Unix convention: uppercase ".C" / ".H" are C++, not C.
_CASE_SENSITIVE_CODE_SUFFIXES = {".C": "text/x-c++src", ".H": "text/x-c++hdr"}