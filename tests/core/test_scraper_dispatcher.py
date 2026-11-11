import pytest

from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import SourceStatus, SourceType
from src.adapters.scrapers import trafilatura_adapter
from src.adapters.scrapers.pdf_adapter import PdfScraperAdapter
from src.adapters.scrapers.trafilatura_adapter import TrafilaturaScraperAdapter
from src.core.scraper_dispatcher import ScraperDispatcher
from src.ports.scraper_port import ContentMismatchError, ScraperPort, UnsupportedSourceError
from tests.conftest import build_pdf


class FakeScraper(ScraperPort):
    """Accepts the given source types; behaves like a real adapter with the Source state machine."""

    def __init__(self, types, mismatch=None, error=None, with_payload=True):
        self.types = set(types)
        self.mismatch = mismatch          # ContentMismatchError to raise from fetch
        self.error = error                # other exception to raise from fetch
        self.with_payload = with_payload  # implements fetch_payload?
        self.fetched = []
        self.payloads = []

    def supports(self, source):
        return source.type in self.types

    def fetch(self, source):
        source.mark_crawling()
        if self.mismatch:
            source.release()
            raise self.mismatch
        if self.error:
            source.mark_failed()
            raise self.error
        self.fetched.append(source.url)
        source.mark_done()
        return RawDocument(id="r", source_id=source.id, content="c", metadata={"origin": "fetch"})

    def fetch_payload(self, source, payload):
        if not self.with_payload:
            return super().fetch_payload(source, payload)
        source.mark_crawling()
        self.payloads.append(payload)
        source.mark_done()
        return RawDocument(id="r", source_id=source.id, content="c", metadata={"origin": "payload"})


class TestRouting:
    def test_first_supporting_scraper_wins(self, make_source):
        html = FakeScraper({SourceType.URL})
        other = FakeScraper({SourceType.URL})
        dispatcher = ScraperDispatcher([html, other])

        doc = dispatcher.fetch(make_source("https://example.com", SourceType.URL))

        assert html.fetched == ["https://example.com"]
        assert other.fetched == []
        assert doc.metadata["scraper"] == "FakeScraper"

    def test_routes_by_type(self, make_source):
        html = FakeScraper({SourceType.URL, SourceType.FILE})
        pdf = FakeScraper({SourceType.PDF})
        dispatcher = ScraperDispatcher([html, pdf])

        dispatcher.fetch(make_source("https://example.com/a.pdf", SourceType.PDF))
        dispatcher.fetch(make_source("https://example.com/page", SourceType.URL))

        assert pdf.fetched == ["https://example.com/a.pdf"]
        assert html.fetched == ["https://example.com/page"]

    def test_supports_reflects_registered_scrapers(self, make_source):
        dispatcher = ScraperDispatcher([FakeScraper({SourceType.URL})])
        assert dispatcher.supports(make_source("u", SourceType.URL))
        assert not dispatcher.supports(make_source("u", SourceType.PDF))

    def test_needs_at_least_one_scraper(self):
        with pytest.raises(ValueError):
            ScraperDispatcher([])


class TestUnsupported:
    def test_no_match_raises_and_leaves_source_pending(self, make_source):
        dispatcher = ScraperDispatcher([FakeScraper({SourceType.URL})])
        source = make_source("https://example.com/a.pdf", SourceType.PDF)

        with pytest.raises(UnsupportedSourceError, match="pdf"):
            dispatcher.fetch(source)
        assert source.status == SourceStatus.PENDING

    def test_sitemap_error_points_to_sitemap_port(self, make_source):
        dispatcher = ScraperDispatcher([FakeScraper({SourceType.URL})])
        with pytest.raises(UnsupportedSourceError, match="SitemapPort"):
            dispatcher.fetch(make_source("https://example.com/sitemap.xml", SourceType.SITEMAP))

    def test_scraper_errors_propagate_untouched(self, make_source):
        dispatcher = ScraperDispatcher([FakeScraper({SourceType.URL}, error=ValueError("boom"))])
        source = make_source("https://example.com", SourceType.URL)
        with pytest.raises(ValueError, match="boom"):
            dispatcher.fetch(source)
        assert source.status == SourceStatus.FAILED


class TestReroute:
    def mismatch(self, payload=b"%PDF-1.4"):
        return ContentMismatchError("application/pdf", payload=payload)

    def test_reroutes_with_payload(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch())
        pdf = FakeScraper({SourceType.PDF})
        source = make_source("https://example.com/download?id=42", SourceType.URL)

        doc = ScraperDispatcher([html, pdf]).fetch(source)

        assert pdf.payloads == [b"%PDF-1.4"]
        assert pdf.fetched == []                  # no second download
        assert source.type == SourceType.PDF      # corrected for later runs
        assert source.status == SourceStatus.DONE
        assert doc.metadata["rerouted_from"] == "url"
        assert doc.metadata["scraper"] == "FakeScraper"

    def test_falls_back_to_fetch_without_payload_support(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch())
        pdf = FakeScraper({SourceType.PDF}, with_payload=False)
        source = make_source("https://example.com/download?id=42", SourceType.URL)

        doc = ScraperDispatcher([html, pdf]).fetch(source)

        assert pdf.fetched == ["https://example.com/download?id=42"]
        assert doc.metadata["origin"] == "fetch"
        assert source.status == SourceStatus.DONE

    def test_falls_back_to_fetch_without_payload(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch(payload=None))
        pdf = FakeScraper({SourceType.PDF})
        ScraperDispatcher([html, pdf]).fetch(make_source("https://example.com/d", SourceType.URL))
        assert pdf.fetched == ["https://example.com/d"]

    def test_no_scraper_for_detected_content(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch())
        source = make_source("https://example.com/d", SourceType.URL)

        with pytest.raises(UnsupportedSourceError, match="application/pdf"):
            ScraperDispatcher([html]).fetch(source)
        assert source.type == SourceType.URL      # not changed when nothing can take it
        assert source.status == SourceStatus.PENDING

    def test_unknown_detected_mime(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=ContentMismatchError("application/zip"))
        source = make_source("https://example.com/d.zip", SourceType.URL)
        with pytest.raises(UnsupportedSourceError, match="application/zip"):
            ScraperDispatcher([html, FakeScraper({SourceType.PDF})]).fetch(source)
        assert source.status == SourceStatus.PENDING

    def test_reroutes_only_once(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch())
        pdf = FakeScraper({SourceType.PDF}, mismatch=self.mismatch(), with_payload=False)
        source = make_source("https://example.com/d", SourceType.URL)

        with pytest.raises(ContentMismatchError):
            ScraperDispatcher([html, pdf]).fetch(source)

    def test_failure_in_new_scraper_propagates(self, make_source):
        html = FakeScraper({SourceType.URL}, mismatch=self.mismatch())
        pdf = FakeScraper({SourceType.PDF}, error=ValueError("corrupt"), with_payload=False)
        source = make_source("https://example.com/d", SourceType.URL)
        with pytest.raises(ValueError, match="corrupt"):
            ScraperDispatcher([html, pdf]).fetch(source)
        assert source.status == SourceStatus.FAILED


class TestRealAdapters:
    """The production setup: trafilatura first, then the PDF scraper."""

    @pytest.fixture
    def dispatcher(self):
        return ScraperDispatcher([TrafilaturaScraperAdapter(), PdfScraperAdapter()])

    def test_url_serving_pdf_ends_up_as_pdf_document(self, dispatcher, make_source, monkeypatch):
        pdf_bytes = build_pdf([{"text": "std::vector<int> v;"}])

        class Response:
            data = pdf_bytes
            status = 200
            headers = {"content-type": "application/pdf"}

        calls = []
        monkeypatch.setattr(trafilatura_adapter, "fetch_response", lambda url, decode=False: calls.append(url) or Response())
        source = make_source("https://example.com/download?id=42", SourceType.URL)

        doc = dispatcher.fetch(source)

        assert calls == ["https://example.com/download?id=42"]   # downloaded once
        assert doc.mime_type == "application/pdf"
        assert "std::vector<int> v;" in doc.content
        assert doc.metadata["scraper"] == "PdfScraperAdapter"
        assert doc.metadata["rerouted_from"] == "url"
        assert doc.metadata["url"] == "https://example.com/download?id=42"
        assert source.type == SourceType.PDF
        assert source.status == SourceStatus.DONE

    def test_local_pdf_file_goes_to_pdf_scraper(self, dispatcher, write_pdf, make_source):
        path = write_pdf([{"text": "int main() { return 0; }"}])
        doc = dispatcher.fetch(make_source(path, SourceType.FILE))
        assert doc.metadata["scraper"] == "PdfScraperAdapter"
        assert "int main()" in doc.content

    def test_local_cpp_file_goes_to_trafilatura(self, dispatcher, tmp_path, make_source):
        path = tmp_path / "main.cpp"
        path.write_text("int main() {}", encoding="utf-8")
        doc = dispatcher.fetch(make_source(str(path), SourceType.FILE))
        assert doc.metadata["scraper"] == "TrafilaturaScraperAdapter"
        assert doc.mime_type == "text/x-c++src"
