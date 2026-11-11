import pytest

from src.adapters.models.source import SourceStatus, SourceType
from src.adapters.scrapers import trafilatura_adapter
from src.adapters.scrapers.trafilatura_adapter import TrafilaturaScraperAdapter
from src.ports.scraper_port import ContentMismatchError


HTML = "<html><head><title>T</title></head><body><p>Hello   world</p></body></html>"


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        self.data = data
        self.status = status
        self.headers = headers if headers is not None else {"content-type": "text/html; charset=utf-8"}

    def __bool__(self):
        return self.data is not None


@pytest.fixture
def fake_fetch_url(monkeypatch):
    """Stubs trafilatura's download. `result` is a str (HTML), bytes, a FakeResponse or None."""
    calls = []

    def install(result):
        def _fetch(url, decode=False):
            calls.append(url)
            if result is None or isinstance(result, FakeResponse):
                return result
            data = result.encode("utf-8") if isinstance(result, str) else result
            return FakeResponse(data)

        monkeypatch.setattr(trafilatura_adapter, "fetch_response", _fetch)
        return calls

    return install


class TestUrlSource:
    def test_returns_raw_html(self, fake_fetch_url, make_source):
        calls = fake_fetch_url(HTML)
        source = make_source("https://example.com/page", SourceType.URL, id="s1")

        doc = TrafilaturaScraperAdapter().fetch(source)

        assert calls == ["https://example.com/page"]
        assert doc.content == HTML  # raw, not extracted/normalized
        assert doc.mime_type == "text/html"
        assert doc.source_id == "s1"
        assert doc.metadata == {"url": "https://example.com/page"}
        assert source.status == SourceStatus.DONE

    @pytest.mark.parametrize("result", [None, ""])
    def test_failed_download_marks_failed(self, fake_fetch_url, make_source, result):
        fake_fetch_url(result)
        source = make_source("https://example.com/down", SourceType.URL)

        with pytest.raises(ValueError, match="Failed to download"):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED


    def test_non_200_marks_failed(self, fake_fetch_url, make_source):
        fake_fetch_url(FakeResponse(b"<html>gone</html>", status=404))
        source = make_source("https://example.com/gone", SourceType.URL)
        with pytest.raises(ValueError, match="HTTP 404"):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED


class TestPdfBehindUrl:
    """A URL without .pdf can serve a PDF: hand it back with the bytes instead of failing."""

    @pytest.mark.parametrize("response", [
        FakeResponse(b"%PDF-1.4 ...", headers={"content-type": "application/pdf"}),
        FakeResponse(b"%PDF-1.4 ...", headers={"content-type": "application/octet-stream"}),  # sniffed
        FakeResponse(b"\n\n%PDF-1.7 ...", headers={}),
        FakeResponse(b"not really", headers={"content-type": "Application/PDF; qs=0.9"}),  # header wins
    ])
    def test_raises_mismatch_and_releases_source(self, fake_fetch_url, make_source, response):
        fake_fetch_url(response)
        source = make_source("https://example.com/download?id=42", SourceType.URL)

        with pytest.raises(ContentMismatchError) as info:
            TrafilaturaScraperAdapter().fetch(source)

        assert info.value.detected_mime == "application/pdf"
        assert info.value.payload == response.data
        assert source.status == SourceStatus.PENDING

    def test_html_mentioning_pdf_magic_late_is_html(self, fake_fetch_url, make_source):
        html = "<html>" + "x" * 2000 + "%PDF-1.4</html>"
        fake_fetch_url(html)
        doc = TrafilaturaScraperAdapter().fetch(make_source("https://example.com/p", SourceType.URL))
        assert doc.content == html


class TestSupports:
    @pytest.mark.parametrize("url, source_type, expected", [
        ("https://example.com/page", SourceType.URL, True),
        ("https://example.com/paper.pdf", SourceType.URL, True),   # URL type: content decides
        ("/tmp/a.cpp", SourceType.FILE, True),
        ("/tmp/a.html", SourceType.FILE, True),
        ("/tmp/a.pdf", SourceType.FILE, False),
        ("/tmp/A.PDF", SourceType.FILE, False),
        ("https://example.com/a.pdf", SourceType.PDF, False),
        ("https://example.com/sitemap.xml", SourceType.SITEMAP, False),
    ])
    def test_supports(self, make_source, url, source_type, expected):
        assert TrafilaturaScraperAdapter().supports(make_source(url, source_type)) is expected


class TestFileSource:
    @pytest.mark.parametrize("suffix, mime", [
        (".html", "text/html"),
        (".htm", "text/html"),
        (".HTML", "text/html"),
        (".md", "text/markdown"),
        (".markdown", "text/markdown"),
        (".txt", "text/plain"),
        ("", "text/plain"),
        (".c", "text/x-csrc"),
        (".h", "text/x-chdr"),
        (".cpp", "text/x-c++src"),
        (".CPP", "text/x-c++src"),
        (".cc", "text/x-c++src"),
        (".cxx", "text/x-c++src"),
        (".c++", "text/x-c++src"),
        (".ipp", "text/x-c++src"),
        (".tpp", "text/x-c++src"),
        (".inl", "text/x-c++src"),
        (".ixx", "text/x-c++src"),
        (".cppm", "text/x-c++src"),
        (".hpp", "text/x-c++hdr"),
        (".hh", "text/x-c++hdr"),
        (".hxx", "text/x-c++hdr"),
        (".h++", "text/x-c++hdr"),
        (".C", "text/x-c++src"),   # Unix convention: uppercase .C is C++
        (".H", "text/x-c++hdr"),
        (".py", "text/plain"),     # only C/C++ get a code mime type for now
    ])
    def test_reads_file_and_sets_mime_type(self, tmp_path, make_source, suffix, mime):
        path = tmp_path / f"doc{suffix}"
        path.write_text("content here", encoding="utf-8")
        source = make_source(str(path), SourceType.FILE)

        doc = TrafilaturaScraperAdapter().fetch(source)

        assert doc.content == "content here"
        assert doc.mime_type == mime
        assert doc.metadata == {"path": str(path), "encoding": "utf-8"}
        assert source.status == SourceStatus.DONE

    def test_utf8_content_preserved(self, tmp_path, make_source):
        path = tmp_path / "page.html"
        path.write_text("<p>café – مرحبا – 你好</p>", encoding="utf-8")
        doc = TrafilaturaScraperAdapter().fetch(make_source(str(path), SourceType.FILE))
        assert doc.content == "<p>café – مرحبا – 你好</p>"

    def test_missing_file_marks_failed(self, tmp_path, make_source):
        source = make_source(str(tmp_path / "missing.html"), SourceType.FILE)
        with pytest.raises(FileNotFoundError):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_binary_file_marks_failed(self, tmp_path, make_source):
        path = tmp_path / "image.html"
        path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR binary")
        source = make_source(str(path), SourceType.FILE)
        with pytest.raises(ValueError, match="binary"):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED


class TestFileEncoding:
    """Legacy C/C++ files are often not UTF-8; they must still load."""

    def read(self, tmp_path, make_source, data: bytes, name="legacy.cpp"):
        path = tmp_path / name
        path.write_bytes(data)
        return TrafilaturaScraperAdapter().fetch(make_source(str(path), SourceType.FILE))

    def test_windows_1252_file(self, tmp_path, make_source):
        code = "// Auteur : Fran\u00e7ois \u2014 \u00a9 1998\nint main() { return 0; }\n"
        doc = self.read(tmp_path, make_source, code.encode("cp1252"))
        assert doc.content == code
        assert doc.metadata["encoding"] == "cp1252"
        assert doc.mime_type == "text/x-c++src"

    def test_latin1_bytes_undefined_in_cp1252(self, tmp_path, make_source):
        doc = self.read(tmp_path, make_source, b"// \x81\x8d\nint x;\n")
        assert doc.content == "// \x81\x8d\nint x;\n"
        assert doc.metadata["encoding"] == "latin-1"

    def test_utf8_bom_removed(self, tmp_path, make_source):
        doc = self.read(tmp_path, make_source, b"\xef\xbb\xbf#include <vector>\n")
        assert doc.content == "#include <vector>\n"
        assert doc.metadata["encoding"] == "utf-8"

    @pytest.mark.parametrize("encoding, bom", [
        ("utf-16-le", b"\xff\xfe"),
        ("utf-16-be", b"\xfe\xff"),
        ("utf-32-le", b"\xff\xfe\x00\x00"),
    ])
    def test_utf16_and_utf32_with_bom(self, tmp_path, make_source, encoding, bom):
        code = "int main() { return 0; } // \u00e9\n"
        doc = self.read(tmp_path, make_source, bom + code.encode(encoding))
        assert doc.content == code
        assert doc.metadata["encoding"] == encoding

    def test_invalid_data_after_bom_marks_failed(self, tmp_path, make_source):
        path = tmp_path / "broken.cpp"
        path.write_bytes(b"\xff\xfe\x00\x81\x9f")  # UTF-16 BOM + odd byte count
        source = make_source(str(path), SourceType.FILE)
        with pytest.raises(ValueError, match="BOM"):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_utf8_preferred_over_cp1252(self, tmp_path, make_source):
        doc = self.read(tmp_path, make_source, "// caf\u00e9\n".encode("utf-8"))
        assert doc.content == "// caf\u00e9\n"
        assert doc.metadata["encoding"] == "utf-8"


class TestUnsupported:
    @pytest.mark.parametrize("source_type", [SourceType.SITEMAP, SourceType.PDF])
    def test_unsupported_types_mark_failed(self, make_source, fake_fetch_url, source_type):
        calls = fake_fetch_url(HTML)
        source = make_source("https://example.com/x", source_type)

        with pytest.raises(NotImplementedError):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED
        assert calls == []

    def test_source_not_pending_is_rejected(self, make_source, fake_fetch_url):
        fake_fetch_url(HTML)
        source = make_source("https://example.com/x", SourceType.URL)
        source.status = SourceStatus.DONE
        with pytest.raises(ValueError):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.DONE
