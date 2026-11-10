import pytest

from src.adapters.models.source import SourceStatus, SourceType
from src.adapters.scrapers import trafilatura_adapter
from src.adapters.scrapers.trafilatura_adapter import TrafilaturaScraperAdapter


HTML = "<html><head><title>T</title></head><body><p>Hello   world</p></body></html>"


@pytest.fixture
def fake_fetch_url(monkeypatch):
    calls = []

    def install(result):
        def _fetch(url):
            calls.append(url)
            return result

        monkeypatch.setattr(trafilatura_adapter.trafilatura, "fetch_url", _fetch)
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
        assert doc.metadata == {"path": str(path)}
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
        path.write_bytes(b"\xff\xfe\x00\x81\x9f binary")
        source = make_source(str(path), SourceType.FILE)
        with pytest.raises(UnicodeDecodeError):
            TrafilaturaScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED


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
