import logging

import pytest
import requests

from src.adapters.models.source import SourceStatus, SourceType
from src.adapters.scrapers import pdf_adapter
from src.adapters.scrapers.pdf_adapter import PAGE_SEPARATOR, PdfScraperAdapter
from tests.conftest import LONG_TEXT, build_pdf, encrypt_pdf


class FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def fake_get(monkeypatch):
    """Replaces requests.get in the adapter; records calls and returns the configured response."""
    calls = []

    def install(response: FakeResponse):
        def _get(url, timeout=None):
            calls.append({"url": url, "timeout": timeout})
            return response

        monkeypatch.setattr(pdf_adapter.requests, "get", _get)
        return calls

    return install


# --------------------------------------------------------------------------- #
# Main job: reading text out of PDFs
# --------------------------------------------------------------------------- #

class TestLocalPdf:
    def test_extracts_text_from_every_page_joined_by_separator(self, write_pdf, make_source):
        path = write_pdf([{"text": "First page content"}, {"text": "Second page content"}])
        source = make_source(path, SourceType.FILE)

        doc = PdfScraperAdapter().fetch(source)

        pages = doc.content.split(PAGE_SEPARATOR)
        assert len(pages) == 2
        assert "First page content" in pages[0]
        assert "Second page content" in pages[1]

    def test_builds_raw_document(self, write_pdf, make_source):
        path = write_pdf([{"text": LONG_TEXT}], title="Annual Report")
        source = make_source(path, SourceType.FILE, id="abc")

        doc = PdfScraperAdapter().fetch(source)

        assert doc.source_id == "abc"
        assert doc.id
        assert doc.mime_type == "application/pdf"
        assert doc.metadata["path"] == path
        assert "url" not in doc.metadata
        assert doc.metadata["page_count"] == 1
        assert doc.metadata["page_separator"] == PAGE_SEPARATOR
        assert doc.metadata["pdf_metadata"]["Title"] == "Annual Report"

    def test_each_fetch_gets_a_new_document_id(self, write_pdf, make_source):
        path = write_pdf([{"text": LONG_TEXT}])
        first = PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        second = PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        assert first.id != second.id

    def test_marks_source_done(self, write_pdf, make_source):
        source = make_source(write_pdf([{"text": LONG_TEXT}]), SourceType.FILE)
        PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.DONE

    def test_pdf_type_with_local_path_reads_file(self, write_pdf, make_source, fake_get):
        calls = fake_get(FakeResponse(b"should not be used"))
        path = write_pdf([{"text": LONG_TEXT}])
        source = make_source(path, SourceType.PDF)

        doc = PdfScraperAdapter().fetch(source)

        assert calls == []
        assert doc.metadata["path"] == path
        assert source.status == SourceStatus.DONE

    def test_text_is_not_normalized(self, write_pdf, make_source):
        # Odd spacing / casing must pass through untouched; cleanup is the normalizer's job.
        path = write_pdf([{"text": "  MiXeD   CaSe   spacing  "}])
        doc = PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        assert "MiXeD   CaSe   spacing" in doc.content

    def test_pdf_without_info_dict_has_empty_pdf_metadata(self, write_pdf, make_source):
        doc = PdfScraperAdapter().fetch(make_source(write_pdf([{"text": LONG_TEXT}]), SourceType.FILE))
        assert doc.metadata["pdf_metadata"] == {}


class TestRemotePdf:
    @pytest.mark.parametrize("source_type", [SourceType.URL, SourceType.PDF])
    def test_downloads_and_extracts(self, make_source, fake_get, source_type):
        calls = fake_get(FakeResponse(build_pdf([{"text": LONG_TEXT}])))
        url = "https://example.com/docs/report.pdf"
        source = make_source(url, source_type)

        doc = PdfScraperAdapter(timeout=5).fetch(source)

        assert calls == [{"url": url, "timeout": 5}]
        assert LONG_TEXT in doc.content
        assert doc.metadata["url"] == url
        assert "path" not in doc.metadata
        assert source.status == SourceStatus.DONE

    def test_uppercase_scheme_is_treated_as_remote(self, make_source, fake_get):
        calls = fake_get(FakeResponse(build_pdf([{"text": LONG_TEXT}])))
        PdfScraperAdapter().fetch(make_source("HTTPS://example.com/a.pdf", SourceType.PDF))
        assert len(calls) == 1

    def test_http_error_marks_failed(self, make_source, fake_get):
        fake_get(FakeResponse(status_code=404))
        source = make_source("https://example.com/missing.pdf", SourceType.PDF)

        with pytest.raises(requests.HTTPError):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_empty_response_marks_failed(self, make_source, fake_get):
        fake_get(FakeResponse(b""))
        source = make_source("https://example.com/empty.pdf", SourceType.URL)

        with pytest.raises(ValueError, match="Empty response"):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_html_instead_of_pdf_marks_failed(self, make_source, fake_get):
        fake_get(FakeResponse(b"<html><body>Login required</body></html>"))
        source = make_source("https://example.com/paywalled.pdf", SourceType.PDF)

        with pytest.raises(ValueError, match="does not look like a PDF"):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_network_exception_marks_failed(self, make_source, monkeypatch):
        def boom(*args, **kwargs):
            raise requests.ConnectionError("unreachable")

        monkeypatch.setattr(pdf_adapter.requests, "get", boom)
        source = make_source("https://example.com/a.pdf", SourceType.PDF)

        with pytest.raises(requests.ConnectionError):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED


# --------------------------------------------------------------------------- #
# Edge cases: bad inputs and source state
# --------------------------------------------------------------------------- #

class TestInputErrors:
    def test_missing_file_marks_failed(self, tmp_path, make_source):
        source = make_source(str(tmp_path / "nope.pdf"), SourceType.FILE)
        with pytest.raises(FileNotFoundError):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_non_pdf_file_marks_failed(self, write_pdf, make_source):
        path = write_pdf([], raw=b"just some plain text, not a pdf")
        source = make_source(path, SourceType.FILE)
        with pytest.raises(ValueError, match="does not look like a PDF"):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_header_preceded_by_junk_is_accepted(self, write_pdf, make_source):
        path = write_pdf([], raw=b"\r\n\x00junk" + build_pdf([{"text": LONG_TEXT}]))
        doc = PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        assert LONG_TEXT in doc.content

    def test_corrupt_pdf_marks_failed(self, write_pdf, make_source):
        path = write_pdf([], raw=b"%PDF-1.4\n this is not really a pdf body \n%%EOF")
        source = make_source(path, SourceType.FILE)
        with pytest.raises(Exception):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    def test_unsupported_source_type_marks_failed(self, make_source):
        source = make_source("https://example.com/sitemap.xml", SourceType.SITEMAP)
        with pytest.raises(NotImplementedError):
            PdfScraperAdapter().fetch(source)
        assert source.status == SourceStatus.FAILED

    @pytest.mark.parametrize("status", [SourceStatus.CRAWLING, SourceStatus.DONE, SourceStatus.FAILED])
    def test_source_not_pending_is_rejected_and_left_unchanged(self, write_pdf, make_source, status):
        source = make_source(write_pdf([{"text": LONG_TEXT}]), SourceType.FILE)
        source.status = status
        with pytest.raises(ValueError):
            PdfScraperAdapter().fetch(source)
        assert source.status == status

    def test_zero_page_pdf(self, write_pdf, make_source):
        source = make_source(write_pdf([]), SourceType.FILE)
        doc = PdfScraperAdapter().fetch(source)

        assert doc.content == ""
        assert doc.metadata["page_count"] == 0
        assert doc.metadata["text_layer"] == {
            "needs_ocr": False, "ocr_pages": [], "empty_pages": [], "chars_per_page": [],
        }
        assert source.status == SourceStatus.DONE


class TestEncryption:
    def test_empty_user_password_is_opened_automatically(self, write_pdf, make_source):
        raw = encrypt_pdf(build_pdf([{"text": LONG_TEXT}]), user_password="")
        doc = PdfScraperAdapter().fetch(make_source(write_pdf([], raw=raw), SourceType.FILE))
        assert LONG_TEXT in doc.content

    def test_correct_password_opens_pdf(self, write_pdf, make_source):
        raw = encrypt_pdf(build_pdf([{"text": LONG_TEXT}]), user_password="s3cret")
        doc = PdfScraperAdapter(password="s3cret").fetch(make_source(write_pdf([], raw=raw), SourceType.FILE))
        assert LONG_TEXT in doc.content

    @pytest.mark.parametrize("password", [None, "wrong"])
    def test_missing_or_wrong_password_marks_failed(self, write_pdf, make_source, password):
        raw = encrypt_pdf(build_pdf([{"text": LONG_TEXT}]), user_password="s3cret")
        source = make_source(write_pdf([], raw=raw), SourceType.FILE)

        with pytest.raises(PermissionError):
            PdfScraperAdapter(password=password).fetch(source)
        assert source.status == SourceStatus.FAILED


# --------------------------------------------------------------------------- #
# OCR detection (text_layer metadata)
# --------------------------------------------------------------------------- #

def _text_layer(write_pdf, make_source, pages, **adapter_kwargs):
    source = make_source(write_pdf(pages), SourceType.FILE)
    doc = PdfScraperAdapter(**adapter_kwargs).fetch(source)
    return doc, source, doc.metadata["text_layer"]


class TestOcrDetection:
    def test_clean_document_needs_no_ocr(self, write_pdf, make_source):
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": LONG_TEXT}, {"text": LONG_TEXT}])
        assert layer["needs_ocr"] is False
        assert layer["ocr_pages"] == []
        assert layer["empty_pages"] == []
        assert all(n > 20 for n in layer["chars_per_page"])

    def test_image_only_page_is_flagged(self, write_pdf, make_source):
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": LONG_TEXT}, {"image": True}])
        assert layer["needs_ocr"] is True
        assert layer["ocr_pages"] == [1]
        assert layer["chars_per_page"][1] == 0

    def test_image_nested_in_form_xobject_is_flagged(self, write_pdf, make_source):
        _, _, layer = _text_layer(write_pdf, make_source, [{"form_image": True}])
        assert layer["ocr_pages"] == [0]

    def test_blank_page_is_empty_not_ocr(self, write_pdf, make_source):
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": LONG_TEXT}, {}])
        assert layer["needs_ocr"] is False
        assert layer["empty_pages"] == [1]

    def test_short_text_page_without_image_is_not_flagged(self, write_pdf, make_source):
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": "Title"}])
        assert layer["ocr_pages"] == []
        assert layer["empty_pages"] == []

    def test_short_text_over_image_is_flagged(self, write_pdf, make_source):
        # e.g. a scanner stamping a page number on top of the scan
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": "p. 3", "image": True}])
        assert layer["ocr_pages"] == [0]

    def test_long_text_with_image_is_not_flagged(self, write_pdf, make_source):
        # e.g. a normal page with a photo or logo
        _, _, layer = _text_layer(write_pdf, make_source, [{"text": LONG_TEXT, "image": True}])
        assert layer["needs_ocr"] is False

    def test_mixed_document_flags_only_scanned_pages(self, write_pdf, make_source):
        pages = [{"text": LONG_TEXT}, {"image": True}, {}, {"form_image": True}, {"text": LONG_TEXT}]
        doc, _, layer = _text_layer(write_pdf, make_source, pages)
        assert layer["ocr_pages"] == [1, 3]
        assert layer["empty_pages"] == [2]
        assert len(layer["chars_per_page"]) == 5
        # text from good pages is still extracted
        assert doc.content.split(PAGE_SEPARATOR)[0].strip() == LONG_TEXT

    def test_fully_scanned_document_is_still_a_successful_fetch(self, write_pdf, make_source):
        _, source, layer = _text_layer(write_pdf, make_source, [{"image": True}, {"image": True}])
        assert source.status == SourceStatus.DONE
        assert layer["ocr_pages"] == [0, 1]

    def test_min_chars_threshold_is_configurable(self, write_pdf, make_source):
        pages = [{"text": LONG_TEXT, "image": True}]
        _, _, layer = _text_layer(write_pdf, make_source, pages, min_chars_per_page=500)
        assert layer["ocr_pages"] == [0]

    def test_warning_logged_when_ocr_needed(self, write_pdf, make_source, caplog):
        path = write_pdf([{"text": LONG_TEXT}, {"image": True}])
        with caplog.at_level(logging.WARNING, logger=pdf_adapter.__name__):
            PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "needs OCR" in message
        assert "1/2" in message
        assert path in message

    def test_no_warning_for_clean_document(self, write_pdf, make_source, caplog):
        path = write_pdf([{"text": LONG_TEXT}])
        with caplog.at_level(logging.WARNING, logger=pdf_adapter.__name__):
            PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        assert caplog.records == []

    def test_malformed_resources_count_as_images(self, write_pdf, make_source, monkeypatch):
        def broken(cls, resources, depth):
            raise KeyError("bad resources")

        monkeypatch.setattr(PdfScraperAdapter, "_resources_have_images", classmethod(broken))
        _, source, layer = _text_layer(write_pdf, make_source, [{}])
        assert layer["ocr_pages"] == [0]
        assert source.status == SourceStatus.DONE


class TestAnalyzeTextLayer:
    """Direct tests of the classifier, for cases hard to produce in a real PDF."""

    def test_garbled_private_use_text_is_flagged(self):
        garbled = "" * 10 + "abc"
        layer = PdfScraperAdapter()._analyze_text_layer([garbled], [False])
        assert layer["ocr_pages"] == [0]

    def test_replacement_characters_are_flagged(self):
        layer = PdfScraperAdapter()._analyze_text_layer(["�" * 40], [False])
        assert layer["ocr_pages"] == [0]

    def test_a_few_odd_characters_are_tolerated(self):
        text = "Normal readable sentence with one odd glyph � in the middle."
        layer = PdfScraperAdapter()._analyze_text_layer([text], [False])
        assert layer["needs_ocr"] is False

    def test_garbled_ratio_threshold_is_configurable(self):
        text = "a" * 80 + "�" * 20  # 20% junk
        assert PdfScraperAdapter()._analyze_text_layer([text], [False])["needs_ocr"] is False
        strict = PdfScraperAdapter(max_garbled_ratio=0.1)
        assert strict._analyze_text_layer([text], [False])["needs_ocr"] is True

    def test_non_latin_text_is_not_garbled(self):
        arabic = "هذا نص عربي عادي يحتوي على عدد كاف من الأحرف"
        chinese = "这是一段普通的中文文本内容，包含足够多的字符用于测试"
        layer = PdfScraperAdapter()._analyze_text_layer([arabic, chinese], [False, False])
        assert layer["needs_ocr"] is False

    def test_whitespace_only_page_is_empty(self):
        layer = PdfScraperAdapter()._analyze_text_layer(["   \n\t  "], [False])
        assert layer["empty_pages"] == [0]
        assert layer["chars_per_page"] == [0]


# --------------------------------------------------------------------------- #
# Layout mode (indentation for code)
# --------------------------------------------------------------------------- #

CODE_LINES = [(0, "def add(a, b):"), (24, "if a > b:"), (48, "return a - b"), (24, "return a + b")]


def _indents(text: str) -> dict[str, int]:
    return {line.strip(): len(line) - len(line.lstrip()) for line in text.split("\n") if line.strip()}


class TestLayoutMode:
    def test_layout_mode_keeps_positional_indentation(self, write_pdf, make_source):
        doc = PdfScraperAdapter().fetch(make_source(write_pdf([{"lines": CODE_LINES}]), SourceType.FILE))
        indents = _indents(doc.content)
        base = indents["def add(a, b):"]
        assert base < indents["if a > b:"] < indents["return a - b"]
        assert indents["return a + b"] == indents["if a > b:"]

    def test_plain_mode_loses_positional_indentation(self, write_pdf, make_source):
        # documents why layout mode is the default
        doc = PdfScraperAdapter(layout=False).fetch(make_source(write_pdf([{"lines": CODE_LINES}]), SourceType.FILE))
        indents = _indents(doc.content)
        assert indents["return a - b"] <= indents["if a > b:"] + 1

    def test_falls_back_to_plain_when_layout_fails(self, write_pdf, make_source, monkeypatch):
        from pypdf import PageObject

        original = PageObject.extract_text

        def flaky(self, *args, extraction_mode="plain", **kwargs):
            if extraction_mode == "layout":
                raise ValueError("layout engine choked")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(PageObject, "extract_text", flaky)
        source = make_source(write_pdf([{"text": LONG_TEXT}]), SourceType.FILE)
        doc = PdfScraperAdapter().fetch(source)
        assert LONG_TEXT in doc.content
        assert source.status == SourceStatus.DONE

    def test_ocr_char_counts_unaffected_by_layout_padding(self, write_pdf, make_source):
        path = write_pdf([{"text": LONG_TEXT}])
        layout = PdfScraperAdapter().fetch(make_source(path, SourceType.FILE))
        plain = PdfScraperAdapter(layout=False).fetch(make_source(path, SourceType.FILE))
        assert layout.metadata["text_layer"]["chars_per_page"] == plain.metadata["text_layer"]["chars_per_page"]
