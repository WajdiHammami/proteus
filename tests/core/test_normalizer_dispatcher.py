import pytest

from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.normalizers.html_normalizer import TrafilaturaHtmlNormalizer
from src.adapters.normalizers.pdf_normalizer import PdfNormalizer
from src.adapters.normalizers.text_normalizer import TextNormalizer
from src.core.normalizer_dispatcher import NormalizerDispatcher
from src.ports.normalization_port import NormalizationPort, UnsupportedDocumentError


class FakeNormalizer(NormalizationPort):
    def __init__(self, mimes, name="fake"):
        self.mimes = set(mimes)
        self.name = name
        self.seen = []

    def supports(self, raw):
        return raw.mime_type in self.mimes

    def normalize(self, raw):
        self.seen.append(raw.id)
        return NormalizedDocument(
            id="n", raw_document_id=raw.id, source_id=raw.source_id,
            content=self.name, format=TextFormat.PLAIN, metadata=dict(raw.metadata),
        )


def raw(mime, content="x", id="r1"):
    return RawDocument(id=id, source_id="s1", content=content, mime_type=mime)


class TestRouting:
    def test_first_supporting_normalizer_wins(self):
        first = FakeNormalizer({"text/plain"}, "first")
        second = FakeNormalizer({"text/plain"}, "second")

        doc = NormalizerDispatcher([first, second]).normalize(raw("text/plain"))

        assert doc.content == "first"
        assert second.seen == []
        assert doc.metadata["normalizer"] == "FakeNormalizer"

    def test_routes_by_mime(self):
        html = FakeNormalizer({"text/html"}, "html")
        pdf = FakeNormalizer({"application/pdf"}, "pdf")
        dispatcher = NormalizerDispatcher([html, pdf])
        assert dispatcher.normalize(raw("application/pdf")).content == "pdf"
        assert dispatcher.normalize(raw("text/html")).content == "html"

    def test_unknown_mime_fails(self):
        dispatcher = NormalizerDispatcher([FakeNormalizer({"text/html"})])
        assert not dispatcher.supports(raw("text/csv"))
        with pytest.raises(UnsupportedDocumentError, match="text/csv"):
            dispatcher.normalize(raw("text/csv"))

    def test_raw_metadata_is_not_tagged(self):
        document = raw("text/plain")
        NormalizerDispatcher([FakeNormalizer({"text/plain"})]).normalize(document)
        assert "normalizer" not in document.metadata

    def test_needs_at_least_one_normalizer(self):
        with pytest.raises(ValueError):
            NormalizerDispatcher([])


class TestRealNormalizers:
    @pytest.fixture
    def dispatcher(self):
        return NormalizerDispatcher([TrafilaturaHtmlNormalizer(), PdfNormalizer(), TextNormalizer()])

    @pytest.mark.parametrize("mime, expected", [
        ("text/html", "TrafilaturaHtmlNormalizer"),
        ("text/html; charset=utf-8", "TrafilaturaHtmlNormalizer"),
        ("application/pdf", "PdfNormalizer"),
        ("text/plain", "TextNormalizer"),
        ("text/markdown", "TextNormalizer"),
        ("text/x-c++src", "TextNormalizer"),
        ("text/x-c++hdr", "TextNormalizer"),
    ])
    def test_each_mime_reaches_its_normalizer(self, dispatcher, mime, expected):
        content = "<html><body><p>hi</p></body></html>" if "html" in mime else "int main() {}"
        document = RawDocument(
            id="r1", source_id="s1", content=content, mime_type=mime,
            metadata={"page_count": 1, "page_separator": "\f"} if mime == "application/pdf" else {},
        )
        assert dispatcher.normalize(document).metadata["normalizer"] == expected

    @pytest.mark.parametrize("mime", ["text/csv", "application/json", "image/png", ""])
    def test_unsupported_mimes_fail(self, dispatcher, mime):
        with pytest.raises(UnsupportedDocumentError):
            dispatcher.normalize(raw(mime))
