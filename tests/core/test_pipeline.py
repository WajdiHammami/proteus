import pytest

from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.raw_document import RawDocument
from src.adapters.models.source import Source, SourceStatus, SourceType
from src.adapters.extractors.snippet_extractor import HeuristicSnippetExtractor
from src.core.pipeline import ExtractionPipeline
from src.ports.normalization_port import NormalizationPort
from src.ports.scraper_port import ScraperPort
from src.ports.sitemap_port import SitemapPort


CPP_PAGE = "## Example\n\nA vector:\n\n```cpp\n#include <vector>\nint main() {\n    std::vector<int> v;\n    return 0;\n}\n```"


class FakeScraper(ScraperPort):
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)

    def supports(self, source):
        return True

    def fetch(self, source):
        source.mark_crawling()
        if source.url in self.fail_on:
            source.mark_failed()
            raise ValueError("down")
        source.mark_done()
        return RawDocument(id="r-" + source.id, source_id=source.id, content=CPP_PAGE, metadata={"url": source.url})


class FakeNormalizer(NormalizationPort):
    def supports(self, raw):
        return True

    def normalize(self, raw):
        return NormalizedDocument(id="n-" + raw.id, raw_document_id=raw.id, source_id=raw.source_id,
                                  content=raw.content, format=TextFormat.MARKDOWN, metadata=dict(raw.metadata))


class FakeSitemap(SitemapPort):
    def __init__(self, urls):
        self.urls = urls

    def expand(self, source):
        source.mark_crawling()
        source.mark_done()
        return [Source(id=f"c{i}", url=u, parent_id=source.id) for i, u in enumerate(self.urls)]


def pipeline(**kwargs):
    return ExtractionPipeline(
        scraper=kwargs.pop("scraper", FakeScraper()), normalizer=FakeNormalizer(),
        snippet_extractor=HeuristicSnippetExtractor(), schema_path="schemas/cpp_snippet.yaml", **kwargs,
    )


def test_single_source_gives_records():
    result = pipeline().run(Source(id="s1", url="https://example.com/a"))
    [record] = result.records
    assert record.data["heading"] == "Example"
    assert record.data["url"] == "https://example.com/a"
    assert record.data["includes"] == ["<vector>"]


def test_sitemap_children_processed_and_failures_skipped():
    p = pipeline(scraper=FakeScraper(fail_on={"https://example.com/b"}),
                 sitemap=FakeSitemap(["https://example.com/a", "https://example.com/b", "https://example.com/c"]))
    result = p.run(Source(id="map", url="https://example.com/sitemap.xml", type=SourceType.SITEMAP))
    assert sorted(r.data["url"] for r in result.records) == ["https://example.com/a", "https://example.com/c"]


def test_sitemap_without_port_fails():
    with pytest.raises(ValueError, match="SitemapPort"):
        pipeline().run(Source(id="map", url="https://example.com/sitemap.xml", type=SourceType.SITEMAP))


# --------------------------------------------------------------------------- #
# Filling is the last step, for all sources at once, after confirmation
# --------------------------------------------------------------------------- #

from src.adapters.extractors.llm_filler import LLMFieldFiller
from tests.fakes.fake_llm import FakeLLM


def test_filling_runs_once_after_all_sources_collected():
    log = []

    class LoggingScraper(FakeScraper):
        def fetch(self, source):
            log.append(f"fetch {source.url}")
            return super().fetch(source)

    def confirm(estimates):
        log.append(f"confirm {estimates[0].calls} calls")
        return True

    llm = FakeLLM([{"description": "d1", "concepts": ["x"]}, {"description": "d2", "concepts": ["y"]}])
    p = pipeline(scraper=LoggingScraper(), fillers=[LLMFieldFiller(llm)], confirm=confirm)
    result = p.run([Source(id="a", url="https://example.com/a"), Source(id="b", url="https://example.com/b")])

    assert log == ["fetch https://example.com/a", "fetch https://example.com/b", "confirm 2 calls"]
    assert sorted(r.data["description"] for r in result.records) == ["d1", "d2"]
    assert result.fill.ran and result.fill.spent > 0


def test_declined_keeps_free_fields():
    llm = FakeLLM()
    result = pipeline(fillers=[LLMFieldFiller(llm)], confirm=lambda e: False).run(Source(id="s1", url="https://example.com/a"))
    [record] = result.records
    assert record.data["code"].startswith("#include <vector>")
    assert record.data["description"] is None
    assert llm.requests == [] and result.fill.skipped_reason == "declined"


def test_debug_fills_without_asking():
    llm = FakeLLM([{"description": "d", "concepts": None}])
    def confirm(estimates):
        raise AssertionError("must not ask in debug mode")
    result = pipeline(fillers=[LLMFieldFiller(llm)], confirm=confirm, debug=True).run(Source(id="s1", url="https://example.com/a"))
    assert result.records[0].data["description"] == "d"


def test_without_fillers_nothing_is_asked():
    def confirm(estimates):
        raise AssertionError("nothing to confirm")
    result = pipeline(confirm=confirm).run(Source(id="s1", url="https://example.com/a"))
    assert result.fill.skipped_reason == "no filler configured"
