import pytest

from src.adapters.models.source import Source, SourceStatus, SourceType
from src.adapters.scrapers import sitemap_adapter
from src.adapters.scrapers.sitemap_adapter import TrafilaturaSitemapAdapter


SITEMAP_URL = "https://example.com/sitemap.xml"


@pytest.fixture
def fake_search(monkeypatch):
    """Replaces trafilatura's sitemap_search; records calls and returns the given URLs."""
    calls = []

    def install(urls=None, error: Exception | None = None):
        def _search(url, **kwargs):
            calls.append({"url": url, **kwargs})
            if error:
                raise error
            return list(urls or [])

        monkeypatch.setattr(sitemap_adapter, "sitemap_search", _search)
        return calls

    return install


@pytest.fixture
def sitemap_source():
    return Source(id="sitemap-1", url=SITEMAP_URL, type=SourceType.SITEMAP)


def urls_of(children):
    return [c.url for c in children]


# --------------------------------------------------------------------------- #
# Main job: expanding a sitemap into child sources
# --------------------------------------------------------------------------- #

class TestExpand:
    def test_returns_one_child_per_url(self, fake_search, sitemap_source):
        found = ["https://example.com/a", "https://example.com/b"]
        fake_search(found)

        children = TrafilaturaSitemapAdapter().expand(sitemap_source)

        assert urls_of(children) == found

    def test_children_are_pending_and_linked_to_parent(self, fake_search, sitemap_source):
        fake_search(["https://example.com/a", "https://example.com/b"])

        children = TrafilaturaSitemapAdapter().expand(sitemap_source)

        for child in children:
            assert child.status == SourceStatus.PENDING
            assert child.parent_id == "sitemap-1"

    def test_children_get_unique_ids(self, fake_search, sitemap_source):
        fake_search([f"https://example.com/{i}" for i in range(50)])
        children = TrafilaturaSitemapAdapter().expand(sitemap_source)
        ids = [c.id for c in children]
        assert len(set(ids)) == len(ids)
        assert "sitemap-1" not in ids

    def test_marks_sitemap_done(self, fake_search, sitemap_source):
        fake_search(["https://example.com/a"])
        TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert sitemap_source.status == SourceStatus.DONE

    def test_passes_options_to_trafilatura(self, fake_search, sitemap_source):
        calls = fake_search(["https://example.com/a"])

        TrafilaturaSitemapAdapter(
            target_lang="fr", include_external=True, max_sitemaps=7, sleep_time=0.5,
        ).expand(sitemap_source)

        assert calls == [{
            "url": SITEMAP_URL,
            "target_lang": "fr",
            "external": True,
            "sleep_time": 0.5,
            "max_sitemaps": 7,
        }]


class TestTypeInference:
    @pytest.mark.parametrize("url", [
        "https://example.com/docs/manual.pdf",
        "https://example.com/docs/MANUAL.PDF",
        "https://example.com/report.pdf?version=2",
        "https://example.com/report.pdf#page=3",
    ])
    def test_pdf_links_become_pdf_sources(self, fake_search, sitemap_source, url):
        fake_search([url])
        [child] = TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert child.type == SourceType.PDF

    @pytest.mark.parametrize("url", [
        "https://example.com/blog/post",
        "https://example.com/pdf/",                  # "pdf" in path, not an extension
        "https://example.com/view?file=report.pdf",  # .pdf only in the query string
        "https://example.com/report.pdf.html",
    ])
    def test_other_links_become_url_sources(self, fake_search, sitemap_source, url):
        fake_search([url])
        [child] = TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert child.type == SourceType.URL

    def test_mixed_sitemap(self, fake_search, sitemap_source):
        fake_search(["https://example.com/a", "https://example.com/b.pdf", "https://example.com/c"])
        children = TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert [c.type for c in children] == [SourceType.URL, SourceType.PDF, SourceType.URL]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

class TestFiltering:
    URLS = [
        "https://example.com/blog/a",
        "https://example.com/blog/b",
        "https://example.com/tag/x",
        "https://example.com/docs/c.pdf",
    ]

    def test_include_pattern(self, fake_search, sitemap_source):
        fake_search(self.URLS)
        children = TrafilaturaSitemapAdapter(include_pattern=r"/blog/").expand(sitemap_source)
        assert urls_of(children) == self.URLS[:2]

    def test_exclude_pattern(self, fake_search, sitemap_source):
        fake_search(self.URLS)
        children = TrafilaturaSitemapAdapter(exclude_pattern=r"/tag/").expand(sitemap_source)
        assert urls_of(children) == [self.URLS[0], self.URLS[1], self.URLS[3]]

    def test_include_and_exclude_combined(self, fake_search, sitemap_source):
        fake_search(self.URLS)
        adapter = TrafilaturaSitemapAdapter(include_pattern=r"/(blog|tag)/", exclude_pattern=r"/b$")
        assert urls_of(adapter.expand(sitemap_source)) == [self.URLS[0], self.URLS[2]]

    def test_max_urls_caps_after_filtering(self, fake_search, sitemap_source):
        fake_search(self.URLS)
        adapter = TrafilaturaSitemapAdapter(exclude_pattern=r"/blog/a", max_urls=2)
        assert urls_of(adapter.expand(sitemap_source)) == [self.URLS[1], self.URLS[2]]

    def test_max_urls_larger_than_result(self, fake_search, sitemap_source):
        fake_search(self.URLS)
        assert len(TrafilaturaSitemapAdapter(max_urls=100).expand(sitemap_source)) == 4

    def test_duplicates_are_removed_preserving_order(self, fake_search, sitemap_source):
        fake_search(["https://example.com/a", "https://example.com/b", "https://example.com/a"])
        children = TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert urls_of(children) == ["https://example.com/a", "https://example.com/b"]

    def test_duplicates_do_not_count_towards_max_urls(self, fake_search, sitemap_source):
        fake_search(["https://example.com/a", "https://example.com/a", "https://example.com/b"])
        children = TrafilaturaSitemapAdapter(max_urls=2).expand(sitemap_source)
        assert urls_of(children) == ["https://example.com/a", "https://example.com/b"]

    def test_filters_removing_everything_is_still_done(self, fake_search, sitemap_source):
        # The sitemap itself was fine; the user's filters just matched nothing.
        fake_search(self.URLS)
        children = TrafilaturaSitemapAdapter(include_pattern=r"/nothing-matches/").expand(sitemap_source)
        assert children == []
        assert sitemap_source.status == SourceStatus.DONE

    def test_invalid_regex_fails_at_construction(self):
        with pytest.raises(Exception):
            TrafilaturaSitemapAdapter(include_pattern="(unclosed")


# --------------------------------------------------------------------------- #
# Failure paths and source state
# --------------------------------------------------------------------------- #

class TestFailures:
    def test_no_urls_found_marks_failed(self, fake_search, sitemap_source):
        fake_search([])
        with pytest.raises(ValueError, match="No URLs found"):
            TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert sitemap_source.status == SourceStatus.FAILED

    def test_trafilatura_exception_marks_failed(self, fake_search, sitemap_source):
        fake_search(error=RuntimeError("parser crashed"))
        with pytest.raises(RuntimeError, match="parser crashed"):
            TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert sitemap_source.status == SourceStatus.FAILED

    @pytest.mark.parametrize("source_type", [SourceType.URL, SourceType.FILE, SourceType.PDF])
    def test_non_sitemap_source_is_rejected_without_state_change(self, fake_search, source_type):
        calls = fake_search(["https://example.com/a"])
        source = Source(id="x", url="https://example.com/a", type=source_type)

        with pytest.raises(NotImplementedError):
            TrafilaturaSitemapAdapter().expand(source)
        assert source.status == SourceStatus.PENDING
        assert calls == []

    @pytest.mark.parametrize("status", [SourceStatus.CRAWLING, SourceStatus.DONE, SourceStatus.FAILED])
    def test_already_processed_sitemap_is_rejected(self, fake_search, sitemap_source, status):
        calls = fake_search(["https://example.com/a"])
        sitemap_source.status = status

        with pytest.raises(ValueError):
            TrafilaturaSitemapAdapter().expand(sitemap_source)
        assert sitemap_source.status == status
        assert calls == []
