import pytest

from src.domain import Source, SourceStatus, SourceType


def make_source(**kwargs) -> Source:
    return Source(id="s1", url="https://example.com", **kwargs)


def test_defaults():
    source = make_source()
    assert source.type == SourceType.URL
    assert source.status == SourceStatus.PENDING


def test_mark_crawling_from_pending():
    source = make_source()
    source.mark_crawling()
    assert source.status == SourceStatus.CRAWLING


@pytest.mark.parametrize(
    "status", [SourceStatus.CRAWLING, SourceStatus.DONE, SourceStatus.FAILED]
)
def test_mark_crawling_rejects_non_pending(status):
    source = make_source(status=status)
    with pytest.raises(ValueError, match="Cannot crawl source"):
        source.mark_crawling()
    assert source.status == status


def test_mark_done():
    source = make_source()
    source.mark_crawling()
    source.mark_done()
    assert source.status == SourceStatus.DONE


def test_mark_failed():
    source = make_source()
    source.mark_crawling()
    source.mark_failed()
    assert source.status == SourceStatus.FAILED


@pytest.mark.parametrize("method", ["mark_done", "mark_failed"])
@pytest.mark.parametrize(
    "status", [SourceStatus.PENDING, SourceStatus.DONE, SourceStatus.FAILED]
)
def test_finishing_requires_crawling(method, status):
    source = make_source(status=status)
    with pytest.raises(ValueError, match="Cannot"):
        getattr(source, method)()
    assert source.status == status
