import pytest

from src.adapters.models.source import Source, SourceStatus, SourceType


def test_defaults():
    source = Source(id="1", url="https://example.com")
    assert source.type == SourceType.URL
    assert source.status == SourceStatus.PENDING
    assert source.parent_id is None


def test_parent_id_can_be_set():
    child = Source(id="2", url="https://example.com/a.pdf", type=SourceType.PDF, parent_id="1")
    assert child.parent_id == "1"
    assert child.type == SourceType.PDF


def test_pdf_type_value():
    assert SourceType("pdf") is SourceType.PDF


def test_happy_path_transitions():
    source = Source(id="1", url="u")
    source.mark_crawling()
    assert source.status == SourceStatus.CRAWLING
    source.mark_done()
    assert source.status == SourceStatus.DONE


def test_failure_transition():
    source = Source(id="1", url="u")
    source.mark_crawling()
    source.mark_failed()
    assert source.status == SourceStatus.FAILED


@pytest.mark.parametrize("action", ["mark_done", "mark_failed"])
def test_cannot_finish_without_crawling(action):
    source = Source(id="1", url="u")
    with pytest.raises(ValueError):
        getattr(source, action)()
    assert source.status == SourceStatus.PENDING


@pytest.mark.parametrize("status", [SourceStatus.CRAWLING, SourceStatus.DONE, SourceStatus.FAILED])
def test_cannot_crawl_twice(status):
    source = Source(id="1", url="u", status=status)
    with pytest.raises(ValueError):
        source.mark_crawling()


def test_release_returns_crawling_source_to_pending():
    source = Source(id="1", url="u")
    source.mark_crawling()
    source.release()
    assert source.status == SourceStatus.PENDING
    source.mark_crawling()  # can be picked up again
    assert source.status == SourceStatus.CRAWLING


@pytest.mark.parametrize("status", [SourceStatus.PENDING, SourceStatus.DONE, SourceStatus.FAILED])
def test_release_requires_crawling(status):
    source = Source(id="1", url="u", status=status)
    with pytest.raises(ValueError):
        source.release()
    assert source.status == status
