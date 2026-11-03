import pytest

from src.domain import RawDocument


def make_doc(content: str, **kwargs) -> RawDocument:
    return RawDocument(id="d1", source_id="s1", content=content, **kwargs)


def test_defaults():
    doc = make_doc("hello")
    assert doc.mime_type == "text/html"
    assert doc.metadata == {}


def test_metadata_default_not_shared():
    a, b = make_doc("a"), make_doc("b")
    a.metadata["k"] = "v"
    assert b.metadata == {}


@pytest.mark.parametrize("content", ["", "   ", "\n\t  \n"])
def test_is_empty_true_for_blank(content):
    assert make_doc(content).is_empty()


def test_is_empty_false_for_text():
    assert not make_doc("  hi  ").is_empty()


@pytest.mark.parametrize(
    "content, expected",
    [
        ("", 0),
        ("   ", 0),
        ("one", 1),
        ("one two three", 3),
        ("  spaced\tout\n\nwords  ", 3),
    ],
)
def test_word_count(content, expected):
    assert make_doc(content).word_count() == expected
