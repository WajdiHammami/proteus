from src.adapters.models.normalized_document import NormalizedDocument, TextFormat


def make(content: str) -> NormalizedDocument:
    return NormalizedDocument(id="n", raw_document_id="r", source_id="s", content=content)


def test_defaults():
    doc = make("hello")
    assert doc.format == TextFormat.PLAIN
    assert doc.metadata == {}


def test_metadata_not_shared_between_instances():
    a, b = make("a"), make("b")
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_is_empty():
    assert make("").is_empty()
    assert make(" \n\t ").is_empty()
    assert not make("x").is_empty()


def test_word_count():
    assert make("").word_count() == 0
    assert make("one two\nthree").word_count() == 3
