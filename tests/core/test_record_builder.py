import pytest

from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.schema import Schema
from src.adapters.models.value_objects import FieldType, SchemaField
from src.core.record_builder import RecordBuilder, fill_pending
from src.ports.extraction_port import FieldFillerPort


def schema(*fields: SchemaField) -> Schema:
    s = Schema(id="schema-1", name="cpp_snippet")
    for f in fields:
        s.add_field(f)
    return s


def document(**metadata) -> NormalizedDocument:
    return NormalizedDocument(
        id="n1", raw_document_id="r1", source_id="s1", content="", format=TextFormat.MARKDOWN,
        metadata={"url": "https://example.com/p", "page_metadata": {"title": "Guide", "author": "Ann"}, **metadata},
    )


def snippet(**overrides) -> CodeSnippet:
    values = dict(code="int main() {\n  return 0;\n}", language="cpp", index=0, heading="Intro",
                  context="An example:", includes=["<vector>"], functions=["main"], classes=[])
    values.update(overrides)
    return CodeSnippet(**values)


class FakeFiller(FieldFillerPort):
    def __init__(self, values):
        self.values = values
        self.calls = []

    def fill(self, snippet, fields, known, document):
        self.calls.append({"fields": [f.name for f in fields], "known": known})
        return {name: value for name, value in self.values.items()}


def build_filled(schema, fillers, doc, snippets):
    builder = RecordBuilder(schema)
    pending = builder.prepare(doc, snippets)
    fill_pending(pending, fillers)
    return builder.finalize(pending)


class TestFreeFields:
    def test_names_map_automatically(self):
        builder = RecordBuilder(schema(
            SchemaField("code", FieldType.STR),
            SchemaField("language", FieldType.STR),
            SchemaField("url", FieldType.STR),
            SchemaField("title", FieldType.STR),
            SchemaField("heading", FieldType.STR),
            SchemaField("line_count", FieldType.INT),
            SchemaField("includes", FieldType.LIST),
        ))
        [record] = builder.build(document(), [snippet()]).records
        assert record.data == {
            "code": "int main() {\n  return 0;\n}", "language": "cpp", "url": "https://example.com/p",
            "title": "Guide", "heading": "Intro", "line_count": 3, "includes": ["<vector>"],
        }
        assert record.document_id == "n1"
        assert record.schema_id == "schema-1"

    def test_explicit_source(self):
        builder = RecordBuilder(schema(
            SchemaField("snippet_text", FieldType.STR, source="snippet.code"),
            SchemaField("page", FieldType.STR, source="document.url"),
            SchemaField("written_by", FieldType.STR, source="document.author"),
        ))
        [record] = builder.build(document(), [snippet()]).records
        assert record.data == {"snippet_text": "int main() {\n  return 0;\n}", "page": "https://example.com/p", "written_by": "Ann"}

    def test_url_falls_back_to_path(self):
        builder = RecordBuilder(schema(SchemaField("url", FieldType.STR)))
        doc = document()
        doc.metadata = {"path": "/src/main.cpp"}
        assert builder.build(doc, [snippet()]).records[0].data["url"] == "/src/main.cpp"

    def test_list_into_str_field_is_joined(self):
        builder = RecordBuilder(schema(SchemaField("includes", FieldType.STR)))
        record = builder.build(document(), [snippet(includes=["<vector>", "<map>"])]).records[0]
        assert record.data["includes"] == "<vector>, <map>"

    def test_unknown_source_fails_at_construction(self):
        with pytest.raises(ValueError, match="unknown source 'snippet.nope'"):
            RecordBuilder(schema(SchemaField("x", FieldType.STR, source="snippet.nope")))

    def test_one_record_per_snippet(self):
        builder = RecordBuilder(schema(SchemaField("code", FieldType.STR), SchemaField("index", FieldType.INT)))
        result = builder.build(document(), [snippet(index=0), snippet(code="void f() {\n}", index=1)])
        assert [r.data["index"] for r in result.records] == [0, 1]
        assert len({r.id for r in result.records}) == 2


class TestLlmFields:
    def fields(self):
        return schema(
            SchemaField("code", FieldType.STR),
            SchemaField("description", FieldType.STR, required=False),           # unknown name -> llm
            SchemaField("concepts", FieldType.LIST, required=False, source="llm"),
        )

    def test_without_fillers_optional_llm_fields_are_none(self):
        [record] = RecordBuilder(self.fields()).build(document(), [snippet()]).records
        assert record.data["description"] is None and record.data["concepts"] is None

    def test_filler_gets_only_missing_fields_and_known_values(self):
        filler = FakeFiller({"description": "Returns 0.", "concepts": ["main"]})
        [record] = build_filled(self.fields(), [filler], document(), [snippet()]).records
        assert record.data["description"] == "Returns 0."
        assert record.data["concepts"] == ["main"]
        assert filler.calls[0]["fields"] == ["description", "concepts"]
        assert filler.calls[0]["known"]["code"].startswith("int main")

    def test_cascade_cheapest_first(self):
        cheap = FakeFiller({"description": "cheap"})
        expensive = FakeFiller({"description": "expensive", "concepts": ["x"]})
        [record] = build_filled(self.fields(), [cheap, expensive], document(), [snippet()]).records
        assert record.data["description"] == "cheap"
        assert expensive.calls[0]["fields"] == ["concepts"]

    def test_later_fillers_skipped_when_done(self):
        cheap = FakeFiller({"description": "d", "concepts": ["x"]})
        expensive = FakeFiller({})
        build_filled(self.fields(), [cheap, expensive], document(), [snippet()])
        assert expensive.calls == []

    def test_filler_values_are_coerced(self):
        filler = FakeFiller({"description": ["a", "b"], "concepts": "RAII"})
        [record] = build_filled(self.fields(), [filler], document(), [snippet()]).records
        assert record.data["description"] == "a, b"
        assert record.data["concepts"] == ["RAII"]

    def test_fields_the_filler_did_not_ask_for_are_ignored(self):
        filler = FakeFiller({"description": "d", "code": "overwritten?"})
        [record] = build_filled(self.fields(), [filler], document(), [snippet()]).records
        assert record.data["code"].startswith("int main")


class TestRejected:
    def test_missing_required_llm_field_rejects_snippet(self):
        builder = RecordBuilder(schema(SchemaField("code", FieldType.STR), SchemaField("summary", FieldType.STR)))
        result = builder.build(document(), [snippet()])
        assert result.records == []
        [rejected] = result.rejected
        assert "summary" in rejected.error
        assert rejected.snippet.index == 0

    def test_missing_required_free_field(self):
        builder = RecordBuilder(schema(SchemaField("heading", FieldType.STR)))
        result = builder.build(document(), [snippet(heading=None), snippet(index=1)])
        assert len(result.records) == 1 and len(result.rejected) == 1

    def test_wrong_type_from_filler_rejects(self):
        result = build_filled(schema(SchemaField("code", FieldType.STR), SchemaField("rating", FieldType.INT)),
                              [FakeFiller({"rating": "very good"})], document(), [snippet()])
        assert result.records == [] and "rating" in result.rejected[0].error


def test_example_schema_end_to_end():
    from src.adapters.infrastructure.yaml_loader import load_schema_from_yaml
    builder = RecordBuilder(load_schema_from_yaml("schemas/cpp_snippet.yaml"))
    [record] = builder.build(document(), [snippet()]).records
    assert record.data["code"].startswith("int main")
    assert record.data["url"] == "https://example.com/p"
    assert record.data["description"] is None


class TestPhases:
    def test_prepare_lists_missing_llm_fields(self):
        builder = RecordBuilder(schema(SchemaField("code", FieldType.STR), SchemaField("summary", FieldType.STR, source="llm")))
        [item] = builder.prepare(document(), [snippet()])
        assert item.values == {"code": "int main() {\n  return 0;\n}"}
        assert [f.name for f in item.missing] == ["summary"]
        job = item.job()
        assert job.snippet is item.snippet and [f.name for f in job.fields] == ["summary"]

    def test_prepare_has_no_missing_without_llm_fields(self):
        builder = RecordBuilder(schema(SchemaField("code", FieldType.STR)))
        assert builder.prepare(document(), [snippet()])[0].missing == []
