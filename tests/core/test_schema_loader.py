import dataclasses

import pytest
from pydantic import ValidationError

from src.adapters.infrastructure.schema_sanitizer import SchemaValidationError
from src.adapters.models.schema import Schema
from src.adapters.models.value_objects import FieldType, SchemaField
from src.core.schema_loader import build_record_class, load_record_class, validator_for


def schema(*fields: SchemaField, name="cpp_snippet") -> Schema:
    s = Schema(id="schema-1", name=name)
    for f in fields:
        s.add_field(f)
    return s


class TestBuildRecordClass:
    def test_is_a_dataclass_mirroring_the_schema(self):
        cls = build_record_class(schema(
            SchemaField("code", FieldType.STR),
            SchemaField("line_count", FieldType.INT),
            SchemaField("score", FieldType.FLOAT, required=False),
            SchemaField("valid", FieldType.BOOL, required=False),
            SchemaField("tags", FieldType.LIST, required=False),
        ))
        assert dataclasses.is_dataclass(cls)
        assert cls.__name__ == "CppSnippet"
        assert [f.name for f in dataclasses.fields(cls)] == ["code", "line_count", "score", "valid", "tags"]
        types = {f.name: f.type for f in dataclasses.fields(cls)}
        assert types["code"] is str
        assert types["tags"] == list[str] | None

    def test_required_fields_have_no_default(self):
        cls = build_record_class(schema(SchemaField("code", FieldType.STR), SchemaField("tags", FieldType.LIST, required=False)))
        with pytest.raises(TypeError):
            cls()
        record = cls(code="int x;")
        assert record.tags is None

    def test_optional_before_required_keeps_yaml_order(self):
        cls = build_record_class(schema(SchemaField("note", FieldType.STR, required=False), SchemaField("code", FieldType.STR)))
        assert [f.name for f in dataclasses.fields(cls)] == ["note", "code"]
        assert cls(code="x").note is None

    @pytest.mark.parametrize("name", [
        "class", "1code", "my-field", "has space", "__init__", "__class__", "_private", "self",
        "Code", "x" * 64, "a=1):\n    import os  #", "café",
    ])
    def test_unsafe_field_names_rejected(self, name):
        # Field names go into source code that make_dataclass exec()s: checked even without the YAML loader.
        with pytest.raises(SchemaValidationError):
            build_record_class(schema(SchemaField(name, FieldType.STR)))

    def test_empty_schema_rejected(self):
        with pytest.raises(SchemaValidationError, match="no fields"):
            build_record_class(schema())

    @pytest.mark.parametrize("name, expected", [("cpp_snippet", "CppSnippet"), ("my schema v2", "MySchemaV2"), ("2024 data", "Record2024Data"), ("", "Record"), ("café menu", "CafMenu")])
    def test_class_names(self, name, expected):
        assert build_record_class(schema(SchemaField("a", FieldType.STR), name=name)).__name__ == expected


class TestValidator:
    @pytest.fixture
    def validator(self):
        return validator_for(build_record_class(schema(
            SchemaField("code", FieldType.STR),
            SchemaField("line_count", FieldType.INT),
            SchemaField("tags", FieldType.LIST, required=False),
        )))

    def test_valid_dict(self, validator):
        record = validator.validate_python({"code": "x", "line_count": 3, "tags": ["a"]})
        assert record.line_count == 3 and record.tags == ["a"]

    def test_coerces_lax_values(self, validator):
        assert validator.validate_python({"code": "x", "line_count": "3"}).line_count == 3

    def test_missing_required(self, validator):
        with pytest.raises(ValidationError, match="line_count"):
            validator.validate_python({"code": "x"})

    def test_wrong_type(self, validator):
        with pytest.raises(ValidationError):
            validator.validate_python({"code": "x", "line_count": "many"})

    def test_json_schema_for_llm(self, validator):
        js = validator.json_schema()
        assert set(js["properties"]) == {"code", "line_count", "tags"}
        assert set(js["required"]) == {"code", "line_count"}


def test_load_record_class_from_yaml(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: cpp_snippet\nfields:\n"
        "  - {name: code, type: str, vector: true}\n"
        "  - {name: summary, type: str, required: false, source: llm, description: What it does}\n"
    )
    loaded, cls = load_record_class(str(path))
    assert [f.name for f in dataclasses.fields(cls)] == ["code", "summary"]
    assert loaded.fields[0].vector is True
    assert loaded.fields[1].source == "llm"
    assert loaded.fields[1].description == "What it does"


def test_example_schema_loads():
    loaded, cls = load_record_class("schemas/cpp_snippet.yaml")
    assert "code" in {f.name for f in dataclasses.fields(cls)}
