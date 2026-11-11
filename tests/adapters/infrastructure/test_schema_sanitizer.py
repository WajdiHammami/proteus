import pytest

from src.adapters.infrastructure.schema_sanitizer import (
    MAX_DESCRIPTION_CHARS, MAX_FIELDS, SchemaValidationError, read_schema_yaml, sanitize_schema_dict,
)
from src.adapters.infrastructure.yaml_loader import load_schema_from_yaml


VALID = "name: cpp_snippet\nfields:\n  - {name: code, type: str}\n"


@pytest.fixture
def write(tmp_path):
    def _write(text: str | bytes, name="schema.yaml"):
        path = tmp_path / name
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
        return str(path)
    return _write


def problems(data) -> list[str]:
    with pytest.raises(SchemaValidationError) as info:
        sanitize_schema_dict(data)
    return info.value.problems


def field(**overrides):
    return {"name": "code", "type": "str", **overrides}


class TestYamlSafety:
    def test_valid_file(self, write):
        data = read_schema_yaml(write(VALID))
        assert data["name"] == "cpp_snippet"
        assert data["fields"] == [{"name": "code", "type": "str", "required": True, "vector": False,
                                   "description": None, "source": None}]

    @pytest.mark.parametrize("payload", [
        "!!python/object/apply:os.system ['echo pwned']",
        "!!python/name:os.system",
        "name: !!python/object/new:dict {}\nfields: []",
    ])
    def test_python_tags_never_executed(self, write, payload):
        with pytest.raises(SchemaValidationError, match="YAML"):
            read_schema_yaml(write(payload))

    def test_aliases_refused(self, write):
        bomb = 'a: &a ["x","x","x","x","x","x","x","x","x"]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\nc: [*b,*b,*b,*b,*b,*b,*b,*b,*b]\n'
        with pytest.raises(SchemaValidationError, match="aliases"):
            read_schema_yaml(write(bomb))

    def test_file_size_limit(self, write):
        with pytest.raises(SchemaValidationError, match="limit"):
            read_schema_yaml(write(VALID + "#" * (300 * 1024)))

    def test_not_utf8(self, write):
        with pytest.raises(SchemaValidationError, match="UTF-8"):
            read_schema_yaml(write(b"name: caf\xe9\nfields: []\n"))

    def test_directory_or_missing_file(self, tmp_path):
        with pytest.raises(SchemaValidationError, match="regular file"):
            read_schema_yaml(str(tmp_path))
        with pytest.raises(SchemaValidationError, match="regular file"):
            read_schema_yaml(str(tmp_path / "missing.yaml"))

    def test_syntax_error(self, write):
        with pytest.raises(SchemaValidationError, match="syntax"):
            read_schema_yaml(write("name: [unclosed\n"))


class TestNames:
    @pytest.mark.parametrize("name", [
        "__init__", "__class__", "__post_init__", "_hidden",           # would override dataclass machinery
        "class", "return", "match", "self", "cls",                     # Python keywords / reserved
        "code; DROP TABLE users", "code\"--", "a b", "my-field",        # SQL / code injection attempts
        "Code", "1code", "x" * 64, "", "naïve",                         # not plain snake_case
    ])
    def test_unsafe_field_names(self, name):
        assert any("field" in p for p in problems({"name": "s", "fields": [field(name=name)]}))

    @pytest.mark.parametrize("name", [None, 3, ["code"]])
    def test_non_string_names(self, name):
        assert problems({"name": "s", "fields": [field(name=name)]})

    @pytest.mark.parametrize("name", ["users; DROP TABLE x", "My Schema", "__main__", None])
    def test_unsafe_schema_names(self, name):
        assert any("schema name" in p for p in problems({"name": name, "fields": [field()]}))

    def test_max_length_name_ok(self):
        assert sanitize_schema_dict({"name": "s", "fields": [field(name="a" * 63)]})

    def test_duplicates(self):
        assert any("more than once" in p for p in problems({"name": "s", "fields": [field(), field()]}))


class TestStructure:
    @pytest.mark.parametrize("data", [None, [], "text", 3])
    def test_top_level_must_be_mapping(self, data):
        assert problems(data)

    @pytest.mark.parametrize("fields", [None, [], "code", {"name": "code"}])
    def test_fields_must_be_non_empty_list(self, fields):
        assert any("fields" in p for p in problems({"name": "s", "fields": fields}))

    def test_too_many_fields(self):
        fields = [field(name=f"f{i}") for i in range(MAX_FIELDS + 1)]
        assert any("limit" in p for p in problems({"name": "s", "fields": fields}))

    def test_unknown_keys_rejected(self):
        found = problems({"name": "s", "fields": [field(requird=False)], "extra": 1})
        assert any("'requird'" in p for p in found)
        assert any("'extra'" in p for p in found)

    @pytest.mark.parametrize("value", ["no", "false", 0, 1, None])
    def test_booleans_must_be_real_booleans(self, value):
        assert any("required" in p for p in problems({"name": "s", "fields": [field(required=value)]}))

    @pytest.mark.parametrize("field_type", ["dict", "object", "eval", None, "STR"])
    def test_types_whitelisted(self, field_type):
        assert any("type" in p for p in problems({"name": "s", "fields": [field(type=field_type)]}))

    @pytest.mark.parametrize("source", ["llm", "snippet.code", "document.url"])
    def test_valid_sources(self, source):
        assert sanitize_schema_dict({"name": "s", "fields": [field(source=source)]})["fields"][0]["source"] == source

    @pytest.mark.parametrize("source", ["os.system", "snippet.__class__", "snippet", "llm; rm -rf", 3])
    def test_invalid_sources(self, source):
        assert any("source" in p for p in problems({"name": "s", "fields": [field(source=source)]}))

    def test_all_problems_reported_at_once(self):
        found = problems({"name": "Bad Name", "fields": [field(name="__init__", type="dict", vector="yes")]})
        assert len(found) == 4


class TestDescriptions:
    def test_control_and_invisible_characters_removed(self):
        text = "Explain‮ the​ code\x00\x1b[31m.\nSecond line\tok"
        cleaned = sanitize_schema_dict({"name": "s", "fields": [field(description=text)]})["fields"][0]["description"]
        assert cleaned == "Explain the code[31m.\nSecond line\tok"

    def test_length_capped(self):
        assert any("limit" in p for p in problems({"name": "s", "fields": [field(description="x" * (MAX_DESCRIPTION_CHARS + 1))]}))

    def test_must_be_string(self):
        assert any("description" in p for p in problems({"name": "s", "fields": [field(description=["a"])]}))


class TestLoader:
    def test_loader_uses_sanitizer(self, write):
        with pytest.raises(SchemaValidationError, match="__init__"):
            load_schema_from_yaml(write("name: s\nfields:\n  - {name: __init__, type: str}\n"))

    def test_example_schema_is_valid(self):
        schema = load_schema_from_yaml("schemas/cpp_snippet.yaml")
        assert schema.name == "cpp_snippet"
        assert {f.source for f in schema.fields} >= {"llm", "document.title"}
