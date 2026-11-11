from src.adapters.extractors.llm_prompt import (
    SYSTEM_PROMPT, build_user_prompt, expected_output_tokens, response_schema,
)
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.value_objects import FieldType, SchemaField


FIELDS = [
    SchemaField("description", FieldType.STR, description="One sentence on what the snippet does."),
    SchemaField("concepts", FieldType.LIST, description="C++ concepts illustrated."),
    SchemaField("lines", FieldType.INT),
    SchemaField("score", FieldType.FLOAT),
    SchemaField("compiles", FieldType.BOOL),
]


def snippet(**overrides):
    values = dict(code="int main() {\n  return 0;\n}", language="cpp", index=0, heading="Intro", context="An example:")
    values.update(overrides)
    return CodeSnippet(**values)


def document(title="C++ Guide"):
    return NormalizedDocument(id="n", raw_document_id="r", source_id="s", content="", format=TextFormat.MARKDOWN,
                              metadata={"page_metadata": {"title": title} if title else {}})


class TestResponseSchema:
    def test_strict_shape(self):
        schema = response_schema(FIELDS)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["description", "concepts", "lines", "score", "compiles"]

    def test_types_are_nullable(self):
        props = response_schema(FIELDS)["properties"]
        assert props["description"] == {"type": ["string", "null"]}
        assert props["concepts"] == {"type": ["array", "null"], "items": {"type": "string"}}
        assert props["lines"]["type"] == ["integer", "null"]
        assert props["score"]["type"] == ["number", "null"]
        assert props["compiles"]["type"] == ["boolean", "null"]

    def test_only_given_fields(self):
        assert list(response_schema(FIELDS[:1])["properties"]) == ["description"]


class TestUserPrompt:
    def test_contains_fields_context_and_code(self):
        prompt = build_user_prompt(snippet(), FIELDS[:2], document())
        assert "- description (string): One sentence on what the snippet does." in prompt
        assert "- concepts (list of strings): C++ concepts illustrated." in prompt
        assert "Page title: C++ Guide" in prompt
        assert "Heading: Intro" in prompt
        assert "Context: An example:" in prompt
        assert "```cpp\nint main() {\n  return 0;\n}\n```" in prompt

    def test_missing_parts_left_out(self):
        prompt = build_user_prompt(snippet(heading=None, context=""), FIELDS[2:3], document(title=None))
        assert "Heading" not in prompt and "Context" not in prompt and "Page title" not in prompt
        assert "- lines (integer)" in prompt

    def test_long_code_truncated(self):
        prompt = build_user_prompt(snippet(code="x;\n" * 5000), FIELDS[:1], document(), max_code_chars=100)
        assert "// ... (truncated)" in prompt
        assert len(prompt) < 600

    def test_backticks_in_code_get_longer_fence(self):
        prompt = build_user_prompt(snippet(code="// ```\nint x;"), FIELDS[:1], document())
        assert "````cpp" in prompt

    def test_system_prompt_treats_content_as_data(self):
        assert "not instructions" in SYSTEM_PROMPT
        assert "null" in SYSTEM_PROMPT


def test_expected_output_tokens_grow_with_fields():
    assert expected_output_tokens(FIELDS[:1]) < expected_output_tokens(FIELDS)
