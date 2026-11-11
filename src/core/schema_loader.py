import dataclasses
from typing import Any

from pydantic import TypeAdapter

from src.adapters.infrastructure.schema_sanitizer import SchemaValidationError, identifier_problems
from src.adapters.infrastructure.yaml_loader import load_schema_from_yaml
from src.adapters.models.schema import Schema
from src.adapters.models.value_objects import FieldType


_PYTHON_TYPES: dict[FieldType, Any] = {
    FieldType.STR: str,
    FieldType.INT: int,
    FieldType.FLOAT: float,
    FieldType.BOOL: bool,
    FieldType.LIST: list[str],
}


def build_record_class(schema: Schema) -> type:
    """
    Builds a dataclass that mirrors the schema: same field names, same order, same
    types. Required fields have no default (they must be given); optional ones are
    `T | None = None`. Fields are keyword-only, so the YAML order is kept as is.

        name: cpp_snippet          ->   @dataclass(kw_only=True)
        fields:                         class CppSnippet:
          - {name: code, type: str}         code: str
          - {name: tags, type: list,        tags: list[str] | None = None
             required: false}
    """
    # Checked again here (not only in the YAML loader): field names end up in source code
    # that make_dataclass exec()s, and a Schema can also be built in code.
    problems = [p for f in schema.fields for p in identifier_problems(f.name, "field name")]
    problems += [f"field {f.name!r}: unsupported type {f.type!r}" for f in schema.fields if f.type not in _PYTHON_TYPES]
    if not schema.fields:
        problems.append(f"schema {schema.name!r} has no fields")
    if problems:
        raise SchemaValidationError(problems)

    definitions = []
    for schema_field in schema.fields:
        name = schema_field.name
        python_type = _PYTHON_TYPES[schema_field.type]
        if schema_field.required:
            definitions.append((name, python_type))
        else:
            definitions.append((name, python_type | None, dataclasses.field(default=None)))

    return dataclasses.make_dataclass(_class_name(schema.name), definitions, kw_only=True)


def load_record_class(path: str) -> tuple[Schema, type]:
    """Reads the YAML schema and returns it with its dataclass."""
    schema = load_schema_from_yaml(path)
    return schema, build_record_class(schema)


def validator_for(record_class: type) -> TypeAdapter:
    """
    Validates/coerces a dict into an instance of the record dataclass ("3" -> 3, missing
    required fields and wrong types raise pydantic.ValidationError). `.json_schema()`
    on it gives the JSON schema for LLM structured outputs.
    """
    return TypeAdapter(record_class)


def _class_name(schema_name: str) -> str:
    # Only letters and digits survive, so the class name is always a safe identifier.
    parts = [part for part in "".join(c if c.isascii() and c.isalnum() else " " for c in schema_name).split() if part]
    name = "".join(part[:1].upper() + part[1:] for part in parts) or "Record"
    return name if name[0].isalpha() else f"Record{name}"
