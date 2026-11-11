"""
Validation of user-supplied schemas, before anything is built from them.

Why this matters:
  - YAML: only plain data is accepted (safe loader, no tags), and aliases are refused,
    so a small file can't expand into a huge structure ("billion laughs").
  - Code generation: dataclasses.make_dataclass writes the field names into Python
    source that it exec()s. Names are therefore limited to plain snake_case
    identifiers: no dunders/underscore prefix (which could override __init__,
    __class__, __post_init__, ...), no Python keywords, no `self`/`cls`.
  - Storage: field and schema names become SQL column/table names and vector-store
    keys, so the same strict pattern also rules out SQL injection through identifiers
    (63 chars max = PostgreSQL's identifier limit). The storage layer must still quote
    identifiers and use bound parameters for values.
  - Structure: unknown keys, wrong value types and oversized content are rejected
    instead of being silently coerced (e.g. `required: "no"` would be truthy).
  - Descriptions end up in LLM prompts: control characters are removed and length is capped.
"""

import keyword
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


MAX_FILE_BYTES = 256 * 1024
MAX_FIELDS = 200
MAX_DESCRIPTION_CHARS = 1000
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
SOURCE = re.compile(r"^(?:llm|(?:snippet|document)\.[a-z][a-z0-9_]{0,62})$")
_RESERVED_NAMES = frozenset({"self", "cls"})
_FIELD_TYPES = frozenset({"str", "int", "float", "bool", "list"})

_SCHEMA_KEYS = {"name", "description", "fields"}
_FIELD_KEYS = {"name", "type", "required", "vector", "description", "source"}


class SchemaValidationError(ValueError):
    """Raised when a schema is unsafe or malformed. `problems` lists every issue found."""

    def __init__(self, problems: list[str], location: str = "schema"):
        self.problems = problems
        super().__init__(f"Invalid {location}:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# YAML reading
# --------------------------------------------------------------------------- #

class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases (anchors are harmless without them)."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            raise SchemaValidationError([f"YAML aliases are not allowed (line {event.start_mark.line + 1})"])
        return super().compose_node(parent, index)


def read_schema_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a schema file safely and returns the validated, cleaned mapping."""
    path = Path(path)
    if not path.is_file():
        raise SchemaValidationError([f"'{path}' is not a regular file"])
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SchemaValidationError([f"file is {size} bytes, the limit is {MAX_FILE_BYTES}"])
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise SchemaValidationError([f"file is not valid UTF-8 ({error.reason} at byte {error.start})"]) from error
    try:
        data = yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 - strict *safe* loader
    except yaml.YAMLError as error:
        raise SchemaValidationError([f"YAML syntax error: {error}"]) from error
    return sanitize_schema_dict(data, location=str(path))


# --------------------------------------------------------------------------- #
# Structure validation
# --------------------------------------------------------------------------- #

def sanitize_schema_dict(data: Any, location: str = "schema") -> dict[str, Any]:
    """
    Validates the raw schema mapping and returns a cleaned copy:
    {"name": str, "description": str | None, "fields": [ {name, type, required, vector, description, source} ]}
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        raise SchemaValidationError(["the top level must be a mapping with 'name' and 'fields'"], location)

    _unknown_keys(data, _SCHEMA_KEYS, "schema", problems)
    name = data.get("name")
    problems.extend(identifier_problems(name, "schema name"))
    description = _description(data.get("description"), "schema description", problems)

    raw_fields = data.get("fields")
    fields: list[dict[str, Any]] = []
    if not isinstance(raw_fields, list) or not raw_fields:
        problems.append("'fields' must be a non-empty list")
        raw_fields = []
    elif len(raw_fields) > MAX_FIELDS:
        problems.append(f"{len(raw_fields)} fields, the limit is {MAX_FIELDS}")
        raw_fields = []

    seen: set[str] = set()
    for position, raw in enumerate(raw_fields, start=1):
        label = f"field #{position}"
        if not isinstance(raw, dict):
            problems.append(f"{label} must be a mapping")
            continue
        if isinstance(raw.get("name"), str):
            label = f"field '{raw['name'][:40]}'"
        fields.append(_sanitize_field(raw, label, problems))
        field_name = raw.get("name")
        if isinstance(field_name, str):
            if field_name in seen:
                problems.append(f"{label} is defined more than once")
            seen.add(field_name)

    if problems:
        raise SchemaValidationError(problems, location)
    return {"name": name, "description": description, "fields": fields}


def _sanitize_field(raw: dict[str, Any], label: str, problems: list[str]) -> dict[str, Any]:
    _unknown_keys(raw, _FIELD_KEYS, label, problems)
    problems.extend(identifier_problems(raw.get("name"), f"{label} name"))

    field_type = raw.get("type")
    if field_type not in _FIELD_TYPES:
        problems.append(f"{label}: type must be one of {sorted(_FIELD_TYPES)}, got {field_type!r}")

    required = _bool(raw, "required", True, label, problems)
    vector = _bool(raw, "vector", False, label, problems)

    source = raw.get("source")
    if source is not None and (not isinstance(source, str) or not SOURCE.match(source)):
        problems.append(f"{label}: source must be 'llm', 'snippet.<name>' or 'document.<name>', got {source!r}")

    return {
        "name": raw.get("name"),
        "type": field_type,
        "required": required,
        "vector": vector,
        "description": _description(raw.get("description"), f"{label} description", problems),
        "source": source,
    }


def identifier_problems(name: Any, what: str) -> list[str]:
    """Problems with a name that will become Python code and an SQL identifier ([] = safe)."""
    if not isinstance(name, str):
        return [f"{what} must be a string, got {type(name).__name__}"]
    if not IDENTIFIER.match(name):
        return [f"{what} {name[:40]!r} must be snake_case: a lowercase letter, then lowercase letters, "
                f"digits or '_', at most 63 characters"]
    if keyword.iskeyword(name) or keyword.issoftkeyword(name) or name in _RESERVED_NAMES:
        return [f"{what} {name!r} is a reserved word"]
    return []


def _bool(raw: dict[str, Any], key: str, default: bool, label: str, problems: list[str]) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        problems.append(f"{label}: '{key}' must be true or false, got {value!r}")
        return default
    return value


def _description(value: Any, what: str, problems: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        problems.append(f"{what} must be a string")
        return None
    # Drop control/format characters (incl. bidi overrides and zero-width chars) except newlines/tabs.
    cleaned = "".join(
        ch for ch in value if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Co", "Cs"}
    ).strip()
    if len(cleaned) > MAX_DESCRIPTION_CHARS:
        problems.append(f"{what} is {len(cleaned)} characters, the limit is {MAX_DESCRIPTION_CHARS}")
    return cleaned or None


def _unknown_keys(mapping: dict, allowed: set[str], label: str, problems: list[str]) -> None:
    for key in mapping:
        if not isinstance(key, str):
            problems.append(f"{label}: keys must be strings, got {key!r}")
        elif key not in allowed:
            problems.append(f"{label}: unknown key {key!r} (allowed: {', '.join(sorted(allowed))})")
