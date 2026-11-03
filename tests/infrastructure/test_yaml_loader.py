import uuid
from pathlib import Path

import pytest

from src.domain import Schema
from src.domain.value_objects import FieldType
from src.infrastructure.yaml_loader import load_schema_from_yaml


def write_yaml(tmp_path: Path, text: str) -> str:
    path = tmp_path / "schema.yaml"
    path.write_text(text)
    return str(path)


def test_loads_full_schema(tmp_path):
    path = write_yaml(tmp_path, """
name: products
description: Product listings
fields:
  - name: title
    type: str
    vector: true
    description: Product name
  - name: price
    type: float
    required: false
  - name: tags
    type: list
""")
    schema = load_schema_from_yaml(path)

    assert isinstance(schema, Schema)
    uuid.UUID(schema.id)  # valid uuid
    assert schema.name == "products"
    assert schema.description == "Product listings"
    assert [f.name for f in schema.fields] == ["title", "price", "tags"]

    title, price, tags = schema.fields
    assert title.type == FieldType.STR
    assert title.vector is True
    assert title.required is True
    assert title.description == "Product name"
    assert price.type == FieldType.FLOAT
    assert price.required is False
    assert price.vector is False
    assert price.description is None
    assert tags.type == FieldType.LIST


def test_description_optional(tmp_path):
    path = write_yaml(tmp_path, "name: x\nfields: []\n")
    schema = load_schema_from_yaml(path)
    assert schema.description is None
    assert schema.fields == []


def test_each_load_gets_new_id(tmp_path):
    path = write_yaml(tmp_path, "name: x\nfields: []\n")
    assert load_schema_from_yaml(path).id != load_schema_from_yaml(path).id


def test_invalid_field_type_raises(tmp_path):
    path = write_yaml(tmp_path, "name: x\nfields:\n  - name: a\n    type: datetime\n")
    with pytest.raises(ValueError):
        load_schema_from_yaml(path)


def test_duplicate_field_raises(tmp_path):
    path = write_yaml(tmp_path, """
name: x
fields:
  - name: a
    type: str
  - name: a
    type: int
""")
    with pytest.raises(ValueError, match="already exists"):
        load_schema_from_yaml(path)


@pytest.mark.parametrize("text", ["fields: []\n", "name: x\n"])
def test_missing_required_keys_raise(tmp_path, text):
    with pytest.raises(KeyError):
        load_schema_from_yaml(write_yaml(tmp_path, text))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_schema_from_yaml(str(tmp_path / "nope.yaml"))
