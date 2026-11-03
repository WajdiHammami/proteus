import pytest

from src.domain import Schema
from src.domain.value_objects import FieldType, SchemaField


@pytest.fixture
def schema() -> Schema:
    return Schema(id="sch1", name="products")


def test_defaults(schema):
    assert schema.description is None
    assert schema.fields == []


def test_fields_default_not_shared_between_instances():
    a = Schema(id="a", name="a")
    b = Schema(id="b", name="b")
    a.add_field(SchemaField(name="x", type=FieldType.STR))
    assert b.fields == []


def test_add_field(schema):
    f = SchemaField(name="title", type=FieldType.STR)
    schema.add_field(f)
    assert schema.fields == [f]


def test_add_duplicate_field_raises(schema):
    schema.add_field(SchemaField(name="title", type=FieldType.STR))
    with pytest.raises(ValueError, match="'title' already exists"):
        schema.add_field(SchemaField(name="title", type=FieldType.INT))
    assert len(schema.fields) == 1


def test_remove_field(schema):
    schema.add_field(SchemaField(name="title", type=FieldType.STR))
    schema.add_field(SchemaField(name="price", type=FieldType.FLOAT))
    schema.remove_field("title")
    assert [f.name for f in schema.fields] == ["price"]


def test_remove_missing_field_is_noop(schema):
    schema.add_field(SchemaField(name="title", type=FieldType.STR))
    schema.remove_field("nope")
    assert [f.name for f in schema.fields] == ["title"]


def test_field_can_be_re_added_after_removal(schema):
    schema.add_field(SchemaField(name="title", type=FieldType.STR))
    schema.remove_field("title")
    schema.add_field(SchemaField(name="title", type=FieldType.INT))
    assert schema.fields[0].type == FieldType.INT


def test_required_and_vector_fields(schema):
    title = SchemaField(name="title", type=FieldType.STR, vector=True)
    price = SchemaField(name="price", type=FieldType.FLOAT, required=False)
    tags = SchemaField(name="tags", type=FieldType.LIST, required=False, vector=True)
    for f in (title, price, tags):
        schema.add_field(f)

    assert schema.required_fields() == [title]
    assert schema.vector_fields() == [title, tags]
    assert schema.sql_fields() == [title, price, tags]


def test_sql_fields_returns_copy(schema):
    schema.add_field(SchemaField(name="title", type=FieldType.STR))
    result = schema.sql_fields()
    result.append(SchemaField(name="injected", type=FieldType.STR))
    assert [f.name for f in schema.fields] == ["title"]
