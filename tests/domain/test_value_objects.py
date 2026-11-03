import dataclasses

import pytest

from src.domain.value_objects import (
    FieldType,
    PipelineStatus,
    SchemaField,
    SourceStatus,
    SourceType,
    StageResult,
)


@pytest.mark.parametrize(
    "enum_cls, value",
    [
        (SourceStatus, "pending"),
        (SourceType, "sitemap"),
        (PipelineStatus, "partial"),
        (FieldType, "list"),
    ],
)
def test_enums_construct_from_value(enum_cls, value):
    assert enum_cls(value).value == value


def test_field_type_invalid_value():
    with pytest.raises(ValueError):
        FieldType("datetime")


def test_stage_result_defaults():
    r = StageResult(stage="ingestion", status=PipelineStatus.DONE)
    assert r.items_processed == 0
    assert r.items_failed == 0
    assert r.error is None


def test_stage_result_is_frozen():
    r = StageResult(stage="ingestion", status=PipelineStatus.DONE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.items_processed = 5


def test_schema_field_defaults():
    f = SchemaField(name="title", type=FieldType.STR)
    assert f.required is True
    assert f.vector is False
    assert f.description is None


def test_schema_field_is_frozen_and_hashable():
    f = SchemaField(name="title", type=FieldType.STR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.name = "other"
    assert f == SchemaField(name="title", type=FieldType.STR)
    assert len({f, SchemaField(name="title", type=FieldType.STR)}) == 1
