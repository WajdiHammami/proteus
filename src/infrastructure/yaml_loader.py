# infrastructure/yaml_loader.py
import yaml
import uuid
from src.domain import Schema
from src.domain.value_objects import SchemaField, FieldType


def load_schema_from_yaml(path: str) -> Schema:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    schema = Schema(
        id=str(uuid.uuid4()),
        name=data["name"],
        description=data.get("description"),
    )

    for field_data in data["fields"]:
        schema.add_field(SchemaField(
            name=field_data["name"],
            type=FieldType(field_data["type"]),
            required=field_data.get("required", True),
            vector=field_data.get("vector", False),
            description=field_data.get("description"),
        ))

    return schema