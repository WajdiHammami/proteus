# infrastructure/yaml_loader.py
import uuid

from src.adapters.infrastructure.schema_sanitizer import read_schema_yaml
from src.adapters.models import Schema
from src.adapters.models.value_objects import FieldType, SchemaField


def load_schema_from_yaml(path: str) -> Schema:
    """
    Loads a schema file. Everything is validated first (see schema_sanitizer):
    raises SchemaValidationError listing every problem if the file is unsafe or malformed.
    """
    data = read_schema_yaml(path)

    schema = Schema(
        id=str(uuid.uuid4()),
        name=data["name"],
        description=data["description"],
    )
    for field_data in data["fields"]:
        schema.add_field(SchemaField(
            name=field_data["name"],
            type=FieldType(field_data["type"]),
            required=field_data["required"],
            vector=field_data["vector"],
            description=field_data["description"],
            source=field_data["source"],
        ))
    return schema
