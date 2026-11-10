# domain/schema.py
from dataclasses import dataclass, field
from .value_objects import SchemaField


@dataclass
class Schema:
    id: str
    name: str
    description: str | None = None
    fields: list[SchemaField] = field(default_factory=list)

    def add_field(self, f: SchemaField):
        if any(existing.name == f.name for existing in self.fields):
            raise ValueError(f"Field '{f.name}' already exists")
        self.fields.append(f)

    def remove_field(self, name: str):
        self.fields = [f for f in self.fields if f.name != name]

    def required_fields(self) -> list[SchemaField]:
        return [f for f in self.fields if f.required]

    def vector_fields(self) -> list[SchemaField]:
        return [f for f in self.fields if f.vector]

    def sql_fields(self) -> list[SchemaField]:
        return list(self.fields)