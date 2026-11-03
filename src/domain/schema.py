from dataclasses import dataclass, field
from typing import Any


@dataclass
class Schema:
    id: str
    name: str
    columns: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_column(self, name: str, type: str):
        if name in self.columns:
            raise ValueError(f"Column '{name}' already exists")
        self.columns[name] = type

    def remove_column(self, name: str):
        if name not in self.columns:
            raise ValueError(f"Column '{name}' not found")
        del self.columns[name]