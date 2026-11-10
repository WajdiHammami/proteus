from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedRecord:
    id: str
    document_id: str
    schema_id: str
    data: dict[str, Any] = field(default_factory=dict)

    def is_complete(self, required_columns: list[str]) -> bool:
        return all(self.data.get(col) is not None for col in required_columns)

    def get_field(self, name: str) -> Any:
        return self.data.get(name)