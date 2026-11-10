from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    id: str
    source_id: str
    content: str
    mime_type: str = "text/html"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.content.strip()

    def word_count(self) -> int:
        return len(self.content.split())