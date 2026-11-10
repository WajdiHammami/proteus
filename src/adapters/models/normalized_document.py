from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TextFormat(Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"


@dataclass
class NormalizedDocument:
    """
    Clean, extraction-ready text produced from a RawDocument.

    `metadata` carries forward whatever downstream stages need from the raw
    document (e.g. title, language, page boundaries, PDF `text_layer` OCR flags).
    """
    id: str
    raw_document_id: str
    source_id: str
    content: str
    format: TextFormat = TextFormat.PLAIN
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.content.strip()

    def word_count(self) -> int:
        return len(self.content.split())
