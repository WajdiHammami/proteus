from dataclasses import dataclass, field
from enum import Enum


class SourceStatus(Enum):
    PENDING = "pending"
    CRAWLING = "crawling"
    DONE = "done"
    FAILED = "failed"

class SourceType(Enum):
    URL = "url"
    FILE = "file"
    SITEMAP = "sitemap"
    PDF = "pdf"  # remote URL or local path pointing to a PDF



class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PARTIAL = "partial"  # some sources succeeded, some failed



@dataclass(frozen=True)
class StageResult:
    """Tracks the result of one stage in the pipeline."""
    stage: str              # "ingestion", "extraction", "storage"
    status: PipelineStatus
    items_processed: int = 0
    items_failed: int = 0
    error: str | None = None



class FieldType(Enum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"


@dataclass(frozen=True)
class SchemaField:
    name: str
    type: FieldType
    required: bool = True
    vector: bool = False        # should this field be embedded?
    description: str | None = None   # helps LLM understand what to extract
    source: str | None = None        # where the value comes from: "snippet.code", "document.url", "llm", ...