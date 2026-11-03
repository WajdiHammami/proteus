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



class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PARTIAL = "partial"  # some sources succeeded, some failed