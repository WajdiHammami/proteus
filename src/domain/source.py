from dataclasses import dataclass
from src.domain.value_objects import SourceStatus, SourceType


@dataclass
class Source:
    id: str
    url: str
    type: SourceType = SourceType.URL
    status: SourceStatus = SourceStatus.PENDING

    def mark_crawling(self):
        if self.status != SourceStatus.PENDING:
            raise ValueError(f"Cannot crawl source with status {self.status}")
        self.status = SourceStatus.CRAWLING

    def mark_done(self):
        self._require_crawling("finish")
        self.status = SourceStatus.DONE

    def mark_failed(self):
        self._require_crawling("fail")
        self.status = SourceStatus.FAILED

    def _require_crawling(self, action: str):
        if self.status != SourceStatus.CRAWLING:
            raise ValueError(f"Cannot {action} source with status {self.status}")