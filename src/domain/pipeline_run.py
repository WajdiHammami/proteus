from src.domain.value_objects import PipelineStatus, StageResult
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PipelineRun:
    id: str
    schema_id: str
    source_ids: list[str] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    stages: list[StageResult] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def start(self):
        if self.status != PipelineStatus.PENDING:
            raise ValueError("Pipeline already started")
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.now()

    def add_stage_result(self, result: StageResult):
        self.stages.append(result)

    def complete(self):
        self._require_running("complete")
        self.status = PipelineStatus.DONE
        self.finished_at = datetime.now()

    def partial(self):
        self._require_running("mark partial")
        self.status = PipelineStatus.PARTIAL
        self.finished_at = datetime.now()

    def fail(self, error: str):
        self._require_running("fail")
        self.status = PipelineStatus.FAILED
        self.error = error
        self.finished_at = datetime.now()

    def _require_running(self, action: str):
        if self.status != PipelineStatus.RUNNING:
            raise ValueError(f"Cannot {action} pipeline with status {self.status}")
