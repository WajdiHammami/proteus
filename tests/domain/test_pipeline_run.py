from datetime import datetime

import pytest

from src.domain import PipelineRun, PipelineStatus, StageResult


@pytest.fixture
def run() -> PipelineRun:
    return PipelineRun(id="p1", schema_id="sch1")


def test_defaults(run):
    assert run.status == PipelineStatus.PENDING
    assert run.source_ids == []
    assert run.stages == []
    assert run.started_at is None
    assert run.finished_at is None
    assert run.error is None


def test_start_sets_running_and_timestamp(run):
    before = datetime.now()
    run.start()
    assert run.status == PipelineStatus.RUNNING
    assert before <= run.started_at <= datetime.now()


def test_start_twice_raises(run):
    run.start()
    with pytest.raises(ValueError, match="already started"):
        run.start()


@pytest.mark.parametrize(
    "status",
    [PipelineStatus.RUNNING, PipelineStatus.DONE, PipelineStatus.FAILED, PipelineStatus.PARTIAL],
)
def test_start_rejects_non_pending(status):
    run = PipelineRun(id="p1", schema_id="sch1", status=status)
    with pytest.raises(ValueError):
        run.start()


def test_add_stage_result_preserves_order(run):
    first = StageResult(stage="ingestion", status=PipelineStatus.DONE, items_processed=3)
    second = StageResult(stage="extraction", status=PipelineStatus.FAILED, error="boom")
    run.add_stage_result(first)
    run.add_stage_result(second)
    assert run.stages == [first, second]


def test_complete(run):
    run.start()
    run.complete()
    assert run.status == PipelineStatus.DONE
    assert run.finished_at >= run.started_at
    assert run.error is None


def test_partial(run):
    run.start()
    run.partial()
    assert run.status == PipelineStatus.PARTIAL
    assert run.finished_at >= run.started_at


def test_fail_records_error(run):
    run.start()
    run.fail("network down")
    assert run.status == PipelineStatus.FAILED
    assert run.error == "network down"
    assert run.finished_at >= run.started_at


FINISHERS = {
    "complete": lambda r: r.complete(),
    "partial": lambda r: r.partial(),
    "fail": lambda r: r.fail("boom"),
}


@pytest.mark.parametrize("finisher", FINISHERS.values(), ids=FINISHERS.keys())
def test_finishing_before_start_raises(run, finisher):
    with pytest.raises(ValueError, match="Cannot"):
        finisher(run)
    assert run.status == PipelineStatus.PENDING
    assert run.finished_at is None


@pytest.mark.parametrize("finisher", FINISHERS.values(), ids=FINISHERS.keys())
def test_finishing_twice_raises(run, finisher):
    run.start()
    run.complete()
    finished_at = run.finished_at
    with pytest.raises(ValueError, match="Cannot"):
        finisher(run)
    assert run.status == PipelineStatus.DONE
    assert run.finished_at == finished_at
