import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.adapters.models.llm_fill import CostEstimate
from src.core.record_builder import PendingRecord, fill_pending
from src.ports.extraction_port import FieldFillerPort


logger = logging.getLogger(__name__)

# Shown the estimates (one per filler), returns True to go ahead.
ConfirmCallback = Callable[[list[CostEstimate]], bool]


@dataclass
class FillReport:
    estimates: list[CostEstimate] = field(default_factory=list)
    jobs: int = 0                  # snippets that had fields to fill
    confirmed: bool = False
    ran: bool = False
    skipped_reason: str | None = None
    spent: float | None = 0.0      # actual cost, None if a filler couldn't tell


class FillingStage:
    """
    The last extraction step, run once for all collected snippets:

        estimate (no calls) -> confirmation -> fill

    With `debug=True` the confirmation is skipped. Without fillers, or when nothing
    is missing, nothing is asked and nothing is spent. If the user declines, records
    are finalized with their LLM fields empty.
    """

    def __init__(self, fillers: Sequence[FieldFillerPort], confirm: ConfirmCallback | None, debug: bool = False):
        self.fillers = list(fillers)
        self.confirm = confirm
        self.debug = debug

    def run(self, pending: Sequence[PendingRecord]) -> FillReport:
        todo = [item for item in pending if item.missing]
        report = FillReport(jobs=len(todo))
        if not todo:
            report.skipped_reason = "nothing to fill"
            return report
        if not self.fillers:
            report.skipped_reason = "no filler configured"
            return report

        jobs = [item.job() for item in todo]
        # Upper bound: each filler is estimated on every job (later fillers usually get fewer).
        report.estimates = [filler.estimate(jobs) for filler in self.fillers]

        if self.debug:
            logger.info("Debug mode: filling %d snippet(s) without confirmation", len(todo))
            report.confirmed = True
        elif self.confirm is None:
            report.skipped_reason = "no confirmation callback (pass confirm=... or debug=True)"
            return report
        else:
            report.confirmed = bool(self.confirm(report.estimates))
            if not report.confirmed:
                report.skipped_reason = "declined"
                logger.info("Filling declined; LLM fields stay empty")
                return report

        fill_pending(todo, self.fillers)
        report.ran = True
        spent = [filler.spent for filler in self.fillers]
        report.spent = None if any(s is None for s in spent) else sum(spent)
        return report
