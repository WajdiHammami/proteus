import pytest

from src.adapters.extractors.llm_filler import LLMFieldFiller
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.schema import Schema
from src.adapters.models.value_objects import FieldType, SchemaField
from src.core.filling import FillingStage
from src.core.record_builder import RecordBuilder
from tests.fakes.fake_llm import FakeLLM


def builder():
    s = Schema(id="schema-1", name="cpp_snippet")
    s.add_field(SchemaField("code", FieldType.STR))
    s.add_field(SchemaField("description", FieldType.STR, required=False, source="llm"))
    return RecordBuilder(s)


def pending(n=2):
    doc = NormalizedDocument(id="n", raw_document_id="r", source_id="s", content="", format=TextFormat.MARKDOWN)
    snippets = [CodeSnippet(code=f"int f{i}() {{\n  return {i};\n}}", language="cpp", index=i) for i in range(n)]
    return builder().prepare(doc, snippets)


class Recorder:
    def __init__(self, answer):
        self.answer = answer
        self.seen = []

    def __call__(self, estimates):
        self.seen.append(estimates)
        return self.answer


def test_confirmed_fills_after_estimate():
    llm = FakeLLM([{"description": "one"}, {"description": "two"}])
    confirm = Recorder(True)
    items = pending()
    report = FillingStage([LLMFieldFiller(llm)], confirm=confirm).run(items)

    [estimates] = confirm.seen
    assert estimates[0].calls == 2
    assert report.confirmed and report.ran and report.jobs == 2
    assert [i.values["description"] for i in items] == ["one", "two"]
    assert report.spent > 0


def test_estimate_shown_before_any_call():
    llm = FakeLLM([{"description": "x"}] * 2)

    def confirm(estimates):
        assert llm.requests == []       # nothing spent yet
        return True

    FillingStage([LLMFieldFiller(llm)], confirm=confirm).run(pending())
    assert len(llm.requests) == 2


def test_declined_makes_no_calls():
    llm = FakeLLM()
    items = pending()
    report = FillingStage([LLMFieldFiller(llm)], confirm=Recorder(False)).run(items)
    assert llm.requests == []
    assert report.confirmed is False and report.ran is False and report.skipped_reason == "declined"
    assert all("description" not in i.values for i in items)
    assert report.estimates[0].calls == 2      # the estimate is still reported


def test_debug_skips_confirmation():
    llm = FakeLLM([{"description": "x"}] * 2)
    confirm = Recorder(False)
    report = FillingStage([LLMFieldFiller(llm)], confirm=confirm, debug=True).run(pending())
    assert confirm.seen == []
    assert report.ran and len(llm.requests) == 2


def test_no_confirm_callback_means_no_spending():
    llm = FakeLLM()
    report = FillingStage([LLMFieldFiller(llm)], confirm=None).run(pending())
    assert not report.ran and "confirmation" in report.skipped_reason
    assert llm.requests == []


def test_nothing_missing_asks_nothing():
    confirm = Recorder(True)
    items = pending()
    for item in items:
        item.missing = []
    report = FillingStage([LLMFieldFiller(FakeLLM())], confirm=confirm).run(items)
    assert confirm.seen == [] and report.skipped_reason == "nothing to fill"


def test_no_fillers_asks_nothing():
    confirm = Recorder(True)
    report = FillingStage([], confirm=confirm).run(pending())
    assert confirm.seen == [] and report.skipped_reason == "no filler configured"


def test_one_estimate_per_filler():
    confirm = Recorder(True)
    FillingStage([LLMFieldFiller(FakeLLM()), LLMFieldFiller(FakeLLM())], confirm=confirm).run(pending())
    assert len(confirm.seen[0]) == 2


def test_spent_unknown_without_prices():
    report = FillingStage([LLMFieldFiller(FakeLLM([{}] * 2, pricing=None))], confirm=None, debug=True).run(pending())
    assert report.ran and report.spent is None
