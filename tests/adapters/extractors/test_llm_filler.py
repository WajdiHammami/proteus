import pytest

from src.adapters.extractors.llm_filler import LLMFieldFiller
from src.adapters.models.code_snippet import CodeSnippet
from src.adapters.models.llm_fill import FillJob
from src.adapters.models.normalized_document import NormalizedDocument, TextFormat
from src.adapters.models.value_objects import FieldType, SchemaField
from src.ports.llm_port import (
    LLMConfigurationError, LLMOutputError, LLMUnavailableError, ModelPricing, TokenUsage,
)
from tests.fakes.fake_llm import FakeLLM


FIELDS = [SchemaField("description", FieldType.STR), SchemaField("concepts", FieldType.LIST)]


def snippet(index=0):
    return CodeSnippet(code="int main() {\n  return 0;\n}", language="cpp", index=index)


def document():
    return NormalizedDocument(id="n", raw_document_id="r", source_id="s", content="", format=TextFormat.MARKDOWN)


def fill(filler, fields=FIELDS):
    return filler.fill(snippet(), fields, {}, document())


class TestFill:
    def test_returns_requested_non_null_values(self):
        llm = FakeLLM([{"description": "Returns 0.", "concepts": None, "extra": "ignored"}])
        assert fill(LLMFieldFiller(llm)) == {"description": "Returns 0."}

    def test_request_is_provider_neutral(self):
        llm = FakeLLM([{}])
        fill(LLMFieldFiller(llm, max_output_tokens=150))
        [request] = llm.requests
        assert request.json_schema["required"] == ["description", "concepts"]
        assert request.max_output_tokens == 150
        assert "int main()" in request.user

    def test_no_fields_no_call(self):
        llm = FakeLLM()
        assert fill(LLMFieldFiller(llm), fields=[]) == {}
        assert llm.requests == []

    @pytest.mark.parametrize("error", [LLMUnavailableError("timeout"), LLMOutputError("refused")])
    def test_recoverable_errors_leave_fields_empty(self, error, caplog):
        filler = LLMFieldFiller(FakeLLM([error]))
        assert fill(filler) == {}
        assert filler.stats.failures == 1
        assert "LLM fill failed" in caplog.text

    def test_configuration_error_stops(self):
        filler = LLMFieldFiller(FakeLLM([LLMConfigurationError("bad key")]))
        with pytest.raises(LLMConfigurationError):
            fill(filler)

    def test_usage_and_cost_tracked(self):
        llm = FakeLLM([{}, {}], pricing=ModelPricing(1.0, 2.0), usage=TokenUsage(1000, 100))
        filler = LLMFieldFiller(llm)
        fill(filler); fill(filler)
        assert filler.stats.calls == 2
        assert filler.stats.usage == TokenUsage(2000, 200)
        assert filler.spent == pytest.approx((2000 * 1.0 + 200 * 2.0) / 1e6)

    def test_cost_unknown_without_pricing(self):
        filler = LLMFieldFiller(FakeLLM([{}], pricing=None))
        fill(filler)
        assert filler.spent is None


class TestEstimate:
    def jobs(self, n=3, fields=FIELDS):
        return [FillJob(snippet(i), list(fields), {}, document()) for i in range(n)]

    def test_counts_calls_tokens_and_cost(self):
        filler = LLMFieldFiller(FakeLLM(), max_output_tokens=200)
        estimate = filler.estimate(self.jobs(3))
        assert estimate.calls == 3
        assert estimate.input_tokens > 0
        assert 0 < estimate.expected_output_tokens < estimate.max_output_tokens == 600
        assert 0 < estimate.expected_cost < estimate.max_cost
        assert estimate.fields == ["description", "concepts"]
        assert estimate.filler == "LLMFieldFiller(fake-model)"

    def test_cost_formula_uses_model_prices(self):
        llm = FakeLLM(pricing=ModelPricing(input_per_million=1_000_000, output_per_million=0))
        estimate = LLMFieldFiller(llm).estimate(self.jobs(1))
        assert estimate.expected_cost == pytest.approx(estimate.input_tokens)   # $1 per token

    def test_makes_no_calls(self):
        llm = FakeLLM()
        LLMFieldFiller(llm).estimate(self.jobs(5))
        assert llm.requests == []

    def test_input_tokens_match_what_fill_sends(self):
        llm = FakeLLM([{}])
        filler = LLMFieldFiller(llm)
        estimate = filler.estimate(self.jobs(1))
        fill(filler)
        request = llm.requests[0]
        sent = sum(llm.count_tokens(t) for t in (request.system, request.user))
        assert sent < estimate.input_tokens   # estimate also counts the schema and message overhead

    def test_unpriced_model(self):
        estimate = LLMFieldFiller(FakeLLM(pricing=None)).estimate(self.jobs(2))
        assert estimate.priced is False and estimate.calls == 2

    def test_jobs_without_fields_ignored(self):
        assert LLMFieldFiller(FakeLLM()).estimate(self.jobs(2, fields=[])).calls == 0
