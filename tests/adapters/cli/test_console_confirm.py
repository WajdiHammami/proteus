import pytest

from src.adapters.cli.console_confirm import console_confirm, format_estimates
from src.adapters.models.llm_fill import CostEstimate


ESTIMATE = CostEstimate(filler="LLMFieldFiller(gpt-4.1-mini)", calls=42, input_tokens=25300,
                        expected_output_tokens=3800, max_output_tokens=8400,
                        expected_cost=0.0228, max_cost=0.0331, fields=["description", "concepts"])


def test_format():
    text = format_estimates([ESTIMATE])
    assert "LLMFieldFiller(gpt-4.1-mini)" in text
    assert "description, concepts" in text
    assert "42" in text and "25,300" in text
    assert "~3,800 expected, 8,400 max" in text
    assert "~$0.0228 expected, $0.0331 max" in text


def test_format_unpriced():
    text = format_estimates([CostEstimate(filler="x", calls=1)])
    assert "unknown" in text


@pytest.mark.parametrize("answer, expected", [("y", True), ("YES", True), (" yes ", True), ("", False), ("n", False), ("maybe", False)])
def test_answers(answer, expected):
    out = []
    assert console_confirm([ESTIMATE], input_fn=lambda prompt: answer, output=out.append, interactive=True) is expected
    assert "cost estimate" in out[0]


def test_eof_is_no():
    def eof(prompt):
        raise EOFError
    assert console_confirm([ESTIMATE], input_fn=eof, output=lambda s: None, interactive=True) is False


def test_non_interactive_is_no_without_asking():
    asked = []
    out = []
    assert console_confirm([ESTIMATE], input_fn=asked.append, output=out.append, interactive=False) is False
    assert asked == []
    assert "skipped" in out[-1]
