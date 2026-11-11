import sys
from collections.abc import Callable

from src.adapters.models.llm_fill import CostEstimate


def format_estimates(estimates: list[CostEstimate]) -> str:
    lines = ["LLM filling: cost estimate (before any call)"]
    for estimate in estimates:
        lines += [
            f"  filler:         {estimate.filler}",
            f"  fields:         {', '.join(estimate.fields) or '-'}",
            f"  calls:          {estimate.calls:,}",
            f"  input tokens:   {estimate.input_tokens:,}",
            f"  output tokens:  ~{estimate.expected_output_tokens:,} expected, {estimate.max_output_tokens:,} max",
        ]
        if estimate.priced:
            lines.append(
                f"  cost:           ~{_money(estimate.expected_cost, estimate.currency)} expected, "
                f"{_money(estimate.max_cost, estimate.currency)} max"
            )
        else:
            lines.append("  cost:           unknown (no prices configured for this model)")
    total_expected = [e.expected_cost for e in estimates]
    if len(estimates) > 1 and all(c is not None for c in total_expected):
        lines.append(f"  total:          ~{_money(sum(total_expected), estimates[0].currency)} expected")
    return "\n".join(lines)


def console_confirm(
    estimates: list[CostEstimate],
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    interactive: bool | None = None,
) -> bool:
    """Prints the estimate and asks y/N. Never says yes on its own: no terminal = no."""
    output(format_estimates(estimates))
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        output("No terminal to confirm in: filling skipped (debug mode fills without confirmation).")
        return False
    try:
        answer = input_fn("Proceed with filling? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _money(amount: float, currency: str) -> str:
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{amount:.4f}" if amount < 1 else f"{symbol}{amount:,.2f}"
