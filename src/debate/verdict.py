"""Final verdict + transcript assembly (SPEC section 9.3).

Produces the human-readable final plan text (endorsed, amended-as-text, or
rejected/conservative) and, on disagreement, the overruled model's reasoning.

Note (SPEC hard requirement 8 / section 9 note): an 'amend' reports the
amended plan AS TEXT only. It never re-runs the pipeline.
"""

from src.models import ActionPlan, DebateResult
from src.verdicts import most_conservative


def _last_verdict_per_model(result: DebateResult) -> dict:
    latest = {}
    for turn in result.transcript:
        latest[turn.model] = turn.verdict
    return latest


def _latest_amended_text(result: DebateResult) -> str | None:
    for turn in reversed(result.transcript):
        if turn.verdict == "amend" and turn.amended_plan:
            return turn.amended_plan
    return None


def format_transcript(result: DebateResult) -> str:
    """Human-readable debate transcript + verdict + concessions (debate.txt)."""
    lines = ["=== VIPER DEBATE TRANSCRIPT ===", ""]
    for turn in result.transcript:
        lines.append(f"[round {turn.round}] {turn.model} -> {turn.verdict}")
        lines.append(f"    {turn.reasoning}")
    lines += [
        "",
        f"round-1 verdicts : {result.round1_solo}",
        f"concessions      : {result.concessions}",
        f"converged        : {result.converged} (rounds used: {result.rounds_used})",
        f"winner           : {result.winner_model}",
        f"final verdict    : {result.final_verdict}",
    ]
    if result.loser_reasoning:
        lines += ["", f"overruled        : {result.loser_reasoning}"]
    lines += ["", "--- final plan ---", result.final_plan_text]
    return "\n".join(lines)


def assemble(result: DebateResult, plan: ActionPlan) -> tuple[str, str | None]:
    """Return ``(final_plan_text, loser_reasoning)``."""
    base = (
        f"{plan.rationale}\n"
        f"Debate verdict: {result.final_verdict} "
        f"(winner: {result.winner_model}, rounds: {result.rounds_used}, "
        f"concessions: {result.concessions})."
    )

    if result.converged:
        verdict = result.final_verdict
        if verdict == "endorse":
            text = f"ENDORSED.\n{base}"
        elif verdict == "amend":
            amended = _latest_amended_text(result) or "(no amendment text provided)"
            text = f"AMENDED (text only, plan not re-run).\nProposed amendment: {amended}\n{base}"
        else:  # reject
            text = f"REJECTED by consensus.\n{base}"
    else:
        # No consensus -> conservative fallback across the two latest verdicts.
        latest = _last_verdict_per_model(result)
        conservative = most_conservative(*latest.values())
        text = (
            f"NO CONSENSUS - conservative fallback applied: '{conservative}'.\n{base}"
        )

    return text, result.loser_reasoning
