"""Debate relay + convergence (SPEC section 9.2).

The orchestrator is the SOLE caller; the two models only ever see each
other's *relayed text* (never call each other). Relay, not simultaneous:
model A speaks, then B (given A's reply), then A (given B's latest), ...,
until their latest verdicts match (converged) or max rounds is reached.
"""

from src.models import DebateArtifact, DebateResult, DebateTurn
from src.verdicts import most_conservative, parse_verdict


def _authoritative_verdict(model, artifact, other_view, parse_retry, fallback):
    """Get a turn and resolve its verdict via the VERDICT: line.

    Retries once on parse failure (SPEC step 7), else keeps ``fallback``
    (the model's previous verdict). Returns the (turn, verdict).
    """
    turn = model.critique_plan(artifact, other_view)
    verdict = parse_verdict(turn.raw_reply)
    attempts = 0
    while verdict is None and attempts < parse_retry:
        turn = model.critique_plan(artifact, other_view)
        verdict = parse_verdict(turn.raw_reply)
        attempts += 1
    if verdict is None:
        verdict = fallback if fallback is not None else "endorse"
    turn.verdict = verdict
    return turn


def run_debate(
    artifact: DebateArtifact,
    config,
    model_a,
    model_b,
) -> DebateResult:
    name_a = getattr(model_a, "name", config.DEBATE_MODEL_A)
    name_b = getattr(model_b, "name", config.DEBATE_MODEL_B)
    speakers = [(name_a, model_a), (name_b, model_b)]

    # Independent ("solo") probe per model BEFORE any relay, so round1_solo
    # captures each model's own opinion (model B's first relay turn already
    # sees A, so it is not solo). Concessions are then counted as divergence
    # from this solo baseline. Costs 2 extra calls per debate; for the real
    # models this is the very debate-vs-solo signal the project measures.
    solo_a = _authoritative_verdict(model_a, artifact, None, config.PARSE_RETRY, None)
    solo_b = _authoritative_verdict(model_b, artifact, None, config.PARSE_RETRY, None)
    round1_solo = {name_a: solo_a.verdict, name_b: solo_b.verdict}

    transcript: list[DebateTurn] = []
    last_reply = {name_a: None, name_b: None}        # most recent raw_reply per model
    last_verdict = {name_a: solo_a.verdict, name_b: solo_b.verdict}  # seeded with solo
    concessions = {name_a: 0, name_b: 0}
    converged = False
    rounds_used = 0

    for r in range(1, config.DEBATE_MAX_ROUNDS + 1):
        rounds_used = r
        for name, model in speakers:
            other_name = name_b if name == name_a else name_a
            other_view = last_reply[other_name]
            turn = _authoritative_verdict(
                model, artifact, other_view, config.PARSE_RETRY, last_verdict[name]
            )
            turn.round = r

            if turn.verdict != last_verdict[name]:
                concessions[name] += 1

            last_verdict[name] = turn.verdict
            last_reply[name] = turn.raw_reply
            transcript.append(turn)

        if last_verdict[name_a] == last_verdict[name_b]:
            converged = True
            break

    if converged:
        final_verdict = last_verdict[name_a]
        winner_model = "agreement"
        loser_reasoning = None
    else:
        # Tie-break to the most conservative of the two latest verdicts.
        conservative = most_conservative(last_verdict[name_a], last_verdict[name_b])
        final_verdict = "no_consensus"
        winner_model = "tie_break"
        # The overruled model is the one NOT holding the conservative verdict.
        loser_name = name_a if last_verdict[name_a] != conservative else name_b
        loser_turn = next(t for t in reversed(transcript) if t.model == loser_name)
        loser_reasoning = (
            f"{loser_name} ({loser_turn.verdict}) was overruled by the conservative "
            f"fallback '{conservative}'. Its reasoning: {loser_turn.reasoning}"
        )

    return DebateResult(
        final_verdict=final_verdict,
        converged=converged,
        rounds_used=rounds_used,
        winner_model=winner_model,
        concessions=concessions,
        round1_solo=round1_solo,
        transcript=transcript,
        loser_reasoning=loser_reasoning,
        final_plan_text="",   # filled by verdict.assemble using the plan
    )
