"""Phase 5 — verdict assembly tests (SPEC sections 9.3, 13 Phase 5)."""

from PIL import Image

from src.debate.verdict import assemble, format_transcript
from src.models import (
    ActionPlan,
    CostBreakdown,
    DebateResult,
    DebateTurn,
    GroundedLocations,
    Point,
    SceneUnderstanding,
    SimulationResult,
    Trajectory,
)


def _plan():
    traj = Trajectory(id=1, points=[Point(0.1, 0.1), Point(0.9, 0.9)])
    cost = CostBreakdown(1, 0.0, 0.0, 0.5, 0.5)
    img = Image.new("RGB", (16, 16))
    return ActionPlan(
        best_trajectory=traj,
        locations=GroundedLocations(Point(0.1, 0.1), Point(0.9, 0.9), "A", "B", []),
        cost=cost,
        all_costs=[cost],
        simulation=SimulationResult(1, Point(0.9, 0.9), 0.5, False, []),
        rationale="Chose T1.",
        candidates_image=img,
        selected_image=img,
        final_image=img,
        traversal_gif_path="x.gif",
        scene=SceneUnderstanding([], "d", "g"),
    )


def _turn(model, verdict, amended=None):
    return DebateTurn(
        round=1, model=model, verdict=verdict, amended_plan=amended,
        reasoning=f"{model} says {verdict}", raw_reply=f"VERDICT: {verdict}",
    )


def _result(final, converged, winner, transcript, loser=None):
    return DebateResult(
        final_verdict=final, converged=converged, rounds_used=1, winner_model=winner,
        concessions={"claude": 0, "molmo": 0},
        round1_solo={"claude": transcript[0].verdict, "molmo": transcript[1].verdict},
        transcript=transcript, loser_reasoning=loser, final_plan_text="",
    )


def test_assemble_endorse():
    res = _result("endorse", True, "agreement",
                  [_turn("claude", "endorse"), _turn("molmo", "endorse")])
    text, loser = assemble(res, _plan())
    assert "ENDORSED" in text
    assert "Chose T1." in text
    assert loser is None


def test_assemble_amend_is_text_only():
    res = _result("amend", True, "agreement",
                  [_turn("claude", "amend", amended="shift path left"),
                   _turn("molmo", "amend", amended="shift path left")])
    text, _ = assemble(res, _plan())
    assert "AMENDED" in text
    assert "plan not re-run" in text
    assert "shift path left" in text


def test_assemble_reject():
    res = _result("reject", True, "agreement",
                  [_turn("claude", "reject"), _turn("molmo", "reject")])
    text, _ = assemble(res, _plan())
    assert "REJECTED" in text


def test_assemble_no_consensus_conservative():
    res = _result("no_consensus", False, "tie_break",
                  [_turn("claude", "endorse"), _turn("molmo", "reject")],
                  loser="claude (endorse) was overruled by the conservative fallback 'reject'.")
    text, loser = assemble(res, _plan())
    assert "NO CONSENSUS" in text
    assert "reject" in text         # conservative pick surfaced
    assert loser is not None


def test_format_transcript_contains_verdicts():
    res = _result("endorse", True, "agreement",
                  [_turn("claude", "endorse"), _turn("molmo", "endorse")])
    res.final_plan_text = "ENDORSED."
    out = format_transcript(res)
    assert "claude -> endorse" in out
    assert "final verdict    : endorse" in out
    assert "concessions" in out
