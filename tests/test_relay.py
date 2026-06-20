"""Phase 5 — debate relay tests (SPEC sections 9.2, 13 Phase 5)."""

from PIL import Image

from src.debate.relay import run_debate
from src.models import DebateArtifact
from src.vlm.mock_vlm import MockVLM

from .conftest import make_cfg


def _artifact():
    return DebateArtifact(
        selected_image=Image.new("RGB", (32, 32)),
        goal="from A to B",
        trajectory_summary="A->B",
        cost_summary="low",
        candidate_summary="others lost",
    )


def test_converge_when_both_endorse():
    # seed 42 -> endorse, seed 45 -> endorse (45 % 3 == 0)
    cfg = make_cfg(DEBATE_MAX_ROUNDS=3)
    a = MockVLM(seed=42, name="claude")
    b = MockVLM(seed=45, name="molmo")
    res = run_debate(_artifact(), cfg, a, b)
    assert res.converged is True
    assert res.final_verdict == "endorse"
    assert res.winner_model == "agreement"
    assert res.rounds_used == 1


def test_deadlock_conservative_tie_break():
    # endorse (42) vs reject (44); neither concedes -> no consensus, conservative.
    cfg = make_cfg(DEBATE_MAX_ROUNDS=3)
    a = MockVLM(seed=42, concede=False, name="claude")  # endorse
    b = MockVLM(seed=44, concede=False, name="molmo")   # reject
    res = run_debate(_artifact(), cfg, a, b)
    assert res.converged is False
    assert res.winner_model == "tie_break"
    assert res.final_verdict == "no_consensus"
    # the endorser (claude) is overruled by the conservative 'reject'
    assert res.loser_reasoning is not None
    assert "claude" in res.loser_reasoning


def test_concession_counted_and_converges():
    # claude endorses (42); molmo rejects (44) but concedes -> molmo moves toward
    # claude's view? No: molmo concedes to MORE conservative. claude=endorse,
    # molmo base=reject; molmo only concedes to something more conservative than
    # reject (none), so molmo stays reject. Instead make molmo the endorser that
    # concedes to claude's reject.
    cfg = make_cfg(DEBATE_MAX_ROUNDS=3)
    a = MockVLM(seed=44, concede=False, name="claude")        # reject
    b = MockVLM(seed=42, concede=True, name="molmo")          # endorse -> concedes
    res = run_debate(_artifact(), cfg, a, b)
    assert res.round1_solo["claude"] == "reject"
    assert res.round1_solo["molmo"] == "endorse"
    # molmo sees claude's reject and concedes to reject -> converge on reject.
    assert res.converged is True
    assert res.final_verdict == "reject"
    assert res.concessions["molmo"] >= 1
    assert res.concessions["claude"] == 0


def test_transcript_completeness_and_rounds():
    cfg = make_cfg(DEBATE_MAX_ROUNDS=2)
    a = MockVLM(seed=42, concede=False, name="claude")  # endorse
    b = MockVLM(seed=44, concede=False, name="molmo")   # reject
    res = run_debate(_artifact(), cfg, a, b)
    # 2 rounds * 2 speakers = 4 turns when it never converges
    assert len(res.transcript) == 4
    assert {t.model for t in res.transcript} == {"claude", "molmo"}
    assert all(t.round in (1, 2) for t in res.transcript)
