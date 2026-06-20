"""Phase 7 — evaluation metrics + resumable batch (SPEC sections 12, 13 Phase 7)."""

import os

from src.evaluation import metrics
from src.vlm.mock_vlm import MockVLM

from .conftest import make_cfg


def _rec(goal_distance, total_cost, collision, path_length, straight, debate=None):
    return {
        "frame": "f.png",
        "goal_distance": goal_distance,
        "total_cost": total_cost,
        "collision": collision,
        "path_length": path_length,
        "straight_dist": straight,
        "debate": debate,
    }


def test_trajectory_metrics():
    recs = [
        _rec(0.0, 1.0, False, 1.0, 0.8),
        _rec(0.0, 3.0, True, 2.0, 1.0),
    ]
    m = metrics.trajectory_metrics(recs)
    assert m["avg_path_cost"] == 2.0
    assert m["collision_rate"] == 0.5
    assert m["avg_path_efficiency"] == (0.8 / 1.0 + 1.0 / 2.0) / 2


def test_outcome_metrics():
    recs = [
        _rec(0.05, 1.0, False, 1.0, 1.0),   # success (close, no collision)
        _rec(0.05, 1.0, True, 1.0, 1.0),    # fail (collision)
        _rec(0.5, 1.0, False, 1.0, 1.0),    # fail (too far)
    ]
    m = metrics.outcome_metrics(recs, threshold=0.1)
    assert m["task_success_rate"] == 1 / 3
    assert m["avg_goal_distance_error"] == (0.05 + 0.05 + 0.5) / 3


def test_ensemble_metrics():
    d1 = {"final_verdict": "endorse", "converged": True, "rounds_used": 1,
          "concessions": {"claude": 0, "molmo": 1},
          "round1_solo": {"claude": "endorse", "molmo": "reject"}}
    d2 = {"final_verdict": "no_consensus", "converged": False, "rounds_used": 3,
          "concessions": {"claude": 0, "molmo": 0},
          "round1_solo": {"claude": "endorse", "molmo": "reject"}}
    recs = [_rec(0, 0, False, 1, 1, debate=d1), _rec(0, 0, False, 1, 1, debate=d2)]
    m = metrics.ensemble_metrics(recs)
    assert m["convergence_rate"] == 0.5
    assert m["avg_rounds"] == 2.0
    assert m["total_concessions"] == {"claude": 0, "molmo": 1}
    # debate-final matches claude solo (endorse) in d1 only -> 0.5
    assert m["debate_vs_claude_solo_agreement"] == 0.5
    assert m["debate_vs_molmo_solo_agreement"] == 0.0


def test_ensemble_metrics_empty_when_no_debate():
    recs = [_rec(0, 0, False, 1, 1, debate=None)]
    assert metrics.ensemble_metrics(recs) == {}


def test_run_batch_and_resume(frames_dir, zones_file, tmp_path):
    cfg = make_cfg(
        GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file,
        TRAVERSAL_FRAMES=4, DEBATE_ENABLED=True, DEBATE_MAX_ROUNDS=2,
    )
    gen = MockVLM(seed=cfg.SEED, name="mock")
    a = MockVLM(seed=cfg.SEED, name="claude")
    b = MockVLM(seed=cfg.SEED + 1, concede=True, name="molmo")
    batch_dir = str(tmp_path / "batch")

    s1 = metrics.run_batch(frames_dir, "receiving bay", "dock 4", cfg, gen, a, b, batch_dir)
    assert s1["n_frames"] == len(frames_dir)
    assert os.path.exists(os.path.join(batch_dir, "summary.json"))
    assert "ensemble" in s1 and s1["ensemble"]  # debate ran
    # per-frame logs written
    assert os.path.exists(os.path.join(batch_dir, "frame_0000", "log.json"))

    # Resume: deleting one frame's log forces only that frame to re-run; summary identical.
    os.remove(os.path.join(batch_dir, "frame_0001", "log.json"))
    s2 = metrics.run_batch(frames_dir, "receiving bay", "dock 4", cfg, gen, a, b, batch_dir)
    assert s2 == s1
