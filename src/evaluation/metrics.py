"""Evaluation metrics + resumable batch runner (SPEC section 12).

Metrics operate on lightweight per-frame *records* (plain dicts) so a batch
can be resumed purely from the per-frame ``log.json`` files on disk — no need
to keep heavy ActionPlan/DebateResult objects in memory.

Metric groups:
  Outcome     : Task Success Rate, Goal Distance Error.
  Trajectory  : avg Path Cost, Collision Rate, Path Efficiency.
  Ensemble    : convergence rate, avg rounds, concessions, debate-vs-solo
                verdict agreement (deference check).
"""

import json
import math
import os

from src.evaluation.logger import log_run
from src.models import ViperResult
from src.runner import persist_outputs, run_viper


# ---------- record extraction ----------
def record_from_result(result: ViperResult) -> dict:
    plan = result.plan
    a, b = plan.locations.place_a, plan.locations.place_b
    straight = math.hypot(b.x - a.x, b.y - a.y)
    rec = {
        "frame": result.frame_path,
        "goal_distance": plan.cost.goal_distance,
        "total_cost": plan.cost.total_cost,
        "collision": plan.cost.collision_penalty > 0,
        "path_length": plan.simulation.path_length,
        "straight_dist": straight,
        "debate": None,
    }
    if result.debate is not None:
        d = result.debate
        rec["debate"] = {
            "final_verdict": d.final_verdict,
            "converged": d.converged,
            "rounds_used": d.rounds_used,
            "concessions": dict(d.concessions),
            "round1_solo": dict(d.round1_solo),
        }
    return rec


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------- metric groups ----------
def trajectory_metrics(records: list[dict]) -> dict:
    return {
        "avg_path_cost": _mean(r["total_cost"] for r in records),
        "collision_rate": _mean(1.0 if r["collision"] else 0.0 for r in records),
        "avg_path_efficiency": _mean(
            (r["straight_dist"] / r["path_length"]) if r["path_length"] > 1e-9 else 0.0
            for r in records
        ),
    }


def outcome_metrics(records: list[dict], threshold: float) -> dict:
    def success(r):
        return (r["goal_distance"] <= threshold) and (not r["collision"])

    return {
        "task_success_rate": _mean(1.0 if success(r) else 0.0 for r in records),
        "avg_goal_distance_error": _mean(r["goal_distance"] for r in records),
    }


def ensemble_metrics(records: list[dict]) -> dict:
    debates = [r["debate"] for r in records if r["debate"] is not None]
    if not debates:
        return {}
    model_names = list(debates[0]["concessions"].keys())
    out = {
        "convergence_rate": _mean(1.0 if d["converged"] else 0.0 for d in debates),
        "avg_rounds": _mean(d["rounds_used"] for d in debates),
        "total_concessions": {
            m: sum(d["concessions"].get(m, 0) for d in debates) for m in model_names
        },
    }
    # Agreement of the debate's final verdict with each model's solo verdict.
    for m in model_names:
        out[f"debate_vs_{m}_solo_agreement"] = _mean(
            1.0 if d["final_verdict"] == d["round1_solo"].get(m) else 0.0 for d in debates
        )
    return out


def summarize(records: list[dict], threshold: float) -> dict:
    return {
        "n_frames": len(records),
        "outcome": outcome_metrics(records, threshold),
        "trajectory": trajectory_metrics(records),
        "ensemble": ensemble_metrics(records),
    }


# ---------- resumable batch ----------
def run_batch(
    frames: list[str],
    place_a: str,
    place_b: str,
    config,
    gen_vlm,
    model_a=None,
    model_b=None,
    batch_dir: str = "outputs/batch",
) -> dict:
    """Plan on every frame; resumable (skips frames whose log.json exists).

    Each frame is planned individually (its frame is the planning frame) while
    all frames remain available for the traversal animation. Per-frame records
    accumulate into ``<batch_dir>/summary.json``.
    """
    os.makedirs(batch_dir, exist_ok=True)
    records: list[dict] = []

    for i, _frame in enumerate(frames):
        frame_dir = os.path.join(batch_dir, f"frame_{i:04d}")
        log_path = os.path.join(frame_dir, "log.json")

        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as fh:
                records.append(json.load(fh)["record"])
            continue

        ordered = [frames[i]] + frames[:i] + frames[i + 1 :]
        result = run_viper(
            ordered, place_a, place_b, config, gen_vlm, model_a, model_b, run_dir=frame_dir
        )
        persist_outputs(result.plan, frame_dir)
        rec = record_from_result(result)
        log_run({"record": rec}, log_path)
        records.append(rec)

    summary = summarize(records, config.SUCCESS_DISTANCE_THRESHOLD)
    with open(os.path.join(batch_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
