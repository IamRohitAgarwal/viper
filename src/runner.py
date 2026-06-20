"""GENERATE pipeline orchestration (SPEC section 8.7).

Order: pick planning frame -> ground A/B/obstacles -> generate candidates
(A->B) -> draw -> select (VLM/fallback) -> simulate each selected -> cost
each -> pick best -> animate traversal across real frames -> build images +
rationale -> ActionPlan.
"""

import dataclasses
import os
from datetime import datetime

from PIL import Image

from src.debate import artifact as artifact_mod
from src.debate import verdict as verdict_mod
from src.debate.relay import run_debate
from src.grounding.locate import ground_locations
from src.models import ActionPlan, ViperResult
from src.pivot import generator, visual_prompt
from src.vlmpc import cost_function, rollout, validator
from src.visualization import animate, draw


def _new_run_dir() -> str:
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join("outputs", run_id)
    os.makedirs(path, exist_ok=True)
    return path


def run_pipeline(
    frames: list[str],
    place_a: str,
    place_b: str,
    config,
    vlm,
    run_dir: str | None = None,
) -> ActionPlan:
    """Run the full GENERATE pipeline and return a populated ActionPlan.

    ``frames`` are all extracted frame paths; the first is used for planning
    and all are used for the traversal animation (resolved Q4).
    """
    if not frames:
        raise ValueError("run_pipeline requires at least one frame")
    run_dir = run_dir or _new_run_dir()
    goal = f"from {place_a} to {place_b}" if place_a else (place_b or "")

    planning_frame = Image.open(frames[0]).convert("RGB")

    # Ground A, B, obstacles.
    locations = ground_locations(planning_frame, place_a, place_b, config, vlm)

    # Scene understanding; inject grounded obstacles so the rollout sees them.
    scene = vlm.understand_scene(planning_frame, goal)
    scene = dataclasses.replace(scene, objects=scene.objects + locations.obstacles)

    # PIVOT: candidates A->B, then annotate.
    candidates = generator.generate_candidates(
        planning_frame, locations, config.NUM_CANDIDATES, config
    )
    proposal = visual_prompt.draw_candidates(planning_frame, candidates)

    # VLM / fallback selection.
    selected = vlm.select_candidates(proposal.image, goal, candidates, scene)
    by_id = {t.id: t for t in candidates}
    shortlist = [by_id[i] for i in selected.ids if i in by_id]

    # VLMPC: simulate + cost each shortlisted trajectory.
    sims = {t.id: rollout.simulate_trajectory(t, scene, config, planning_frame) for t in shortlist}
    costs = [cost_function.compute_cost(sims[t.id], locations, config) for t in shortlist]

    best_id = validator.select_best(costs)
    best_traj = by_id[best_id]
    best_sim = sims[best_id]
    best_cost = next(c for c in costs if c.trajectory_id == best_id)

    # Imagery.
    candidates_image = draw.draw_candidates(planning_frame, candidates, selected_ids=selected.ids)
    selected_image = draw.draw_selected(planning_frame, best_traj, locations)
    final_image = draw.draw_final(planning_frame, best_traj, locations, scene)

    # Traversal GIF across the real frames (KEY output).
    traversal_path = os.path.join(run_dir, "traversal.gif")
    animate.generate_traversal(frames, best_traj, locations, config, traversal_path)

    plan = ActionPlan(
        best_trajectory=best_traj,
        locations=locations,
        cost=best_cost,
        all_costs=costs,
        simulation=best_sim,
        rationale="",
        candidates_image=candidates_image,
        selected_image=selected_image,
        final_image=final_image,
        traversal_gif_path=traversal_path,
        scene=scene,
    )
    plan.rationale = vlm.generate_rationale(planning_frame, plan, goal)
    return plan


def run_viper(
    frames: list[str],
    place_a: str,
    place_b: str,
    config,
    gen_vlm,
    model_a=None,
    model_b=None,
    run_dir: str | None = None,
) -> ViperResult:
    """Full system: GENERATE -> (optional) DEBATE (SPEC section 9.4).

    Signature adapted from the SPEC's ``run_viper(frame_image, goal, ...)`` to
    take the same (frames, place_a, place_b) inputs as ``run_pipeline``.
    """
    run_dir = run_dir or _new_run_dir()
    plan = run_pipeline(frames, place_a, place_b, config, gen_vlm, run_dir=run_dir)
    log_path = os.path.join(run_dir, "log.json")

    if not config.DEBATE_ENABLED:
        return ViperResult(plan=plan, debate=None, frame_path=frames[0], log_path=log_path)

    if model_a is None or model_b is None:
        raise ValueError("DEBATE_ENABLED requires both model_a and model_b")

    goal = f"from {place_a} to {place_b}" if place_a else (place_b or "")
    artifact = artifact_mod.package(plan, goal)
    debate = run_debate(artifact, config, model_a, model_b)
    final_text, loser = verdict_mod.assemble(debate, plan)
    debate.final_plan_text = final_text
    debate.loser_reasoning = loser
    return ViperResult(plan=plan, debate=debate, frame_path=frames[0], log_path=log_path)


def persist_outputs(plan: ActionPlan, run_dir: str) -> dict:
    """Write candidates/selected/final PNGs + rollout GIF; return their paths."""
    os.makedirs(run_dir, exist_ok=True)
    paths = {
        "candidates": os.path.join(run_dir, "candidates.png"),
        "selected": os.path.join(run_dir, "selected.png"),
        "final": os.path.join(run_dir, "final.png"),
        "trajectory": os.path.join(run_dir, "trajectory.gif"),
        "traversal": plan.traversal_gif_path,
    }
    plan.candidates_image.save(paths["candidates"])
    plan.selected_image.save(paths["selected"])
    plan.final_image.save(paths["final"])
    animate.generate_gif(plan.simulation.frames, paths["trajectory"])
    return paths
