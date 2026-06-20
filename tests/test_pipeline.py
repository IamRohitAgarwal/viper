"""Phase 4 — GENERATE pipeline tests (SPEC sections 8, 13 Phase 4)."""

import os

import pytest
from PIL import Image

from src.grounding.locate import ground_locations
from src.models import (
    ActionPlan,
    GroundedLocations,
    Point,
    SceneObject,
    SceneUnderstanding,
    Trajectory,
)
from src.pivot import generator, visual_prompt
from src.runner import persist_outputs, run_pipeline
from src.visualization import animate
from src.vlm.mock_vlm import MockVLM
from src.vlmpc import cost_function, rollout, validator

from .conftest import make_cfg


# ---------- grounding ----------
def test_ground_locations_centers(zones_file):
    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file)
    loc = ground_locations(None, "receiving bay", "dock 4", cfg)
    assert loc.place_a.x == pytest.approx(0.15)   # 0.10 + 0.10/2
    assert loc.place_a.y == pytest.approx(0.20)   # 0.15 + 0.10/2
    assert loc.place_b.x == pytest.approx(0.83)
    assert loc.place_b.y == pytest.approx(0.80)
    assert loc.place_a_label == "receiving bay"
    assert len(loc.obstacles) == 1
    assert loc.obstacles[0].role == "obstacle"


def test_ground_locations_empty_a_defaults_center(zones_file):
    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file)
    loc = ground_locations(None, "", "dock 4", cfg)
    assert loc.place_a == Point(0.5, 0.5)
    assert loc.place_a_label == "start"


def test_ground_locations_unknown_place_raises(zones_file):
    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file)
    with pytest.raises(ValueError):
        ground_locations(None, "receiving bay", "nowhere", cfg)


def test_ground_locations_no_destination_raises(zones_file):
    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file)
    with pytest.raises(ValueError):
        ground_locations(None, "receiving bay", "", cfg)


# ---------- generator ----------
def _locations():
    return GroundedLocations(
        place_a=Point(0.15, 0.20),
        place_b=Point(0.83, 0.80),
        place_a_label="bay",
        place_b_label="dock",
        obstacles=[],
    )


def test_generator_count_and_start():
    cfg = make_cfg()
    cands = generator.generate_candidates(None, _locations(), cfg.NUM_CANDIDATES, cfg)
    assert len(cands) == cfg.NUM_CANDIDATES
    for c in cands:
        assert c.points[0] == Point(0.15, 0.20)   # starts at A


def test_generator_clipped():
    cfg = make_cfg()
    cands = generator.generate_candidates(None, _locations(), cfg.NUM_CANDIDATES, cfg)
    for c in cands:
        for p in c.points:
            assert 0.05 <= p.x <= 0.95
            assert 0.05 <= p.y <= 0.95


def test_generator_deterministic():
    cfg = make_cfg()
    a = generator.generate_candidates(None, _locations(), 5, cfg)
    b = generator.generate_candidates(None, _locations(), 5, cfg)
    assert a == b


def test_visual_prompt_returns_proposal():
    cfg = make_cfg()
    img = Image.new("RGB", (120, 90), (0, 0, 0))
    cands = generator.generate_candidates(img, _locations(), 5, cfg)
    prop = visual_prompt.draw_candidates(img, cands)
    assert prop.image.size == img.size
    assert prop.candidates == cands


# ---------- rollout ----------
def _scene_with_obstacle():
    return SceneUnderstanding(
        objects=[SceneObject("pallet", 0.4, 0.4, 0.2, 0.2, "obstacle")],
        description="d",
        goal_interpretation="g",
    )


def test_rollout_collision_detected():
    cfg = make_cfg()
    traj = Trajectory(id=1, points=[Point(0.1, 0.1), Point(0.5, 0.5), Point(0.9, 0.9)])
    sim = rollout.simulate_trajectory(traj, _scene_with_obstacle(), cfg)
    assert sim.collision is True
    assert len(sim.frames) == len(traj.points)
    assert sim.final_position == Point(0.9, 0.9)


def test_rollout_no_collision():
    cfg = make_cfg()
    traj = Trajectory(id=1, points=[Point(0.05, 0.05), Point(0.05, 0.9)])
    scene = SceneUnderstanding(objects=[], description="d", goal_interpretation="g")
    sim = rollout.simulate_trajectory(traj, scene, cfg)
    assert sim.collision is False
    assert sim.path_length > 0


# ---------- cost ----------
def test_cost_breakdown():
    cfg = make_cfg(COLLISION_PENALTY=100.0, PATH_LENGTH_WEIGHT=1.0, GOAL_DISTANCE_WEIGHT=1.0)
    sim = rollout.SimulationResult(
        trajectory_id=1,
        final_position=Point(0.83, 0.80),  # exactly at B -> goal_distance 0
        path_length=0.5,
        collision=False,
        frames=[],
    )
    cost = cost_function.compute_cost(sim, _locations(), cfg)
    assert cost.goal_distance == pytest.approx(0.0, abs=1e-9)
    assert cost.collision_penalty == 0.0
    assert cost.path_length_penalty == pytest.approx(0.5)
    assert cost.total_cost == pytest.approx(0.5)


def test_cost_collision_applied():
    cfg = make_cfg(COLLISION_PENALTY=100.0)
    sim = rollout.SimulationResult(1, Point(0.83, 0.80), 0.1, True, [])
    cost = cost_function.compute_cost(sim, _locations(), cfg)
    assert cost.collision_penalty == 100.0
    assert cost.total_cost > 100.0


# ---------- validator ----------
def test_validator_picks_min():
    from src.models import CostBreakdown

    costs = [
        CostBreakdown(1, 0.5, 0, 0.5, 1.0),
        CostBreakdown(2, 0.1, 0, 0.2, 0.3),
        CostBreakdown(3, 0.9, 100, 0.5, 101.4),
    ]
    assert validator.select_best(costs) == 2


# ---------- traversal ----------
def test_generate_traversal_creates_gif(frames_dir, tmp_path):
    cfg = make_cfg(TRAVERSAL_FRAMES=8)
    traj = Trajectory(id=1, points=[Point(0.15, 0.2), Point(0.83, 0.8)])
    out = animate.generate_traversal(frames_dir, traj, _locations(), cfg, str(tmp_path / "trav.gif"))
    assert os.path.exists(out)
    with Image.open(out) as gif:
        assert gif.n_frames == 8


# ---------- runner integration ----------
def test_run_pipeline_full(frames_dir, zones_file, tmp_path):
    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=zones_file, TRAVERSAL_FRAMES=6)
    vlm = MockVLM(seed=cfg.SEED, name="mock")
    run_dir = str(tmp_path / "run")
    plan = run_pipeline(frames_dir, "receiving bay", "dock 4", cfg, vlm, run_dir=run_dir)

    assert isinstance(plan, ActionPlan)
    assert plan.best_trajectory is not None
    assert plan.cost.trajectory_id == plan.best_trajectory.id
    assert len(plan.all_costs) >= 1
    assert plan.rationale
    assert os.path.exists(plan.traversal_gif_path)
    # best cost is the minimum among all evaluated
    assert plan.cost.total_cost == min(c.total_cost for c in plan.all_costs)

    paths = persist_outputs(plan, run_dir)
    for p in paths.values():
        assert os.path.exists(p)
