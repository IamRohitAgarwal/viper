"""Deterministic mock VLM + offline rule-based fallback (SPEC sections 7, 10).

MockVLM serves two purposes:
  1. The deterministic fake "brain" for tests (same seed -> identical output).
  2. The rule-based fallback used whenever ``USE_VLM = False`` — the whole
     GENERATE pipeline must run offline with no network through this class.

All randomness flows through ``numpy.random.default_rng(seed)``.
"""

import numpy as np
from PIL import Image

from src.models import (
    ActionPlan,
    DebateArtifact,
    DebateTurn,
    Point,
    SceneObject,
    SceneUnderstanding,
    SelectedCandidates,
    Trajectory,
)
from src.verdicts import CONSERVATISM, VERDICTS, parse_verdict
from src.vlm.interface import VLMInterface


def _goal_region(scene: SceneUnderstanding) -> Point:
    """Centroid of the goal object, or image centre if none is tagged."""
    for obj in scene.objects:
        if obj.role == "goal":
            return Point(obj.x + obj.w / 2, obj.y + obj.h / 2)
    return Point(0.5, 0.5)


class MockVLM(VLMInterface):
    """Deterministic, offline implementation of the full VLM interface.

    Args:
        seed: drives all randomness; identical seed -> identical outputs.
        concede: if True, in a debate this model defers to a more
            conservative opposing verdict (used in Phase 5 to exercise
            concession tracking). Default False -> never changes its mind.
        name: label used in debate turns ("claude" | "molmo" | ...).
    """

    def __init__(self, seed: int = 42, concede: bool = False, name: str = "mock"):
        self.seed = seed
        self.concede = concede
        self.name = name

    # ---- Stage 1: perception ----
    def understand_scene(self, image: Image.Image, goal: str) -> SceneUnderstanding:
        rng = np.random.default_rng(self.seed)
        n = int(rng.integers(2, 6))  # 2..5 objects
        objects: list[SceneObject] = []

        # Always a target and a goal; optionally an obstacle if the goal hints it.
        objects.append(self._rand_object(rng, "target", "object_0"))
        objects.append(self._rand_object(rng, "goal", "goal_region"))
        wants_obstacle = any(
            kw in goal.lower() for kw in ("avoid", "obstacle", "around", "without")
        )
        roles = (["obstacle"] if wants_obstacle else []) + ["background"] * n
        for i in range(len(objects), n):
            role = roles[i - len(objects)] if (i - len(objects)) < len(roles) else "background"
            objects.append(self._rand_object(rng, role, f"object_{i}"))

        return SceneUnderstanding(
            objects=objects,
            description=f"Mock scene with {len(objects)} objects.",
            goal_interpretation=f"Interpreted goal: {goal!r}",
        )

    def _rand_object(self, rng, role: str, name: str) -> SceneObject:
        x, y = rng.uniform(0.05, 0.75, size=2)
        w, h = rng.uniform(0.08, 0.2, size=2)
        return SceneObject(name=name, x=float(x), y=float(y), w=float(w), h=float(h), role=role)

    # ---- Stage 3: selection (rule-based fallback) ----
    def select_candidates(
        self,
        annotated_image: Image.Image,
        goal: str,
        candidates: list[Trajectory],
        scene: SceneUnderstanding,
    ) -> SelectedCandidates:
        from config import SELECT_TOP_K

        target = _goal_region(scene)

        def endpoint_dist(traj: Trajectory) -> float:
            end = traj.points[-1]
            return float(np.hypot(end.x - target.x, end.y - target.y))

        ranked = sorted(candidates, key=endpoint_dist)
        top = ranked[: max(1, SELECT_TOP_K)]
        ids = [t.id for t in top]
        return SelectedCandidates(
            ids=ids,
            reasoning=(
                f"Kept {len(ids)} trajectory(ies) whose endpoints are nearest "
                f"the goal region at ({target.x:.2f}, {target.y:.2f})."
            ),
        )

    # ---- Stage output: rationale ----
    def generate_rationale(self, image: Image.Image, plan: ActionPlan, goal: str) -> str:
        loc = plan.locations
        return (
            f"Selected trajectory T{plan.best_trajectory.id} to go from "
            f"{loc.place_a_label or 'start'} to {loc.place_b_label or 'goal'}. "
            f"Total cost {plan.cost.total_cost:.2f} "
            f"(goal_distance={plan.cost.goal_distance:.2f}, "
            f"collision_penalty={plan.cost.collision_penalty:.0f}, "
            f"path_length_penalty={plan.cost.path_length_penalty:.2f}). "
            f"It best satisfies {goal!r} among the evaluated candidates."
        )

    # ---- Debate ----
    def critique_plan(self, artifact: DebateArtifact, other_view: str | None) -> DebateTurn:
        base = VERDICTS[self.seed % 3]
        verdict = base
        reasoning = f"Base assessment from seed {self.seed}: {base}."

        if other_view is not None and self.concede:
            other = parse_verdict(other_view)
            if other is not None and CONSERVATISM[other] > CONSERVATISM[base]:
                verdict = other
                reasoning = (
                    f"Conceding from {base} to the more conservative {other} "
                    f"after seeing the other model's view."
                )

        amended = None
        if verdict == "amend":
            amended = f"Amended plan (text only): tighten path away from obstacles for {artifact.goal!r}."

        raw_reply = f"{reasoning}\nVERDICT: {verdict}"
        return DebateTurn(
            round=0,  # the relay sets the real round number
            model=self.name,
            verdict=verdict,
            amended_plan=amended,
            reasoning=reasoning,
            raw_reply=raw_reply,
        )
