"""Package an ActionPlan into the relayable DebateArtifact (SPEC section 9.1).

This is ALL the models see in the debate: the selected-trajectory image, the
goal, and human-readable summaries. NO pipeline internals, NO ground truth.
"""

from src.models import ActionPlan, DebateArtifact


def _trajectory_summary(plan: ActionPlan) -> str:
    loc = plan.locations
    pts = plan.best_trajectory.points
    return (
        f"Route T{plan.best_trajectory.id} from {loc.place_a_label} "
        f"({loc.place_a.x:.2f},{loc.place_a.y:.2f}) to {loc.place_b_label} "
        f"({loc.place_b.x:.2f},{loc.place_b.y:.2f}) via {len(pts)} waypoints."
    )


def _cost_summary(plan: ActionPlan) -> str:
    c = plan.cost
    collided = "yes" if c.collision_penalty > 0 else "no"
    return (
        f"Total cost {c.total_cost:.2f} (goal_distance={c.goal_distance:.3f}, "
        f"path_length_penalty={c.path_length_penalty:.3f}, collision={collided})."
    )


def _candidate_summary(plan: ActionPlan) -> str:
    others = [c for c in plan.all_costs if c.trajectory_id != plan.best_trajectory.id]
    if not others:
        return "No alternative candidates were evaluated."
    parts = [
        f"T{c.trajectory_id} (cost {c.total_cost:.2f}"
        + (", collided" if c.collision_penalty > 0 else "")
        + ")"
        for c in others
    ]
    return "Alternatives that lost: " + ", ".join(parts) + "."


def package(plan: ActionPlan, goal: str) -> DebateArtifact:
    return DebateArtifact(
        selected_image=plan.selected_image,
        goal=goal,
        trajectory_summary=_trajectory_summary(plan),
        cost_summary=_cost_summary(plan),
        candidate_summary=_candidate_summary(plan),
    )
