"""VLMPC cost function (SPEC section 8.5).

total = GOAL_DISTANCE_WEIGHT * goal_distance
        + (COLLISION_PENALTY if collision else 0)
        + PATH_LENGTH_WEIGHT * path_length
"""

import numpy as np

from src.models import CostBreakdown, GroundedLocations, SimulationResult


def compute_cost(
    sim_result: SimulationResult,
    locations: GroundedLocations,
    config,
) -> CostBreakdown:
    final = sim_result.final_position
    goal_distance = float(
        np.hypot(final.x - locations.place_b.x, final.y - locations.place_b.y)
    )
    collision_penalty = config.COLLISION_PENALTY if sim_result.collision else 0.0
    path_length_penalty = config.PATH_LENGTH_WEIGHT * sim_result.path_length
    total = (
        config.GOAL_DISTANCE_WEIGHT * goal_distance
        + collision_penalty
        + path_length_penalty
    )
    return CostBreakdown(
        trajectory_id=sim_result.trajectory_id,
        goal_distance=goal_distance,
        collision_penalty=float(collision_penalty),
        path_length_penalty=float(path_length_penalty),
        total_cost=float(total),
    )
