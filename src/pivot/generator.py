"""PIVOT candidate trajectory generation (SPEC section 8.1).

Randomized (NOT learned) sampling, seeded by ``config.SEED``. Each
trajectory STARTS at ``locations.place_a`` and aims at ``locations.place_b``,
with sampled intermediate waypoints so candidates differ. Points are clipped
to [0.05, 0.95].
"""

import numpy as np

from src.models import GroundedLocations, Point, Trajectory

_CLIP_LO = 0.05
_CLIP_HI = 0.95


def generate_candidates(
    image,
    locations: GroundedLocations,
    num_candidates: int,
    config,
) -> list[Trajectory]:
    """Sample ``num_candidates`` trajectories from A toward B.

    ``image`` is unused here (kept for interface symmetry / future use).
    """
    rng = np.random.default_rng(config.SEED)
    n_pts = max(2, config.MAX_TRAJECTORY_LEN)
    a = np.array([locations.place_a.x, locations.place_a.y])
    b = np.array([locations.place_b.x, locations.place_b.y])

    # Perpendicular direction for lateral spread.
    direction = b - a
    norm = np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]]) / norm if norm > 1e-6 else np.array([0.0, 1.0])

    candidates: list[Trajectory] = []
    ts = np.linspace(0.0, 1.0, n_pts)
    for i in range(num_candidates):
        # Lateral offset profile: zero at A and B, max in the middle.
        amplitude = rng.uniform(-0.25, 0.25)
        bow = np.sin(ts * np.pi) * amplitude
        jitter = rng.normal(0.0, 0.015, size=(n_pts, 2))
        jitter[0] = 0.0          # pin the start exactly at A
        points = []
        for k, t in enumerate(ts):
            base = a + direction * t + perp * bow[k]
            p = np.clip(base + jitter[k], _CLIP_LO, _CLIP_HI)
            points.append(Point(float(p[0]), float(p[1])))
        # Ensure the start is exactly A (post-clip safety).
        points[0] = Point(
            float(np.clip(a[0], _CLIP_LO, _CLIP_HI)),
            float(np.clip(a[1], _CLIP_LO, _CLIP_HI)),
        )
        candidates.append(Trajectory(id=i + 1, points=points, action_type="move"))
    return candidates
