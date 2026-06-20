"""VLMPC forward rollout (SPEC section 8.4).

Step-by-step simulation: move the agent along the trajectory waypoints,
check collisions against obstacle bboxes, accumulate path length, and record
a frame per step (for the rollout GIF).

The collision model is bbox containment along the path (resolved open
question: simplest bbox check first).
"""

import numpy as np

from src.models import Point, SceneObject, SceneUnderstanding, SimulationResult, Trajectory
from src.visualization import draw


def _point_in_bbox(p: Point, o: SceneObject) -> bool:
    return (o.x <= p.x <= o.x + o.w) and (o.y <= p.y <= o.y + o.h)


def simulate_trajectory(
    trajectory: Trajectory,
    scene: SceneUnderstanding,
    config,
    base_image=None,
) -> SimulationResult:
    """Roll out one trajectory.

    ``base_image`` is the frame the agent is drawn onto for the GIF; if None,
    a small blank canvas is used. Obstacles are read from ``scene.objects``
    (the runner injects grounded obstacles into the scene).
    """
    from PIL import Image

    if base_image is None:
        base_image = Image.new("RGB", (160, 120), (20, 20, 30))

    obstacles = [o for o in scene.objects if o.role == "obstacle"]
    pts = trajectory.points

    path_length = 0.0
    collision = False
    frames = []
    for k, p in enumerate(pts):
        if k > 0:
            prev = pts[k - 1]
            path_length += float(np.hypot(p.x - prev.x, p.y - prev.y))
        if any(_point_in_bbox(p, o) for o in obstacles):
            collision = True
        frames.append(draw.draw_agent(base_image, p, trajectory=trajectory))

    return SimulationResult(
        trajectory_id=trajectory.id,
        final_position=pts[-1],
        path_length=path_length,
        collision=collision,
        frames=frames,
    )
