"""GIF generation (SPEC section 11).

Phase 3 provides ``generate_gif`` (rollout GIF). ``generate_traversal`` —
the agent marker moving A->B over real frames — is added in Phase 4.
"""

import os

import numpy as np
from PIL import Image

import config
from src.models import GroundedLocations, Point, Trajectory
from src.visualization import draw


def generate_gif(frames: list[Image.Image], path: str, fps: int | None = None) -> str:
    """Write ``frames`` to an animated GIF at ``path``; return ``path``."""
    if not frames:
        raise ValueError("generate_gif requires at least one frame")
    fps = fps or config.GIF_FPS
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rgb = [f.convert("RGB") for f in frames]
    duration_ms = int(1000 / fps)
    rgb[0].save(
        path,
        save_all=True,
        append_images=rgb[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _interpolate(trajectory: Trajectory, n: int) -> list[Point]:
    """Resample a trajectory to ``n`` evenly spaced points along its path."""
    pts = np.array([[p.x, p.y] for p in trajectory.points])
    if len(pts) == 1:
        return [Point(float(pts[0][0]), float(pts[0][1]))] * n
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 1e-9 else 1.0
    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return [Point(float(x), float(y)) for x, y in zip(xs, ys)]


def _load(frame) -> Image.Image:
    return Image.open(frame).convert("RGB") if isinstance(frame, str) else frame.convert("RGB")


def generate_traversal(
    real_frames: list,
    trajectory: Trajectory,
    locations: GroundedLocations,
    config,
    path: str,
) -> str:
    """Animate the agent moving A->B as an overlay on successive real frames.

    ``real_frames`` may be file paths or PIL images. The agent marker steps
    along ``trajectory`` over ``config.TRAVERSAL_FRAMES`` steps, each drawn on
    the next real frame (cycling if there are fewer frames than steps).
    """
    if not real_frames:
        raise ValueError("generate_traversal requires at least one real frame")
    n = max(2, config.TRAVERSAL_FRAMES)
    agent_path = _interpolate(trajectory, n)
    out_frames = []
    for i, p in enumerate(agent_path):
        base = _load(real_frames[i % len(real_frames)])
        out_frames.append(draw.draw_agent(base, p, trajectory=trajectory))
    return generate_gif(out_frames, path)
