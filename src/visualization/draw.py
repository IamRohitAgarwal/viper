"""All image overlays (SPEC sections 5, 8.2; golden rule 5/6).

This is the ONLY module that draws on images. Every function takes a PIL
image and returns a NEW one (never mutates in place). Colors and thickness
come from ``config`` — no hard-coded values.
"""

import math

from PIL import Image, ImageDraw, ImageFont

import config
from src.models import GroundedLocations, SceneObject, SceneUnderstanding, Trajectory

_ROLE_COLORS = {
    "target": config.COLOR_GOAL,
    "goal": config.COLOR_GOAL,
    "obstacle": config.COLOR_OBSTACLE,
    "agent": config.COLOR_SELECTED,
    "background": config.COLOR_REJECTED,
}


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(round(x * w)), int(round(y * h))


def _draw_arrowhead(draw, p0, p1, color, thickness):
    """Draw a small arrowhead at p1, pointing along p0->p1."""
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    size = 8
    for da in (math.radians(150), math.radians(-150)):
        ex = p1[0] + size * math.cos(angle + da)
        ey = p1[1] + size * math.sin(angle + da)
        draw.line([p1, (ex, ey)], fill=color, width=thickness)


def _draw_trajectory(draw, traj: Trajectory, color, w, h, thickness, label=None):
    pts = [_px(p.x, p.y, w, h) for p in traj.points]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=thickness)
        _draw_arrowhead(draw, pts[-2], pts[-1], color, thickness)
    if label:
        draw.text((pts[0][0] + 3, pts[0][1] - 10), label, fill=color, font=_font())


def _draw_bbox(draw, obj: SceneObject, w, h, color):
    x0, y0 = _px(obj.x, obj.y, w, h)
    x1, y1 = _px(obj.x + obj.w, obj.y + obj.h, w, h)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=config.ARROW_THICKNESS)
    draw.text((x0 + 2, y0 + 2), obj.name, fill=color, font=_font())


def draw_candidates(
    image: Image.Image,
    candidates: list[Trajectory],
    selected_ids: list[int] | None = None,
) -> Image.Image:
    """Draw every candidate trajectory, labelled T1..Tn.

    Selected ids -> green; if a selection is given, others -> grey; with no
    selection, all candidates -> blue.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    selected = set(selected_ids or [])
    for traj in candidates:
        if traj.id in selected:
            color = config.COLOR_SELECTED
        elif selected_ids is not None:
            color = config.COLOR_REJECTED
        else:
            color = config.COLOR_CANDIDATE
        _draw_trajectory(draw, traj, color, w, h, config.ARROW_THICKNESS, f"T{traj.id}")
    return out


def draw_selected(
    image: Image.Image,
    trajectory: Trajectory,
    locations: GroundedLocations | None = None,
) -> Image.Image:
    """Highlight the winning trajectory in green; mark A and B if given."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    _draw_trajectory(
        draw, trajectory, config.COLOR_SELECTED, w, h, config.ARROW_THICKNESS + 1,
        f"T{trajectory.id}",
    )
    if locations is not None:
        _mark_endpoints(draw, locations, w, h)
    return out


def draw_scene(image: Image.Image, scene: SceneUnderstanding) -> Image.Image:
    """Draw bounding boxes for detected objects, colour-coded by role."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for obj in scene.objects:
        color = _ROLE_COLORS.get(obj.role, config.COLOR_REJECTED)
        _draw_bbox(draw, obj, w, h, color)
    return out


def draw_final(
    image: Image.Image,
    trajectory: Trajectory,
    locations: GroundedLocations,
    scene: SceneUnderstanding,
) -> Image.Image:
    """Final annotated frame: winning path + A/B markers + obstacles."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for obj in scene.objects:
        if obj.role == "obstacle":
            _draw_bbox(draw, obj, w, h, config.COLOR_OBSTACLE)
    for obj in locations.obstacles:
        _draw_bbox(draw, obj, w, h, config.COLOR_OBSTACLE)
    _draw_trajectory(
        draw, trajectory, config.COLOR_SELECTED, w, h, config.ARROW_THICKNESS + 1,
        f"T{trajectory.id}",
    )
    _mark_endpoints(draw, locations, w, h)
    return out


def draw_agent(
    image: Image.Image,
    point,
    color: tuple | None = None,
    trajectory: Trajectory | None = None,
) -> Image.Image:
    """Draw the agent marker at ``point`` (optionally with its path so far).

    Used by the rollout and the traversal animation.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    color = color or config.COLOR_SELECTED
    if trajectory is not None:
        _draw_trajectory(draw, trajectory, config.COLOR_REJECTED, w, h, config.ARROW_THICKNESS)
    cx, cy = _px(point.x, point.y, w, h)
    r = 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return out


def _mark_endpoints(draw, locations: GroundedLocations, w, h):
    for point, label, color in (
        (locations.place_a, locations.place_a_label or "A", config.COLOR_CANDIDATE),
        (locations.place_b, locations.place_b_label or "B", config.COLOR_GOAL),
    ):
        cx, cy = _px(point.x, point.y, w, h)
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        draw.text((cx + r + 2, cy - 5), label, fill=color, font=_font())
