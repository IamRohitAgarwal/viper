"""Phase 3 — visualization + animation tests (SPEC section 13 Phase 3)."""

import os

import pytest
from PIL import Image

import config
from src.models import (
    GroundedLocations,
    Point,
    SceneObject,
    SceneUnderstanding,
    Trajectory,
)
from src.visualization import draw
from src.visualization.animate import generate_gif

SIZE = (80, 60)


def _img():
    return Image.new("RGB", SIZE, (0, 0, 0))


def _candidates(n=4):
    return [
        Trajectory(id=i + 1, points=[Point(0.1, 0.1), Point(0.2 + 0.15 * i, 0.8)])
        for i in range(n)
    ]


def _scene():
    return SceneUnderstanding(
        objects=[
            SceneObject("t", 0.1, 0.1, 0.1, 0.1, "target"),
            SceneObject("g", 0.8, 0.8, 0.1, 0.1, "goal"),
            SceneObject("o", 0.45, 0.45, 0.12, 0.18, "obstacle"),
        ],
        description="d",
        goal_interpretation="g",
    )


def _locations():
    return GroundedLocations(
        place_a=Point(0.1, 0.1),
        place_b=Point(0.85, 0.85),
        place_a_label="bay",
        place_b_label="dock",
        obstacles=[SceneObject("pallet", 0.45, 0.45, 0.12, 0.18, "obstacle")],
    )


def _is_new_image(out, size=SIZE):
    assert isinstance(out, Image.Image)
    assert out.size == size


def _has_color(img, color):
    return color in {px[:3] for px in img.convert("RGB").getdata()}


def test_draw_candidates_returns_new_image():
    out = draw.draw_candidates(_img(), _candidates())
    _is_new_image(out)
    assert _has_color(out, config.COLOR_CANDIDATE)


def test_draw_candidates_selection_colors():
    out = draw.draw_candidates(_img(), _candidates(), selected_ids=[1])
    assert _has_color(out, config.COLOR_SELECTED)   # the selected one
    assert _has_color(out, config.COLOR_REJECTED)   # the others


def test_draw_candidates_does_not_mutate_input():
    src = _img()
    before = list(src.getdata())
    draw.draw_candidates(src, _candidates())
    assert list(src.getdata()) == before


def test_draw_selected():
    out = draw.draw_selected(_img(), _candidates()[0], _locations())
    _is_new_image(out)
    assert _has_color(out, config.COLOR_SELECTED)


def test_draw_scene():
    out = draw.draw_scene(_img(), _scene())
    _is_new_image(out)
    assert _has_color(out, config.COLOR_OBSTACLE)


def test_draw_final():
    out = draw.draw_final(_img(), _candidates()[0], _locations(), _scene())
    _is_new_image(out)
    assert _has_color(out, config.COLOR_SELECTED)
    assert _has_color(out, config.COLOR_OBSTACLE)


def test_generate_gif_creates_file(tmp_path):
    frames = [Image.new("RGB", SIZE, (i * 20, 0, 0)) for i in range(5)]
    path = str(tmp_path / "roll.gif")
    out = generate_gif(frames, path)
    assert out == path
    assert os.path.exists(path)
    with Image.open(path) as gif:
        assert getattr(gif, "n_frames", 1) == 5


def test_generate_gif_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_gif([], str(tmp_path / "x.gif"))
