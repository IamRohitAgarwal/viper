"""Tests for the generic goal parser + spatial grounding (real-video prompts)."""

import pytest

from main import parse_goal
from src.grounding.locate import _match_spatial, ground_locations
from src.models import Point

from .conftest import make_cfg


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("from receiving bay to dock 4", ("receiving bay", "dock 4")),
        ("Move the red block to the bottom left", ("red block", "bottom left")),
        ("move the bottle to the sink", ("bottle", "sink")),
        ("take the box to the loading dock", ("box", "loading dock")),
        ("put the cup on the table", ("cup", "table")),
        ("push the cart to the right", ("cart", "right")),
        ("go to the exit", ("", "exit")),
        ("navigate to the top right", ("", "top right")),
        ("the blue bin to the corner", ("blue bin", "corner")),
    ],
)
def test_parse_goal_variants(goal, expected):
    assert parse_goal(goal) == expected


def test_parse_goal_empty():
    assert parse_goal("") == ("", "")
    assert parse_goal(None) == ("", "")


def test_match_spatial():
    assert _match_spatial("the bottom left") == Point(0.15, 0.85)
    assert _match_spatial("TOP RIGHT corner") == Point(0.85, 0.15)
    assert _match_spatial("center") == Point(0.5, 0.5)
    assert _match_spatial("the red block") is None   # not spatial


class _FakeVLM:
    """Minimal VLM for grounding tests: points objects, returns no obstacles."""

    name = "fake"

    def locate_points(self, frame, labels):
        return {labels[0]: Point(0.3, 0.4)}   # pretend the object is here

    def understand_scene(self, frame, goal):
        from src.models import SceneUnderstanding

        return SceneUnderstanding(objects=[], description="d", goal_interpretation=goal)


def test_vlm_grounding_spatial_dest_and_object_start():
    cfg = make_cfg(GROUNDING_MODE="vlm")
    loc = ground_locations(None, "red block", "bottom left", cfg, vlm=_FakeVLM())
    # destination resolved by spatial lexicon (precise, no VLM call)
    assert loc.place_b == Point(0.15, 0.85)
    # start ("red block") resolved by the VLM's pointing
    assert loc.place_a == Point(0.3, 0.4)
    assert loc.place_b_label == "bottom left"
