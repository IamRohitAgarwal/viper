"""Tests for the zone-tagging helper's pure geometry + grounding round-trip."""

import json

import pytest

from scripts.tag_zones import build_zones, obstacle_box, place_box
from src.grounding.locate import ground_locations

from .conftest import make_cfg


def test_place_box_centered():
    box = place_box(0.5, 0.4, 0.10)
    assert box["x"] == pytest.approx(0.45)
    assert box["y"] == pytest.approx(0.35)
    assert box["w"] == pytest.approx(0.10)
    assert box["h"] == pytest.approx(0.10)


def test_obstacle_box_from_corners():
    box = obstacle_box((0.6, 0.5), (0.4, 0.3), "obstacle_0")
    assert box["x"] == pytest.approx(0.4)
    assert box["y"] == pytest.approx(0.3)
    assert box["w"] == pytest.approx(0.2)
    assert box["h"] == pytest.approx(0.2)
    assert box["name"] == "obstacle_0"


def test_build_zones_roundtrips_through_grounding(tmp_path):
    zones = build_zones(
        clip="c.mp4",
        a=(0.15, 0.20),
        b=(0.80, 0.75),
        obstacle_corners=[((0.4, 0.3), (0.55, 0.5))],
        place_size=0.10,
        label_a="receiving bay",
        label_b="dock 4",
    )
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zones), encoding="utf-8")

    cfg = make_cfg(GROUNDING_MODE="pretagged", PRETAG_FILE=str(path))
    loc = ground_locations(None, "receiving bay", "dock 4", cfg)
    assert loc.place_a_label == "receiving bay"
    assert loc.place_b_label == "dock 4"
    assert len(loc.obstacles) == 1
    assert loc.obstacles[0].role == "obstacle"
