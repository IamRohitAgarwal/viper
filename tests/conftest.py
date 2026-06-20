"""Shared Phase 4 test fixtures: synthetic frames + a zones.json."""

import json
import types

import pytest
from PIL import Image

import config as base_config


def make_cfg(**overrides):
    """A mutable config namespace copied from the real config module."""
    keys = [k for k in dir(base_config) if k.isupper()]
    ns = types.SimpleNamespace(**{k: getattr(base_config, k) for k in keys})
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def frames_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    paths = []
    for i in range(5):
        img = Image.new("RGB", (120, 90), (10 * i, 20, 40))
        p = d / f"frame_{i + 1:04d}.png"
        img.save(p)
        paths.append(str(p))
    return paths


@pytest.fixture
def zones_file(tmp_path):
    zones = {
        "clip": "synthetic.mp4",
        "places": {
            "receiving bay": {"x": 0.10, "y": 0.15, "w": 0.10, "h": 0.10},
            "dock 4": {"x": 0.78, "y": 0.75, "w": 0.10, "h": 0.10},
        },
        "obstacles": [
            {"name": "pallet stack", "x": 0.42, "y": 0.40, "w": 0.14, "h": 0.18}
        ],
    }
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(zones), encoding="utf-8")
    return str(p)
