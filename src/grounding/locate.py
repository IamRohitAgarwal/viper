"""Location grounding (SPEC section 8.0b).

Resolves the two named places (and obstacles) to points on the frame.
- ``pretagged`` mode (offline, free): read labeled regions from a zones
  JSON file (one-time manual tag per demo clip). Used in Phases 0-5.
- ``vlm`` mode (real models): the VLM returns coordinates. Phase 6.
"""

import json

from src.models import GroundedLocations, Point, SceneObject


def _center(region: dict) -> Point:
    return Point(region["x"] + region["w"] / 2, region["y"] + region["h"] / 2)


def _match_place(label: str, places: dict) -> str | None:
    """Case-insensitive match of a label against the zones place keys."""
    label = label.strip().lower()
    for key in places:
        if key.strip().lower() == label:
            return key
    return None


def ground_locations(
    frame,
    place_a_text: str,
    place_b_text: str,
    config,
    vlm=None,
    pretag_file: str | None = None,
) -> GroundedLocations:
    """Resolve place A, place B, and obstacles for a frame.

    ``frame`` is unused in pretagged mode (kept for the vlm-mode signature).
    ``pretag_file`` overrides ``config.PRETAG_FILE`` (useful for tests).
    """
    mode = config.GROUNDING_MODE

    if mode == "pretagged":
        path = pretag_file or config.PRETAG_FILE
        with open(path, encoding="utf-8") as fh:
            zones = json.load(fh)
        places = zones.get("places", {})

        # Destination is required.
        if not place_b_text:
            raise ValueError("no destination (place B) given for grounding")
        b_key = _match_place(place_b_text, places)
        if b_key is None:
            raise ValueError(
                f"destination {place_b_text!r} not found in {path}; "
                f"available places: {sorted(places)}"
            )
        place_b = _center(places[b_key])
        b_label = b_key

        # Start: fall back to image centre when no place A was parsed (Q5).
        if place_a_text:
            a_key = _match_place(place_a_text, places)
            if a_key is None:
                raise ValueError(
                    f"start {place_a_text!r} not found in {path}; "
                    f"available places: {sorted(places)}"
                )
            place_a = _center(places[a_key])
            a_label = a_key
        else:
            place_a = Point(0.5, 0.5)
            a_label = "start"

        obstacles = [
            SceneObject(
                name=o.get("name", f"obstacle_{i}"),
                x=o["x"], y=o["y"], w=o["w"], h=o["h"], role="obstacle",
            )
            for i, o in enumerate(zones.get("obstacles", []))
        ]
        return GroundedLocations(place_a, place_b, a_label, b_label, obstacles)

    if mode == "vlm":
        # Phase 6: the VLM grounds named places via its point output.
        if vlm is None or not hasattr(vlm, "locate_points"):
            raise ValueError("vlm grounding mode requires a VLM with locate_points()")
        if not place_b_text:
            raise ValueError("no destination (place B) given for grounding")
        labels = [place_b_text] + ([place_a_text] if place_a_text else [])
        points = vlm.locate_points(frame, labels)
        if place_b_text not in points:
            raise ValueError(f"VLM could not locate destination {place_b_text!r}")
        place_b = points[place_b_text]
        if place_a_text and place_a_text in points:
            place_a, a_label = points[place_a_text], place_a_text
        else:
            place_a, a_label = Point(0.5, 0.5), "start"
        scene = vlm.understand_scene(frame, f"from {place_a_text} to {place_b_text}")
        obstacles = [o for o in scene.objects if o.role == "obstacle"]
        return GroundedLocations(place_a, place_b, a_label, place_b_text, obstacles)

    raise ValueError(f"unknown GROUNDING_MODE: {mode!r}")
