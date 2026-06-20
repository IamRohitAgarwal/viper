"""Interactive zone tagger: click A/B/obstacles on a frame -> data/zones.json.

Pretagged grounding (Phases 0-5) needs a zones.json giving the normalised
[0,1] regions for place A, place B, and obstacles on a clip. This tool shows
the first frame and lets you click them, then writes the JSON.

Usage:
    python scripts/tag_zones.py --video data/videos/myclip.mp4 \
        --place-a "receiving bay" --place-b "dock 4" --out data/zones.json

Interaction:
    1. Single-click the START (place A).
    2. Single-click the DESTINATION (place B).
    3. For each obstacle: click two opposite corners to draw its box.
    4. Close the window when done.
"""

import argparse
import json
import os


# ---------- pure, testable helpers ----------
def place_box(cx: float, cy: float, size: float) -> dict:
    """Square region centred on a normalised click."""
    return {"x": cx - size / 2, "y": cy - size / 2, "w": size, "h": size}


def obstacle_box(p1: tuple[float, float], p2: tuple[float, float], name: str) -> dict:
    """Axis-aligned box from two opposite normalised corners."""
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    return {"name": name, "x": x, "y": y, "w": abs(p2[0] - p1[0]), "h": abs(p2[1] - p1[1])}


def build_zones(clip, a, b, obstacle_corners, place_size, label_a, label_b) -> dict:
    """Assemble the zones dict from normalised inputs."""
    return {
        "clip": clip,
        "places": {
            label_a: place_box(a[0], a[1], place_size),
            label_b: place_box(b[0], b[1], place_size),
        },
        "obstacles": [
            obstacle_box(c1, c2, f"obstacle_{i}")
            for i, (c1, c2) in enumerate(obstacle_corners)
        ],
    }


def load_first_frame(path: str):
    """Return (rgb_array, width, height, clip_name) from a video/dir/image."""
    import cv2

    if os.path.isdir(path):
        from src.ingest.frames import list_frames

        frames = list_frames(path)
        if not frames:
            raise RuntimeError(f"no frames in {path}")
        bgr = cv2.imread(frames[0])
        clip = os.path.basename(frames[0])
    elif path.lower().endswith((".png", ".jpg", ".jpeg")):
        bgr = cv2.imread(path)
        clip = os.path.basename(path)
    else:
        cap = cv2.VideoCapture(path)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read first frame of {path}")
        clip = os.path.basename(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return rgb, w, h, clip


# ---------- interactive tagging ----------
def tag_interactive(path, label_a, label_b, place_size, out):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    rgb, w, h, clip = load_first_frame(path)
    state = {"a": None, "b": None, "pending_corner": None, "obstacles": []}

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    title = ax.set_title("Click START (place A)")

    def norm(event):
        return (event.xdata / w, event.ydata / h)

    def redraw():
        if state["a"] is None:
            title.set_text("Click START (place A)")
        elif state["b"] is None:
            title.set_text(f"Click DESTINATION ({label_b})")
        else:
            title.set_text("Drag obstacle corners (2 clicks each). Close window to finish.")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        p = norm(event)
        if state["a"] is None:
            state["a"] = p
            _draw_box(ax, patches, place_box(p[0], p[1], place_size), w, h, "tab:blue", label_a)
        elif state["b"] is None:
            state["b"] = p
            _draw_box(ax, patches, place_box(p[0], p[1], place_size), w, h, "gold", label_b)
        elif state["pending_corner"] is None:
            state["pending_corner"] = p
        else:
            c1, c2 = state["pending_corner"], p
            state["obstacles"].append((c1, c2))
            state["pending_corner"] = None
            box = obstacle_box(c1, c2, f"obstacle_{len(state['obstacles']) - 1}")
            _draw_box(ax, patches, box, w, h, "red", box["name"])
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    redraw()
    plt.show()

    if state["a"] is None or state["b"] is None:
        raise SystemExit("Tagging cancelled: need at least START and DESTINATION.")

    zones = build_zones(clip, state["a"], state["b"], state["obstacles"], place_size, label_a, label_b)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(zones, fh, indent=2)
    print(f"wrote {out}:")
    print(json.dumps(zones, indent=2))


def _draw_box(ax, patches, box, w, h, color, label):
    rect = patches.Rectangle(
        (box["x"] * w, box["y"] * h), box["w"] * w, box["h"] * h,
        linewidth=2, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(box["x"] * w, box["y"] * h - 4, label, color=color, fontsize=9)
    ax.figure.canvas.draw_idle()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Click-to-tag zones.json for a clip.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file (first frame is used)")
    src.add_argument("--frames-dir", help="directory of extracted frames")
    src.add_argument("--image", help="a single image frame")
    parser.add_argument("--place-a", default="receiving bay")
    parser.add_argument("--place-b", default="dock 4")
    parser.add_argument("--place-size", type=float, default=0.10, help="place box side (normalised)")
    parser.add_argument("--out", default="data/zones.json")
    args = parser.parse_args(argv)

    path = args.video or args.frames_dir or args.image
    tag_interactive(path, args.place_a, args.place_b, args.place_size, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
