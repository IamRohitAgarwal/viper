"""Stage 0 — video -> scene frames (SPEC section 8.0).

The planning pipeline operates on single images, so this one-time
preprocessing step samples frames from a video and stores them as PNGs.
"""

import glob
import os

import cv2

FRAMES_DIR = "data/frames"
_FRAME_GLOB = "frame_*.png"


def _frame_path(frames_dir: str, index: int) -> str:
    return os.path.join(frames_dir, f"frame_{index:04d}.png")


def extract_frames(
    video_path: str,
    frame_stride: int,
    max_frames: int,
    frames_dir: str = FRAMES_DIR,
) -> list[str]:
    """Sample 1 of every ``frame_stride`` frames (up to ``max_frames``).

    Writes ``frame_0001.png``, ``frame_0002.png``, ... to ``frames_dir`` and
    returns their paths. Idempotent: any pre-existing ``frame_*.png`` in the
    target directory is cleared first so a re-run overwrites cleanly.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")

    os.makedirs(frames_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(frames_dir, _FRAME_GLOB)):
        os.remove(stale)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    saved: list[str] = []
    try:
        src_index = 0
        while len(saved) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if src_index % frame_stride == 0:
                out = _frame_path(frames_dir, len(saved) + 1)
                cv2.imwrite(out, frame)
                saved.append(out)
            src_index += 1
    finally:
        cap.release()

    if not saved:
        raise RuntimeError(f"no frames extracted from: {video_path}")
    return saved


def list_frames(frames_dir: str = FRAMES_DIR) -> list[str]:
    """Return existing extracted frame paths, sorted by index."""
    return sorted(glob.glob(os.path.join(frames_dir, _FRAME_GLOB)))


def pick_random_frame(frames_dir: str, seed: int) -> str:
    """Pick a frame reproducibly (same seed -> same choice)."""
    import numpy as np

    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"no extracted frames in: {frames_dir}")
    rng = np.random.default_rng(seed)
    return frames[int(rng.integers(0, len(frames)))]
