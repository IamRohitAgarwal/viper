"""Phase 1 — frame ingestion tests (SPEC section 13 Phase 1).

Uses a synthetic video generated in-test so the suite needs no bundled
media. To exercise a real clip instead, drop a file in data/videos/ and
point SAMPLE_VIDEO at it (see _resolve_video).
"""

import os

import cv2
import numpy as np
import pytest

from src.ingest import frames as frames_mod

# Optional: set to a real clip path to test against actual footage.
SAMPLE_VIDEO: str | None = None


def _make_synthetic_video(path: str, n_frames: int = 60, size: int = 64) -> str:
    """Write a tiny .mp4 with a moving square, so frames visibly differ."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (size, size))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = (i * 2) % (size - 10)
        frame[10:20, x : x + 10] = (0, 255, 0)
        writer.write(frame)
    writer.release()
    return path


def _resolve_video(tmp_path) -> str:
    if SAMPLE_VIDEO and os.path.exists(SAMPLE_VIDEO):
        return SAMPLE_VIDEO
    return _make_synthetic_video(str(tmp_path / "synthetic.mp4"))


def test_extract_frames_writes_pngs(tmp_path):
    video = _resolve_video(tmp_path)
    out_dir = str(tmp_path / "frames")
    paths = frames_mod.extract_frames(
        video, frame_stride=10, max_frames=5, frames_dir=out_dir
    )
    assert len(paths) >= 1
    assert len(paths) <= 5
    for p in paths:
        assert os.path.exists(p)
        assert p.endswith(".png")


def test_extract_frames_respects_max(tmp_path):
    video = _resolve_video(tmp_path)
    out_dir = str(tmp_path / "frames")
    paths = frames_mod.extract_frames(
        video, frame_stride=1, max_frames=3, frames_dir=out_dir
    )
    assert len(paths) == 3


def test_extract_frames_idempotent(tmp_path):
    video = _resolve_video(tmp_path)
    out_dir = str(tmp_path / "frames")
    first = frames_mod.extract_frames(video, 5, 4, frames_dir=out_dir)
    second = frames_mod.extract_frames(video, 5, 2, frames_dir=out_dir)
    # Re-run cleared stale frames: only the new 2 remain on disk.
    assert second == first[:2]
    assert len(frames_mod.list_frames(out_dir)) == 2


def test_pick_random_frame_reproducible(tmp_path):
    video = _resolve_video(tmp_path)
    out_dir = str(tmp_path / "frames")
    frames_mod.extract_frames(video, 2, 5, frames_dir=out_dir)
    a = frames_mod.pick_random_frame(out_dir, seed=42)
    b = frames_mod.pick_random_frame(out_dir, seed=42)
    assert a == b


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        frames_mod.extract_frames("does_not_exist.mp4", 1, 1)


def test_empty_frames_dir_random_raises(tmp_path):
    with pytest.raises(RuntimeError):
        frames_mod.pick_random_frame(str(tmp_path / "empty"), seed=42)
