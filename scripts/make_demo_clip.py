"""Regenerate the synthetic top-down demo clip used by the Phase 4 gate.

Writes browser-playable H.264 via imageio-ffmpeg (so it previews in the
Gradio UI). Falls back to OpenCV's mp4v if ffmpeg is unavailable — note that
mp4v clips will NOT preview in a browser, though the pipeline still reads them.

The real clip lives under data/videos/ (gitignored). Run:
    python scripts/make_demo_clip.py
Matches the zones in data/zones.json (receiving bay / dock 4 / pallet stack).
"""

import numpy as np

OUT = "data/videos/demo_topdown.mp4"
N_FRAMES = 80
FPS = 20


def _frames():
    bg = np.full((240, 320, 3), (50, 55, 60), np.uint8)  # RGB
    bg[30:70, 30:70] = (90, 150, 90)        # receiving bay
    bg[170:210, 245:285] = (180, 90, 90)    # dock 4
    bg[95:140, 130:180] = (160, 40, 40)     # pallet obstacle
    for i in range(N_FRAMES):
        frame = bg.copy()
        x = min(30 + int(2.5 * i), 290)
        y = 40 + i
        frame[max(0, y - 7):y + 7, max(0, x - 7):x + 7] = (255, 230, 0)  # moving agent
        yield frame


def main() -> None:
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(OUT, fps=FPS, codec="libx264", macro_block_size=16) as w:
            for f in _frames():
                w.append_data(f)
        print(f"wrote {OUT} (H.264, browser-playable)")
    except Exception as exc:  # noqa: BLE001 - fall back to OpenCV mp4v
        import cv2

        writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (320, 240))
        for f in _frames():
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"wrote {OUT} (mp4v fallback; NOT browser-playable). Reason: {exc}")


if __name__ == "__main__":
    main()
