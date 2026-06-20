# VIPER — Verifiable Iterative Planning with Ensemble Reasoning

Fuses **PIVOT** (visual proposal) + **VLM** (visual reasoning) + **VLMPC**
(predictive validation) into a route-planning pipeline, then validates the
plan through a two-model **Claude ↔ Molmo** debate.

Input is a **video** of a physical scene and a goal of the form
`"from {place A} to {place B}"`. Output is a *validated traversal* — a route
animated as an overlay on the real footage, cross-checked by two independent
models.

> See `SPEC.md` (source of truth) and `plan.md` (build order).

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_demo_clip.py                         # synthetic demo clip
python main.py --video data/videos/demo_topdown.mp4 \
    --goal "from receiving bay to dock 4"                # plan a route
python -m pytest -q                                      # 61 tests, offline
```

Outputs land in `outputs/<run_id>/`: `candidates.png`, `selected.png`,
`final.png`, `trajectory.gif`, **`traversal.gif`** (the key A→B overlay), and
`log.json`. Set `DEBATE_ENABLED = True` in `config.py` to also write
`debate.txt`.

Phases 0–5 + 7 run **fully offline** on a deterministic mock/fallback VLM — no
GPU, no API key, no network. Phase 6 (real Claude + Molmo) requires a CUDA GPU
and runs on Kaggle/Colab only — see [`docs/KAGGLE.md`](docs/KAGGLE.md).

## Commands

| Command | What |
|---|---|
| `python main.py --video <clip> --goal "from A to B"` | plan one route |
| `python main.py --frames-dir data/frames --random` | plan on a random extracted frame |
| `python main.py --video <clip> --goal "..." --batch` | resumable metrics over all frames |
| `python scripts/tag_zones.py --video <clip>` | click-to-tag `zones.json` for a real clip |
| `python scripts/make_demo_clip.py` | regenerate the synthetic demo clip (H.264) |
| `python app.py` | optional Gradio UI (`pip install gradio` first) |

## Using a real clip

A real top-down clip needs a matching `data/zones.json` (the A/B/obstacle
labels) when using offline `pretagged` grounding. Tag it interactively:

```bash
python scripts/tag_zones.py --video data/videos/yourclip.mp4 \
    --place-a "receiving bay" --place-b "dock 4"
```

The `--place-a`/`--place-b` labels must match the words in your `--goal`. In
Phase 6 `vlm` grounding mode, the VLM locates named places directly and no
tagging is needed.

## Honest scope

- Input is real video, but the planner reasons on extracted single frames as
  a 2D-trajectory proxy — not full temporal control or a deployed robot.
- The forward rollout is a lightweight simulation (waypoint motion + bbox
  collision), not a learned video-prediction model.
- The A→B traversal is an **overlay animation**: a synthetic agent marker
  drawn moving over the real frames. It is not the real vehicle moving and is
  not robot control.
- In the debate, Molmo-7B-D is smaller than the Claude model; some
  disagreement reflects a capability gap, tracked via concession counts.
- Steep **top-down** footage is required for trajectory/cost validity; oblique
  footage degrades results.

## Data

Place a short (5–15s) top-down clip under `data/videos/`. Free,
no-attribution sources: Pexels, Pixabay, Coverr (search "warehouse top
view" / "warehouse aerial"). Academic top-down trajectory datasets: Stanford
Drone Dataset, inD/rounD/highD, VIRAT. Cite the source here.

A steep top-down (near-90°) angle is required for the 2D trajectory and cost
function to be valid; oblique/eye-level footage degrades results.
