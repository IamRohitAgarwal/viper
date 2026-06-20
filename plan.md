# plan.md — VIPER build plan for Claude Code

> **Companion to `SPEC.md`.** SPEC.md is the source of truth for *what* to build and *what "correct" means*. This plan.md is the *execution order*: what to do, in what sequence, and the exact gate that must pass before moving on.
> **Read `SPEC.md` fully before starting any phase.** If anything here conflicts with SPEC.md, SPEC.md wins.

**VIPER** = **V**erifiable **I**terative **P**lanning with **E**nsemble **R**easoning.
*PIVOT (proposal) + VLM (reasoning) + VLMPC (validation), then a Claude<->Molmo debate. Input is a video.*

---

## How to work through this plan

1. Build **one phase at a time**, top to bottom. Do not start a phase until the previous phase's gate passes.
2. After each phase, **run its gate** and confirm the "Done when" criterion, then stop and report status before continuing.
3. Build the **simplest thing that passes the gate**. Anything marked `[OPTIONAL]` waits until the base system works end-to-end.
4. **Single sources of truth:** data shapes only from `src/models.py`; VLM calls only through `src/vlm/interface.py`; drawing only through `src/visualization/draw.py`; tunables only in `config.py`.
5. The three layers are independent: **GENERATE** must run and be testable with the mock/offline VLM *before* **DEBATE** is touched.

---

## Architecture recap (so the build order makes sense)

```
  video + goal
      |
      v
  [Stage 0] extract frames  ---------------------------------+
      |                                                       |
      v                                                       |
+--------------------------------------------------+          |
| LAYER 1 -- GENERATE  (per frame)                 |          |
|   GROUND : locate place A, place B, obstacles    |          |
|   PIVOT  : candidate trajectories  A -> B         |          |
|   draw   : annotate frame (T1..Tn)               |          |
|   VLM    : shortlist promising candidates        |          |
|   VLMPC  : simulate + cost(dist to B) + pick best|          |
|   ANIM   : traverse A->B over real frames        |          |
|   -> ActionPlan (+ candidates/selected/traversal)|          |
+--------------------------------------------------+          |
      |  (packaged as DebateArtifact)                         |
      v                                                       |
+--------------------------------------------------+          |
| LAYER 2 -- DEBATE                                |          |
|   relay: Claude -> Molmo -> Claude ... converge  |          |
|   -> DebateResult (verdict + transcript)         |          |
+--------------------------------------------------+          |
      |                                                       |
      v                                                       |
  ViperResult  -> outputs/<run_id>/ (+ log.json) <------------+
```

Build order is **bottom-up**: shared types -> ingestion -> mock brain -> drawing/GIF -> GENERATE pipeline -> DEBATE (still on mocks) -> real models -> evaluation/UI.

---

## Phase 0 -- Scaffold
**Goal:** an empty-but-running skeleton with the CLI.
Tasks:
- Full directory tree from SPEC section 5, with `__init__.py` in every package.
- `requirements.txt`: opencv-python, pillow, imageio, numpy, matplotlib, anthropic, transformers, gradio, pytest, ruff.
- `config.py` with ALL defaults from SPEC section 7 (frame, PIVOT, VLM, VLMPC cost, DEBATE, visualization).
- `src/models.py` with EVERY dataclass from SPEC section 6 (both layers).
- `main.py` CLI shell parsing `--video`, `--frames-dir`, `--goal`, `--frame N`, `--random` (stubbed actions).
**Gate:** `pip install -r requirements.txt` succeeds; `python main.py --help` prints the options; no import errors.

## Phase 1 -- Frame ingestion (Stage 0)
**Goal:** turn a video into scene frames.
Tasks:
- `src/ingest/frames.py: extract_frames(video_path, frame_stride, max_frames) -> list[str]` using `cv2.VideoCapture`; write `data/frames/frame_0001.png` ...
- Wire `main.py --video ... ` to call it; wire `--frames-dir --random` to pick an existing frame.
- `tests/test_frames.py`: generate a tiny synthetic video in the test (or use a bundled sample), assert N PNGs written.
**Gate:** `extract_frames` produces PNGs from a video; `--frames-dir --random` selects one. `pytest tests/test_frames.py` passes.

## Phase 2 -- VLM interface + Mock/offline VLM
**Goal:** a deterministic fake brain that also serves as the offline fallback.
Tasks:
- `src/vlm/interface.py`: `VLMInterface` ABC with `understand_scene`, `select_candidates`, `generate_rationale`, and `critique_plan` (SPEC section 10).
- `src/vlm/mock_vlm.py`: deterministic (numpy `default_rng(SEED)`); also the rule-based fallback used when `USE_VLM=False`. `critique_plan` endorses/amends by a simple deterministic rule.
- `tests/test_mock_vlm.py`: same seed -> identical outputs; correct return types.
**Gate:** `pytest tests/test_mock_vlm.py` passes; two same-seed runs identical.

## Phase 3 -- Visualization + animation
**Goal:** all drawing and the GIF, in one place.
Tasks:
- `src/visualization/draw.py`: draw candidates (labelled T1..Tn), highlight selected, draw scene roles, final annotated frame. Each returns a NEW PIL image; colors from config.
- `src/visualization/animate.py: generate_gif(frames, path) -> str`.
- `tests/test_draw.py`: outputs are PIL images of expected size; GIF file is created.
**Gate:** `pytest tests/test_draw.py` passes; a sample GIF opens and shows motion.

## Phase 4 -- GENERATE pipeline (grounding + PIVOT + VLM + VLMPC + traversal)
**Goal:** the full A->B pipeline runs end-to-end on the offline VLM and writes all outputs, including the traversal GIF.
Build in dependency order, each with a test (SPEC section 8):
1. `grounding/locate.py` (pretagged) -- read `data/zones.json`; return `GroundedLocations` (A, B, obstacles). Test: known zones -> correct points.
2. `pivot/generator.py` -- candidates START at A, aim at B; seeded; clipped to [0.05,0.95]; ids T1..Tn.
3. `pivot/visual_prompt.py` -- delegates to draw; returns `AnnotatedProposal`.
4. selector (fallback) -- shortlist sane; returns `SelectedCandidates`.
5. `vlmpc/rollout.py` -- step rollout; collision flag on obstacle overlap; records frames.
6. `vlmpc/cost_function.py` -- goal_dist = distance to B; full `CostBreakdown`.
7. `vlmpc/validator.py` -- returns min-cost trajectory id.
8. `visualization/animate.py: generate_traversal(...)` -- agent marker moves A->B overlaid on successive real frames -> `traversal.gif`.
9. runner (`run_pipeline`) -- grounds -> candidates -> select -> simulate -> cost -> best -> traversal -> `ActionPlan`.
Also hand-create a small `data/zones.json` for one demo clip (A/B/obstacle tags).
**Gate:** `pytest tests/` (pipeline) green; `python main.py --video <clip> --goal "from A to B"` writes candidates.png, selected.png, trajectory.gif, **traversal.gif**, final.png, log.json; the traversal GIF shows an agent moving A->B over the real footage.

## Phase 5 -- DEBATE layer (on mocks)
**Goal:** the Claude->Molmo relay runs deterministically using MockVLM for BOTH debaters (different seeds).
Tasks (SPEC section 9):
- `debate/artifact.py: package(plan, goal) -> DebateArtifact` (no internals, no ground truth).
- `debate/relay.py: run_debate(...)` -- relay, convergence on matching verdicts, conservative tie-break (reject>amend>endorse), concession tracking, `VERDICT:` parse + 1 retry, transcript appended live, round1_solo recorded.
- `debate/verdict.py: assemble(...)` -- final plan text + loser reasoning.
- `run_viper` chaining GENERATE -> package -> DEBATE, gated by `DEBATE_ENABLED`.
- `tests/test_relay.py` + `tests/test_verdict.py`.
**Gate:** those tests pass; with `DEBATE_ENABLED=True` a `debate.txt` (transcript + verdict + concessions) is written; deadlock yields the conservative pick.

## Phase 6 -- Real model swap-in `[after Phase 5 stable]`
**Goal:** real Claude vision and a real Claude<->Molmo debate.
Tasks (SPEC section 13 Phase 6):
- `vlm/anthropic_vlm.py` + `vlm/molmo_vlm.py` implementing the full interface incl. `critique_plan`.
- Molmo loaded ONCE, 4-bit (CUDA only, `trust_remote_code=True`), reused; Claude images base64 PNG, downscaled; keys from env / Kaggle Secret.
- Confirm `VLM_BACKEND` and `DEBATE_ENABLED` switch with NO code edits.
- Run on a GPU host (Kaggle: Internet on, key as secret, progressive/resumable save). **Kaggle/Colab-only — Molmo needs a CUDA GPU and will not run on CPU-only or non-NVIDIA local hardware.**
- **Cheap config:** Molmo does pipeline reasoning where possible (native `<point>` grounding); Claude (Sonnet) for debate turns; downscale images.
**Gate:** with real keys+GPU, GENERATE runs on a real VLM AND the Claude<->Molmo debate produces a verdict end-to-end on at least one frame.

## Phase 7 -- Evaluation + optional UI
**Goal:** numbers + a demo.
Tasks:
- `evaluation/metrics.py` (SPEC section 12): batch-run over extracted frames; progressive, resumable saving (skip frames already done).
- Emit a summary: debate vs claude-solo vs molmo-solo, convergence rate, avg rounds, concession totals; trajectory metrics (path cost, collision rate, efficiency).
- `[OPTIONAL]` `app.py` Gradio: upload video, set goal, view candidates/selected/GIF/debate tabs.
**Gate:** a batch run produces a metrics summary and is resumable; (optional) the UI demos a video end-to-end.

---

## Phase order summary

| Phase | What | Layer | Gate |
|---|---|---|---|
| 0 | Scaffold + CLI | -- | `python main.py --help` |
| 1 | Frame ingestion | 0 | PNGs from video; `pytest tests/test_frames.py` |
| 2 | VLM interface + Mock/offline | -- | `pytest tests/test_mock_vlm.py` |
| 3 | Visualization + GIF | -- | `pytest tests/test_draw.py`; GIF opens |
| 4 | GENERATE: ground + PIVOT + VLMPC + traversal | 1 | `pytest tests/`; A->B outputs + traversal.gif; runs offline |
| 5 | DEBATE on mocks | 2 | `pytest tests/test_relay.py test_verdict.py`; debate.txt |
| 6 | Real model swap-in | 1+2 | real Claude+Molmo verdict end-to-end |
| 7 | Evaluation + UI | -- | metrics summary; resumable; (opt) UI demo |

Phases 0-5 run entirely **offline on the mock/fallback VLM, for free** -- the whole architecture, demoable end-to-end including a GIF and a (mock) debate. Phases 6-7 need the **Kaggle GPU + Claude API**. A rock-solid Phases 0-5 is a complete, defensible project on its own.

---

## Hard rules (from SPEC sections 3 / 15 -- do not violate)
1. Models never call each other; orchestrator is sole caller in DEBATE.
2. Ground truth never shown to a model -- scoring only.
3. Mock VLM deterministic given a seed.
4. All VLM calls through `VLMInterface`; no direct SDK calls outside `src/vlm/`.
5. All drawing through `visualization/draw.py`.
6. Every cross-module value is a dataclass from `models.py`; no bare dicts.
7. All tunables in `config.py`; no hard-coded numbers in `src/`.
8. Molmo loads once, reused; never per-frame.
9. API key from env / Kaggle Secret only; never a literal.
10. Offline path (`USE_VLM=False`) always works with no network.
11. "Amend" reports text only -- no unbounded pipeline re-run.

---

## Open questions to resolve before the relevant phase (SPEC section 17)
- **Before Phase 4:** collision model = bbox overlap (recommended, simplest). Location grounding = `pretagged` mode reading `data/zones.json` (manual A/B/obstacle tags per clip) offline; `vlm` mode in Phase 6. (Resolves the old scene-understanding gap.)
- **Before Phase 5:** confirm amend-as-text (no pipeline re-run).
- **Before Phase 6:** guard against Molmo always conceding; use concession counts to detect deference vs consensus.

---

## Data source (README guidance, not code)
Input is a generic video file under `data/videos/`. Use a steep **top-down** clip so 2D trajectories and the cost function stay valid (oblique/eye-level footage degrades results). Free, no-attribution sources include Pexels, Pixabay, and Coverr -- search "warehouse top view" / "warehouse aerial". A 5-15s clip yields plenty of frames. Mute/ignore audio; cite the source in the README.

---

## What "the whole thing works" looks like (SPEC section 14)
- `pip install -r requirements.txt` then `python main.py --video <clip> --goal "from A to B"` writes `outputs/<run_id>/` with candidates.png, selected.png, trajectory.gif, **traversal.gif**, final.png, log.json.
- Different frames/runs -> different trajectories.
- `DEBATE_ENABLED=True` also writes debate.txt (transcript + verdict + concessions).
- `pytest tests/` green; `ruff check` clean.
- Flipping `VLM_BACKEND` / `DEBATE_ENABLED` activates real models with no code changes.
- GENERATE runs fully offline.
