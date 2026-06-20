# SPEC.md — VIPER: Verifiable Iterative Planning with Ensemble Reasoning

> **Source of truth for this project.** Claude Code reads this file at the start of every session.
> All design decisions, module contracts, and build phases are defined here. Read it fully before building any phase.

**VIPER** = **V**erifiable **I**terative **P**lanning with **E**nsemble **R**easoning.
*Fuses PIVOT (visual proposal) + VLM (visual reasoning) + VLMPC (predictive validation) into a planning pipeline, then validates the plan through a two-model Claude<->Molmo debate.*

---

## 1. Purpose

VIPER takes a **video** of a physical scene and a **task instruction of the form "go from place A to place B (avoiding obstacles)"**, and produces a **validated traversal** — a route that is not just visually plausible but predicted to actually work, animated across the real footage, then cross-checked by two independent models.

> **Scope of "execute":** VIPER *plans, validates, and animates* the route an agent would take from A to B, rendered as an overlay on the real video frames. It does **not** drive a physical vehicle or track/move the real object in the footage — the traversing agent is a synthetic marker drawn on top. This is a route-planning-and-visualization system for logistics, not a robot controller.

It answers: *"Which action not only looks correct, but actually works — and do two independent AI models agree it works?"*

The system runs **three layers in sequence**, per processed frame:

1. **GENERATE — the planning pipeline** (PIVOT + VLM + VLMPC):
   - **PIVOT** — generate candidate trajectories and draw them on the frame.
   - **VLM** — reason over the annotated frame and shortlist promising candidates.
   - **VLMPC** — simulate each shortlisted trajectory, score it with a cost function, select the lowest-cost trajectory.
   - **Output:** an `ActionPlan` (selected trajectory + cost breakdown + annotated images + GIF + rationale).

2. **DEBATE — ensemble validation:** the `ActionPlan` is packaged and relayed to two different models in turn — first **Claude**, then **Molmo** — which critique it and converge (over up to N rounds) on a verdict: endorse, amend, or reject.

3. **FINAL OUTPUT:** the validated plan, the debate transcript, the verdict, and (on disagreement) why the overruled model lost.

**Why this design:** PIVOT is strong at visual proposal but weak at validation. VLMPC is strong at predictive validation but weak at grounding. Fusing them yields a good plan; relaying that plan through two independent models turns it into a *trustworthy* one — and surfaces disagreement as a trust signal. No model training is required (zero-shot).

---

## 2. Input: Video

The system's input is a **video file** (e.g. `.mp4`). VIPER does not assume any particular source — any video of a physical scene with a goal expressible as a 2D trajectory is valid.

**Frame extraction (Stage 0).** Because the planning pipeline operates on single images, a one-time preprocessing step samples frames from the video and stores them as scene images:

```
extract_frames(video_path, frame_stride, max_frames) -> list[str]   # returns frame paths
```

- Sample one frame every `frame_stride` frames (config), up to `max_frames`.
- Save to `data/frames/` as `frame_0001.png`, `frame_0002.png`, ...
- Each extracted frame plays the role of a single "scene image" for the rest of the pipeline.

**Guidance (README, not enforced in code):** a steep top-down camera angle makes 2D trajectories and the cost function most meaningful; near-eye-level footage degrades trajectory validity. A short clip (5-15s) yields hundreds of frames — more than enough.

**Prompt format.** The instruction names a start and a destination (and optionally an obstacle):
`"from {place_a} to {place_b}"`, e.g. `"from receiving bay to dock 4"`. `main.py` parses it into `place_a` and `place_b`. Free-form goals still work but the A->B form drives the traversal feature.

**Run modes (CLI):**
- `python main.py --video data/videos/clip.mp4 --goal "from receiving bay to dock 4"` — extract frames, ground A and B, then plan + animate the traversal.
- `python main.py --frames-dir data/frames/ --random` — skip extraction; pick a random already-extracted frame. Fully offline, reproducible.

---

## 3. Golden Rules

1. **Build one phase at a time.** Each phase has a "done when" gate. Do not start the next until it passes.
2. **The three layers are separable.** GENERATE runs and is testable (with the mock/fallback VLM) without DEBATE. DEBATE consumes a finished `ActionPlan` and never reaches into pipeline internals.
3. **Models never call each other.** In DEBATE the orchestrator is the sole caller; Claude and Molmo only ever see each other's *relayed text*.
4. **One VLM entry point.** Every VLM call goes through `src/vlm/interface.py`. Swapping mock<->real is a config change, never a code edit.
5. **One drawing module.** All image overlays come from `src/visualization/draw.py`. No other module draws.
6. **Typed contracts only.** Every cross-module value is a dataclass from `src/models.py`. No bare dicts.
7. **All tunables in `config.py`.** No hard-coded numbers elsewhere.
8. **Mock VLM is deterministic** given a seed. Tests must be repeatable.
9. **Ground truth (if present) is never shown to any model.** Scoring only.
10. **Offline-first.** The whole GENERATE pipeline must run with `USE_VLM = False` (rule-based fallback) and no network.

---

## 4. Tech Stack

| Concern | Tool |
|---|---|
| Language | Python 3.11+ |
| Video / frames | OpenCV (cv2.VideoCapture) |
| Image processing | OpenCV + Pillow |
| Visualization | OpenCV / Pillow / matplotlib |
| Animation (GIF) | Pillow / imageio |
| Numerical / sampling | NumPy |
| VLM — mock & fallback | built-in (deterministic, offline) |
| VLM — real (pipeline) | Anthropic SDK (Claude), optional |
| Debate model A | Anthropic SDK (Claude) |
| Debate model B | Molmo-7B-D (transformers, 4-bit) |
| UI (optional) | Gradio 4.x |
| Tests | pytest |
| Lint | ruff |

**Runtime:** Development runs fully offline on the mock/fallback VLM (free, deterministic). The real-model experiment (Claude and/or Molmo pipeline, and/or Claude<->Molmo debate) runs where a CUDA GPU is available for Molmo — e.g. a Kaggle notebook with Internet enabled and the API key as a secret; Molmo is loaded once and reused. **Molmo requires CUDA (bitsandbytes 4-bit); it cannot run on CPU-only or non-NVIDIA hardware in practice, so Phase 6 is Kaggle/Colab-only.** Molmo is chosen over generic VLMs because it natively emits 2D points (`<point>`), which suits grounding and trajectory tasks.

---

## 5. Repository Layout

```
viper/
|-- SPEC.md                      # source of truth
|-- plan.md                      # build order (companion)
|-- README.md                    # quick start, data-source guidance, honest scope
|-- main.py                      # entry point (CLI)
|-- config.py                    # all parameters
|-- requirements.txt
|-- data/
|   |-- videos/                  # input video(s)
|   |-- frames/                  # extracted frames (scene images)
|-- src/
|   |-- __init__.py
|   |-- models.py                # all dataclasses / typed contracts
|   |-- ingest/
|   |   |-- frames.py            # Stage 0: video -> frames
|   |-- vlm/
|   |   |-- interface.py         # VLMInterface ABC — sole VLM entry point
|   |   |-- mock_vlm.py          # deterministic mock + rule-based fallback
|   |   |-- anthropic_vlm.py     # real Claude (pipeline + debate)
|   |   |-- molmo_vlm.py         # real Molmo (debate + optional pipeline)
|   |-- pivot/
|   |   |-- generator.py         # candidate trajectory generation
|   |   |-- visual_prompt.py     # draw candidates on frame
|   |-- vlmpc/
|   |   |-- rollout.py           # simulate a trajectory
|   |   |-- cost_function.py     # cost = goal_dist + collision + path_length
|   |   |-- validator.py         # select minimum-cost trajectory
|   |-- debate/
|   |   |-- artifact.py          # package ActionPlan -> DebateArtifact
|   |   |-- relay.py             # Claude->Molmo->... relay + convergence
|   |   |-- verdict.py           # final verdict + transcript assembly
|   |-- visualization/
|   |   |-- draw.py              # all image overlays
|   |   |-- animate.py           # GIF generation
|   |-- evaluation/
|       |-- metrics.py           # evaluation metrics
|       |-- logger.py            # trace logging -> log.json
|-- outputs/                     # per-run artifacts (see section 11)
|-- tests/
|-- app.py                       # optional Gradio UI
```

---

## 6. Data Models (`src/models.py`)

All cross-module contracts live here. No module defines its own.

```python
from dataclasses import dataclass
from typing import Optional
from PIL import Image

# ---------- shared ----------
@dataclass
class Point:
    x: float        # normalised [0,1]
    y: float

@dataclass
class Trajectory:
    id: int                       # T1, T2, ...
    points: list[Point]           # ordered waypoints
    action_type: str = "move"     # "move" | "push" | "navigate"

@dataclass
class SceneObject:
    name: str
    x: float; y: float; w: float; h: float    # bbox, normalised
    role: str                     # "target" | "obstacle" | "goal" | "agent" | "background"

@dataclass
class SceneUnderstanding:
    objects: list[SceneObject]
    description: str
    goal_interpretation: str

@dataclass
class GroundedLocations:
    place_a: Point                # resolved start, normalised [0,1]
    place_b: Point                # resolved destination
    place_a_label: str            # e.g. "receiving bay"
    place_b_label: str            # e.g. "dock 4"
    obstacles: list[SceneObject]  # things to avoid en route

# ---------- LAYER 1: GENERATE ----------
@dataclass
class AnnotatedProposal:
    image: Image.Image
    candidates: list[Trajectory]

@dataclass
class SelectedCandidates:
    ids: list[int]                # shortlisted trajectory ids
    reasoning: str

@dataclass
class SimulationResult:
    trajectory_id: int
    final_position: Point
    path_length: float
    collision: bool
    frames: list[Image.Image]     # rollout frames, for the GIF

@dataclass
class CostBreakdown:
    trajectory_id: int
    goal_distance: float
    collision_penalty: float
    path_length_penalty: float
    total_cost: float

@dataclass
class ActionPlan:
    best_trajectory: Trajectory            # path from place_a to place_b
    locations: GroundedLocations           # resolved A, B, obstacles
    cost: CostBreakdown
    all_costs: list[CostBreakdown]
    simulation: SimulationResult           # rollout of the winner
    rationale: str
    candidates_image: Image.Image          # all candidates drawn
    selected_image: Image.Image            # winner highlighted
    final_image: Image.Image               # final annotated frame
    traversal_gif_path: str                # agent moving A->B across real frames
    scene: SceneUnderstanding

# ---------- LAYER 2: DEBATE ----------
@dataclass
class DebateArtifact:
    """The plan, packaged for the debate. This is ALL the models see."""
    selected_image: Image.Image
    goal: str
    trajectory_summary: str        # the chosen path in words
    cost_summary: str              # human-readable cost rationale
    candidate_summary: str         # what alternatives lost and why

@dataclass
class DebateTurn:
    round: int
    model: str                     # "claude" | "molmo"
    verdict: str                   # "endorse" | "amend" | "reject"
    amended_plan: Optional[str]
    reasoning: str
    raw_reply: str

@dataclass
class DebateResult:
    final_verdict: str             # "endorse" | "amend" | "reject" | "no_consensus"
    converged: bool
    rounds_used: int
    winner_model: str              # "agreement" | "claude" | "molmo" | "tie_break"
    concessions: dict              # {"claude": int, "molmo": int}
    round1_solo: dict              # {"claude": verdict, "molmo": verdict}
    transcript: list[DebateTurn]
    loser_reasoning: Optional[str]
    final_plan_text: str

@dataclass
class ViperResult:
    plan: ActionPlan
    debate: Optional[DebateResult]   # None if debate disabled
    frame_path: str
    log_path: str
```

---

## 7. Configuration (`config.py`)

```python
# ---- frame ingestion ----
FRAME_STRIDE        = 15        # sample 1 of every N video frames
MAX_FRAMES          = 20        # cap on extracted frames

# ---- location grounding ----
GROUNDING_MODE      = "pretagged"   # "pretagged" (offline) | "vlm" (real models)
PRETAG_FILE         = "data/zones.json"  # per-clip labeled A/B/obstacle regions
# ---- traversal animation ----
TRAVERSAL_FRAMES    = 12       # frames the agent takes to move A->B
TRAVERSAL_OVERLAY   = True     # render path/agent as overlay on real video frames

# ---- PIVOT ----
NUM_CANDIDATES      = 5         # trajectories proposed per frame
MAX_TRAJECTORY_LEN  = 10        # waypoints per trajectory
SEED                = 42        # reproducible sampling

# ---- VLM (pipeline) ----
USE_VLM             = False     # False = offline rule-based fallback
VLM_BACKEND         = "mock"    # "mock" | "anthropic"
CLAUDE_MODEL        = "claude-sonnet-4-6"   # cheap default for pipeline reasoning

# ---- VLMPC cost ----
COLLISION_PENALTY   = 100.0
PATH_LENGTH_WEIGHT  = 1.0
GOAL_DISTANCE_WEIGHT= 1.0

# ---- DEBATE ----
DEBATE_ENABLED      = False     # off by default; turn on for the ensemble experiment
DEBATE_MAX_ROUNDS   = 3
DEBATE_MODEL_A      = "claude"  # first in the relay
DEBATE_MODEL_B      = "molmo"   # second in the relay
DEBATE_CLAUDE_MODEL = "claude-sonnet-4-6"
MOLMO_MODEL_ID      = "allenai/Molmo-7B-D-0924"
MOLMO_REVISION      = "main"    # pin a commit hash for true reproducibility
MOLMO_TRUST_REMOTE_CODE = True  # Molmo ships custom modeling code
MOLMO_QUANTIZATION  = "4bit"
PARSE_RETRY         = 1

# ---- visualization ----
COLOR_CANDIDATE     = (100, 160, 255)
COLOR_SELECTED      = (86, 211, 100)
COLOR_REJECTED      = (150, 150, 150)
COLOR_OBSTACLE      = (239, 68, 68)
COLOR_GOAL          = (250, 204, 21)
ARROW_THICKNESS     = 2
GIF_FPS             = 5
```

Changing `VLM_BACKEND`, `DEBATE_ENABLED`, or any model id must require **no other code changes**.

---

## 8. Component Contracts — LAYER 1: GENERATE

### 8.0 ingest/frames.py
```
extract_frames(video_path: str, frame_stride: int, max_frames: int) -> list[str]
```
Reads the video with OpenCV, samples frames, writes PNGs to `data/frames/`, returns their paths. Idempotent: re-running overwrites cleanly.

### 8.0b grounding/locate.py  (behind VLMInterface for the "vlm" mode)
```
ground_locations(frame, place_a_text, place_b_text, config, vlm=None) -> GroundedLocations
```
Resolves the two named places (and obstacles) to pixel points on the frame.
- `GROUNDING_MODE="pretagged"` (offline, free): read labeled A/B/obstacle regions for this clip from `PRETAG_FILE` (`data/zones.json`). One-time manual tagging per demo clip. Used in Phases 0-5.
- `GROUNDING_MODE="vlm"` (real models): the VLM returns coordinates for each named place. Used in Phase 6.
- Fills the OPEN-2 scene-understanding gap. Ground truth, if any, stays separate from this.

### 8.1 pivot/generator.py
```
generate_candidates(image, locations, num_candidates, config) -> list[Trajectory]
```
- Randomized sampling (NOT learned), seeded by `SEED`.
- Each `Trajectory` STARTS at `locations.place_a` and aims at `locations.place_b`, with sampled intermediate waypoints (spread/noise) so candidates differ.
- Clip points to [0.05, 0.95].

### 8.2 pivot/visual_prompt.py
```
draw_candidates(image, candidates) -> AnnotatedProposal
```
Delegates drawing to `visualization/draw.py`. Labels each candidate T1, T2, ... Returns the annotated image + the candidate list.

### 8.3 vlm/selector.py behavior (lives behind VLMInterface)
```
select_candidates(annotated_image, goal, candidates, scene) -> SelectedCandidates
```
- If `USE_VLM` / real backend: the VLM reads the visual prompt and returns shortlisted ids + reasoning.
- If offline: rule-based fallback (e.g. keep the K trajectories whose endpoints are nearest the goal region).

### 8.4 vlmpc/rollout.py
```
simulate_trajectory(trajectory, scene, config) -> SimulationResult
```
Step-by-step forward rollout. Moves the agent/target along waypoints; checks collisions against obstacle bboxes; accumulates path length; records a frame per step (for the GIF). Returns final position, path length, collision flag, frames.

### 8.5 vlmpc/cost_function.py
```
compute_cost(sim_result, locations, config) -> CostBreakdown
```
`goal_distance` = distance from the rollout's final position to `locations.place_b`.
`total = GOAL_DISTANCE_WEIGHT*goal_distance + (COLLISION_PENALTY if collision else 0) + PATH_LENGTH_WEIGHT*path_length`. Returns the full breakdown, for transparency.

### 8.6 vlmpc/validator.py
```
select_best(costs: list[CostBreakdown]) -> int      # winning trajectory id
```
Returns the id of the minimum total-cost trajectory.

### 8.7 pipeline runner (in main.py or src/runner.py)
```
run_pipeline(frames, place_a, place_b, config, vlm) -> ActionPlan
```
Order: pick planning frame -> **ground_locations(A, B, obstacles)** -> generate candidates (A->B) -> draw -> select (VLM/fallback) -> simulate each selected -> cost each (distance to B) -> pick best -> **animate traversal across `frames`** (agent marker moving A->B, overlaid on successive real frames) -> build images + rationale -> ActionPlan.

---

## 9. Component Contracts — LAYER 2: DEBATE

### 9.1 debate/artifact.py
```
package(plan: ActionPlan, goal: str) -> DebateArtifact
```
Builds the relayable package: selected-trajectory image, goal, trajectory summary, cost summary, candidate summary. NO pipeline internals, NO ground truth.

### 9.2 debate/relay.py
```
run_debate(artifact, config, model_a, model_b) -> DebateResult
```
Relay (not simultaneous):
1. Turn 1 — Claude (model_a) critiques the artifact (other_view=None) -> DebateTurn (endorse/amend/reject + reasoning).
2. Turn 2 — Molmo (model_b) critiques, given Claude's turn as other_view.
3. Turn 3 — Claude critiques, given Molmo's latest; if its verdict changes, increment Claude concessions. Concessions are tracked symmetrically: any model changing from its round-1 verdict increments that model's count.
4. Alternate until latest verdicts match (converged) OR max_rounds reached.
5. Converged -> final_verdict = agreed verdict; winner_model="agreement".
6. Not converged -> tie-break to the MOST CONSERVATIVE verdict (reject > amend > endorse), winner_model="tie_break", final_verdict="no_consensus" recorded alongside the conservative pick.
7. Parse each reply's `VERDICT:` line; retry once on failure, else keep that model's previous verdict.
8. Append every turn to the transcript as it happens. Record round1_solo verdicts.

### 9.3 debate/verdict.py
```
assemble(result: DebateResult, plan: ActionPlan) -> (final_plan_text, loser_reasoning)
```
Produces the human-readable final plan (endorsed or amended-as-text) and, on disagreement, the overruled model's final reasoning.

### 9.4 top-level
```
run_viper(frame_image, goal, config, gen_vlm, model_a=None, model_b=None) -> ViperResult
  plan = run_pipeline(frame_image, goal, config, gen_vlm)
  if not config.DEBATE_ENABLED: return ViperResult(plan, None, ...)
  artifact = package(plan, goal)
  debate   = run_debate(artifact, config, model_a, model_b)
  return ViperResult(plan, debate, ...)
```

**Note on "amend":** an amend verdict reports the amended plan AS TEXT only. It does NOT re-run the pipeline (that risks non-termination). Pipeline re-run on amend is a future extension and must carry a hard iteration cap if ever added.

---

## 10. VLM Interface (`src/vlm/interface.py`)

The ONLY files that talk to a model are `interface.py` and its implementations.

```python
from abc import ABC, abstractmethod

class VLMInterface(ABC):
    @abstractmethod
    def understand_scene(self, image, goal) -> "SceneUnderstanding": ...
    @abstractmethod
    def select_candidates(self, annotated_image, goal, candidates, scene) -> "SelectedCandidates": ...
    @abstractmethod
    def generate_rationale(self, image, plan, goal) -> str: ...
    # debate capability:
    @abstractmethod
    def critique_plan(self, artifact: "DebateArtifact", other_view: "str | None") -> "DebateTurn":
        """Reply must end with a parseable line: 'VERDICT: <endorse|amend|reject>'."""
```

- `MockVLM` implements all methods deterministically (numpy default_rng(seed)) and doubles as the offline rule-based fallback.
- `AnthropicVLM` and `MolmoVLM` implement all methods incl. `critique_plan`.
- Molmo loads ONCE, 4-bit (CUDA only), reused; uses its own `AutoProcessor` (`trust_remote_code=True`), not a chat template. Its native `<point>` output is parsed for grounding/selection. Claude images sent base64 PNG, downscaled to control token cost. Keys from env / Kaggle Secret.

---

## 11. Outputs

Each run writes to `outputs/<run_id>/`:
```
candidates.png      # all candidate trajectories drawn
selected.png        # selected A->B trajectory highlighted
trajectory.gif      # rollout of the winner
traversal.gif       # agent moving A->B over real frames (KEY output)
final.png           # final annotated frame
debate.txt          # debate transcript + verdict (only if DEBATE_ENABLED)
log.json            # full trace: input, candidates, shortlist, sim results,
                    #             costs, chosen trajectory, debate result
```

`evaluation/logger.py: log_run(data) -> None` writes `log.json`.
`visualization/animate.py`:
- `generate_gif(frames, path) -> str` — rollout GIF (single-frame).
- `generate_traversal(real_frames, trajectory, locations, config) -> str` — the A->B traversal: step the agent marker along the selected trajectory, drawn as an overlay on successive real video frames, write `traversal.gif`.

---

## 12. Evaluation Metrics (`src/evaluation/metrics.py`)

Outcome: Task Success Rate, Goal Distance Error.
Trajectory: Path Cost, Collision Rate, Path Efficiency.
System: VLM Selection Accuracy, Simulation Correction Rate.
Ensemble (debate): debate-final vs claude-solo vs molmo-solo verdict agreement with outcome; convergence rate; concession counts (deference check).

If a frame has an optional ground-truth goal region, "success" = winner's final position within it. Ground truth is used ONLY here, never shown to a model.

---

## 13. Build Phases

### Phase 0 — Scaffold
Directory tree + `__init__.py`; `requirements.txt` (opencv-python, pillow, imageio, numpy, matplotlib, anthropic, transformers, gradio, pytest, ruff); `config.py` with all defaults; `src/models.py` with ALL dataclasses (both layers); `main.py` CLI shell parsing `--video/--frames-dir/--goal/--frame/--random`.
**Done when:** `python main.py --help` works; imports clean; `pip install -r requirements.txt` succeeds.

### Phase 1 — Frame ingestion (Stage 0)
`ingest/frames.py`. Test on any short sample video (or a synthetic one generated in the test).
**Done when:** `extract_frames` writes N PNGs to `data/frames/` from a video; `--frames-dir --random` picks one.

### Phase 2 — VLM interface + Mock/fallback
`vlm/interface.py` (incl. `critique_plan`); `vlm/mock_vlm.py` deterministic + rule-based fallback. `tests/test_mock_vlm.py`: same seed -> identical outputs; correct return types.
**Done when:** `pytest tests/test_mock_vlm.py` passes.

### Phase 3 — Visualization + animation
`visualization/draw.py` (candidates, selected, scene, final) + `animate.py` (GIF). Tests assert new PIL images of right size and a GIF file is produced.
**Done when:** `pytest tests/test_draw.py` passes; a sample GIF opens.

### Phase 4 — GENERATE pipeline (grounding + PIVOT + VLM + VLMPC + traversal)
Implement in order, each with a test:
1. `grounding/locate.py` (pretagged mode) — reads `data/zones.json`, returns `GroundedLocations` for A, B, obstacles. Test: known zones file -> correct points.
2. `pivot/generator.py` — candidates START at A, aim at B; clipped & seeded.
3. `pivot/visual_prompt.py` — annotate (T1..Tn).
4. selector (fallback) — shortlist sane.
5. `vlmpc/rollout.py` — collision flag on obstacle overlap; records frames.
6. `vlmpc/cost_function.py` — goal_distance = distance to B; collision penalty correct.
7. `vlmpc/validator.py` — picks min cost.
8. `visualization/animate.py: generate_traversal(...)` — agent marker moves A->B overlaid on successive real frames.
9. runner — returns a fully-populated `ActionPlan` incl. `traversal_gif_path`.
Also create a small `data/zones.json` for one demo clip (manual A/B/obstacle tags).
**Done when:** `pytest tests/` (pipeline) green; `python main.py --video <clip> --goal "from A to B"` writes candidates/selected/trajectory/**traversal**/final/log; the traversal GIF shows an agent moving A->B over the real footage.

### Phase 5 — DEBATE layer (on mocks)
`debate/artifact.py`, `relay.py`, `verdict.py`; `run_viper` chaining; use MockVLM as BOTH debaters (different seeds so they can disagree). Tests: converge when both endorse; conservative tie-break on deadlock; concession counting; transcript completeness; amend-as-text.
**Done when:** `pytest tests/test_relay.py tests/test_verdict.py` pass; with `DEBATE_ENABLED=True` a `debate.txt` transcript + verdict is written.

### Phase 6 — Real model swap-in (optional, after Phase 5 stable)
`anthropic_vlm.py` + `molmo_vlm.py` implementing the interface incl. `critique_plan`. Molmo loaded once 4-bit (CUDA only, `trust_remote_code=True`); Claude base64 PNG downscaled; keys from secret. Verify config switches activate real models with NO code edits. Run on a GPU host (Kaggle: Internet on, key as secret, progressive save). **Kaggle/Colab-only — Molmo will not run on CPU-only or non-NVIDIA local hardware.**
**Recommended cheap config:** Molmo runs the pipeline reasoning where possible (its `<point>` output grounds places directly); Claude (Sonnet) for debate turns; downscale images. Keeps a full 50-frame run to a few dollars.
**Done when:** with real keys+GPU, GENERATE runs on a real VLM AND a real Claude<->Molmo debate yields a verdict end-to-end.

### Phase 7 — Evaluation + optional Gradio UI
`evaluation/metrics.py`; batch-run over all extracted frames with progressive, resumable saving; emit a summary (debate vs solo, convergence, concessions). Optionally wire `app.py` (Gradio) to upload a video, set a goal, and view candidates/selected/GIF/debate tabs.
**Done when:** a batch run produces a metrics summary; (optional) the UI demos a video end-to-end.

---

## 14. Definition of Done (overall)

- `pip install -r requirements.txt` then `python main.py --video <clip> --goal "<goal>"` runs end-to-end and writes `outputs/<run_id>/` with candidates.png, selected.png, trajectory.gif, final.png, log.json.
- Different frames/runs produce different trajectories (randomization works).
- With `DEBATE_ENABLED=True`, a `debate.txt` with transcript + verdict + concessions is also written.
- `pytest tests/` green; `ruff check` clean.
- Flipping `VLM_BACKEND` and `DEBATE_ENABLED` activates real models with NO code changes.
- The whole GENERATE pipeline runs fully offline (`USE_VLM=False`).

---

## 15. Hard Requirements (non-negotiable)

1. No model-to-model communication; orchestrator is sole caller in DEBATE.
2. Ground truth never shown to a model.
3. Molmo loads once, reused.
4. API key from env / Kaggle Secret only; never a literal.
5. All VLM traffic through `VLMInterface`; no direct SDK calls outside `src/vlm/`.
6. All drawing through `visualization/draw.py`.
7. Offline path (`USE_VLM=False`) must always work with no network.
8. "Amend" reports text only — no unbounded pipeline re-run.

---

## 16. Honest Scope Notes (must appear in README)

- Input is real video, but the planner reasons on extracted single frames as a 2D-trajectory proxy; it is not full video/temporal control or a deployed robot.
- The forward rollout is a lightweight simulation (waypoint motion + bbox collision), not a learned video-prediction model; the architecture supports upgrading it later.
- In the debate, Molmo-7B-D is smaller than the Claude model used; some disagreement reflects a capability gap, not just independent judgement — hence concession tracking.
- "Agreement" can mean genuine consensus or the weaker model deferring; the report distinguishes these via concession counts.
- Steep top-down footage is required for trajectory/cost validity; oblique footage degrades results. This is a data-quality assumption, stated openly.
- The A->B traversal is an **overlay animation**: a synthetic agent marker drawn moving over the real frames. It is not the real vehicle in the footage moving, and it is not robot control. For a logistics demo this shows the *planned, validated route*, animated on real footage.

### zones.json format (pretagged grounding)
```json
{
  "clip": "warehouse_topdown.mp4",
  "places": {
    "receiving bay": {"x": 0.12, "y": 0.20, "w": 0.10, "h": 0.10},
    "dock 4":        {"x": 0.80, "y": 0.75, "w": 0.10, "h": 0.10}
  },
  "obstacles": [
    {"name": "pallet stack", "x": 0.45, "y": 0.40, "w": 0.12, "h": 0.18}
  ]
}
```
All coordinates normalised [0,1]. `ground_locations` matches `place_a_text`/`place_b_text` against the `places` keys.

---

## 17. Open Questions (resolve before the relevant phase)

- **Before Phase 4:** rollout collision model = bbox overlap along the path (recommended) vs point-in-polygon; choose the simpler bbox check first.
- **RESOLVED (was OPEN-2):** location/obstacle grounding in offline mode is handled by `grounding/locate.py` in `pretagged` mode, reading `data/zones.json` (one-time manual tag of A/B/obstacle regions per demo clip). In `vlm` mode the VLM grounds named places. This replaces the earlier color-blob suggestion.
- **Before Phase 5:** confirm amend-as-text (no pipeline re-run).
- **Before Phase 6:** guard against Molmo always conceding; rely on concession counts to detect deference.
