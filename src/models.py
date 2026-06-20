"""All cross-module typed contracts (SPEC section 6).

No module defines its own data structures for pipeline data — everything
crossing a module boundary is one of these dataclasses.
"""

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
    x: float
    y: float
    w: float
    h: float                      # bbox, normalised
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
