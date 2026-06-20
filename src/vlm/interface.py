"""The sole VLM entry point (SPEC section 10).

Every VLM call in the codebase goes through a ``VLMInterface`` instance.
Swapping mock <-> real is a config change (``VLM_BACKEND``), never a code
edit. The only files allowed to talk to a model are this one and its
implementations under ``src/vlm/``.
"""

from abc import ABC, abstractmethod

from PIL import Image

from src.models import (
    ActionPlan,
    DebateArtifact,
    DebateTurn,
    SceneUnderstanding,
    SelectedCandidates,
    Trajectory,
)


class VLMInterface(ABC):
    """Provider-agnostic contract for every model interaction."""

    @abstractmethod
    def understand_scene(self, image: Image.Image, goal: str) -> SceneUnderstanding:
        """Detect objects + roles and interpret the goal."""

    @abstractmethod
    def select_candidates(
        self,
        annotated_image: Image.Image,
        goal: str,
        candidates: list[Trajectory],
        scene: SceneUnderstanding,
    ) -> SelectedCandidates:
        """Shortlist promising candidate trajectories with reasoning."""

    @abstractmethod
    def generate_rationale(
        self, image: Image.Image, plan: ActionPlan, goal: str
    ) -> str:
        """Human-readable explanation of the chosen plan."""

    @abstractmethod
    def critique_plan(
        self, artifact: DebateArtifact, other_view: str | None
    ) -> DebateTurn:
        """Critique a packaged plan in the debate.

        Reply must end with a parseable line:
        ``VERDICT: <endorse|amend|reject>``.
        """
