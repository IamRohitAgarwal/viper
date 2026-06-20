"""Real Claude VLM (SPEC sections 10, 13 Phase 6).

Implements the full VLMInterface (incl. critique_plan) against the Anthropic
Messages API. Used for pipeline reasoning and/or as debate model A.

Heavy deps (`anthropic`) are imported lazily so this module imports cleanly
on machines without the SDK; it is only needed when instantiated (Kaggle).

Key from env / Kaggle Secret only: ANTHROPIC_API_KEY. Images are sent as
downscaled base64 PNG to control token cost.
"""

import base64
import io
import json
import re

from src.models import (
    ActionPlan,
    DebateArtifact,
    DebateTurn,
    Point,
    SceneObject,
    SceneUnderstanding,
    SelectedCandidates,
    Trajectory,
)
from src.verdicts import parse_verdict
from src.vlm.interface import VLMInterface

_MAX_IMAGE_DIM = 768   # downscale longest side to this before sending


def _extract_json(text: str):
    """Pull the first JSON object/array out of a model reply."""
    match = re.search(r"(\{.*\}|\[.*\])", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON found in reply: {text!r}")
    return json.loads(match.group(1))


class AnthropicVLM(VLMInterface):
    def __init__(self, model: str = "claude-sonnet-4-6", name: str = "claude", max_tokens: int = 1024):
        import os

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Claude backend needs the SDK: pip install anthropic") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (env / Kaggle Secret)")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = name
        self.max_tokens = max_tokens

    # ---- helpers ----
    def _encode(self, image) -> str:
        img = image.convert("RGB").copy()
        img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _ask(self, image, prompt: str) -> str:
        content = []
        if image is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": self._encode(image)},
            })
        content.append({"type": "text", "text": prompt})
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    # ---- VLMInterface ----
    def understand_scene(self, image, goal: str) -> SceneUnderstanding:
        prompt = (
            f"You are analysing a top-down image for the goal: {goal!r}.\n"
            "Return ONLY JSON of the form:\n"
            '{"description": "...", "goal_interpretation": "...", "objects": '
            '[{"name": "...", "x": 0-1, "y": 0-1, "w": 0-1, "h": 0-1, '
            '"role": "target|obstacle|goal|agent|background"}]}\n'
            "Coordinates are normalised [0,1] with (0,0) at the top-left."
        )
        data = _extract_json(self._ask(image, prompt))
        objects = [
            SceneObject(
                name=o.get("name", "object"),
                x=float(o["x"]), y=float(o["y"]), w=float(o["w"]), h=float(o["h"]),
                role=o.get("role", "background"),
            )
            for o in data.get("objects", [])
        ]
        return SceneUnderstanding(
            objects=objects,
            description=data.get("description", ""),
            goal_interpretation=data.get("goal_interpretation", ""),
        )

    def select_candidates(
        self, annotated_image, goal: str, candidates: list[Trajectory], scene: SceneUnderstanding
    ) -> SelectedCandidates:
        ids = [t.id for t in candidates]
        prompt = (
            f"The image shows candidate trajectories labelled T1..T{max(ids)} for the "
            f"goal: {goal!r}. Pick the most promising ones (those that reach the goal "
            "while avoiding obstacles).\n"
            "Reply with a line 'IDS: <comma-separated numbers>' then a short reason."
        )
        reply = self._ask(annotated_image, prompt)
        m = re.search(r"IDS:\s*([0-9,\s]+)", reply)
        chosen = (
            [int(n) for n in re.findall(r"\d+", m.group(1))] if m else ids[: max(1, len(ids) // 2)]
        )
        chosen = [i for i in chosen if i in ids] or ids[:1]
        return SelectedCandidates(ids=chosen, reasoning=reply.strip())

    def generate_rationale(self, image, plan: ActionPlan, goal: str) -> str:
        prompt = (
            f"Goal: {goal!r}. The image shows the selected route. In 2-3 sentences, "
            f"explain why trajectory T{plan.best_trajectory.id} (total cost "
            f"{plan.cost.total_cost:.2f}, goal_distance {plan.cost.goal_distance:.3f}, "
            f"collision {'yes' if plan.cost.collision_penalty > 0 else 'no'}) is a good plan."
        )
        return self._ask(image, prompt).strip()

    def critique_plan(self, artifact: DebateArtifact, other_view: str | None) -> DebateTurn:
        prompt = (
            f"You are critiquing a proposed route plan for the goal: {artifact.goal!r}.\n"
            f"Trajectory: {artifact.trajectory_summary}\n"
            f"Cost: {artifact.cost_summary}\n"
            f"Alternatives: {artifact.candidate_summary}\n"
        )
        if other_view:
            prompt += f"\nAnother model said:\n{other_view}\n"
        prompt += (
            "\nDecide whether to endorse, amend, or reject the plan. If amending, give "
            "the amendment as text on an 'AMENDED: ...' line. End your reply with exactly "
            "one line: 'VERDICT: <endorse|amend|reject>'."
        )
        reply = self._ask(artifact.selected_image, prompt)
        verdict = parse_verdict(reply) or "endorse"
        amended = None
        am = re.search(r"AMENDED:\s*(.+)", reply)
        if verdict == "amend" and am:
            amended = am.group(1).strip()
        return DebateTurn(
            round=0, model=self.name, verdict=verdict,
            amended_plan=amended, reasoning=reply.strip(), raw_reply=reply,
        )

    # ---- Phase 6 vlm-mode grounding ----
    def locate_points(self, image, labels: list[str]) -> dict[str, Point]:
        prompt = (
            "Locate each of these places in the top-down image: "
            f"{labels}. Return ONLY JSON mapping each label to a point, e.g. "
            '{"label": {"x": 0-1, "y": 0-1}}. Coordinates normalised, (0,0) top-left.'
        )
        data = _extract_json(self._ask(image, prompt))
        return {k: Point(float(v["x"]), float(v["y"])) for k, v in data.items()}
