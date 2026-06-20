"""Real Molmo-7B-D VLM (SPEC sections 10, 13 Phase 6).

Implements the full VLMInterface (incl. critique_plan) using allenai/Molmo-7B-D.
Used as debate model B and, optionally, for pipeline reasoning. Molmo natively
emits 2D points (``<point x=.. y=..>``), which we parse for grounding/selection.

CUDA-only (4-bit bitsandbytes). Heavy deps (torch/transformers) are imported
lazily, and the underlying model is loaded ONCE per model id (cached at module
level) and reused, per SPEC hard requirement 3.

Molmo emits point coordinates as PERCENTAGES of image dimensions (0-100); we
divide by 100 to normalise. Verify against your transformers/Molmo revision.
"""

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

# Cache: model_id -> (model, processor). Ensures Molmo loads once and is reused.
_MODEL_CACHE: dict = {}

_POINT_RE = re.compile(r'x\s*=\s*"?([0-9.]+)"?\s+y\s*=\s*"?([0-9.]+)"?', re.IGNORECASE)


def _load_molmo(model_id: str, revision: str, trust_remote_code: bool, quantization: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Molmo backend needs GPU deps: "
            "pip install transformers accelerate bitsandbytes einops torchvision"
        ) from exc

    quant_config = None
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code, torch_dtype="auto", device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code,
        torch_dtype="auto", device_map="auto", quantization_config=quant_config,
    )
    _MODEL_CACHE[model_id] = (model, processor)
    return model, processor


class MolmoVLM(VLMInterface):
    def __init__(self, config, name: str = "molmo", max_new_tokens: int = 256):
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = _load_molmo(
            config.MOLMO_MODEL_ID, config.MOLMO_REVISION,
            config.MOLMO_TRUST_REMOTE_CODE, config.MOLMO_QUANTIZATION,
        )

    def _ask(self, image, prompt: str) -> str:
        from transformers import GenerationConfig

        img = image.convert("RGB") if image is not None else None
        inputs = self.processor.process(images=[img] if img else None, text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=self.max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )
        generated = output[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_points(self, text: str) -> list[Point]:
        return [Point(float(x) / 100.0, float(y) / 100.0) for x, y in _POINT_RE.findall(text)]

    # ---- VLMInterface ----
    def understand_scene(self, image, goal: str) -> SceneUnderstanding:
        desc = self._ask(image, f"Describe this top-down scene for the goal: {goal!r}.")
        objects: list[SceneObject] = []
        # Best-effort: point at the goal/target so the pipeline has anchors.
        for role, query in (("goal", "the destination area"), ("target", "the main object to move")):
            pts = self._parse_points(self._ask(image, f"Point to {query}."))
            if pts:
                p = pts[0]
                objects.append(SceneObject(role, p.x, p.y, 0.08, 0.08, role))
        return SceneUnderstanding(objects=objects, description=desc.strip(), goal_interpretation=goal)

    def select_candidates(
        self, annotated_image, goal: str, candidates: list[Trajectory], scene: SceneUnderstanding
    ) -> SelectedCandidates:
        ids = [t.id for t in candidates]
        reply = self._ask(
            annotated_image,
            f"Trajectories T1..T{max(ids)} are drawn for goal {goal!r}. Which reach the "
            "goal while avoiding obstacles? Reply 'IDS: <numbers>' then a brief reason.",
        )
        m = re.search(r"IDS:\s*([0-9,\s]+)", reply)
        chosen = [int(n) for n in re.findall(r"\d+", m.group(1))] if m else ids[: max(1, len(ids) // 2)]
        chosen = [i for i in chosen if i in ids] or ids[:1]
        return SelectedCandidates(ids=chosen, reasoning=reply.strip())

    def generate_rationale(self, image, plan: ActionPlan, goal: str) -> str:
        return self._ask(
            image,
            f"Goal {goal!r}. Explain in 2-3 sentences why route T{plan.best_trajectory.id} "
            f"(cost {plan.cost.total_cost:.2f}) is a good plan.",
        ).strip()

    def critique_plan(self, artifact: DebateArtifact, other_view: str | None) -> DebateTurn:
        prompt = (
            f"Critique this route plan for goal {artifact.goal!r}.\n"
            f"Trajectory: {artifact.trajectory_summary}\nCost: {artifact.cost_summary}\n"
            f"Alternatives: {artifact.candidate_summary}\n"
        )
        if other_view:
            prompt += f"\nAnother model said:\n{other_view}\n"
        prompt += (
            "\nEndorse, amend, or reject. If amending, add an 'AMENDED: ...' line. "
            "End with one line: 'VERDICT: <endorse|amend|reject>'."
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

    # ---- Phase 6 vlm-mode grounding (Molmo's native pointing) ----
    def locate_points(self, image, labels: list[str]) -> dict[str, Point]:
        out: dict[str, Point] = {}
        for label in labels:
            pts = self._parse_points(self._ask(image, f"Point to {label}."))
            if pts:
                out[label] = pts[0]
        return out
