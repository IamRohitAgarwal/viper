"""VIPER CLI entry point (SPEC section 2).

Phase 0: argument-parsing shell with stubbed actions. Real behavior is
wired in later phases (frame ingestion in Phase 1, pipeline in Phase 4,
debate in Phase 5).
"""

import argparse
import json
import os
import re
import sys

import config
from src.debate.verdict import format_transcript
from src.evaluation.logger import log_run
from src.ingest import frames as frames_mod
from src.runner import persist_outputs, run_viper
from src.vlm.mock_vlm import MockVLM


def build_vlm(config):
    """Construct the pipeline VLM backend named in config.

    mock (Phases 0-5, offline) | anthropic (Claude) | molmo (Phase 6, Kaggle GPU).
    """
    backend = config.VLM_BACKEND
    if backend == "mock" or not config.USE_VLM:
        return MockVLM(seed=config.SEED, name="mock")
    if backend == "anthropic":
        from src.vlm.anthropic_vlm import AnthropicVLM

        return AnthropicVLM(model=config.CLAUDE_MODEL, name="claude")
    if backend == "molmo":
        from src.vlm.molmo_vlm import MolmoVLM

        return MolmoVLM(config, name="molmo")
    raise ValueError(f"unknown VLM_BACKEND: {backend!r}")


def build_debaters(config):
    """Two debate models.

    Phases 0-5 (mock/offline): two MockVLMs with different seeds -> different
    base verdicts (seed % 3) so they can disagree; model_b concedes to a more
    conservative view to exercise concession tracking.

    Phase 6 (real): Claude as model A, Molmo as model B. Molmo is cached so it
    loads once even if the pipeline backend is also molmo.
    """
    if config.VLM_BACKEND == "mock" or not config.USE_VLM:
        model_a = MockVLM(seed=config.SEED, name=config.DEBATE_MODEL_A)
        model_b = MockVLM(seed=config.SEED + 1, concede=True, name=config.DEBATE_MODEL_B)
        return model_a, model_b

    from src.vlm.anthropic_vlm import AnthropicVLM
    from src.vlm.molmo_vlm import MolmoVLM

    model_a = AnthropicVLM(model=config.DEBATE_CLAUDE_MODEL, name=config.DEBATE_MODEL_A)
    model_b = MolmoVLM(config, name=config.DEBATE_MODEL_B)
    return model_a, model_b


def parse_goal(goal: str) -> tuple[str, str]:
    """Parse a goal of the form 'from {place_a} to {place_b}'.

    Falls back (per resolved Q5): if the string does not match, the whole
    string is treated as the destination (place_b) and the start (place_a)
    is left empty for the runner to default to the image centre.
    """
    if goal is None:
        return "", ""
    match = re.match(r"\s*from\s+(.+?)\s+to\s+(.+?)\s*$", goal, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", goal.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viper",
        description="VIPER - Verifiable Iterative Planning with Ensemble Reasoning. "
        "Plan + validate + animate a route from place A to place B over video.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--video",
        metavar="PATH",
        help="input video file; frames are extracted to data/frames/",
    )
    src.add_argument(
        "--frames-dir",
        metavar="DIR",
        help="use already-extracted frames in DIR (skip extraction)",
    )
    parser.add_argument(
        "--goal",
        metavar='"from A to B"',
        help='task instruction, e.g. "from receiving bay to dock 4"',
    )
    parser.add_argument(
        "--frame",
        type=int,
        metavar="N",
        help="plan on a specific extracted frame index",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="pick a random already-extracted frame (reproducible, offline)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="plan on every extracted frame and emit a resumable metrics summary",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use real Claude for the pipeline + VLM grounding (needs ANTHROPIC_API_KEY); "
        "debate stays off (it needs Molmo on a GPU -- use scripts/kaggle_run.py for that)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --real: flip to the live Claude pipeline locally (no GPU, no debate).
    if args.real:
        config.USE_VLM = True
        config.VLM_BACKEND = "anthropic"
        config.GROUNDING_MODE = "vlm"
        config.DEBATE_ENABLED = False
        print("Real mode: Claude pipeline + VLM grounding (debate disabled).")

    place_a, place_b = parse_goal(args.goal)

    # Stage 0 — obtain scene frames.
    if args.video:
        frames = frames_mod.extract_frames(
            args.video, config.FRAME_STRIDE, config.MAX_FRAMES
        )
        print(f"Extracted {len(frames)} frame(s) to {frames_mod.FRAMES_DIR}/")
    elif args.frames_dir:
        frames = frames_mod.list_frames(args.frames_dir)
        print(f"Found {len(frames)} existing frame(s) in {args.frames_dir}/")
    else:
        print("No --video or --frames-dir given; nothing to do.")
        return 0

    if not frames:
        print("No frames available; aborting.")
        return 1

    # If a specific planning frame was requested, put it first.
    if args.frame is not None:
        frames = [frames[args.frame]] + frames
    elif args.random:
        chosen = frames_mod.pick_random_frame(os.path.dirname(frames[0]), config.SEED)
        frames = [chosen] + [f for f in frames if f != chosen]

    if not place_b:
        print("No destination given (need --goal \"from A to B\"); cannot plan.")
        return 1

    vlm = build_vlm(config)
    model_a, model_b = (build_debaters(config) if config.DEBATE_ENABLED else (None, None))

    if args.batch:
        from src.evaluation.metrics import run_batch

        summary = run_batch(frames, place_a, place_b, config, vlm, model_a, model_b)
        print("\nBatch metrics summary -> outputs/batch/summary.json")
        print(json.dumps(summary, indent=2))
        return 0

    result = run_viper(frames, place_a, place_b, config, vlm, model_a, model_b)
    plan = result.plan
    run_dir = os.path.dirname(plan.traversal_gif_path)
    paths = persist_outputs(plan, run_dir)

    log_data = {
        "goal": args.goal,
        "place_a": place_a,
        "place_b": place_b,
        "best_trajectory_id": plan.best_trajectory.id,
        "cost": plan.cost,
        "all_costs": plan.all_costs,
        "rationale": plan.rationale,
    }
    if result.debate is not None:
        log_data["debate"] = result.debate
        debate_path = os.path.join(run_dir, "debate.txt")
        with open(debate_path, "w", encoding="utf-8") as fh:
            fh.write(format_transcript(result.debate))
        paths["debate"] = debate_path
    log_run(log_data, os.path.join(run_dir, "log.json"))

    print(f"\nPlan complete -> {run_dir}/")
    for name, p in paths.items():
        print(f"  {name:11s} {p}")
    print(f"  log.json    {os.path.join(run_dir, 'log.json')}")
    print(f"\nRationale: {plan.rationale}")
    if result.debate is not None:
        print(f"Debate verdict: {result.debate.final_verdict} "
              f"(winner: {result.debate.winner_model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
