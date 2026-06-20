"""Phase 6 entry point for Kaggle/Colab (real Claude + Molmo).

Run this in a GPU notebook with Internet enabled and ANTHROPIC_API_KEY set as
a secret. It flips config to real models, grounds via the VLM, and runs a
resumable batch (progressive save: re-running skips finished frames).

Example (notebook cell):
    !python scripts/kaggle_run.py --video data/videos/clip.mp4 \
        --goal "from receiving bay to dock 4" --backend molmo --debate

Cheap config tip (SPEC Phase 6): use --backend molmo for pipeline reasoning
(its native pointing grounds places) and let Claude handle debate turns.
"""

import argparse
import json
import os
import sys

# Allow running as `python scripts/kaggle_run.py` from the repo root: ensure the
# repo root (parent of scripts/) is importable so `import config` / `src...` work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_anthropic_key():
    """Prefer a Kaggle Secret named ANTHROPIC_API_KEY; fall back to env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["ANTHROPIC_API_KEY"] = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    except Exception:  # noqa: BLE001 - not on Kaggle / secret absent
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VIPER real-model run (Kaggle/Colab).")
    parser.add_argument("--video", required=True)
    parser.add_argument("--goal", required=True, help='e.g. "from receiving bay to dock 4"')
    parser.add_argument("--backend", choices=["anthropic", "molmo"], default="molmo")
    parser.add_argument("--grounding", choices=["pretagged", "vlm"], default="vlm")
    parser.add_argument("--debate", action="store_true")
    parser.add_argument("--batch", action="store_true", help="plan over all frames (resumable)")
    args = parser.parse_args(argv)

    import config

    config.USE_VLM = True
    config.VLM_BACKEND = args.backend
    config.GROUNDING_MODE = args.grounding
    config.DEBATE_ENABLED = bool(args.debate)

    _load_anthropic_key()
    if (args.backend == "anthropic" or args.debate) and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found (set it as a Kaggle Secret).")

    from main import build_debaters, build_vlm, parse_goal
    from src.ingest import frames as frames_mod
    from src.runner import persist_outputs, run_viper

    place_a, place_b = parse_goal(args.goal)
    frames = frames_mod.extract_frames(args.video, config.FRAME_STRIDE, config.MAX_FRAMES)
    print(f"Extracted {len(frames)} frames.")

    gen_vlm = build_vlm(config)
    model_a, model_b = build_debaters(config) if config.DEBATE_ENABLED else (None, None)

    if args.batch:
        from src.evaluation.metrics import run_batch

        summary = run_batch(frames, place_a, place_b, config, gen_vlm, model_a, model_b)
        print(json.dumps(summary, indent=2))
        return 0

    result = run_viper(frames, place_a, place_b, config, gen_vlm, model_a, model_b)
    run_dir = os.path.dirname(result.plan.traversal_gif_path)
    persist_outputs(result.plan, run_dir)
    print(f"Done -> {run_dir}/  rationale: {result.plan.rationale}")
    if result.debate is not None:
        print(f"Debate verdict: {result.debate.final_verdict} ({result.debate.winner_model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
