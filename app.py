"""Optional Gradio UI (SPEC section 13 Phase 7).

Upload a video, set an "from A to B" goal, and view the candidates, selected
plan, traversal GIF, and (if enabled) the debate. Runs on the offline mock
VLM by default — no GPU or API key needed.

Run:  python app.py     (requires `pip install gradio`)
"""

import os

import config
from main import build_debaters, build_vlm, parse_goal
from src.debate.verdict import format_transcript
from src.ingest import frames as frames_mod
from src.runner import persist_outputs, run_viper


def _plan_from_video(video_path, goal, enable_debate):
    place_a, place_b = parse_goal(goal)
    if not place_b:
        raise ValueError('Goal must name a destination, e.g. "from A to B".')

    cfg = config
    cfg.DEBATE_ENABLED = bool(enable_debate)
    frames = frames_mod.extract_frames(video_path, cfg.FRAME_STRIDE, cfg.MAX_FRAMES)
    vlm = build_vlm(cfg)
    a, b = build_debaters(cfg) if cfg.DEBATE_ENABLED else (None, None)
    result = run_viper(frames, place_a, place_b, cfg, vlm, a, b)
    run_dir = os.path.dirname(result.plan.traversal_gif_path)
    persist_outputs(result.plan, run_dir)

    debate_text = (
        format_transcript(result.debate) if result.debate is not None
        else "Debate disabled (enable it with the checkbox)."
    )
    plan = result.plan
    return (
        plan.candidates_image,
        plan.selected_image,
        plan.final_image,
        plan.traversal_gif_path,
        plan.rationale,
        debate_text,
    )


def build_ui():
    import gradio as gr

    with gr.Blocks(title="VIPER") as demo:
        gr.Markdown("# VIPER\nVerifiable Iterative Planning with Ensemble Reasoning")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Top-down video")
                goal = gr.Textbox(label="Goal", value="from receiving bay to dock 4")
                debate = gr.Checkbox(label="Run Claude<->Molmo debate", value=False)
                run_btn = gr.Button("Plan", variant="primary")
            with gr.Column():
                with gr.Tab("Candidates"):
                    out_cand = gr.Image(label="Candidate trajectories")
                with gr.Tab("Selected"):
                    out_sel = gr.Image(label="Selected plan")
                with gr.Tab("Final"):
                    out_final = gr.Image(label="Final annotated frame")
                with gr.Tab("Traversal"):
                    out_gif = gr.Image(label="A->B traversal")
                with gr.Tab("Rationale"):
                    out_rat = gr.Textbox(label="Rationale")
                with gr.Tab("Debate"):
                    out_debate = gr.Textbox(label="Debate transcript", lines=20)

        run_btn.click(
            _plan_from_video,
            inputs=[video, goal, debate],
            outputs=[out_cand, out_sel, out_final, out_gif, out_rat, out_debate],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
