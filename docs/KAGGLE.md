# Phase 6 — Running real models on Kaggle

Phases 0–5 + 7 run offline on the mock VLM. Phase 6 swaps in **real Claude**
(via the Anthropic API) and **real Molmo-7B-D** (local 4-bit on GPU). Molmo
needs a CUDA GPU, so this runs on **Kaggle or Colab**, not a CPU-only / Intel
machine.

## 1. Notebook setup

1. New Kaggle Notebook → **Settings → Accelerator → GPU** (T4 x2 or P100).
2. **Settings → Internet → On** (needed to download Molmo + call Claude).
3. **Add-ons → Secrets** → add `ANTHROPIC_API_KEY` (only if you use Claude for
   the pipeline or the debate). Molmo alone needs no key.
4. Upload this repo (or `git clone` it) and your top-down clip into
   `data/videos/`.

## 2. Install the GPU dependencies

These are commented out in `requirements.txt` (they don't install on the
local Windows box). On Kaggle:

```bash
pip install anthropic "transformers==4.45.2" "tokenizers>=0.20,<0.21" \
    accelerate bitsandbytes einops torchvision
```

(`torch` is already present in the Kaggle image.)

> **Pin transformers to 4.45.2.** Molmo-7B-D-0924's modeling code predates
> newer transformers. With a newer version the 4-bit load fails with
> `AttributeError: 'MolmoForCausalLM' object has no attribute
> 'all_tied_weights_keys'`. 4.45.2 is the version Molmo was built against.
> The pip dependency-resolver warnings about RAPIDS (cudf/cuml/dask-cuda) are
> cosmetic — VIPER doesn't use those packages.

## 3. Run

```bash
# Molmo for pipeline + Claude<->Molmo debate, VLM grounding, resumable batch:
python scripts/kaggle_run.py --video data/videos/clip.mp4 \
    --goal "from receiving bay to dock 4" \
    --backend molmo --grounding vlm --debate --batch
```

Flags:
- `--backend molmo|anthropic` — pipeline reasoning model.
- `--grounding vlm|pretagged` — `vlm` grounds named places via Molmo's native
  pointing (no `zones.json` needed); `pretagged` still reads `data/zones.json`.
- `--debate` — enable the Claude↔Molmo ensemble validation.
- `--batch` — plan over all frames with **progressive, resumable** save (re-run
  skips frames whose `log.json` already exists — important if the GPU session
  times out).

## 4. What "done" looks like (SPEC §13 Phase 6)

- `python scripts/kaggle_run.py ... --backend molmo` runs GENERATE on a real
  VLM, and `--debate` yields a real Claude↔Molmo verdict end-to-end.
- Flipping `VLM_BACKEND` / `DEBATE_ENABLED` / `GROUNDING_MODE` activates the
  real models with **no code edits** (the script just sets these in `config`).

## Notes / things to verify on first run

- **Molmo loads once.** The model is cached in `src/vlm/molmo_vlm.py`
  (`_MODEL_CACHE`), so using it for both pipeline and debate loads it a single
  time.
- **Point coordinate scale.** Molmo emits point coords as percentages (0–100);
  `MolmoVLM._parse_points` divides by 100. If your Molmo revision emits pixels
  instead, adjust there.
- **Token cost.** Claude images are downscaled to 768 px (longest side) in
  `AnthropicVLM`. A ~50-frame run with Sonnet for debate stays to a few dollars.
- **Concession probe cost.** The debate does 2 extra solo calls per frame to
  measure debate-vs-solo agreement (see `src/debate/relay.py`).
