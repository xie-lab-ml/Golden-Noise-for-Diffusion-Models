## Overview

This folder provides two entry points:

- `dataset_collection.py`: Collect training data (prompts, seeds, scores) and optionally save intermediate images/latents.
- `main.py`: Train noise solvers and run test generation.

It is recommended to run commands from within this directory, or add it to `PYTHONPATH` to avoid import issues.


## Quick Setup (Metrics and Reward Models)

Download the `metric` and `reward_model` folders from Google Drive and place them under `training/`.

- Download: [Google Drive](https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt?usp=drive_link)
- Final layout example:

```
training/
  metric/
  reward_model/
  utils/
  solver/
  ...
```


## Environment

- Python 3.9+
- PyTorch with CUDA
- Key deps: `diffusers`, `accelerate`, `torchvision`, `numpy`, `Pillow`, `timm`, `scikit-learn`
- Optional metrics: `hpsv2`, `ImageReward` (and `training/reward_model`)

Example install (adjust as needed):

```bash
pip install diffusers accelerate torchvision timm scikit-learn pillow numpy
# Optional if you use HPSv2 / ImageReward
pip install hpsv2 ImageReward
```


## Data Collection (dataset_collection.py)

What it does:
- Uses `StableDiffusionXLPipeline` to generate original and optimized images from a fixed prompt/seed list.
- Computes metrics (default example enables only HPSv2) and writes one JSON object per line (JSONL).
- Can optionally save original/optimized images, side-by-side previews, and latents (see commented code).

Key behaviors:
- Prompts/seeds are read via `load_pick_prompt('train_60000.json')` (path is hardcoded in the current script).
- Outputs a JSONL file named `SDXL_step_10_training.json` (one sample per line). File extension may appear as ".json" but format is JSONL.
- Intermediate output directory defaults to `datasets/Output_test` (auto-created if missing).

Multi-GPU data collection (index management):
- If different GPUs generate different index ranges, shift `idx` to avoid collisions.
- Examples:
  - GPU 0 (range [0, 10000)): keep `idx` unchanged.
  - GPU 1 (range [10000, 20000)): set `idx = idx + 10000` before saving.
- After collection, merge all JSONL files into a single file for training; `.npz` files need no merging.

Run example (from `training/`):

```bash
python dataset_collection.py \
  --inference_step 50 \
  --size 1024 \
  --T_max 1 \
  --RatioT 1.0 \
  --denoising_cfg 5.5 \
  --inversion_cfg 1.0 \
  --method inversion \
  --output_dir ./datasets/Output_test
```

Outputs:
- `SDXL_step_10_training.json` (JSONL), each line includes:
  - `index`: sample index
  - `seed`: the random seed used to generate
  - `caption`: prompt
  - `original_score_list`: scores of the original image
  - `optimized_score_list`: scores of the optimized image

Tip:
- To collect different metrics, modify `cal_score` in `dataset_collection.py`.


## Training and Testing (main.py)

`main.py` builds the diffusion pipeline (e.g., SDXL), constructs the noise solver, and runs training/validation or test generation.

Important flags:
- `--pipeline`: `SDXL` | `SD2.1` | `DS-turbo` | `DiT` (default `SDXL`)
- `--model`: `unet` | `vit` | `svd_unet` | `svd_unet+unet` | `e_unet` | `svd_unet+unet+dit` (default `svd_unet+unet`)
- `--train` / `--test`: booleans to trigger training or testing
- `--prompt-path`: path to the JSONL from data collection (if multiple JSONLs were produced, merge them first)
- `--data-dir`: directory containing `.npz` noise pairs
- `--epochs`, `--batch-size`, `--num-workers`: regular training args
- `--pretrained-path`: directory for pretrained/weights (default `./training/checkpoints`)
- `--save-ckpt-path`: directory to save checkpoints (default `./training/checkpoints/SDXL-10/svd_unet+unet`)

Auto-creation of directories:
- The script will create `--pretrained-path` and `--save-ckpt-path` if they do not exist.

Recommended training settings:
- Use `--model=svd_unet+unet`.
- Typical usage sets `--pick True`. To filter bad samples, add `--discard True`.
- Filtering logic resides in `training/utils/utils.py` function `load_pick_discard_prompt` (edit as needed).
- If you have multiple `.npz` folders, set `--all-file True`; otherwise keep it `False`.

Training example (from `training/`):

```bash
python main.py \
  --train True \
  --pipeline SDXL \
  --model svd_unet+unet \
  --pick True \
  --discard True \
  --all-file True \
  --prompt-path ./SDXL_step_10_training.json \
  --data-dir ./datasets/noise_pairs_SDXL_10_pick_total/ \
  --epochs 30 \
  --batch-size 64
```


## Data and File Formats

- Each training `.npz` should contain keys `arr_0`, `arr_1`, `arr_2`:
  - `arr_0`: original noise; `arr_1`: optimized noise; `arr_2`: index/ID for prompt lookup
- `--prompt-path` JSONL lines should include at least `index`, `caption`, and `seed`.
- When using `--pick`/`--discard`, samples can be filtered based on scores (see `noise_dataset.py` and `utils/utils.py`).


## Notes

- Pipelines and solvers are implemented under `utils/`, `solver/`, and `model/`.
- To fully wire `prompt_dataset`/`output_dir` CLI args in `dataset_collection.py`, replace the hardcoded paths accordingly.
