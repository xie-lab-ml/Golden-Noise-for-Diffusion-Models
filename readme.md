# Golden Noise for Diffusion Models: A Learning Framework

<div align="center">
      
[![arXiv](https://img.shields.io/badge/arXiv-2411.09502-b31b1b.svg)](https://arxiv.org/abs/2411.09502)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model_GoldenNoiseModel)](https://huggingface.co/Klayand/GoldenNoiseModel)

<img src="./logo.png" width=50%/>

![NPNet Demos](./web_demo.png)
</div>

## TODO List

- [x] Uploaded inference code and corresponding model weights.
- [x] Uploaded data collection, model training, and evaluation code.
- [ ] Upload the latest version of the paper.
- [ ] Check for hidden bugs in the codebase.

## Preliminaries

This section will guide you through setting up the environment required to run and develop with this project.

### 1. Python Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environments, but you can also use `venv` or other tools.

#### Using Conda (Recommended)

```bash
# Create a new environment with Python 3.8+
conda create -n golden-noise python=3.8 -y
conda activate golden-noise
```

#### Using venv (Standard Library)

```bash
# Create a new virtual environment
python -m venv golden-noise-env
# Activate the environment (Windows)
golden-noise-env\Scripts\activate
# Activate the environment (Linux/MacOS)
source golden-noise-env/bin/activate
```

### 2. Install PyTorch

Please follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the correct version for your CUDA driver. For example:

```bash
# Example: install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Other Dependencies

Install the required Python packages:

```bash
pip install diffusers PIL numpy timm argparse einops
```

> **Note:** If you encounter issues with package versions, please refer to the requirements in this README or open an issue.

---



## Dataset

The `data/` folder provides several resources for both training and evaluation:

- **drawbench.csv**, **HPD_prompt.csv**, **pickscore.csv**: Three test sets containing prompts and evaluation data for benchmarking model performance.
- **pickscore_train_prompts.json**: A prompt dataset used for collecting training data, containing prompts for generating training pairs.
- **train_60000.json**: An example training dataset, containing prompts and seeds, demonstrating the format and structure for model training.

### Data Preparation for Training

To prepare training data, use the script `training/dataset_collection.py`. This script reads prompts and seeds from the selected prompt dataset (e.g., `train_60000.json`), generates corresponding source noise and target noise for each prompt, and saves the results into `.npz` files for efficient training.

---

## Training Usage

Training scripts are located in the `training/` directory.

### Main Scripts

- `main.py`: Entry point for model training
- `dataset_collection.py`: Collects and processes training data by generating source and target noise pairs from prompt datasets. Run this script before training to prepare your `.npz` training files.

### Example: Prepare Training Data

```bash
python training/dataset_collection.py --prompt_dataset data/train_60000.json --output_dir data/train_npz/
```

This will generate `.npz` files containing the source and target noise for each prompt in the specified dataset.

### Example: Start Training (SDXL NPNet)

```bash
python training/main.py --pipeline=SDXL --model=svd_unet+unet --train=True --pick=True --all-file=True --discard=True --test=True --postfix=_test --evaluate=False --discard=True
```

### main.py Command-Line Arguments Explanation

- `--ddp`: Whether to use Distributed Data Parallel (DDP) training. Default: False.
- `--pipeline`: Model pipeline to use. Options: 'SDXL', 'SD2.1', 'DS-turbo', 'DiT'. Default: 'SDXL'.
- `--model`: Model architecture. Options: 'unet', 'vit', 'svd_unet', 'svd_unet+unet', 'e_unet', 'svd_unet+unet+dit'. Default: 'svd_unet+unet'.
- `--benchmark-type`: Benchmark type. Options: 'pick', 'draw'. Default: 'pick'.
- `--train`: Whether to run training. Default: False.
- `--test`: Whether to run testing/inference. Default: False.
- `--postfix`: Postfix for output files and checkpoints. Default: '_hps_sdxl_step_10_random_noise'.
- `--acculumate-steps`: Number of gradient accumulation steps. Default: 64.
- `--pick`: Whether to use PickScore for filtering or evaluation. Default: False.
- `--do-classifier-free-guidance`: Whether to use classifier-free guidance during generation. Default: True.
- `--inference-step`: Number of inference steps for the diffusion process. Default: 10.
- `--size`: Image size (height and width). Default: 1024.
- `--RatioT`: Ratio parameter for training (custom use). Default: 1.0.
- `--guidance-scale`: Guidance scale for classifier-free guidance. Default: 5.5.
- `--guidance-rescale`: Rescale factor for guidance. Default: 0.0.
- `--all-file`: Whether to use all files in the dataset. Default: False.
- `--epochs`: Number of training epochs. Default: 30.
- `--batch-size`: Batch size for training. Default: 64.
- `--num-workers`: Number of worker processes for data loading. Default: 16.
- `--metric-version`: Metric for evaluation. Options: 'PickScore', 'HPS v2', 'AES', 'ImageReward'. Default: 'PickScore'.
- `--prompt-path`: Path to the prompt JSON file for training. Default: './sdxl_step_10_training_seed.json'.
- `--data-dir`: Directory containing the training data (noise pairs). Default: './datasets/noise_pairs_SDXL_10_pick_total/'.
- `--pretrained-path`: Path to pretrained model weights. Default: './checkpoints/SDXL-10'.
- `--save-ckpt-path`: Path to save model checkpoints. Default: './checkpoints/SDXL-10/svd_unet+unet'.
- `--discard`: Whether to discard bad samples during training. Default: False.


### Example: Evaluate the model performance

We provide evaluation code in `training/metric/cal_metric.py` for quantitative assessment of generated images.

**Usage Notes:**
- You need to prepare the corresponding prompt file and the generated images for evaluation.
- The image folder should follow this structure:

```
images/
  origin/
    0.png
    1.png
    ...
  optim/
    0.png
    1.png
    ...
```
- `origin/` contains images generated by the baseline model, and `optim/` contains images generated by the optimized (e.g., NPNet) model. The file names should correspond to the prompt order.
- Before running the evaluation, make sure you have downloaded the required weights for the reward model you want to use (e.g., PickScore, HPSv2, ImageReward, etc.).

**Example command:**
```bash
python training/metric/cal_metric.py --prompt_file data/your_prompts.csv --image_folder images/ --metric PickScore
```

Replace `--metric` with your desired evaluation metric and adjust paths as needed.


## NPNet Pipeline Inference Usage Guide😄 

### Overview

This guide provides instructions on how to use the NPNet, a noise prompt network aims to transform the random Gaussian noise into golden noise, by adding a small desirable perturbation derived from the text prompt to boost the overall quality and semantic faithfulness of the synthesized images.

Here we provide the inference code which supports different models like ***Stable Diffusion XL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT.***. 

Besides, you can apply the checkpoint of NPNet on SDXL to the models like ***SDXL-Lightning, LCM, and PCM***. The visualizations of these three models are shown below:

![lcm](https://github.com/user-attachments/assets/26d10388-d72d-41f3-b951-1a89a4ca77e0)

We **directly use the checkpoint from SDXL** to ***SDXL-Lightning, LCM, and PCM***, and evaluate them on Geneval dataset:

| Model           | Method    | PickScore↑   | HPSv2↑    | AES↑   | ImageReward↑     | CLIPScore↑ |
|-----------------|-----------|---------|----------|---------|----------|---------|
| SDXL-Lightning(4-step)  | standard  | 22.85 | 29.12 | 5.65 | 59.02 | 0.8093 |
| SDXL-Lightning(4-step)  | ours      | **23.03** | **29.71** | **5.71** | **72.67** | **0.8150** |
| LCM(4-step)             | standard  | 22.30 | 26.52 | 5.49 | 33.21 | 0.8050 |
| LCM(4-step)             | ours      | **22.38** | **26.83** | **5.55** | **37.08** | **0.8123** |
| PCM(8-step)             | standard  | 22.05 | 26.98  | 5.52 | 23.28 | 0.8031 |
| PCM(8-step)             | ours      | **22.22** | **27.59**  | **5.56** | **35.01** | **0.8175** |


The results demonstrate the effectiveness of our NPNet on few-steps image generation.


### Usage👀️ 

To use the NPNet pipeline, you need to run the `npnet_pipeline.py` script with appropriate command-line arguments. Below are the available options:

#### Command-Line Arguments

- `--pipeline`: Select the model pipeline (`SDXL`, `DreamShaper`, `DiT`). Default is `SDXL`.
- `--prompt`: The textual prompt based on which the image will be generated. Default is "A banana on the left of an apple."
- `--inference-step`: Number of inference steps for the diffusion process. Default is 50.
- `--cfg`: Classifier-free guidance scale. Default is 5.5.
- `--pretrained-path`: Path to the pretrained model weights. Default is a specified path in the script.
- `--size`: The size (height and width) of the generated image. Default is 1024.

#### Running the Script

Run the script from the command line by navigating to the directory containing `npnet_pipeline.py` and executing:

```
python npnet_pipeline.py --pipeline SDXL --prompt "A banana on the left of an apple." --size 1024
```

This command will generate an image based on the prompt "A banana on the left of an apple." using the Stable Diffusion XL model with an image size of 1024x1024 pixels.

#### Output🎉️ 

The script will save two images:

- A standard image generated by the diffusion model.
- A golden image generated by the diffusion model with the NPNet.

Both images will be saved in the current directory with names based on the model and prompt.

### Pre-trained Weights Download❤️

We provide the pre-trained NPNet weights of Stable Diffusion XL, DreamShaper-xl-v2-turbo, and Hunyuan-DiT with [google drive](https://drive.google.com/drive/folders/1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt?usp=drive_link)

### Citation:
If you find our code useful for your research, please cite our paper.

```
@inproceedings{zhou2025golden,
      title={Golden Noise for Diffusion Models: A Learning Framework}, 
      author={Zikai Zhou and Shitong Shao and Lichen Bai and Shufei Zhang and Zhiqiang Xu and Bo Han and Zeke Xie},
      booktitle={International Conference on Computer Vision},
      year={2025},
}
```

### 🙏 Acknowledgements

We thank the community and contributors for their invaluable support in developing NPNet. 
We thank @DataCTE for constructing the ComfyUI of NPNet inference code [ComfyUI](https://github.com/DataCTE/ComfyUI_Golden-Noise).
We thank @asagi4 for constructing the ComfyUI of NPNet inference code [ComfyUI](https://github.com/asagi4/ComfyUI-NPNet).

---




