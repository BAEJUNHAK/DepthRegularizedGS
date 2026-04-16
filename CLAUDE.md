# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DRGS (Depth-Regularized Gaussian Splatting) — a CVPRW 2024 research implementation that extends 3D Gaussian Splatting with depth regularization for few-shot 3D scene reconstruction. Built on top of the original 3DGS codebase (Inria).

**Known issue**: The README warns of implementation errors in the depth rasterizer. The first commit contains the original paper code.

## Environment Setup

```bash
# Clone with submodules (ZoeDepth, diff-gaussian-rasterization-depth-acc, simple-knn)
git clone https://github.com/robot0321/DepthRegularizedGS.git --recursive

# Option 1: Full conda environment (Python 3.7, PyTorch 1.12.1, CUDA 11.6)
conda env create --file environment.yml
conda activate DepthRegularizedGS

# Option 2: Minimal install on existing 3DGS setup
pip install -e submodules/diff-gaussian-rasterization-depth-acc
pip install pytorch3d
```

## Key Commands

```bash
# Training — baseline 3DGS
python train.py -s <datadir> --eval --port 6311 --model_path <outdir> --resolution 1 --kshot 5 --seed 3

# Training — with depth regularization (the paper's method)
python train.py -s <datadir> --eval --port 6312 --model_path <outdir> --resolution 1 --kshot 5 --seed 3 --depth --usedepthReg

# Rendering
python render.py -s <datadir> -m <model_path> --iteration 30000

# Metrics evaluation
python metrics.py -m <model_path>

# Batch experiments
python scripts/task_producer.py                                    # generate task list
python scripts/task_consumer.py --tasklist ./scripts/all_tasks.txt --gpu 0  # run tasks
python scripts/task_reducer.py --method <method_id>                # aggregate results

# Dataset preparation
python scripts/convertImagename.py --imgfolder <datadir>/images
python scripts/select_samples.py --dset nerfllff --path <datadir>
```

## Architecture

### Training Pipeline (`train.py`)

The training loop combines three loss terms:
1. **Photometric loss**: L1 + lambda_dssim * SSIM (standard 3DGS)
2. **Depth supervision** (`--depth`): L1 loss between rendered depth and GT depth (projected COLMAP points), weighted by confidence mask
3. **Depth regularization** (`--usedepthReg`): Canny edge-guided smoothness loss via `nearMean_map()` — penalizes depth discontinuities except at image edges

Includes early stopping when depth loss plateaus.

### Core Modules

- **`scene/gaussian_model.py`** — `GaussianModel` class: stores per-Gaussian parameters (xyz, SH features, scaling, rotation, opacity) with densification/pruning logic
- **`scene/dataset_readers.py`** — Dataset loading for COLMAP and Blender formats; handles depth map loading and k-shot sampling via `split_index.json`
- **`scene/cameras.py`** — `Camera` (nn.Module): stores image, depth, depth_weight, canny_mask, and view/projection matrices
- **`gaussian_renderer/__init__.py`** — `render()` function: interfaces with the custom CUDA rasterizer, returns rendered image + depth + accumulation
- **`arguments/__init__.py`** — Three config classes: `ModelParams`, `PipelineParams`, `OptimizationParams` (all argparse-based)
- **`utils/loss_utils.py`** — Loss functions including `nearMean_map()` (3x3 conv smoothness) and `image2canny()`

### Submodules

- **`submodules/diff-gaussian-rasterization-depth-acc`** — Custom CUDA rasterizer that outputs depth maps and accumulation (modified from original 3DGS rasterizer)
- **`submodules/simple-knn`** — Fast KNN for point cloud initialization
- **`ZoeDepth/`** — Monocular depth estimation model used for depth confidence weighting

### Config Defaults

Key parameters in `OptimizationParams`: 30K iterations, densification from iter 100-15K every 100 steps, opacity reset every 3K steps, lambda_dssim=0.2. `ModelParams.kshot` defaults to 1000 (effectively all images); set lower for few-shot.
