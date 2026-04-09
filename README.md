# Multi-GPU YOLOv2 Hyperparameter Optimization

Distributed stochastic multi-GPU hyperparameter optimization for YOLOv2 object detection in traffic surveillance. Implements Algorithm 1 from the paper using a **Nelder-Mead simplex search** with randomized coefficients across 3 GPUs.

## Architecture

- **CPU**: Central Coordinator — aggregates costs, computes simplex operations (Reflection, Expansion, Contraction, Shrink)
- **3 GPUs**: Distributed Workers — each trains YOLOv2-VGG19 on a separate data partition
- **P = 2** hyperparameters: learning rate (mu) and batch size (sigma)
- **K = P + 1 = 3** simplex vertices, one per GPU

## Algorithm Overview

1. Initialize K = 3 random hyperparameter candidates
2. Sample stochastic coefficients: alpha ~ U[0.9, 1.1], gamma ~ U[1.7, 2.3], beta ~ U[0.3, 0.7]
3. Train all 3 GPUs **in parallel** with their assigned candidates and data partitions
4. Random model exchange among GPUs (mitigates overfitting)
5. Evaluate Cost(theta) = 0.5 * (1/Precision + 1/Recall)
6. Perform Nelder-Mead simplex step (Reflection / Expansion / Contraction / Shrink)
7. Repeat until convergence or n_max iterations

## Project Structure

```
distributed_simplex_3gpu.py   # Main optimizer (Algorithm 1)
sanity_yolov2_googlenet.py    # Worker training script (YOLOv2 + VGG19)
run_simplex_3gpu.bat           # Windows batch launcher
utils/
    yolo_model.py              # YOLOv2 architecture
    yolo_loss.py               # YOLO loss function
    yolov2_decode.py           # Bounding box decoding
    yolo_eval_inference.py     # Evaluation & inference
    csv_dataset.py             # CSV-based dataset loader
    dataset.py                 # Dataset utilities
    box_score.py               # Box scoring metrics
    cross_validation.py        # K-fold cross validation
    train_utils.py             # Training utilities
    training_manager.py        # Training loop manager
    model_manager.py           # Model save/load
    param_manager.py           # Hyperparameter management
    federated_utils.py         # Distributed training helpers
tr1_fix.csv                    # GPU 0 training data
tr2_fix.csv                    # GPU 1 training data
tr3_fix.csv                    # GPU 2 training data
valid1_fix.csv                 # Validation data
test1.csv                      # Test data
```

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support)
- 3 CUDA-enabled GPUs
- Pre-trained VGG19 model weights (`best_vgg19.pt`)

## Setup

1. Clone the repository
2. Create a `models/` directory and place `best_vgg19.pt` inside
3. Create an `rgb_/` directory with the training images (referenced by CSV files)

## How to Run

**Windows:**
```bat
run_simplex_3gpu.bat
```

**Linux / Manual:**
```bash
python distributed_simplex_3gpu.py \
    --model models/best_vgg19.pt \
    --training-data tr1_fix.csv tr2_fix.csv tr3_fix.csv \
    --validation-data valid1_fix.csv \
    --gpu-ids 0,1,2 \
    --max-iters 30 \
    --epochs-per-iter 3
```

**Resume from checkpoint:**
```bash
python distributed_simplex_3gpu.py \
    --model models/best_vgg19.pt \
    --training-data tr1_fix.csv tr2_fix.csv tr3_fix.csv \
    --resume-state simplex_work/simplex_state.json
```

## Configuration

Key parameters in `distributed_simplex_3gpu.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_range` | [1e-4, 5e-3] | Learning rate search space |
| `bs_range` | [4, 16] | Batch size search space |
| `max_iters` | 30 | Maximum simplex iterations |
| `epochs_per_iter` | 3 | Training epochs per worker per iteration |
| `cost_goal` | 1.05 | Early stopping threshold |

## Output

- `simplex_work/simplex_state.json` — checkpoint for resuming
- `simplex_work/global_best.pt` — best model weights found
- `simplex_work/simplex_log_*.json` — full optimization history
