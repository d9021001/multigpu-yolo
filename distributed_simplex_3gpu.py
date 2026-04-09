"""
distributed_simplex_3gpu.py
============================
Distributed Stochastic Multi-GPU Hyperparameter Optimization (Algorithm 1)

Implements the full Nelder-Mead simplex search from the paper:
  - CPU = Central Coordinator (aggregates scores, computes simplex operations)
  - 3 GPUs = Workers (train YOLOv2-VGG19 with different hyperparameter candidates)
  - P=2 hyperparameters: learning rate (μ) and batch size (σ)
  - K=P+1=3 simplex vertices, each assigned to one GPU

Steps per iteration:
  1. Each GPU trains its candidate for jᵁ epochs
  2. Random model exchange among GPUs
  3. CPU evaluates Cost(θ) = 0.5*(1/Precision + 1/Recall) via 5-fold CV
  4. Reflection → Expansion / Contraction based on cost comparison
  5. Repeat until convergence or n_max iterations

Usage:
  python distributed_simplex_3gpu.py --model best.pt
  python distributed_simplex_3gpu.py --model best.pt --max-iters 30 --epochs-per-iter 3
"""

import os
import sys
import time
import shutil
import random
import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONFIG = {
    # Hyperparameter search space (paper Section 3.1)
    "lr_range": [1e-4, 5e-3],       # μ ∈ [1×10⁻⁴, 5×10⁻³]
    "bs_range": [4, 16],             # σ ∈ [4, 16]

    # Simplex coefficients (paper Algorithm 1, line 2)
    "alpha_range": [0.9, 1.1],       # Reflection:  α ~ U[0.9, 1.1]
    "gamma_range": [1.7, 2.3],       # Expansion:   γ ~ U[1.7, 2.3]
    "beta_range":  [0.3, 0.7],       # Contraction: β ~ U[0.3, 0.7]

    # Training
    "num_gpus": 3,                   # K = P+1 = 3
    "max_iters": 30,                 # n_max iterations
    "epochs_per_iter": 3,            # jᵁ epochs per training round
    "num_folds": 5,                  # 5-fold CV

    # YOLOv2 / Evaluation
    "backbone": "vgg19",
    "image_size": 224,
    "conf_threshold": 0.1,
    "iou_threshold": 0.75,
    "optimizer": "sgd",

    # Data (per-GPU training splits)
    "training_data": ["tr1_fix.csv", "tr2_fix.csv", "tr3_fix.csv"],
    "validation_data": "valid1_fix.csv",
    "test_data": "test1.csv",

    # Convergence
    "cost_goal": 1.05,              # Stop if cost ≤ this value
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed Stochastic Multi-GPU Simplex Optimizer (Algorithm 1)")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to initial YOLOv2-VGG19 weights (best.pt)")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_CONFIG["max_iters"],
                        help=f"Maximum simplex iterations (default: {DEFAULT_CONFIG['max_iters']})")
    parser.add_argument("--epochs-per-iter", type=int, default=DEFAULT_CONFIG["epochs_per_iter"],
                        help=f"Training epochs per worker per iteration jᵁ (default: {DEFAULT_CONFIG['epochs_per_iter']})")
    parser.add_argument("--cost-goal", type=float, default=DEFAULT_CONFIG["cost_goal"],
                        help=f"Early stop when cost ≤ this value (default: {DEFAULT_CONFIG['cost_goal']})")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2",
                        help="Comma-separated GPU IDs to use (default: 0,1,2)")
    parser.add_argument("--training-data", type=str, nargs='+',
                        default=DEFAULT_CONFIG["training_data"],
                        help="Per-GPU training CSV files (e.g. tr1_fix.csv tr2_fix.csv tr3_fix.csv)")
    parser.add_argument("--validation-data", type=str, default=DEFAULT_CONFIG["validation_data"])
    parser.add_argument("--test-data", type=str, default=DEFAULT_CONFIG["test_data"])
    parser.add_argument("--work-dir", type=str, default="simplex_work",
                        help="Working directory for intermediate outputs")
    parser.add_argument("--resume-state", type=str, default=None,
                        help="Resume from a previous simplex state JSON")

    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================
def clip_hyperparams(lr, bs, config=DEFAULT_CONFIG):
    """Clip hyperparameters to valid search space."""
    lr = float(np.clip(lr, config["lr_range"][0], config["lr_range"][1]))
    bs = int(np.clip(round(bs), config["bs_range"][0], config["bs_range"][1]))
    return lr, bs


def sample_random_coefficients(config=DEFAULT_CONFIG):
    """Sample stochastic simplex coefficients (Algorithm 1, line 2)."""
    alpha = random.uniform(*config["alpha_range"])
    gamma = random.uniform(*config["gamma_range"])
    beta  = random.uniform(*config["beta_range"])
    return alpha, gamma, beta


def compute_cost(precision, recall):
    """
    Cost(θ) = 0.5 * (1/Precision + 1/Recall)
    Paper Equations (5)-(6): harmonic cost penalizing imbalanced P/R.
    """
    if precision > 0 and recall > 0:
        return 0.5 * (1.0 / precision + 1.0 / recall)
    return 50.0  # Penalty for degenerate predictions


def compute_centroid(vertices, costs, exclude_worst=True):
    """
    Compute centroid of all vertices except the worst.
    Paper: θ̄ = (Σθ − θ_worst) / (K-1)
    """
    order = np.argsort(costs)
    if exclude_worst:
        indices = order[:-1]
    else:
        indices = order
    centroid = np.mean([vertices[i] for i in indices], axis=0)
    return centroid


# ============================================================================
# GPU Worker: Train YOLOv2 with given hyperparameters
# ============================================================================
def train_worker(gpu_id, backbone, lr, bs, epochs, tr_data, val_data,
                 model_dir, resume_pt, conf_thresh, iou_thresh):
    """
    Launch a training worker on a specific GPU.
    Calls sanity_yolov2_googlenet.py as a subprocess.

    Returns: (cost, precision, recall, f1, model_path)
    """
    os.makedirs(model_dir, exist_ok=True)

    cmd = [
        sys.executable, "sanity_yolov2_googlenet.py",
        "--base-network", backbone,
        "--gpu-id", str(gpu_id),
        "--training-data", tr_data,
        "--validation-data", val_data,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(bs),
        "--model-save-dir", model_dir,
        "--single-round",
        "--warm-start",
        "--conf-threshold", str(conf_thresh),
        "--iou-threshold", str(iou_thresh),
    ]
    if resume_pt and os.path.exists(resume_pt):
        cmd.extend(["--resume", resume_pt])

    print(f"    [GPU {gpu_id}] Training: lr={lr:.6f}, bs={bs}, "
          f"epochs={epochs}, resume={resume_pt}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, timeout=3600)

        # Parse metrics from stdout
        metrics = {}
        for line in result.stdout.split('\n'):
            if "Precision:" in line:
                metrics['p'] = float(line.split(":")[1].strip())
            if "Recall   :" in line:
                metrics['r'] = float(line.split(":")[1].strip())
            if "F1 Score :" in line:
                metrics['f1'] = float(line.split(":")[1].strip())

        p = metrics.get('p', 0.0)
        r = metrics.get('r', 0.0)
        f1 = metrics.get('f1', 0.0)
        cost = compute_cost(p, r)

        # Find the saved model
        model_path = os.path.join(model_dir, f"last_{backbone}.pt")
        if not os.path.exists(model_path):
            # Try alternative naming
            for fname in os.listdir(model_dir):
                if fname.endswith('.pt'):
                    model_path = os.path.join(model_dir, fname)
                    break

        return cost, p, r, f1, model_path

    except subprocess.TimeoutExpired:
        print(f"    [GPU {gpu_id}] TIMEOUT after 1 hour")
        return 50.0, 0.0, 0.0, 0.0, None
    except subprocess.CalledProcessError as e:
        print(f"    [GPU {gpu_id}] ERROR (exit {e.returncode})")
        if e.stdout:
            print(f"    [stdout tail] {e.stdout[-300:]}")
        if e.stderr:
            print(f"    [stderr tail] {e.stderr[-300:]}")
        return 50.0, 0.0, 0.0, 0.0, None
    except Exception as e:
        print(f"    [GPU {gpu_id}] Exception: {e}")
        return 50.0, 0.0, 0.0, 0.0, None


def train_workers_parallel(gpu_ids, backbone, vertices, epochs,
                           tr_data_list, val_data, work_dir, resume_pts,
                           conf_thresh, iou_thresh):
    """
    Launch K workers in PARALLEL across K GPUs.
    Each GPU trains with its assigned hyperparameter candidate AND its own
    training data partition (tr1_fix.csv, tr2_fix.csv, tr3_fix.csv).

    Args:
        tr_data_list: list of per-GPU training CSV paths

    Returns: list of (cost, precision, recall, f1, model_path)
    """
    import concurrent.futures

    K = len(vertices)
    results = [None] * K

    def worker_task(k):
        lr, bs = clip_hyperparams(vertices[k][0], vertices[k][1])
        model_dir = os.path.join(work_dir, f"gpu{gpu_ids[k]}")
        return train_worker(
            gpu_id=gpu_ids[k],
            backbone=backbone,
            lr=lr, bs=bs,
            epochs=epochs,
            tr_data=tr_data_list[k],
            val_data=val_data,
            model_dir=model_dir,
            resume_pt=resume_pts[k],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )

    # True parallel execution on different GPUs
    with concurrent.futures.ThreadPoolExecutor(max_workers=K) as executor:
        futures = {executor.submit(worker_task, k): k for k in range(K)}
        for future in concurrent.futures.as_completed(futures):
            k = futures[future]
            try:
                results[k] = future.result()
            except Exception as e:
                print(f"    [Worker {k}] Failed: {e}")
                results[k] = (50.0, 0.0, 0.0, 0.0, None)

    return results


# ============================================================================
# Random Model Exchange (Paper Section 3.3, Algorithm 1 line 4)
# ============================================================================
def random_model_exchange(model_paths, work_dir):
    """
    Randomly shuffle and exchange trained models among GPUs.
    Each GPU receives a model trained under different conditions,
    accelerating convergence and mitigating overfitting.
    """
    valid_paths = [(i, p) for i, p in enumerate(model_paths) if p and os.path.exists(p)]
    if len(valid_paths) < 2:
        return model_paths

    indices = [i for i, _ in valid_paths]
    paths = [p for _, p in valid_paths]

    # Random permutation (ensuring at least one exchange)
    shuffled = list(paths)
    while shuffled == paths and len(paths) > 1:
        random.shuffle(shuffled)

    # Copy files to temp then overwrite
    exchange_dir = os.path.join(work_dir, "_exchange_tmp")
    os.makedirs(exchange_dir, exist_ok=True)

    new_paths = list(model_paths)
    for i, (idx, orig_path) in enumerate(valid_paths):
        tmp_path = os.path.join(exchange_dir, f"exchanged_{idx}.pt")
        shutil.copy2(shuffled[i], tmp_path)
        new_paths[idx] = tmp_path

    print(f"    [Exchange] Models shuffled: {[os.path.basename(p) for p in shuffled]}")
    return new_paths


# ============================================================================
# Main Simplex Optimizer Loop (Algorithm 1)
# ============================================================================
def run_simplex(args):
    config = DEFAULT_CONFIG.copy()
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    K = len(gpu_ids)
    assert K >= 3, f"Need at least 3 GPUs, got {K}"

    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(work_dir, f"simplex_log_{timestamp}.json")

    print("=" * 70)
    print("  Distributed Stochastic Multi-GPU Simplex Optimizer")
    print("  (Algorithm 1: Nelder-Mead with Random Coefficients)")
    print("=" * 70)
    print(f"  GPUs         : {gpu_ids}")
    print(f"  Model        : {args.model}")
    print(f"  Max iters    : {args.max_iters}")
    print(f"  Epochs/iter  : {args.epochs_per_iter}")
    print(f"  Search space : μ ∈ {config['lr_range']}, σ ∈ {config['bs_range']}")
    print(f"  Cost goal    : {args.cost_goal}")
    for k, td in enumerate(args.training_data):
        print(f"  GPU {gpu_ids[k]} train : {td}")
    print(f"  Validation   : {args.validation_data}")
    print(f"  Work dir     : {work_dir}")
    print("=" * 70)

    # ---- Algorithm 1, Line 1: Initialize θ = [θ₁, θ₂, θ₃] randomly ----
    if args.resume_state and os.path.exists(args.resume_state):
        with open(args.resume_state) as f:
            state = json.load(f)
        vertices = [np.array(v) for v in state["vertices"]]
        costs = state["costs"]
        n_start = state["iteration"] + 1
        best_cost = state["best_cost"]
        best_model = state["best_model"]
        print(f"\n  [Resume] From iteration {n_start}, best_cost={best_cost:.4f}")
    else:
        vertices = [
            np.array([random.uniform(*config["lr_range"]),
                       random.randint(*config["bs_range"])])
            for _ in range(K)
        ]
        costs = [50.0] * K
        n_start = 0
        best_cost = float('inf')
        best_model = args.model

    # Copy initial model to work dir
    global_best_pt = os.path.join(work_dir, "global_best.pt")
    if not os.path.exists(global_best_pt) or n_start == 0:
        shutil.copy2(args.model, global_best_pt)

    # Per-GPU resume paths
    resume_pts = [global_best_pt] * K

    history = []

    # ---- Algorithm 1, Line 2: Set random factors α, γ, β ----
    # (re-sampled each iteration)

    # ---- Main Loop: Algorithm 1, Lines 3-25 ----
    for n in range(n_start, args.max_iters):
        alpha, gamma, beta = sample_random_coefficients(config)

        print(f"\n{'='*70}")
        print(f"  Iteration {n+1}/{args.max_iters}  "
              f"α={alpha:.3f}  γ={gamma:.3f}  β={beta:.3f}")
        print(f"{'='*70}")

        for k in range(K):
            lr, bs = clip_hyperparams(vertices[k][0], vertices[k][1])
            print(f"  θ_{k+1} = (μ={lr:.6f}, σ={bs})")

        # ---- Line 3: Train GPU_k with θ_k for jᵁ epochs (parallel) ----
        iter_dir = os.path.join(work_dir, f"iter_{n:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        results = train_workers_parallel(
            gpu_ids=gpu_ids,
            backbone=config["backbone"],
            vertices=vertices,
            epochs=args.epochs_per_iter,
            tr_data_list=args.training_data,
            val_data=args.validation_data,
            work_dir=iter_dir,
            resume_pts=resume_pts,
            conf_thresh=config["conf_threshold"],
            iou_thresh=config["iou_threshold"]
        )

        # ---- Line 4: Random model exchange ----
        model_paths = [r[4] for r in results]
        resume_pts = random_model_exchange(model_paths, iter_dir)

        # ---- Line 5: Evaluate Cost(θ_k) ----
        for k in range(K):
            cost_k, p_k, r_k, f1_k, _ = results[k]
            costs[k] = cost_k
            lr, bs = clip_hyperparams(vertices[k][0], vertices[k][1])
            print(f"  θ_{k+1}: Cost={cost_k:.4f}  P={p_k:.4f}  R={r_k:.4f}  "
                  f"F1={f1_k:.4f}  (μ={lr:.6f}, σ={bs})")

        # Track global best
        best_k = int(np.argmin(costs))
        if costs[best_k] < best_cost:
            best_cost = costs[best_k]
            if results[best_k][4] and os.path.exists(results[best_k][4]):
                shutil.copy2(results[best_k][4], global_best_pt)
                print(f"  ★ NEW GLOBAL BEST: Cost={best_cost:.4f} from GPU {gpu_ids[best_k]}")

        # ---- Line 6: Sort vertices by cost ----
        order = np.argsort(costs)
        i_best = order[0]
        i_second = order[1]
        i_worst = order[-1]

        # ---- Centroid: θ̄ = (Σθ − θ_worst) / (K-1) ----
        centroid = compute_centroid(vertices, costs)

        # ---- Line 8: Reflection: θ_R = θ̄ + α·(θ̄ − θ_worst) ----
        theta_worst = np.array(vertices[i_worst])
        theta_reflect = centroid + alpha * (centroid - theta_worst)
        lr_r, bs_r = clip_hyperparams(theta_reflect[0], theta_reflect[1])
        theta_reflect = np.array([lr_r, bs_r])

        print(f"\n  [Reflection] θ_R = (μ={lr_r:.6f}, σ={bs_r})")

        # Train reflection candidate on worst GPU (using that GPU's data partition)
        reflect_dir = os.path.join(iter_dir, "reflect")
        cost_r, p_r, r_r, f1_r, model_r = train_worker(
            gpu_id=gpu_ids[i_worst],
            backbone=config["backbone"],
            lr=lr_r, bs=bs_r,
            epochs=args.epochs_per_iter,
            tr_data=args.training_data[i_worst],
            val_data=args.validation_data,
            model_dir=reflect_dir,
            resume_pt=global_best_pt,
            conf_thresh=config["conf_threshold"],
            iou_thresh=config["iou_threshold"]
        )
        print(f"  [Reflection] Cost_R={cost_r:.4f}  P={p_r:.4f}  R={r_r:.4f}  F1={f1_r:.4f}")

        # ---- Decision Logic (Lines 10-22) ----
        if cost_r < costs[i_best]:
            # Reflection is better than best → try Expansion
            # ---- Line 13: θ_E = θ̄ + γ·(θ_R − θ̄) ----
            theta_expand = centroid + gamma * (theta_reflect - centroid)
            lr_e, bs_e = clip_hyperparams(theta_expand[0], theta_expand[1])
            theta_expand = np.array([lr_e, bs_e])

            print(f"  [Expansion] θ_E = (μ={lr_e:.6f}, σ={bs_e})")

            expand_dir = os.path.join(iter_dir, "expand")
            cost_e, p_e, r_e, f1_e, model_e = train_worker(
                gpu_id=gpu_ids[i_worst],
                backbone=config["backbone"],
                lr=lr_e, bs=bs_e,
                epochs=args.epochs_per_iter,
                tr_data=args.training_data[i_worst],
                val_data=args.validation_data,
                model_dir=expand_dir,
                resume_pt=global_best_pt,
                conf_thresh=config["conf_threshold"],
                iou_thresh=config["iou_threshold"]
            )
            print(f"  [Expansion] Cost_E={cost_e:.4f}  P={p_e:.4f}  R={r_e:.4f}  F1={f1_e:.4f}")

            if cost_e < cost_r:
                # Accept expansion
                vertices[i_worst] = theta_expand
                costs[i_worst] = cost_e
                if model_e: resume_pts[i_worst] = model_e
                print(f"  → Accept EXPANSION (Cost_E={cost_e:.4f} < Cost_R={cost_r:.4f})")
            else:
                # Accept reflection
                vertices[i_worst] = theta_reflect
                costs[i_worst] = cost_r
                if model_r: resume_pts[i_worst] = model_r
                print(f"  → Accept REFLECTION (Cost_R={cost_r:.4f})")

        elif cost_r < costs[i_second]:
            # Reflection is acceptable (better than second-worst)
            vertices[i_worst] = theta_reflect
            costs[i_worst] = cost_r
            if model_r: resume_pts[i_worst] = model_r
            print(f"  → Accept REFLECTION (Cost_R={cost_r:.4f} < Cost_2nd={costs[i_second]:.4f})")

        else:
            # Reflection failed → try Contraction
            # ---- Line 19: θ_C = θ̄ + β·(θ_worst − θ̄) ----
            theta_contract = centroid + beta * (theta_worst - centroid)
            lr_c, bs_c = clip_hyperparams(theta_contract[0], theta_contract[1])
            theta_contract = np.array([lr_c, bs_c])

            print(f"  [Contraction] θ_C = (μ={lr_c:.6f}, σ={bs_c})")

            contract_dir = os.path.join(iter_dir, "contract")
            cost_c, p_c, r_c, f1_c, model_c = train_worker(
                gpu_id=gpu_ids[i_worst],
                backbone=config["backbone"],
                lr=lr_c, bs=bs_c,
                epochs=args.epochs_per_iter,
                tr_data=args.training_data[i_worst],
                val_data=args.validation_data,
                model_dir=contract_dir,
                resume_pt=global_best_pt,
                conf_thresh=config["conf_threshold"],
                iou_thresh=config["iou_threshold"]
            )
            print(f"  [Contraction] Cost_C={cost_c:.4f}  P={p_c:.4f}  R={r_c:.4f}  F1={f1_c:.4f}")

            if cost_c < costs[i_worst]:
                vertices[i_worst] = theta_contract
                costs[i_worst] = cost_c
                if model_c: resume_pts[i_worst] = model_c
                print(f"  → Accept CONTRACTION (Cost_C={cost_c:.4f} < Cost_worst={costs[i_worst]:.4f})")
            else:
                # Shrink: all vertices move toward best
                print(f"  → SHRINK: all vertices toward best")
                for j in range(K):
                    if j != i_best:
                        vertices[j] = vertices[i_best] + 0.5 * (
                            np.array(vertices[j]) - np.array(vertices[i_best]))
                        lr_s, bs_s = clip_hyperparams(vertices[j][0], vertices[j][1])
                        vertices[j] = np.array([lr_s, bs_s])

        # Update global best after this iteration
        iter_best_k = int(np.argmin(costs))
        if costs[iter_best_k] < best_cost:
            best_cost = costs[iter_best_k]
            best_lr, best_bs = clip_hyperparams(
                vertices[iter_best_k][0], vertices[iter_best_k][1])
            print(f"  ★ GLOBAL BEST updated: Cost={best_cost:.4f} "
                  f"(μ={best_lr:.6f}, σ={best_bs})")

        # Log iteration
        iter_record = {
            "iteration": n,
            "best_cost": float(best_cost),
            "costs": [float(c) for c in costs],
            "vertices": [v.tolist() for v in vertices],
            "alpha": alpha, "gamma": gamma, "beta": beta,
            "best_model": global_best_pt,
            "timestamp": datetime.now().isoformat()
        }
        history.append(iter_record)

        # Save state for resume
        state = {
            "iteration": n,
            "vertices": [v.tolist() for v in vertices],
            "costs": [float(c) for c in costs],
            "best_cost": float(best_cost),
            "best_model": global_best_pt,
            "history": history
        }
        state_path = os.path.join(work_dir, "simplex_state.json")
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        # ---- Line 25: Check convergence ----
        if best_cost <= args.cost_goal:
            print(f"\n  ✓ Cost goal reached: {best_cost:.4f} ≤ {args.cost_goal}")
            break

    # ---- Algorithm 1, Line 26: Return θ* with minimum Costᴸ ----
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)

    final_best_k = int(np.argmin(costs))
    final_lr, final_bs = clip_hyperparams(
        vertices[final_best_k][0], vertices[final_best_k][1])
    print(f"  Best Cost      : {best_cost:.4f}")
    print(f"  Best θ*        : μ={final_lr:.6f}, σ={final_bs}")
    print(f"  Best Model     : {global_best_pt}")
    print(f"  Total Iters    : {n+1}")
    print(f"  State saved to : {state_path}")

    # Save full log
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History log    : {log_path}")

    return best_cost, global_best_pt


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    if not os.path.exists("sanity_yolov2_googlenet.py"):
        print("[ERROR] sanity_yolov2_googlenet.py not found in current directory.")
        print("        Please run this script from the project root.")
        sys.exit(1)

    # Validate per-GPU training data
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    if len(args.training_data) != len(gpu_ids):
        print(f"[ERROR] Expected {len(gpu_ids)} training data files (one per GPU), "
              f"got {len(args.training_data)}: {args.training_data}")
        print(f"        Usage: --training-data tr1_fix.csv tr2_fix.csv tr3_fix.csv")
        sys.exit(1)
    for td in args.training_data:
        if not os.path.exists(td):
            print(f"[ERROR] Training data not found: {td}")
            sys.exit(1)

    best_cost, best_model = run_simplex(args)
    print(f"\nDone. Best model: {best_model} (Cost={best_cost:.4f})")
