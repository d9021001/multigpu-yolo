"""
Federated utilities for best-model based Federated Averaging (FedAvg)

This module ONLY operates on best models:
    models/checkpoints/best/best_model_gpu_X.pth

It produces federated models:
    models/checkpoints/federated/federated_round_<N>.pth
"""

import torch
from pathlib import Path
import time


def fedavg_best_models(best_model_paths):
    """
    Perform FedAvg on best model checkpoints.

    Args:
        best_model_paths (list[Path]): paths to best_model_gpu_X.pth

    Returns:
        dict: averaged model_state_dict
    """
    assert len(best_model_paths) > 0, "No best model paths provided"

    avg_state = {}
    n = len(best_model_paths)

    for p in best_model_paths:
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt["model_state_dict"]

        for k, v in state.items():
            if k not in avg_state:
                avg_state[k] = v.clone()
            else:
                avg_state[k] += v

    for k in avg_state:
        avg_state[k] /= n

    return avg_state


def wait_for_files(paths, timeout=600, poll_interval=2):
    """
    Barrier utility: wait until all files exist.

    Args:
        paths (list[Path])
        timeout (int): seconds
    """
    start = time.time()
    while True:
        if all(p.exists() for p in paths):
            return True
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for files: {paths}")
        time.sleep(poll_interval)
