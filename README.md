# Multi-GPU YOLOv2 Hyperparameter Optimization in MATLAB

This project implements a **parallel hyperparameter optimization framework** for training YOLOv2 object detectors using multiple GPUs simultaneously. It utilizes a **Master-Worker architecture** with robust file-based synchronization to distribute training tasks across available hardware resources.

## 🚀 Key Features

*   **Multi-GPU Parallelism**: concurrently trains multiple YOLOv2 models on separate GPUs (specifically targeted for GTX 1050 Ti 4GB cards).
*   **Robust Synchronization**: Uses a low-overhead file polling mechanism (`exist` + `pause`) to coordinate between the Main (Optimizer) and Worker (Trainer) processes. This ensures negligible impact on training performance (<0.01% overhead).
*   **Fault Tolerance**:
    *   **Automatic Fallback**: Switches to simpler network architectures if pretrained models (GoogLeNet/ResNet) are unavailable.
    *   **NaN Recovery**: Automatically detects numerical explosions (`NaN`/`Inf` loss) and recovers by reverting to previous valid states or applying penalties, ensuring the optimization loop never crashes.
    *   **License/Resource Handling**: Designed to be resilient against temporary MATLAB license checkouts.
*   **Manual Cross-Validation**: Implements 5-fold cross-validation manually to remove dependencies on the Statistics and Machine Learning Toolbox.

## 📂 Project Structure

*   **`mainPool.m`**: The **Master** script. It runs the optimization algorithm (Evolutionary/Genetic-style search), generates hyperparameter sets (`x1`, `x2`, `x3`), and aggregates costs.
*   **`trainYolov2gpu1.m`, `gpu2.m`, `gpu3.m`**: The **Worker** scripts. Each is pinned to a specific GPU. They:
    1.  Wait for parameters (`xFolder/xN.mat`).
    2.  Train a YOLOv2 model using those parameters.
    3.  Perform 5-fold cross-validation.
    4.  Return the cost (1/Precision + 1/Recall) to `cFolder/cN.mat`.
*   **`run_*.bat`**: Batch scripts to launch instances in separate processes, ensuring true parallelism (bypassing MATLAB's single-threaded nature).

## 🛠️ Prerequisites

*   **MATLAB R2025a** (or compatible)
*   **Deep Learning Toolbox**
*   **Computer Vision Toolbox**
*   **Parallel Computing Toolbox**
*   **CUDA-enabled GPU(s)** (Tested with NVIDIA GTX 1050 Ti)

## 🚦 How to Run

1.  **Start the Master**:
    Double-click `run_main.bat`. This initializes the optimization loop and waits for workers.

2.  **Start the Workers**:
    Double-click `run_gpu1.bat`, `run_gpu2.bat`, and `run_gpu3.bat`.
    *   Each script will launch a separate MATLAB command window.
    *   They will automatically detect their assigned GPU and begin processing tasks from the Main script.

3.  **Monitor Progress**:
    *   Real-time training progress (Loss/RMSE) is displayed in each Worker's command window.
    *   Logs are saved to `debug_main.log` and `debug_gpuN.log`.

## ⚙️ Configuration

*   **Optimization Settings**: Modified in `mainPool.m`.
*   **Training Options**: Modified in `trainYolov2gpu*.m` (Learning Rate, Batch Size, Epochs, etc.).
*   **Dataset**: The scripts expect `tr1_fix.csv` (Training) and `valid1_fix.csv` (Validation) in the root directory.

## 📊 Performance Note

The file-based synchronization introduces a latency of ~1 second per iteration. Given that a single training run takes ~3.5 hours, the communication overhead is **negligible (<0.01%)** and does not impact optimization efficiency.
