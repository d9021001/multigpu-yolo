"""
sanity_yolov2_googlenet.py
==========================

Sanity check for YOLOv2 + GoogLeNet (single class)
--------------------------------------------------
Purpose:
- Reproduce MATLAB trainYOLOv2ObjectDetector behavior
- Verify YOLOv2 + GoogLeNet can learn (objectness + bbox)
- NO hyperparameter search
- Small data / few epochs / heavy visualization

Data:
- Reuse existing CSVDataset
- Single class
- image size = 224

This script is intentionally SIMPLE and EXPLICIT.
"""

import argparse
import os
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.csv_dataset import CSVDataset, collate_fn_variable_boxes
from utils.yolo_model import build_yolov2_model
from utils.yolo_loss import YOLOv2GridLoss
from utils.yolov2_decode import decode_yolov2
from utils.box_score import compute_iou_xywh, calculate_box_score
from utils.training_manager import create_optimizer, create_scheduler
from utils.param_manager import ParamManager
from utils.yolo_eval_inference import evaluate_inference

# ------------------ Config ------------------
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANCHORS_PX = np.array(
    [
        [8.9, 11.2],
        [14.6, 18.7],
        [25.5, 28.9],
        [49.6, 45.7],
    ],
    dtype=np.float32,
)

NUM_ANCHORS = len(ANCHORS_PX)
NUM_CLASSES = 1

# ------------------ Global Logger ------------------
class Logger:
    def __init__(self, log_path=None):
        self.terminal = sys.stdout
        self.log_file = None
        if log_path:
            # Type hint or explicit assignment to avoid lint error
            f = open(log_path, "a", encoding="utf-8")
            self.log_file = f
            print(f"[INFO] Logging to {log_path}", flush=True)

    def write(self, message):
        self.terminal.write(message)
        f = self.log_file
        if f is not None:
            f.write(message)
            f.flush()

    def flush(self):
        self.terminal.flush()
        f = self.log_file
        if f is not None:
            f.flush()

    def close(self):
        f = self.log_file
        if f is not None:
            f.close()

# ------------------ Visualization ------------------
def draw_boxes(img, boxes, color, box_format='center'):
    """
    Draw boxes on image
    box_format: 'center' -> [cx, cy, w, h] (normalized)
                'topleft' -> [x, y, w, h] (normalized)
    """
    H, W = img.shape[:2]
    for b in boxes:
        if box_format == 'center':
            cx, cy, w, h = b
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
        else: # topleft
            x, y, w, h = b
            x1 = int(x * W)
            y1 = int(y * H)
            x2 = int((x + w) * W)
            y2 = int((y + h) * H)
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# ------------------ Evaluation ------------------
def evaluate_cost(model, dataset, device, conf_threshold=0.3, iou_threshold=0.5):
    """
    評估模型並計算 Cost (複製自 train_yolov2_gpu.py logic)
    Cost formula: (1/P + 1/R) / 2
    """
    model.eval()
    
    box_score_result = calculate_box_score(
        model,
        dataset,
        device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    
    desired_box_n = box_score_result['desired_box_n']
    box_n = box_score_result['box_n']
    candi_n = box_score_result['candi_n']
    
    # 計算指標
    fn = max(0, desired_box_n - box_n)
    tp = max(1, box_n)
    fp = candi_n - box_n
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # === 穩定版成本函數 ===
    EPS = 1e-3
    p = max(precision, EPS)
    r = max(recall, EPS)
    raw_cost = (1 / p + 1 / r) / 2
    cost = min(raw_cost, 50.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cost': cost
    }

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--csv-path", type=str, default=None, help="Legacy: Path to dataset CSV (use --training-data instead)")
    parser.add_argument("--training-data", type=str, default="tr_fix.csv", help="训练資料檔案")
    parser.add_argument("--validation-data", type=str, default="valid1_fix.csv", help="驗證資料檔案")
    parser.add_argument("--test-data", type=str, default="test1.csv", help="測試資料檔案")
    parser.add_argument("--data-root", type=str, default=".", help="資料根目錄")
    parser.add_argument("--skip-test-eval", action="store_true", help="Skip test set evaluation")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu-id", type=int, default=0, choices=[0, 1, 2, 3], help="GPU ID (0=RTX A6000)")
    
    # Model configuration
    parser.add_argument("--base-network", type=str, default="googlenet", 
                        choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 'googlenet', 'mobilenetv2'],
                        help="Base network architecture")
    parser.add_argument("--boxes", type=int, default=3, help="Number of top predicted boxes to visualize")
    parser.add_argument("--model-save-dir", type=str, default="debug_vis/trained_backbones", help="Model save directory")
    
    # Checkpoint/Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--warm-start", action="store_true", help="Load weights only, ignore optimizer state/epoch")
    
    # Optimizer & Scheduler
    parser.add_argument("--optimizer", type=str, default="sgd", choices=['sgd', 'adam', 'adamw'], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", choices=['reduce_on_plateau', 'cosine', 'none'], help="Scheduler type")

    # YOLOv2 Loss & Backbone Control
    parser.add_argument("--lambda-coord", type=float, default=5.0, help="YOLOv2 box/coord loss weight")
    parser.add_argument("--lambda-noobj", type=float, default=0.5, help="YOLOv2 no-object loss weight")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone network")
    
    # Termination Criteria
    parser.add_argument("--target-precision", type=float, default=0.7)
    parser.add_argument("--target-recall", type=float, default=0.7)
    parser.add_argument('--single-round', action='store_true', help='Only run one round')
    parser.add_argument("--log-file", type=str, default=None, help="Path to save log output")
    
    # Thresholds
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold for evaluation")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IOU threshold for evaluation")

    args = parser.parse_args()
    
    # Setup Logger
    if args.log_file:
        sys.stdout = Logger(args.log_file)
    
    # Device Setup
    if torch.cuda.is_available():
        target_device_idx = args.gpu_id
        try:
            DEVICE = torch.device(f"cuda:{target_device_idx}")
            # Test device
            torch.cuda.get_device_name(DEVICE)
        except Exception as e:
            print(f"[ERROR] Invalid device cuda:{target_device_idx}: {e}")
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")

    print(f"[DEVICE] {DEVICE} (Requested GPU ID: {args.gpu_id})")

    # Data Normalization: Only apply for mobilenetv2 to avoid affecting other models
    transform = None
    if args.base_network == "mobilenetv2":
        print("[INFO] Applying ImageNet normalization for MobileNetV2")
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = CSVDataset(args.training_data, image_size=IMAGE_SIZE, root_dir=args.data_root, transform=transform)
    param_manager = ParamManager()
    
    round_id = 0
    while True:
        round_id += 1
        print(f"\n{'='*60}")
        print(f"Round {round_id} - GPU {args.gpu_id} - Backbone: {args.base_network}")
        print(f"{'='*60}")

        current_lr = args.lr
        current_batch_size = args.batch_size
        
        if not args.single_round:
            try:
                x_param = param_manager.load_param(args.gpu_id)
                if x_param is not None:
                    current_batch_size = int(np.ceil(x_param[0]))
                    current_lr = float(x_param[1])
                    print(f"Loaded params: BatchSize={current_batch_size}, LR={current_lr}")
            except: pass

        if current_lr < 1e-4: current_lr = 1e-3

        loader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=True,
            collate_fn=collate_fn_variable_boxes,
        )

        model = build_yolov2_model(
            base_network=args.base_network,
            num_classes=NUM_CLASSES,
            anchor_boxes=ANCHORS_PX,
            image_size=IMAGE_SIZE,
        ).to(DEVICE)

        # Run dummy forward pass to initialize lazy modules
        print("[INIT] Running dummy forward pass to initialize model structure...", flush=True)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
            model(dummy_input)

        # Setup Save Dir
        save_dir = Path(args.model_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # -------- Load Checkpoint (Resume) --------
        start_epoch = 1
        best_loss = float('inf')
        
        resume_path = args.resume
        if not resume_path and args.warm_start:
             potential_best = save_dir / f"best_{args.base_network}.pt"
             if potential_best.exists():
                 resume_path = str(potential_best)

        if resume_path and os.path.exists(resume_path):
            print(f"[RESUME] Loading checkpoint from {resume_path}...", flush=True)
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = int(checkpoint['epoch']) + 1
            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
            print(f"  [*] Loaded weights from epoch {checkpoint.get('epoch', 'unknown')}", flush=True)
        else:
            if resume_path:
                print(f"[WARNING] Checkpoint {resume_path} not found. Starting fresh.", flush=True)

        # Verification Mode (epochs=0)
        if args.epochs == 0:
            print(f"\n[INFO] Verification Mode: Skipping training loop for {args.base_network}...", flush=True)
        else:
            print(f"\n[INFO] Starting training loop for {args.base_network} at epoch {start_epoch}...", flush=True)
            
            criterion = YOLOv2GridLoss(
                anchors=ANCHORS_PX, num_classes=NUM_CLASSES, img_size=IMAGE_SIZE,
                lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj
            )
            
            if args.freeze_backbone:
                print("[INFO] Freezing backbone parameters", flush=True)
                for name, param in model.named_parameters():
                    if args.base_network in name:
                        param.requires_grad = False

            optimizer = create_optimizer(model, current_lr, args.optimizer)
            scheduler = create_scheduler(optimizer, args.scheduler)
            
            if resume_path and os.path.exists(resume_path) and not args.warm_start:
                 if 'optimizer_state_dict' in checkpoint:
                     try:
                         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                         print("  [*] Optimizer state loaded.", flush=True)
                     except:
                         print("  [WARNING] Optimizer state incompatible.", flush=True)

            best_f1 = 0.0
            for epoch in range(start_epoch, start_epoch + args.epochs):
                model.train()
                total_loss = 0.0
                batch_count = 0
                for batch in loader:
                    imgs = batch["image"].to(DEVICE)
                    gts = batch["boxes"]

                    optimizer.zero_grad()
                    preds = model(imgs)
                    loss = criterion(preds, gts)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  [WRN] Loss NaN/Inf at batch {batch_count}", flush=True)
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    batch_count += 1
                    if batch_count % 5 == 0 or batch_count == len(loader):
                        print(f"  [Epoch {epoch}] Batch {batch_count}/{len(loader)} - Loss: {loss.item():.4f}", flush=True)

                avg_loss = total_loss / max(1, batch_count)
                print(f"\n[EPOCH {epoch}] FINISHED. Avg Loss: {avg_loss:.4f}", flush=True)
                
                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_loss)
                    else:
                        scheduler.step()
                
                # Evaluation metrics for saving criteria
                metrics = evaluate_inference(
                    model=model, dataset=dataset, anchors_px=ANCHORS_PX,
                    image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, device=DEVICE,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    use_eval_mode=True # Use eval mode for all
                )
                p, r, f1 = metrics['precision'], metrics['recall'], metrics['f1']
                print(f"  [METRICS] P={p:.4f}, R={r:.4f}, F1={f1:.4f}", flush=True)

                # Save Checkpoints
                if f1 > best_f1:
                    best_f1 = f1
                    print(f"  [*] New best F1: {best_f1:.4f}. Saving best model...", flush=True)
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
                        'backbone': args.base_network, 'metrics': metrics
                    }, save_dir / f"best_{args.base_network}.pt")
                
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
                    'backbone': args.base_network, 'metrics': metrics
                }, save_dir / f"last_{args.base_network}.pt")

                # Criteria check
                if p >= args.target_precision and r >= args.target_recall:
                    print(f"\n[CRITERIA MET] Backbone {args.base_network} reached targets!", flush=True)
                    break

        # Final Evaluation
        print(f"\n[FINAL] Evaluating {args.base_network}...", flush=True)
        best_pt = save_dir / f"best_{args.base_network}.pt"
        if best_pt.exists():
            model.load_state_dict(torch.load(best_pt, map_location=DEVICE)['model_state_dict'])
        
        is_mbv2 = (args.base_network == "mobilenetv2")
        metrics = evaluate_inference(
            model=model, dataset=dataset, anchors_px=ANCHORS_PX,
            image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, device=DEVICE,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            use_eval_mode=True
        )
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")

        if metrics['precision'] >= args.target_precision and metrics['recall'] >= args.target_recall:
            print(f"\n[FINAL VERIFICATION] {args.base_network} Criteria MET!", flush=True)
        else:
            print(f"\n[FINAL VERIFICATION] {args.base_network} Criteria NOT MET!", flush=True)

        param_manager.save_cost(args.gpu_id, 1.0 - metrics['recall'])
        if args.single_round: break
            
    print("SANITY YOLOv2 DONE")

if __name__ == "__main__":
    main()
