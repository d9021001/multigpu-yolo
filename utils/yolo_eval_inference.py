"""
yolo_eval_inference.py
======================

SINGLE SOURCE OF TRUTH for YOLOv2 inference-time evaluation.

Design contract
---------------
- This module defines the *only* evaluation logic that is allowed
  to be used by both:
    - sanity_yolov2_googlenet.py
    - eval_yolov2_inference_safe.py
- If numbers differ, it is a BUG.

Key properties
--------------
- Inference-style decode + NMS
- Explicit conf / IoU thresholds
- VOC-style TP / FP / FN (1 GT matched once)
- No training-only helpers
- Behavior is deterministic and reusable
"""

import numpy as np
import torch
from torchvision.ops import nms

from utils.yolov2_decode import decode_yolov2
from utils.box_score import compute_iou_xywh


def evaluate_inference(
    model,
    dataset,
    anchors_px,
    image_size,
    num_classes,
    conf_threshold=0.3,
    iou_threshold=0.5,
    nms_iou_threshold=0.5,
    device=None,
    use_eval_mode=False,
):
    """
    Unified inference evaluation.

    Returns
    -------
    dict with keys:
        tp, fp, fn, precision, recall, f1
    """

    if device is None:
        device = next(model.parameters()).device

    # Selective mode to follow "don't touch other models" rule
    if use_eval_mode:
        model.eval()
    else:
        # Default legacy behavior for GoogLeNet, VGG, ResNet
        model.train()

    total_gt = 0
    tp = 0
    fp = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            gt = sample["boxes"]  # [x, y, w, h] normalized top-left

            if gt is None or len(gt) == 0:
                continue

            total_gt += len(gt)

            pred = model(img)

            boxes, scores = decode_yolov2(
                pred,
                anchors=anchors_px,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                img_size=image_size,
            )

            if len(boxes) == 0:
                continue

            # ---------- NMS ----------
            boxes_xyxy = []
            for b in boxes:
                cx, cy, w, h = b
                boxes_xyxy.append([
                    cx - w / 2,
                    cy - h / 2,
                    cx + w / 2,
                    cy + h / 2,
                ])

            boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
            scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

            keep = nms(boxes_xyxy, scores_t, nms_iou_threshold)

            boxes = [boxes[i] for i in keep.tolist()]

            # ---------- GT matching ----------
            gt_arr = np.array(gt)
            gt_arr = np.array(gt)
            gt_center = gt_arr.copy()
            # gt is already [cx, cy, w, h] from CSVDataset
            gt_center[:, 0] = gt_arr[:, 0]
            gt_center[:, 1] = gt_arr[:, 1]

            matched_gt = set()

            for b in boxes:
                ious = compute_iou_xywh(b, gt_center)
                if len(ious) == 0:
                    fp += 1
                    continue

                max_iou = np.max(ious)
                gt_idx = int(np.argmax(ious))

                if max_iou >= iou_threshold and gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_idx)
                else:
                    fp += 1

    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    
    # Debug: Check if any boxes are being found at all
    if precision == 0 and recall == 0:
        with torch.no_grad():
            full_max_conf = 0.0
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                img = sample["image"].unsqueeze(0).to(device)
                pred = model(img) # (1, A*(5+C), H, W)
                conf = torch.sigmoid(pred[:, 4::(5+num_classes), :, :])
                m = conf.max().item()
                if m > full_max_conf: full_max_conf = m
            print(f"  [DEBUG] Metrics are zero. Max objectness confidence found: {full_max_conf:.4f} (Threshold: {conf_threshold})", flush=True)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_inference_per_sample(
    model,
    dataset,
    anchors_px,
    image_size,
    num_classes,
    conf_threshold=0.3,
    iou_threshold=0.5,
    nms_iou_threshold=0.5,
    device=None,
    use_eval_mode=False,
):
    """
    Evaluates each sample individually.

    Returns
    -------
    list of dicts, each with keys:
        imageFilename, tp, fp, fn, precision, recall
    """
    if device is None:
        device = next(model.parameters()).device

    if use_eval_mode:
        model.eval()
    else:
        model.train()

    results = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            gt = sample["boxes"]
            image_filename = dataset.data.iloc[i]['imageFilename']

            if gt is None or len(gt) == 0:
                results.append({
                    "imageFilename": image_filename,
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 1.0, "recall": 1.0
                })
                continue

            total_gt_sample = len(gt)
            tp_sample = 0
            fp_sample = 0

            pred = model(img)
            boxes, scores = decode_yolov2(
                pred,
                anchors=anchors_px,
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                img_size=image_size,
            )

            if len(boxes) > 0:
                boxes_xyxy = []
                for b in boxes:
                    cx, cy, w, h = b
                    boxes_xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
                scores_t = torch.tensor(scores, dtype=torch.float32, device=device)
                keep = nms(boxes_xyxy, scores_t, nms_iou_threshold)
                boxes = [boxes[idx] for idx in keep.tolist()]

                gt_center = np.array(gt)
                matched_gt = set()
                for b in boxes:
                    ious = compute_iou_xywh(b, gt_center)
                    if len(ious) == 0:
                        fp_sample += 1
                        continue
                    max_iou = np.max(ious)
                    gt_idx = int(np.argmax(ious))
                    if max_iou >= iou_threshold and gt_idx not in matched_gt:
                        tp_sample += 1
                        matched_gt.add(gt_idx)
                    else:
                        fp_sample += 1

            fn_sample = total_gt_sample - tp_sample
            precision_sample = tp_sample / (tp_sample + fp_sample) if (tp_sample + fp_sample) > 0 else 0.0
            recall_sample = tp_sample / total_gt_sample if total_gt_sample > 0 else 0.0

            results.append({
                "imageFilename": image_filename,
                "tp": tp_sample,
                "fp": fp_sample,
                "fn": fn_sample,
                "precision": precision_sample,
                "recall": recall_sample
            })

    return results
