"""
YOLOv2 decode 模組（debug / validation / finetuning 用）

⚠️ 重要說明（請務必閱讀）：
- 目前你的模型輸出為 S = 1（無 spatial grid）
- 代表 detection head 尚未學會 spatial localization
- decode 後紅框集中在畫面中央是「正確且可預期的現象」
- 這不是 bug，而是 finetuning 前的合理狀態
"""

import torch
import numpy as np


def decode_yolov2(
    raw_output,
    anchors,
    num_classes=1,
    conf_threshold=0.3,
    iou_threshold=0.5,  # 為了與 box_score / debug 呼叫介面一致（本函式內不使用）
    img_size=224
):
    """
    Decode YOLOv2 raw output（支援 S=1 與 S>1）

    參數:
      raw_output:
        - Tensor shape (B, A*5)          -> S = 1
        - Tensor shape (B, A*5, S, S)    -> S > 1
      anchors:
        - ndarray shape (A, 2), pixel size (w, h)
      conf_threshold:
        - objectness threshold
      img_size:
        - model input size (e.g. 224)

    回傳:
      boxes: ndarray (N, 4), normalized (x, y, w, h)
      scores: ndarray (N,)
    """


    # --------- 處理 S x S Grid Output ----------
    # raw_output shape: (B, A*(5+num_classes), S, S)
    if raw_output.dim() != 4:
        # 如果不是 4D，嘗試 view 回去 (for safety)
        # 但理論上現在模型都吐 (B, C, H, W)
        pass 

    B, C_total, H, W = raw_output.shape
    A = anchors.shape[0]
    
    # Check dimensions
    expected_channels = A * (5 + num_classes)
    assert C_total == expected_channels, f"Output channels {C_total} != Expected {expected_channels}"
    
    # Reshape: (B, A, 5+C, H, W)
    raw = raw_output.view(B, A, 5 + num_classes, H, W)
    
    # Permute to (B, A, H, W, 5+C) for easier slicing
    raw = raw.permute(0, 1, 3, 4, 2).contiguous()
    
    # Create grid offsets
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=raw_output.device),
        torch.arange(W, device=raw_output.device),
        indexing="ij"
    )
    
    # Slice outputs
    tx = raw[..., 0]
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    to = raw[..., 4]
    
    # Sigmoid activations
    bx = (torch.sigmoid(tx) + grid_x) / W  # normalized x center
    by = (torch.sigmoid(ty) + grid_y) / H  # normalized y center
    
    # Anchor handling
    # anchors is pixels w, h. Need to normalize by img_size or keep usage consistent.
    # Current code assumes anchors are pixels, output boxes are normalized.
    pw = torch.tensor(anchors[:, 0], device=raw_output.device).view(1, A, 1, 1) / img_size
    ph = torch.tensor(anchors[:, 1], device=raw_output.device).view(1, A, 1, 1) / img_size
    
    bw = torch.exp(tw) * pw
    bh = torch.exp(th) * ph
    
    conf = torch.sigmoid(to)
    
    # Flatten everything to (B, N_anchors, ...) where N_anchors = A*H*W
    bx = bx.view(B, -1)
    by = by.view(B, -1)
    bw = bw.view(B, -1)
    bh = bh.view(B, -1)
    conf = conf.view(B, -1)
    
    boxes = []
    scores = []
    
    # Filter by confidence
    for b in range(B):
        mask = conf[b] >= conf_threshold
        if mask.sum() == 0:
            continue
            
        # Get valid boxes
        cur_bx = bx[b][mask]
        cur_by = by[b][mask]
        cur_bw = bw[b][mask]
        cur_bh = bh[b][mask]
        cur_conf = conf[b][mask]
        
        for i in range(len(cur_conf)):
            boxes.append([
                cur_bx[i].item(),
                cur_by[i].item(),
                cur_bw[i].item(),
                cur_bh[i].item()
            ])
            scores.append(cur_conf[i].item())

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)
