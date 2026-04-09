"""
YOLOv2 Grid Loss 實現
支援 S x S grid matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YOLOv2GridLoss(nn.Module):
    """
    標準 YOLOv2 Grid Loss
    
    機制：
    1. 將 GT boxes 映射到 S x S grid cell
    2. 每個 cell 的負責 anchor 計算座標損失 (MSE)
    3. 負責的 anchor 計算 objectness 損失 (BCE/MSE) -> target=1
    4. 不負責的 anchor 計算 no-obj 損失 (BCE/MSE) -> target=0
    5. 分類損失 (CrossEntropy)
    """
    
    def __init__(self, 
                 anchors, 
                 num_classes=1, 
                 lambda_coord=5.0, 
                 lambda_noobj=0.5,
                 lambda_obj=1.0,
                 lambda_class=1.0,
                 img_size=224):
        super(YOLOv2GridLoss, self).__init__()
        self.anchors = anchors  # list of [w, h] or numpy array
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.img_size = img_size
        
        # Loss functions
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def compute_iou(self, box1, box2):
        """
        計算 box1 與 box2 的 IoU
        box format: [x, y, w, h] (normalized or pixels, assumed same scale)
        """
        b1x1, b1y1 = box1[..., 0] - box1[..., 2]/2, box1[..., 1] - box1[..., 3]/2
        b1x2, b1y2 = box1[..., 0] + box1[..., 2]/2, box1[..., 1] + box1[..., 3]/2
        b2x1, b2y1 = box2[..., 0] - box2[..., 2]/2, box2[..., 1] - box2[..., 3]/2
        b2x2, b2y2 = box2[..., 0] + box2[..., 2]/2, box2[..., 1] + box2[..., 3]/2
        
        inter_rect_x1 = torch.max(b1x1, b2x1)
        inter_rect_y1 = torch.max(b1y1, b2y1)
        inter_rect_x2 = torch.min(b1x2, b2x2)
        inter_rect_y2 = torch.min(b1y2, b2y2)
        
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
        
        b1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
        b2_area = (b2x2 - b2x1) * (b2y2 - b2y1)
        
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
        return iou

    def forward(self, output, targets):
        """
        output: (B, A*(5+C), S, S)  -- conv output
        targets: list of length B, 其中每個 element 是 (N_obj, 4) or (N_obj, 5) tensor
                 format: [cx, cy, w, h, class_id] (normalized 0~1)
        """
        device = output.device
        B, C_out, H, W = output.shape
        A = self.num_anchors
        C = self.num_classes
        
        # Reshape output: (B, A, 5+C, H, W) -> (B, A, H, W, 5+C)
        output = output.view(B, A, 5 + C, H, W)
        output = output.permute(0, 1, 3, 4, 2).contiguous()  # (B, A, H, W, 5+C)
        
        # Predictions
        # tx, ty: sigmoid -> offset from grid top-left (0~1)
        # tw, th: exp -> anchor scale
        # to: sigmoid -> objectness
        # class: probability
        
        pred_tx = output[..., 0]
        pred_ty = output[..., 1]
        pred_tw = output[..., 2]
        pred_th = output[..., 3]
        pred_conf = output[..., 4]
        pred_class = output[..., 5:]
        
        # Sigmoid for x, y, conf, class
        pred_tx = torch.sigmoid(pred_tx)
        pred_ty = torch.sigmoid(pred_ty)
        pred_conf = torch.sigmoid(pred_conf)
        # Note: tw, th output raw values for exponentiation
        
        # Masks for loss calculation
        # obj_mask: 1 if object exists in that cell-anchor, else 0
        # noobj_mask: 1 if no object, but we ignore if IoU > threshold (YOLOv3 style) or just 1-obj (YOLOv2 style)
        # Here we use standard YOLOv2 logic
        obj_mask = torch.zeros(B, A, H, W, device=device)
        noobj_mask = torch.ones(B, A, H, W, device=device)
        
        # Target tensors
        tx_target = torch.zeros(B, A, H, W, device=device)
        ty_target = torch.zeros(B, A, H, W, device=device)
        tw_target = torch.zeros(B, A, H, W, device=device)
        th_target = torch.zeros(B, A, H, W, device=device)
        tconf_target = torch.zeros(B, A, H, W, device=device)
        tcls_target = torch.zeros(B, A, H, W, device=device, dtype=torch.long)
        
        # Grid steps
        stride = self.img_size / H
        grid_x = torch.arange(W, device=device).repeat(H, 1)
        grid_y = torch.arange(H, device=device).repeat(W, 1).t()
        
        # Anchors tensor
        scaled_anchors = torch.tensor(self.anchors, device=device) / stride  # anchors in grid scale
        
        total_loss = 0.0
        n_correct = 0
        n_total_obj = 0
        
        for b in range(B):
            target = targets[b] # (N, 4 or 5)
            if target is None or len(target) == 0:
                continue
                
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target).float()
            
            target = target.to(device)
            # Ensure target has class id if missing (default 0)
            if target.shape[1] == 4:
                target = torch.cat([target, torch.zeros(target.shape[0], 1, device=device)], dim=1)
                
            n_total_obj += len(target)
            
            # GT boxes in grid coordinates
            # target format normalized [0, 1] => * S
            gx = target[:, 0] * W
            gy = target[:, 1] * H
            gw = target[:, 2] * W
            gh = target[:, 3] * H
            
            # Grid indices (integer)
            gi = gx.long().clamp(0, W-1)
            gj = gy.long().clamp(0, H-1)
            
            # Find best anchor for each GT
            gt_box_wh = target[:, 2:4] * self.img_size # pixel scale for IoU matching with anchors
            anchor_shapes = torch.tensor(self.anchors, device=device) # pixel scale
            
            # Calculate IoU between each GT and all anchors (centered)
            # gt: (N, 2), anchors: (A, 2)
            # Expand to (N, A, 2)
            gt_wh_exp = gt_box_wh.unsqueeze(1).repeat(1, A, 1)
            anc_wh_exp = anchor_shapes.unsqueeze(0).repeat(len(target), 1, 1)
            
            inter_w = torch.min(gt_wh_exp[..., 0], anc_wh_exp[..., 0])
            inter_h = torch.min(gt_wh_exp[..., 1], anc_wh_exp[..., 1])
            inter_area = inter_w * inter_h
            union_area = (gt_wh_exp[..., 0] * gt_wh_exp[..., 1]) + \
                         (anc_wh_exp[..., 0] * anc_wh_exp[..., 1]) - inter_area
            ious = inter_area / union_area
            
            # Best anchor index for each GT
            best_ious, best_n = ious.max(1)
            
            for i in range(len(target)):
                a = best_n[i]
                j, k = gj[i], gi[i]
                
                # Mark responsible cell-anchor
                obj_mask[b, a, j, k] = 1
                noobj_mask[b, a, j, k] = 0
                
                # Coordinate targets
                # tx, ty are offsets relative to grid top-left
                tx_target[b, a, j, k] = gx[i] - k.float()
                ty_target[b, a, j, k] = gy[i] - j.float()
                
                # tw, th are log-space scaling of anchor
                # gw = pw * exp(tw)  =>  tw = log(gw / pw)
                # pw is anchor width in grid units
                pw = scaled_anchors[a, 0]
                ph = scaled_anchors[a, 1]
                
                tw_target[b, a, j, k] = torch.log(gw[i] / pw + 1e-16)
                th_target[b, a, j, k] = torch.log(gh[i] / ph + 1e-16)
                
                # Objectness score target
                # Some implementations use IoU, others use 1.0. Let's use 1.0 for simplicity.
                tconf_target[b, a, j, k] = 1.0
                
                # Class target
                tcls_target[b, a, j, k] = target[i, 4].long()
        
        # Loss Calculation
        
        # 1. Coordinate Loss (MSE) - only on obj cells
        # We assume pred_tx, pred_ty are sigmoid output (0-1), so we compare with relative offset (0-1)
        # tw, th are raw outputs, compared with log scaling factor
        loss_x = self.mse(pred_tx[obj_mask.bool()], tx_target[obj_mask.bool()])
        loss_y = self.mse(pred_ty[obj_mask.bool()], ty_target[obj_mask.bool()])
        loss_w = self.mse(pred_tw[obj_mask.bool()], tw_target[obj_mask.bool()])
        loss_h = self.mse(pred_th[obj_mask.bool()], th_target[obj_mask.bool()])
        
        loss_coord = (loss_x + loss_y + loss_w + loss_h) * self.lambda_coord
        
        # 2. Objectness/No-obj Loss (BCE same or MSE)
        # Using MSE for objectness often works better in YOLO v2
        # obj cells
        loss_conf_obj = self.mse(pred_conf[obj_mask.bool()], tconf_target[obj_mask.bool()]) * self.lambda_obj
        
        # noobj cells
        # (Optional: ignore high IoU detections in noobj mask - simplified here to standard v2)
        loss_conf_noobj = self.mse(pred_conf[noobj_mask.bool()], tconf_target[noobj_mask.bool()]) * self.lambda_noobj
        
        loss_conf = loss_conf_obj + loss_conf_noobj
        
        # 3. Class Loss
        loss_cls = 0.0
        if C > 1:
            # Gather class predictions for obj cells
            pred_cls_vec = pred_class[obj_mask.bool()]  # (N_obj, C)
            target_cls_vec = tcls_target[obj_mask.bool()] # (N_obj)
            loss_cls = self.ce(pred_cls_vec, target_cls_vec) * self.lambda_class
            
        total_loss = loss_coord + loss_conf + loss_cls
        
        # Average loss per batch or per object? usually average per batch
        return total_loss / B

def create_yolo_loss(loss_type='grid', **kwargs):
    if loss_type == 'grid':
        return YOLOv2GridLoss(**kwargs)
    else:
        # Fallback or other types
        return YOLOv2GridLoss(**kwargs)
