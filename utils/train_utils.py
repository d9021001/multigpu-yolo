"""
訓練工具函數
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    訓練一個 epoch
    
    返回:
        平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        images = batch['image'].to(device)
        targets = batch['boxes'].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(images)
        
        # 計算損失（這裡是簡化版本，實際需要 YOLOv2 的損失函數）
        # YOLOv2 使用多任務損失：定位損失 + 置信度損失 + 分類損失
        loss = compute_yolo_loss(outputs, targets, criterion)
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_model(model, dataloader, criterion, device):
    """
    驗證模型
    
    返回:
        平均損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            targets = batch['boxes'].to(device)
            
            outputs = model(images)
            loss = compute_yolo_loss(outputs, targets, criterion)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_yolo_loss(outputs, targets, criterion):
    """
    計算 YOLOv2 損失
    
    注意: 這是簡化版本，實際需要完整的 YOLOv2 損失實現
    """
    # 簡化版本：使用 MSE 損失
    # 實際 YOLOv2 損失包括：
    # 1. 定位損失 (MSE for x, y, w, h)
    # 2. 置信度損失 (BCE)
    # 3. 分類損失 (CE)
    
    if isinstance(outputs, torch.Tensor):
        # 簡化處理
        return criterion(outputs, targets.float())
    else:
        # 如果是多個輸出
        total_loss = 0
        for out in outputs:
            if isinstance(out, torch.Tensor):
                total_loss += criterion(out, targets.float())
        return total_loss

