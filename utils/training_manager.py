"""
訓練管理器：提供完整的訓練、驗證和測試功能
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
from .yolo_loss import YOLOv2GridLoss


class EarlyStopping:
    """早停機制"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def load_best_model(self, model):
        """載入最佳模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class TrainingManager:
    """訓練管理器"""
    
    def __init__(self, model, device, optimizer, criterion, 
                 scheduler=None, early_stopping=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        self.train_history = {
            'loss': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.val_history = {
            'loss': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def train_epoch(self, dataloader):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc='Training', leave=False):
            images = batch['image'].to(self.device)
            targets = batch['boxes']  # 現在是列表，不需要轉移到 device
            
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(images)
            
            # 計算損失（處理可變長度的 targets）
            loss = self.compute_loss(outputs, targets)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, dataloader, compute_metrics=False):
        """驗證模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation', leave=False):
                images = batch['image'].to(self.device)
                targets = batch['boxes']  # 現在是列表，不需要轉移到 device
                
                outputs = self.model(images)
                loss = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                if compute_metrics:
                    # 收集預測和目標（需要根據實際模型調整）
                    all_predictions.append(outputs.cpu())
                    # targets 已經是 list，不需要轉移
                    all_targets.append(targets)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {}
        if compute_metrics and len(all_predictions) > 0:
            # 計算指標（需要實現具體的計算邏輯）
            metrics = self.compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train_with_validation(self, train_loader, val_loader, max_epochs, 
                             save_path=None, verbose=True):
        """訓練並驗證"""
        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            # 訓練
            train_loss = self.train_epoch(train_loader)
            
            # 驗證
            val_loss, val_metrics = self.validate(val_loader, compute_metrics=True)
            
            # 更新學習率
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 記錄歷史
            self.train_history['loss'].append(train_loss)
            self.val_history['loss'].append(val_loss)
            
            if val_metrics:
                self.val_history['precision'].append(val_metrics.get('precision', 0))
                self.val_history['recall'].append(val_metrics.get('recall', 0))
                self.val_history['f1'].append(val_metrics.get('f1', 0))
            
            # 顯示進度
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{max_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"LR: {lr:.6f}")
                
                if val_metrics:
                    print(f"  Precision: {val_metrics.get('precision', 0):.4f}, "
                          f"Recall: {val_metrics.get('recall', 0):.4f}, "
                          f"F1: {val_metrics.get('f1', 0):.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics,
                    }, save_path)
            
            # 早停檢查
            if self.early_stopping:
                if self.early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    if self.early_stopping.best_model_state:
                        self.model.load_state_dict(self.early_stopping.best_model_state)
                    break
        
        return self.train_history, self.val_history
    
    def test(self, test_loader, compute_metrics=True):
        """測試模型"""
        test_loss, test_metrics = self.validate(test_loader, compute_metrics=compute_metrics)
        return test_loss, test_metrics
    
    def compute_loss(self, outputs, targets):
        """計算損失 - 使用真正的 YOLOv2 損失"""
        # targets 是 list，每個元素是該圖像的邊界框數組 (N, 4)
        # outputs 是模型輸出，形狀為 (batch_size, output_dim)
        
        if isinstance(targets, list):
            # 使用 YOLOv2 損失函數
            return self.criterion(outputs, targets)
        
        # 如果 targets 是張量（向後兼容）
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            return self.criterion(outputs, targets.float())
        else:
            # 回退
            return self.criterion(outputs, targets)
    
    def compute_metrics(self, predictions, targets):
        """計算評估指標（需要根據實際模型實現）"""
        # 簡化版本，實際需要實現 precision, recall, F1 等
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }


def create_optimizer(model, lr, optimizer_type='sgd', momentum=0.9, weight_decay=1e-4):
    """創建優化器"""
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支援的優化器類型: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_type='reduce_on_plateau', **kwargs):
    """創建學習率調整器"""
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type is None:
        scheduler = None
    else:
        raise ValueError(f"不支援的調度器類型: {scheduler_type}")
    
    return scheduler

