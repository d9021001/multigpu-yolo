"""
五折交叉驗證工具
"""
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset
import torch


def create_kfold_splits(dataset, n_splits=5, shuffle=True, random_state=42):
    """
    創建 K 折交叉驗證分割
    
    參數:
        dataset: PyTorch Dataset
        n_splits: 折數（預設: 5）
        shuffle: 是否打亂資料
        random_state: 隨機種子
    
    返回:
        list: 每個折的 (train_indices, val_indices) 元組列表
    """
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    indices = np.arange(len(dataset))
    splits = []
    
    for train_idx, val_idx in kfold.split(indices):
        splits.append((train_idx, val_idx))
    
    return splits


def get_fold_datasets(dataset, train_indices, val_indices):
    """
    根據索引創建訓練和驗證資料集
    
    參數:
        dataset: 原始資料集
        train_indices: 訓練集索引
        val_indices: 驗證集索引
    
    返回:
        train_dataset, val_dataset
    """
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

