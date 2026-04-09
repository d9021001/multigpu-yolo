"""
YOLO 資料集類
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import pickle


class YOLODataset(Dataset):
    """
    YOLO 格式的資料集
    """
    def __init__(self, data_path, image_size=224, transform=None):
        """
        參數:
            data_path: 資料檔案路徑（pickle 格式）或資料列表
            image_size: 圖像尺寸
            transform: 圖像轉換
        """
        self.image_size = image_size
        self.transform = transform
        
        # 載入資料
        if isinstance(data_path, (str, Path)):
            data_path = Path(data_path)
            if data_path.exists():
                with open(data_path, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                print(f"警告: 資料檔案 {data_path} 不存在，使用空資料集")
                self.data = []
        else:
            self.data = data_path
        
        # 如果資料是表格格式（類似 MATLAB 的 table）
        if isinstance(self.data, list) and len(self.data) > 0:
            # 假設資料格式: [{'image': path, 'boxes': [[x,y,w,h],...]}, ...]
            self.items = self.data
        else:
            self.items = []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 讀取圖像
        if isinstance(item, dict):
            image_path = item.get('image', item.get('imageFilename', ''))
            boxes = item.get('boxes', item.get('vehicle', []))
        else:
            # 假設是元組或列表格式
            image_path, boxes = item[0], item[1]
        
        # 讀取圖像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                # 如果讀取失敗，創建黑色圖像
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(image_path)
        
        # 調整尺寸
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # 轉換為 tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 應用轉換
        if self.transform:
            image = self.transform(image)
        
        # 轉換框的格式
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        elif not isinstance(boxes, np.ndarray):
            boxes = np.array([]).reshape(0, 4)
        
        return {
            'image': image,
            'boxes': boxes,
            'image_path': str(image_path) if isinstance(image_path, (str, Path)) else ''
        }

