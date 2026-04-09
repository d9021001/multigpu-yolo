"""
從 CSV 檔案載入資料集的工具
用於多 GPU YOLOv2 訓練
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


class CSVDataset(Dataset):
    """
    從 CSV 檔案載入的 YOLO 資料集
    
    CSV 格式:
    - imageFilename: 圖像檔案路徑
    - bboxes: JSON 格式的邊界框列表，例如: "[[234,83,38,23], [234,87,37,21]]"
              每個邊界框格式: [x, y, w, h] 或 [x1, y1, x2, y2]
    """
    def __init__(self, csv_file, image_size=224, root_dir=None, transform=None, subset_frac=None):
        """
        參數:
            csv_file: CSV 檔案路徑
            image_size: 目標圖像尺寸
            root_dir: 圖像根目錄（如果 CSV 中的路徑是相對路徑）
            transform: 圖像轉換函數
            root_dir: 圖像根目錄（如果 CSV 中的路徑是相對路徑）
            transform: 圖像轉換函數
            subset_frac: 僅使用資料集的比例（例：0.01 表示 1%）
        """
        self.csv_file = Path(csv_file)
        self.image_size = image_size
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        
        # 載入 CSV
        # print(f"載入 CSV 檔案: {csv_file}")
        self.data = pd.read_csv(csv_file)
        
        if subset_frac and subset_frac < 1.0:
            self.data = self.data.sample(frac=subset_frac, random_state=42).reset_index(drop=True)
            # print(f"套用子集: 僅使用 {len(self.data)} 筆資料 ({subset_frac*100:.1f}%)")
        else:
            # print(f"載入 {len(self.data)} 筆資料")
            pass
        
        # 驗證必要的欄位
        required_columns = ['imageFilename', 'bboxes']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"CSV 檔案缺少必要的欄位: {missing_columns}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 獲取圖像路徑
        img_path = self.data.iloc[idx]['imageFilename']
        
        # 處理路徑
        if self.root_dir:
            # 如果提供了根目錄，組合路徑
            if not Path(img_path).is_absolute():
                img_path = self.root_dir / img_path
            else:
                img_path = Path(img_path)
        else:
            img_path = Path(img_path)
        
        # 讀取圖像
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"無法讀取圖像: {img_path}")
            
            # 確保圖像有 3 個通道（RGB）
            if len(image.shape) == 2:  # 灰度圖像
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"警告: 讀取圖像失敗 {img_path}: {e}")
            # 創建黑色圖像作為替代
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # 調整圖像尺寸
        original_size = image.shape[:2]  # (height, width)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # 再次確保圖像有 3 個通道（以防萬一）
        if len(image.shape) == 2 or image.shape[2] == 1:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 解析邊界框
        bboxes_str = self.data.iloc[idx]['bboxes']
        try:
            bboxes = json.loads(bboxes_str) if isinstance(bboxes_str, str) else bboxes_str
        except json.JSONDecodeError:
            print(f"警告: 無法解析邊界框 JSON: {bboxes_str}")
            bboxes = []
        
        # 轉換邊界框格式並調整尺寸
        if bboxes and len(bboxes) > 0:
            # 計算縮放比例
            scale_x = self.image_size / original_size[1]
            scale_y = self.image_size / original_size[0]
            
            scaled_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 4:
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    # 縮放到新的圖像尺寸
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    w_scaled = w * scale_x
                    h_scaled = h * scale_y

                    # Convert Top-Left to Center (Critical Fix)
                    cx_scaled = x_scaled + w_scaled / 2
                    cy_scaled = y_scaled + h_scaled / 2
                    
                    # 歸一化到 [0, 1]
                    x_norm = cx_scaled / self.image_size
                    y_norm = cy_scaled / self.image_size
                    w_norm = w_scaled / self.image_size
                    h_norm = h_scaled / self.image_size
                    
                    scaled_bboxes.append([x_norm, y_norm, w_norm, h_norm])
            
            bboxes = np.array(scaled_bboxes, dtype=np.float32)
        else:
            bboxes = np.array([], dtype=np.float32).reshape(0, 4)
        
        # 轉換為 tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 應用轉換
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'boxes': bboxes,
            'image_path': str(img_path),
            'original_size': original_size
        }
    
    def get_statistics(self):
        """獲取資料集統計資訊"""
        stats = {
            'total_images': len(self.data),
            'total_bboxes': 0,
            'bboxes_per_image': [],
            'images_with_bboxes': 0,
            'images_without_bboxes': 0
        }
        
        for idx in range(len(self.data)):
            bboxes_str = self.data.iloc[idx]['bboxes']
            try:
                bboxes = json.loads(bboxes_str) if isinstance(bboxes_str, str) else bboxes_str
                num_boxes = len(bboxes) if bboxes else 0
                stats['total_bboxes'] += num_boxes
                stats['bboxes_per_image'].append(num_boxes)
                
                if num_boxes > 0:
                    stats['images_with_bboxes'] += 1
                else:
                    stats['images_without_bboxes'] += 1
            except:
                stats['images_without_bboxes'] += 1
                stats['bboxes_per_image'].append(0)
        
        if stats['bboxes_per_image']:
            stats['avg_bboxes_per_image'] = np.mean(stats['bboxes_per_image'])
            stats['max_bboxes_per_image'] = np.max(stats['bboxes_per_image'])
            stats['min_bboxes_per_image'] = np.min(stats['bboxes_per_image'])
        else:
            stats['avg_bboxes_per_image'] = 0
            stats['max_bboxes_per_image'] = 0
            stats['min_bboxes_per_image'] = 0
        
        return stats


def load_csv_dataset(csv_file, image_size=224, root_dir=None, transform=None):
    """
    載入 CSV 資料集的便利函數
    
    參數:
        csv_file: CSV 檔案路徑
        image_size: 圖像尺寸
        root_dir: 圖像根目錄
        transform: 圖像轉換
    
    返回:
        CSVDataset 實例
    """
    return CSVDataset(csv_file, image_size, root_dir, transform)


def collate_fn_variable_boxes(batch):
    """
    自訂 collate 函數，用於處理可變長度的邊界框
    
    參數:
        batch: 樣本列表，每個樣本都包含 'image', 'boxes' 等鍵
    
    返回:
        合併後的批次字典，其中 'boxes' 是一個列表（而非張量）
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    boxes = [item['boxes'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'image': images,
        'boxes': boxes,  # 保持為列表，而不是張量
        'image_path': image_paths,
        'original_size': original_sizes
    }


if __name__ == '__main__':
    # 測試程式
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python csv_dataset.py <csv_file> [root_dir]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    root_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    dataset = CSVDataset(csv_file, root_dir=root_dir)
    
    print("\n資料集統計:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    
    print("\n測試載入第一個樣本...")
    sample = dataset[0]
    print(f"  圖像形狀: {sample['image'].shape}")
    print(f"  邊界框數量: {len(sample['boxes'])}")
    print(f"  圖像路徑: {sample['image_path']}")

