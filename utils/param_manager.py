"""
參數管理器：處理超參數的保存和載入
用於多 GPU 進程間通信
"""
import os
import pickle
import time
from pathlib import Path
import numpy as np


class ParamManager:
    def __init__(self, x_folder='xFolder', c_folder='cFolder'):
        self.x_folder = Path(x_folder)
        self.c_folder = Path(c_folder)
        
        # 創建資料夾
        self.x_folder.mkdir(exist_ok=True)
        self.c_folder.mkdir(exist_ok=True)
    
    def save_params(self, x1, x2, x3):
        """保存三個超參數組到檔案"""
        # 限制參數範圍
        x1 = self._clamp_params(x1)
        x2 = self._clamp_params(x2)
        x3 = self._clamp_params(x3)
        
        # 保存為 pickle 格式（類似 MATLAB 的 .mat）
        with open(self.x_folder / 'x1.pkl', 'wb') as f:
            pickle.dump(x1, f)
        with open(self.x_folder / 'x2.pkl', 'wb') as f:
            pickle.dump(x2, f)
        with open(self.x_folder / 'x3.pkl', 'wb') as f:
            pickle.dump(x3, f)
    
    def load_param(self, gpu_id, timeout=5):
        """從檔案載入對應 GPU 的超參數
        
        Args:
            gpu_id: GPU ID
            timeout: 等待超時時間（秒）。超時後返回預設值
        
        Returns:
            超參數 [batch_size, learning_rate]
        """
        filename = f'x{gpu_id}.pkl'
        filepath = self.x_folder / filename
        
        start_time = time.time()
        
        while True:
            # 檢查超時
            if time.time() - start_time > timeout:
                # 返回預設值
                return np.array([32, 0.01])
            
            if filepath.exists():
                time.sleep(0.5)  # 等待檔案完全寫入
                try:
                    with open(filepath, 'rb') as f:
                        param = pickle.load(f)
                    # 刪除檔案（類似 MATLAB 版本）
                    filepath.unlink()
                    return param
                except (EOFError, pickle.UnpicklingError):
                    time.sleep(0.1)
                    continue
            time.sleep(0.5)
    
    def save_cost(self, gpu_id, cost):
        """保存成本函數值"""
        filename = f'c{gpu_id}.pkl'
        with open(self.c_folder / filename, 'wb') as f:
            pickle.dump(cost, f)
    
    def load_costs(self):
        """載入三個成本函數值"""
        costs = []
        for i in [1, 2, 3]:
            filename = f'c{i}.pkl'
            filepath = self.c_folder / filename
            
            # 等待檔案存在
            while not filepath.exists():
                time.sleep(0.5)
            
            time.sleep(0.5)  # 等待檔案完全寫入
            with open(filepath, 'rb') as f:
                cost = pickle.load(f)
            costs.append(cost)
            filepath.unlink()  # 刪除檔案
        
        return np.array(costs)
    
    def clear_all(self):
        """清除所有參數和成本檔案"""
        for folder in [self.x_folder, self.c_folder]:
            for file in folder.glob('*.pkl'):
                try:
                    file.unlink()
                except:
                    pass
    
    @staticmethod
    def _clamp_params(x):
        """限制參數範圍"""
        x = np.array(x)
        # MiniBatchSize: 1-32
        x[0] = np.clip(x[0], 1, 32)
        # LearningRate: 1e-6 to 1e-2
        x[1] = np.clip(x[1], 1e-6, 1e-2)
        return x

