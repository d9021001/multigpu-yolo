"""
模型管理器：追蹤和管理全局最佳模型（支援跨 GPU 模型共享和驗證）
"""
import torch
from pathlib import Path
import json
import numpy as np
import time
import shutil


class GlobalBestModelManager:
    """全局最佳模型管理器（支援 Best-model Federated Averaging）"""
    
    def __init__(self, gpu_id, base_network, save_dir='models'):
        """
        初始化全局最佳模型管理器
        
        參數:
            gpu_id: GPU ID
            base_network: 基礎網路名稱
            save_dir: 保存目錄
        """
        self.gpu_id = gpu_id
        self.base_network = base_network
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Federated directories
        self.best_dir = self.save_dir / "best"
        self.federated_dir = self.save_dir / "federated"
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.federated_dir.mkdir(parents=True, exist_ok=True)
        
        # 最佳模型資訊檔案
        self.info_file = self.save_dir / f'best_model_info_gpu_{gpu_id}.json'
        
        # 最佳模型檔案
        # Best model path (for this GPU)
        self.model_file = self.best_dir / f'best_model_gpu_{gpu_id}.pth'
        
        # 載入或初始化最佳模型資訊
        self.best_info = self._load_best_info()
    
    def _load_best_info(self):
        """載入最佳模型資訊"""
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # 初始化
        return {
            'best_cost': float('inf'),
            'best_iter': 0,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'best_f1': 0.0,
            'best_hyperparams': None,
            'update_count': 0
        }
    
    def _save_best_info(self):
        """保存最佳模型資訊"""
        with open(self.info_file, 'w') as f:
            json.dump(self.best_info, f, indent=2)
    
    def update_if_better(self, model, cost, metrics, iter_num, hyperparams=None):
        """
        如果當前模型更好，則更新最佳模型
        
        參數:
            model: PyTorch 模型
            cost: 成本函數值（越小越好）
            metrics: 評估指標字典（precision, recall, f1 等）
            iter_num: 當前迭代次數
            hyperparams: 超參數字典
        
        返回:
            bool: 是否更新了最佳模型
        """
        is_better = cost < self.best_info['best_cost']
        
        if is_better:
            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'iter': iter_num,
                'cost': cost,
                'metrics': metrics,
                'hyperparams': hyperparams,
                'base_network': self.base_network,
            }, self.model_file)
            
            # 更新資訊
            old_cost = self.best_info['best_cost']
            self.best_info['best_cost'] = cost
            self.best_info['best_iter'] = iter_num
            self.best_info['best_precision'] = metrics.get('precision', 0.0)
            self.best_info['best_recall'] = metrics.get('recall', 0.0)
            self.best_info['best_f1'] = metrics.get('f1', 0.0)
            if hyperparams:
                self.best_info['best_hyperparams'] = hyperparams
            self.best_info['update_count'] += 1
            
            # 保存資訊
            self._save_best_info()
            
            print(f"\n{'='*60}")
            print(f"✓ 更新全局最佳模型！")
            print(f"  迭代: {iter_num}")
            print(f"  成本: {old_cost:.4f} → {cost:.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1: {metrics.get('f1', 0):.4f}")
            print(f"{'='*60}\n")
            
            return True
        else:
            print(f"\n當前模型成本 ({cost:.4f}) 未優於最佳模型 ({self.best_info['best_cost']:.4f})")
            return False
    
    def load_best_model(self, model):
        """
        載入最佳模型權重到當前模型
        
        參數:
            model: PyTorch 模型
        
        返回:
            dict: 模型資訊（包含 metrics, hyperparams 等）
        """
        if not self.model_file.exists():
            print(f"警告: 最佳模型檔案不存在: {self.model_file}")
            return None
        
        checkpoint = torch.load(self.model_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"載入最佳模型（迭代 {checkpoint.get('iter', 'unknown')}）")
        return checkpoint
    
    def get_best_info(self):
        """獲取最佳模型資訊"""
        return self.best_info.copy()

    # ================= Federated Learning (Best-model based) =================

    def wait_for_all_best_models(self, all_gpu_ids, timeout=600):
        """
        Barrier: wait until all best_model_gpu_X.pth exist
        """
        from utils.federated_utils import wait_for_files
        paths = [
            self.best_dir / f"best_model_gpu_{gid}.pth"
            for gid in all_gpu_ids
        ]
        wait_for_files(paths, timeout=timeout)

    def federated_average_best_models(self, all_gpu_ids):
        """
        Perform FedAvg on best models from all GPUs
        """
        from utils.federated_utils import fedavg_best_models
        paths = [
            self.best_dir / f"best_model_gpu_{gid}.pth"
            for gid in all_gpu_ids
        ]
        return fedavg_best_models(paths)

    def save_federated_model(self, state_dict, round_id):
        """
        Save federated model for a given round
        """
        path = self.federated_dir / f"federated_round_{round_id}.pth"
        import torch
        torch.save(
            {"model_state_dict": state_dict, "round": round_id},
            path
        )
        return path

    def load_federated_model(self, round_id):
        """
        Load federated model for a given round
        """
        import torch
        path = self.federated_dir / f"federated_round_{round_id}.pth"
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu")
    
    def save_iteration_model(self, model, cost, metrics, iter_num, hyperparams=None):
        """
        保存當前迭代的模型（不會覆蓋最佳模型）
        
        參數:
            model: PyTorch 模型
            cost: 成本函數值
            metrics: 評估指標
            iter_num: 迭代次數
            hyperparams: 超參數
        """
        iter_model_file = self.save_dir / f'model_iter_{iter_num}_gpu_{self.gpu_id}.pth'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'iter': iter_num,
            'cost': cost,
            'metrics': metrics,
            'hyperparams': hyperparams,
            'base_network': self.base_network,
        }, iter_model_file)
        
        return iter_model_file
    
    def cleanup_old_iteration_models(self, keep_last_n=5):
        """
        清理舊的迭代模型，只保留最近的 N 個
        
        參數:
            keep_last_n: 保留最近 N 個迭代的模型
        """
        pattern = f'model_iter_*_gpu_{self.gpu_id}.pth'
        iter_models = sorted(self.save_dir.glob(pattern), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
        
        # 保留最近 N 個和最佳模型
        to_keep = set(iter_models[:keep_last_n])
        to_keep.add(self.model_file)  # 總是保留最佳模型
        
        removed_count = 0
        for model_file in iter_models:
            if model_file not in to_keep:
                try:
                    model_file.unlink()
                    removed_count += 1
                except:
                    pass
        
        if removed_count > 0:
            print(f"清理了 {removed_count} 個舊的迭代模型檔案")
    
    def get_other_gpu_models(self, all_gpu_ids=[1, 2, 3]):
        """
        獲取其他 GPU 的最佳模型資訊
        
        參數:
            all_gpu_ids: 所有 GPU ID 列表
        
        返回:
            list: 其他 GPU 的模型資訊列表
        """
        other_models = []
        
        for gpu_id in all_gpu_ids:
            if gpu_id == self.gpu_id:
                continue
            
            other_info_file = self.save_dir / f'best_model_info_gpu_{gpu_id}.json'
            other_model_file = self.save_dir / f'best_model_gpu_{gpu_id}.pth'
            
            if other_info_file.exists() and other_model_file.exists():
                try:
                    with open(other_info_file, 'r') as f:
                        info = json.load(f)
                    info['gpu_id'] = gpu_id
                    info['model_file'] = other_model_file
                    other_models.append(info)
                except:
                    pass
        
        return other_models
    
    def validate_other_gpu_model(self, other_gpu_id, model, eval_dataset, device, 
                                 eval_function, threshold=0.95):
        """
        驗證其他 GPU 的模型
        
        參數:
            other_gpu_id: 其他 GPU 的 ID
            model: 當前模型（用於載入其他模型的權重）
            eval_dataset: 驗證資料集
            device: 計算設備
            eval_function: 評估函數 (model, dataset, device, threshold) -> metrics
            threshold: 評估閾值
        
        返回:
            dict: 評估結果（包含 metrics 和 cost）
        """
        other_model_file = self.save_dir / f'best_model_gpu_{other_gpu_id}.pth'
        
        if not other_model_file.exists():
            print(f"警告: GPU {other_gpu_id} 的模型檔案不存在")
            return None
        
        try:
            # 載入其他 GPU 的模型權重
            checkpoint = torch.load(other_model_file, map_location='cpu')
            
            # 檢查模型架構是否匹配
            if checkpoint.get('base_network') != self.base_network:
                print(f"警告: GPU {other_gpu_id} 的模型架構不匹配")
                return None
            
            # 載入模型權重
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 評估模型
            metrics = eval_function(model, eval_dataset, device, threshold=threshold)
            
            return {
                'gpu_id': other_gpu_id,
                'metrics': metrics,
                'cost': metrics.get('cost', float('inf')),
                'iter': checkpoint.get('iter', 0),
                'hyperparams': checkpoint.get('hyperparams')
            }
        except Exception as e:
            print(f"驗證 GPU {other_gpu_id} 的模型時發生錯誤: {e}")
            return None
    
    def check_and_update_from_other_gpus(self, model, eval_dataset, device,
                                        eval_function, all_gpu_ids=[1, 2, 3],
                                        threshold=0.95):
        """
        檢查其他 GPU 的模型，如果更好則更新最佳模型
        
        參數:
            model: 當前模型
            eval_dataset: 驗證資料集（用於評估其他 GPU 的模型）
            device: 計算設備
            eval_function: 評估函數
            all_gpu_ids: 所有 GPU ID 列表
            threshold: 評估閾值
        
        返回:
            dict: 更新資訊（如果有更新）
        """
        current_best_cost = self.best_info['best_cost']
        best_other_model = None
        best_other_cost = current_best_cost
        
        print(f"\n檢查其他 GPU 的模型...")
        
        # 獲取其他 GPU 的模型
        other_models_info = self.get_other_gpu_models(all_gpu_ids)
        
        if not other_models_info:
            print("未找到其他 GPU 的模型")
            return None
        
        # 驗證每個其他 GPU 的模型
        validated_models = []
        for other_info in other_models_info:
            other_gpu_id = other_info['gpu_id']
            
            # 先檢查資訊檔案中的成本（快速檢查）
            other_cost_from_info = other_info.get('best_cost', float('inf'))
            
            if other_cost_from_info >= current_best_cost:
                print(f"  GPU {other_gpu_id} 的模型成本 ({other_cost_from_info:.4f}) 未優於當前最佳 ({current_best_cost:.4f})，跳過驗證")
                continue
            
            print(f"  驗證 GPU {other_gpu_id} 的模型...")
            result = self.validate_other_gpu_model(
                other_gpu_id, model, eval_dataset, device, eval_function, threshold
            )
            
            if result:
                validated_models.append(result)
                if result['cost'] < best_other_cost:
                    best_other_cost = result['cost']
                    best_other_model = result
                    cost_val = result['cost']
                    print(f"    GPU {other_gpu_id} 的模型更好: Cost = {cost_val:.4f}")
        
        # 如果找到更好的模型，則更新
        if best_other_model and best_other_cost < current_best_cost:
            print(f"\n{'='*60}")
            gpu_id = best_other_model['gpu_id']
            precision = best_other_model['metrics'].get('precision', 0)
            recall = best_other_model['metrics'].get('recall', 0)
            f1 = best_other_model['metrics'].get('f1', 0)
            print(f"✓ 從 GPU {gpu_id} 更新全局最佳模型！")
            print(f"  成本: {current_best_cost:.4f} → {best_other_cost:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            
            # 複製其他 GPU 的模型檔案
            gpu_id = best_other_model['gpu_id']
            other_model_file = self.save_dir / f'best_model_gpu_{gpu_id}.pth'
            
            # 載入模型並保存為當前 GPU 的最佳模型
            checkpoint = torch.load(other_model_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 更新為當前 GPU 的最佳模型
            self.update_if_better(
                model,
                best_other_cost,
                best_other_model['metrics'],
                best_other_model['iter'],
                best_other_model.get('hyperparams')
            )
            
            # 更新資訊，標記是從其他 GPU 獲得的
            self.best_info['source_gpu'] = best_other_model['gpu_id']
            self.best_info['is_from_other_gpu'] = True
            self._save_best_info()
            
            return {
                'updated': True,
                'source_gpu': best_other_model['gpu_id'],
                'old_cost': current_best_cost,
                'new_cost': best_other_cost
            }
        else:
            if validated_models:
                print(f"驗證了 {len(validated_models)} 個其他 GPU 的模型，但都沒有優於當前最佳模型")
            return {'updated': False}
    
    def share_model_to_others(self):
        """
        將當前最佳模型分享給其他 GPU（標記為可用）
        實際上模型已經在共享目錄中，這只是確保資訊檔案的同步
        """
        # 確保資訊檔案是最新的
        self._save_best_info()
        
        # 模型檔案已經在共享目錄中，其他 GPU 可以直接讀取
        return True


def create_model_manager(gpu_id, base_network, save_dir='models'):
    """創建模型管理器"""
    return GlobalBestModelManager(gpu_id, base_network, save_dir)
