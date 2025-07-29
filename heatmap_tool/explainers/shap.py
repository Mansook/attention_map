import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import numpy as np
import cv2
from itertools import combinations
import random
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import time

class SHAP(nn.Module):
    def __init__(self,model,config_dict):
        super(SHAP,self).__init__()
        self.model = model.eval()
        self.type = config_dict.get('type', 'absolute')
        self.input_size = config_dict.get('input_size',(96,96))
        self.sampling_size = config_dict.get('sampling_size', 100)
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 5)
        self.progress_callback = None
        self.samples_dir = "C:/Users/orgin/XAI-study/heatmap_tool/"
        self.save_samples = True
    def generate(self,input_tensor,class_idx=None):
        start_time = time.time()
        print(f"[SHAP] 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        
        print(f"[SHAP] input_tensor - dtype: {input_tensor.dtype}, shape: {input_tensor.shape}")
        print(f"[SHAP] input_tensor - min: {input_tensor.min()}, max: {input_tensor.max()}")
        
        input_3d = input_tensor.squeeze()
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)
        
        print(f"[SHAP] input_3d - dtype: {input_3d.dtype}, shape: {input_3d.shape}")
        print(f"[SHAP] input_3d - min: {input_3d.min()}, max: {input_3d.max()}")
        
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        
        unique_labels = np.unique(labels)
        num_superpixels = len(unique_labels)
        
        if self.save_samples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            samples_path = os.path.join(self.samples_dir, f"shap_samples_{timestamp}")
            os.makedirs(samples_path, exist_ok=True)
            print(f"[SHAP] 샘플 저장 경로: {samples_path}")
        
        samples_matrix = np.zeros((self.sampling_size, num_superpixels))
        predictions = np.zeros(self.sampling_size)
        
        for i in range(self.sampling_size):
            progress_percent = int((i / self.sampling_size) * 100)
            if self.progress_callback:
                self.progress_callback(progress_percent, "SHAP")
            
            index = np.zeros(num_superpixels)
            sample = np.zeros_like(input_3d)
            random_integer = random.randint(0, num_superpixels-1)
            
            if random_integer > 0:
                ones_indices = random.sample(range(num_superpixels), random_integer)
                index[ones_indices] = 1
            
            samples_matrix[i] = index
            
            for idx, val in enumerate(index):
                mask = labels == idx
                if input_3d.ndim == 3:
                    mask_3d = np.stack([mask] * input_3d.shape[0], axis=0)
                else:
                    mask_3d = mask
                
                if val == 1:
                    sample[mask_3d] = input_3d[mask_3d]
                else:
                    sample[mask_3d] = 0
            
            sample_tensor = torch.from_numpy(sample).unsqueeze(0).float()
            
            if i < 3:
                print(f"[SHAP DEBUG] 모델 입력 텐서 {i} - dtype: {sample_tensor.dtype}, min: {sample_tensor.min()}, max: {sample_tensor.max()}")
            
            with torch.no_grad():
                output = self.model(sample_tensor)
                if class_idx is not None:
                    pred = output[0, class_idx].item()
                else:
                    pred = output.max(1)[1].item()
            predictions[i] = pred
            
            if i < 3:
                print(f"[SHAP DEBUG] 샘플 {i} 예측값: {pred}")
        
        if self.progress_callback:
            self.progress_callback(100, "SHAP")
        
        weights = self._kernel_weights(samples_matrix)
        
        lr = LinearRegression()
        lr.fit(samples_matrix, predictions, sample_weight=weights)
        
        shap_values = lr.coef_
        
        print(f"[SHAP] labels shape: {labels.shape}")
        heatmap = np.zeros_like(labels, dtype=np.float32)
        for idx, shap_value in enumerate(shap_values):
            mask = labels == idx
            heatmap[mask] = shap_value
        heatmap = process_heatmap_by_type(heatmap, self.type)
        print(f"[SHAP] heatmap shape: {heatmap.shape}, min: {heatmap.min()}, max: {heatmap.max()}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[SHAP] 종료 시간: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[SHAP] 총 소요 시간: {elapsed_time:.2f}초")
        
        return heatmap
    
    def _save_sample_as_png(self, sample, sample_idx, save_dir):
        try:
            if sample.ndim == 3:
                if sample.shape[0] == 1:
                    img = sample[0]
                elif sample.shape[0] == 3:
                    img = np.transpose(sample, (1, 2, 0))
                else:
                    img = sample[0]
            else:
                img = sample
            
            print(f"[SHAP] 샘플 {sample_idx} - dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
            
            if img.dtype == np.float32 or img.dtype == np.float64:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                if img.ndim == 3 and img.shape[2] == 3:
                    img = img * std + mean
                
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            filename = f"sample_{sample_idx:04d}.png"
            filepath = os.path.join(save_dir, filename)
            
            cv2.imwrite(filepath, img)
            print(f"[SHAP] 샘플 {sample_idx} 저장됨: {filepath}")
            
        except Exception as e:
            print(f"[SHAP] 샘플 {sample_idx} 저장 실패: {e}")
    
    def _kernel_weights(self, samples_matrix):
        weights = []
        for sample in samples_matrix:
            num_active = np.sum(sample)
            num_total = len(sample)
            
            if num_active == 0 or num_active == num_total:
                weight = 1e6
            else:
                weight = (num_total - 1) / (num_active * (num_total - num_active))
            
            weights.append(weight)
        
        return np.array(weights)
        
        
        
