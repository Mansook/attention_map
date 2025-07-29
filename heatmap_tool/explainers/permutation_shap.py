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

class PERMUTATION_SHAP(nn.Module):
    def __init__(self,model,config_dict):
        super(PERMUTATION_SHAP,self).__init__()
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
        print(f"[DEBUG] input_tensor shape: {input_tensor.shape}")
        
        input_3d = input_tensor.squeeze()
        print(f"[DEBUG] input_3d shape after squeeze: {input_3d.shape}")
        
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)
            print(f"[DEBUG] input_3d shape after unsqueeze: {input_3d.shape}")
        
        # SLIC로 슈퍼픽셀 라벨 추출
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        print(f"[DEBUG] labels shape: {labels.shape}")
        print(f"[DEBUG] unique labels: {np.unique(labels)}")
        
        unique_labels = np.unique(labels)
        num_superpixels = len(unique_labels)
        print(f"[DEBUG] num_superpixels: {num_superpixels}")
        
        # 미리 마스크 생성 (메모리 효율성)
        masks = {}
        for label in unique_labels:
            mask = (labels == label)
            masks[label] = mask
            print(f"[DEBUG] mask for label {label} shape: {mask.shape}")
        
        # 배치 처리를 위한 마스크 텐서 미리 생성
        mask_tensors = {}
        for label, mask in masks.items():
            mask_3d = np.stack([mask] * input_3d.shape[0], axis=0)
            print(f"[DEBUG] mask_3d for label {label} shape: {mask_3d.shape}")
            mask_tensors[label] = torch.from_numpy(mask_3d).bool()
            print(f"[DEBUG] mask_tensor for label {label} shape: {mask_tensors[label].shape}")
        
        shap_values = np.zeros(num_superpixels)
        
        # 순열 생성
        permutation_list = [np.random.permutation(unique_labels) for _ in range(self.sampling_size)]
        basis = torch.zeros_like(input_3d.unsqueeze(0))
        print(f"[DEBUG] basis shape: {basis.shape}")
        print(f"[DEBUG] basis[0] shape: {basis[0].shape}")
        basis_score = 0
        
        with torch.no_grad():
            # 베이스 점수 계산
            output = self.model(basis)
            if class_idx is not None:
                store_basis_score = output[0, class_idx].item()
            else:
                store_basis_score = output.max(1)[1].item()

        for perm_idx, permut in enumerate(permutation_list):
            # 각 순열마다 빈 이미지로 시작
            basis = torch.zeros_like(input_3d.unsqueeze(0))
            basis_score = store_basis_score  # 빈 이미지의 점수
    
            # 순열 순서대로 슈퍼픽셀을 하나씩 누적해서 추가
            for idx, val in enumerate(permut):
                progress_percent = int((perm_idx * num_superpixels + idx) / (self.sampling_size * num_superpixels) * 100)
                if self.progress_callback:
                    self.progress_callback(progress_percent, "PERMUT_SHAP")
                
                # 현재 슈퍼픽셀을 basis에 추가
                mask_tensor = mask_tensors[val]
                print(f"[DEBUG] mask_tensor shape: {mask_tensor.shape}")
                print(f"[DEBUG] input_3d shape: {input_3d.shape}")
                print(f"[DEBUG] basis[0] shape: {basis[0].shape}")
                basis[0][mask_tensor] = input_3d[mask_tensor]
                
                # 추가된 이미지로 모델 추론
                with torch.no_grad():
                    output = self.model(basis)
                    if class_idx is not None:
                        new_score = output[0, class_idx].item()
                    else:
                        new_score = output.max(1)[1].item()
                
                # 점수 변화를 해당 슈퍼픽셀의 SHAP 값에 누적
                shap_values[val] += new_score - basis_score
                basis_score = new_score  # 다음 단계를 위한 점수 업데이트
        
        shap_values /= self.sampling_size
        heatmap = shap_values[labels]
        heatmap = process_heatmap_by_type(heatmap, self.type)
        return heatmap
        
        
