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

class SHAP(nn.Module):
    def __init__(self,model,config_dict):
        super(SHAP,self).__init__()
        self.model = model.eval()
        self.type = config_dict.get('type', 'absolute')
        self.input_size = config_dict.get('input_size',(96,96))
        self.sampling_size = config_dict.get('sampling_size', 100)
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 5)
        self.progress_callback = None  # 진행률 콜백 (XAIWorker에서 설정)
     
    def generate(self,input_tensor,class_idx=None):
        input_3d = input_tensor.squeeze()  # [C, H, W] 또는 [H, W]로 변환
        if input_3d.dim() == 2:  # [H, W]인 경우
            input_3d = input_3d.unsqueeze(0)  # [1, H, W]로 변환
        #print(f"[SHAP] input_tensor 원본 shape: {input_tensor.shape}")
        #print(f"[SHAP] input_3d 변환 후 shape: {input_3d.shape}")
        
        # SLIC로 슈퍼픽셀 라벨과 경계선 추출
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        #print(f"[SHAP] labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
        #print(f"[SHAP] boundary shape: {boundary.shape}")
        
        unique_labels = np.unique(labels)  # 모든 고유한 슈퍼픽셀 라벨들 (예: [0, 1, 2, 3, ...])
        num_superpixels = len(unique_labels)
        #print(f"[SHAP] 슈퍼픽셀 개수: {num_superpixels}")
        
        # 샘플링 행렬과 예측값 저장
        samples_matrix = np.zeros((self.sampling_size, num_superpixels))
        predictions = np.zeros(self.sampling_size)
        
        for i in range(self.sampling_size):
            # 진행률 계산 및 콜백 호출
            progress_percent = int((i / self.sampling_size) * 100)
            if self.progress_callback:
                self.progress_callback(progress_percent, "SHAP")
            
            index = np.zeros(num_superpixels)
            # 3D 이미지로 생성 (원본 input_3d와 같은 shape)
            sample = np.zeros_like(input_3d)  # (C, H, W) 또는 (H, W)
            # 0부터 num_superpixels-1 중에서 랜덤 숫자 하나 선택
            random_integer = random.randint(0, num_superpixels-1)
            #print(f"[SHAP] 샘플 {i}: 랜덤 숫자: {random_integer}")
            # sample 배열에서 1의 개수가 random_integer개가 되도록 랜덤 위치에 1을 할당
            if random_integer > 0:
                ones_indices = random.sample(range(num_superpixels), random_integer)
                index[ones_indices] = 1
            #print(f"[SHAP] 샘플 {i}: index: {index}")
            
            # 샘플링 행렬에 저장
            samples_matrix[i] = index
            
            for idx, val in enumerate(index):
                mask = labels == idx  # (H, W) boolean mask
                # mask를 3차원으로 확장 (C, H, W)
                if input_3d.ndim == 3:
                    mask_3d = np.stack([mask] * input_3d.shape[0], axis=0)  # (C, H, W)
                else:
                    mask_3d = mask  # 2D면 그대로
                
                if val == 1:  # 1이면 원본 이미지
                    sample[mask_3d] = input_3d[mask_3d]
                else:  # 0이면 검은색
                    sample[mask_3d] = 0  # 검은색
            
            # 모델 예측
            sample_tensor = torch.from_numpy(sample).unsqueeze(0).float()
            with torch.no_grad():
                output = self.model(sample_tensor)
                if class_idx is not None:
                    pred = output[0, class_idx].item()
                else:
                    pred = output.max(1)[1].item()
            predictions[i] = pred
            #print(f"[SHAP] 샘플 {i}: 예측값: {pred}")
        
        # 100% 완료 신호
        if self.progress_callback:
            self.progress_callback(100, "SHAP")
        
        # 선형회귀로 SHAP 값 추정
        #print(f"[SHAP] 선형회귀 시작...")
        #print(f"[SHAP] samples_matrix shape: {samples_matrix.shape}")
        #print(f"[SHAP] predictions shape: {predictions.shape}")
        
        # Kernel SHAP 가중치 계산 (Lundberg & Lee, 2017)
        weights = self._kernel_weights(samples_matrix)
        
        # 가중 선형회귀
        lr = LinearRegression()
        lr.fit(samples_matrix, predictions, sample_weight=weights)
        
        # SHAP 값 (편향 제외)
        shap_values = lr.coef_
        #print(f"[SHAP] SHAP 값: {shap_values}")
        
        # 히트맵 생성
        # heatmap의 shape이 어떻게 되는지 확인 (labels와 동일)
        print(f"[SHAP] labels shape: {labels.shape}")  # 예: (H, W), 예를 들어 2x2면 (2, 2)
        heatmap = np.zeros_like(labels, dtype=np.float32)
        for idx, shap_value in enumerate(shap_values):
            mask = labels == idx
            heatmap[mask] = shap_value
        heatmap = process_heatmap_by_type(heatmap, self.type)
        print(f"[SHAP] heatmap shape: {heatmap.shape}, min: {heatmap.min()}, max: {heatmap.max()}")
        return heatmap
    
    def _kernel_weights(self, samples_matrix):
        """Kernel SHAP 가중치 계산 (Lundberg & Lee, 2017)"""
        weights = []
        for sample in samples_matrix:
            # 활성화된 feature 수
            num_active = np.sum(sample)
            # 전체 feature 수
            num_total = len(sample)
            
            # Kernel SHAP 가중치 공식
            if num_active == 0 or num_active == num_total:
                weight = 1e6  # 매우 큰 가중치 (전체 포함/제외)
            else:
                weight = (num_total - 1) / (num_active * (num_total - num_active))
            
            weights.append(weight)
        
        return np.array(weights)
        
        
        
