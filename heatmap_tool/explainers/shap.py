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
        self.samples_dir = "C:/Users/orgin/XAI-study/heatmap_tool/"
        self.save_samples = True
    def generate(self,input_tensor,class_idx=None):
        # 입력 텐서 정보 출력
        print(f"[SHAP] input_tensor - dtype: {input_tensor.dtype}, shape: {input_tensor.shape}")
        print(f"[SHAP] input_tensor - min: {input_tensor.min()}, max: {input_tensor.max()}")
        
        input_3d = input_tensor.squeeze()  # [C, H, W] 또는 [H, W]로 변환
        if input_3d.dim() == 2:  # [H, W]인 경우
            input_3d = input_3d.unsqueeze(0)  # [1, H, W]로 변환
        
        # 변환 후 정보 출력
        print(f"[SHAP] input_3d - dtype: {input_3d.dtype}, shape: {input_3d.shape}")
        print(f"[SHAP] input_3d - min: {input_3d.min()}, max: {input_3d.max()}")
        
        # SLIC로 슈퍼픽셀 라벨과 경계선 추출
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        #print(f"[SHAP] labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
        #print(f"[SHAP] boundary shape: {boundary.shape}")
        
        unique_labels = np.unique(labels)  # 모든 고유한 슈퍼픽셀 라벨들 (예: [0, 1, 2, 3, ...])
        num_superpixels = len(unique_labels)
        #print(f"[SHAP] 슈퍼픽셀 개수: {num_superpixels}")
        
        # 샘플 저장 디렉토리 생성
        #if self.save_samples:
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #samples_path = os.path.join(self.samples_dir, f"shap_samples_{timestamp}")
            #os.makedirs(samples_path, exist_ok=True)
            #print(f"[SHAP] 샘플 저장 경로: {samples_path}")
        
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
                #print("mask_3d shape", mask_3d.shape)
                #print("input_3d shape", input_3d.shape)
                #print("sample shape", sample.shape)
            # 샘플을 PNG로 저장
            #self._save_sample_as_png(sample, i, samples_path)
            
       
            # 모델 예측
            sample_tensor = torch.from_numpy(sample).unsqueeze(0).float()
            
            # 디버깅: 모델 입력 텐서 검증
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
    
    def _save_sample_as_png(self, sample, sample_idx, save_dir):
        """샘플을 PNG 파일로 저장"""
        try:
            # sample을 이미지 형태로 변환
            if sample.ndim == 3:  # (C, H, W)
                # 채널이 1개인 경우 (그레이스케일)
                if sample.shape[0] == 1:
                    img = sample[0]  # (H, W)
                # 채널이 3개인 경우 (RGB)
                elif sample.shape[0] == 3:
                    img = np.transpose(sample, (1, 2, 0))  # (H, W, C)
                else:
                    # 첫 번째 채널만 사용
                    img = sample[0]
            else:  # (H, W)
                img = sample
            
            # 데이터 정보 출력
            print(f"[SHAP] 샘플 {sample_idx} - dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
            
            # ImageNet 정규화 역변환
            if img.dtype == np.float32 or img.dtype == np.float64:
                # ImageNet 평균과 표준편차
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # 역정규화: (x * std) + mean
                if img.ndim == 3 and img.shape[2] == 3:  # RGB 이미지
                    img = img * std + mean
                
                # 0~1 범위로 클리핑 후 0~255로 변환
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            # 파일명 생성
            filename = f"sample_{sample_idx:04d}.png"
            filepath = os.path.join(save_dir, filename)
            
            # PNG로 저장
            cv2.imwrite(filepath, img)
            print(f"[SHAP] 샘플 {sample_idx} 저장됨: {filepath}")
            
        except Exception as e:
            print(f"[SHAP] 샘플 {sample_idx} 저장 실패: {e}")
    
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
        
        
        
