import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import numpy as np
import cv2


class SmoothGrad(nn.Module):
    def __init__(self, model, config_dict):
        super(SmoothGrad, self).__init__()
        self.model = model.eval()
        
        # config_dict에서 설정 가져오기
        self.input_size = tuple(config_dict.get('input_size', (96, 96)))
        self.num_sample = config_dict.get('num_sample',10)
        self.var = config_dict.get('var',0.125)
        self.type = config_dict.get('type', 'both')  # type 추가
        self.slic = config_dict.get('slic', False)
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 10)

    def generate(self, input_tensor, class_idx=None):
        
        self.model.zero_grad()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).clone().detach()
        
        total_gradients = torch.zeros_like(input_tensor)
        
        for _ in range(self.num_sample):
            noise = torch.normal(0,self.var,size=input_tensor.shape).to(device)
            noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)
            
            output = self.model(noisy_input)
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()

            score = output[:, class_idx]
            self.model.zero_grad()
            score.backward()
            
            gradients = noisy_input.grad
            if gradients is not None:
                total_gradients += gradients
            else:
                raise RuntimeError("Gradient 계산 실패")
        
        
        avg_gradients = total_gradients/self.num_sample
        saliency = avg_gradients.mean(dim=1)  # shape: (B, H, W)
       
        # utils의 통합 처리 함수 사용
        saliency = process_heatmap_by_type(saliency.squeeze(), self.type)
        if self.slic is True:
            print("[SmoothGrad] Slic Lets go")
            # input_tensor를 squeeze해서 2D 배열로 변환
            input_2d = input_tensor.squeeze()  # [C, H, W] 또는 [H, W]로 변환
            if input_2d.dim() == 3:  # [C, H, W]인 경우
                input_2d = input_2d.mean(dim=0)  # 채널 평균을 취해서 [H, W]로 변환
            print(f"[SmoothGrad] input_tensor 원본 shape: {input_tensor.shape}")
            print(f"[SmoothGrad] input_2d 변환 후 shape: {input_2d.shape}")
            
            # SLIC로 슈퍼픽셀 라벨과 경계선 추출
            labels, boundary = superpixel_mean_map(input_2d, region_size=self.slic_size, ruler=self.slic_ruler)
            print(f"[SmoothGrad] labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
            print(f"[SmoothGrad] boundary shape: {boundary.shape}")
            
            # saliency를 numpy로 변환 (tensor인 경우)
            if hasattr(saliency, 'cpu'):
                saliency_np = saliency.cpu().numpy()
            else:
                saliency_np = saliency
            print(f"[SmoothGrad] saliency_np shape: {saliency_np.shape}, min: {saliency_np.min()}, max: {saliency_np.max()}")
            
            # 각 슈퍼픽셀별로 SmoothGrad 기여도 평균 계산
            unique_labels = np.unique(labels)  # 모든 고유한 슈퍼픽셀 라벨들 (예: [0, 1, 2, 3, ...])
            num_superpixels = len(unique_labels)
            print(f"[SmoothGrad] 슈퍼픽셀 개수: {num_superpixels}")
            
            # 슈퍼픽셀별 평균 기여도 계산
            superpixel_contributions = np.zeros(num_superpixels)  # 각 슈퍼픽셀의 평균 기여도를 저장할 배열
            for i, label in enumerate(unique_labels):
                # mask = labels == label: 현재 보고 있는 라벨과 같은 픽셀들만 True(1), 나머지는 False(0)
                # 예: label=2일 때, labels[y,x]==2인 픽셀들만 mask[y,x]=True
                mask = labels == label
                if np.any(mask):  # 해당 라벨의 픽셀이 하나라도 있는지 확인
                    # saliency_np[mask]: mask가 True인 픽셀들의 SmoothGrad 기여도만 추출
                    # .mean(): 추출된 기여도들의 평균을 계산
                    superpixel_contributions[i] = saliency_np[mask].mean()
                    print(f"[SmoothGrad] 슈퍼픽셀 {label}: 평균 기여도 = {superpixel_contributions[i]:.4f}, 픽셀 수 = {np.sum(mask)}")
            
            # 슈퍼픽셀별 기여도로 시각화 맵 생성
            slic_visualization = np.zeros_like(saliency_np)  # 원본 SmoothGrad 맵과 같은 크기의 빈 배열
            for i, label in enumerate(unique_labels):
                # 다시 같은 마스크 생성: 현재 라벨에 해당하는 픽셀들만 True
                mask = labels == label
                # slic_visualization[mask] = superpixel_contributions[i]: 
                # mask가 True인 모든 픽셀에 해당 슈퍼픽셀의 평균 기여도를 할당
                # 결과적으로 같은 슈퍼픽셀 내의 모든 픽셀이 동일한 평균 기여도 값을 가지게 됨
                slic_visualization[mask] = superpixel_contributions[i]
            
            print(f"[SmoothGrad] slic_visualization shape: {slic_visualization.shape}, min: {slic_visualization.min()}, max: {slic_visualization.max()}")
            
            return slic_visualization  # 슈퍼픽셀별 평균 기여도 맵 반환
        return saliency
        
    def set_config_dict(self, config_dict):
        self.input_size = config_dict.get('input_size', self.input_size)
        self.num_sample = config_dict.get('num_sample', self.num_sample)
        self.var = config_dict.get('var',self.var)
        self.type = config_dict.get('type', self.type)  # type 추가
        self.slic = config_dict.get('slic', self.slic)
        self.slic_size = config_dict.get('slic_size', self.slic_size)
        self.slic_ruler = config_dict.get('slic_ruler', self.slic_ruler)

