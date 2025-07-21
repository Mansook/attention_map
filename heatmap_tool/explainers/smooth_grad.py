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
            print("SmoothGrad Slic Lets go")
            sp_map, labels, sp_means = superpixel_mean_map(saliency, region_size=int(self.slic_size), ruler=float(self.slic_ruler))
            sp_map_norm = (sp_map - sp_map.min()) / (sp_map.max() - sp_map.min() + 1e-8)
            sp_map_uint8 = (sp_map_norm * 255).astype(np.uint8)
            sp_map_color = cv2.applyColorMap(sp_map_uint8, cv2.COLORMAP_JET)
            return sp_map_color  # (H, W, 3) RGB 이미지 반환
        return saliency
        
    def set_config_dict(self, config_dict):
        self.input_size = config_dict.get('input_size', self.input_size)
        self.num_sample = config_dict.get('num_sample', self.num_sample)
        self.var = config_dict.get('var',self.var)
        self.type = config_dict.get('type', self.type)  # type 추가
        self.slic = config_dict.get('slic', self.slic)
        self.slic_size = config_dict.get('slic_size', self.slic_size)
        self.slic_ruler = config_dict.get('slic_ruler', self.slic_ruler)

