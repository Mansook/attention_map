from configparser import InterpolationError
import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import numpy as np
class IG(nn.Module):
    def __init__(self,model,config_dict):
        super(IG,self).__init__()
        self.model = model.eval()
        self.input_size = tuple(config_dict.get('input_size',(96,96)))
        self.steps = config_dict.get('steps',10)
        self.type = config_dict.get('type', 'both')  # type 추가
        print("steps : ",self.steps)
        self.baseline = None
        self.feature_maps = None
        self.gradients = None
        self.slic = config_dict.get('slic',False)
        self.slic_size = config_dict.get('slic_size',10)
        self.slic_ruler = config_dict.get('slic_ruler',10)
        
    def make_baseline(self,input_tensor):
        return torch.zeros_like(input_tensor)
    
    def _interpolate_input(self, input_tensor, baseline):
        """입력과 베이스라인 사이를 보간"""
        alphas = torch.linspace(0, 1, self.steps, device=input_tensor.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
         # torch.stack을 사용하므로 반환값은 torch.Tensor입니다.            
        return torch.stack(interpolated_inputs, dim=0)
    
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        self.baseline = self.make_baseline(input_tensor)
        interpolated_inputs = self._interpolate_input(input_tensor, self.baseline)
        interpolated_inputs = interpolated_inputs.to(input_tensor.device)

        gradients_sum = torch.zeros_like(input_tensor)
        for i in range(self.steps):
            interpolated_input = interpolated_inputs[i]
            interpolated_input = interpolated_input.clone().requires_grad_(True)
            output = self.model(interpolated_input)
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            score = output[:, class_idx]
            self.model.zero_grad()
            score.backward()
            gradients = interpolated_input.grad
            if gradients is not None:
                gradients_sum += gradients
            else:
                raise RuntimeError("Gradients are not available")

        avg_gradients = gradients_sum / self.steps
        ig_attribution = (input_tensor - self.baseline) * avg_gradients
        ig_map = torch.mean(ig_attribution, dim=1)  # [1, H, W]
        
        # utils의 통합 처리 함수 사용
        ig_map = process_heatmap_by_type(ig_map.squeeze(), self.type)
        
        if self.slic is True:
            print("IG Slic Lets go")
            sp_map, labels, sp_means = superpixel_mean_map(ig_map,region_size = self.slic_size,ruler=self.slic_ruler)
            # 시각화용 RGB 컬러맵 변환 (예: OpenCV)
            import cv2
            sp_map_norm = (sp_map - sp_map.min()) / (sp_map.max() - sp_map.min() + 1e-8)
            sp_map_uint8 = (sp_map_norm * 255).astype(np.uint8)
            sp_map_color = cv2.applyColorMap(sp_map_uint8, cv2.COLORMAP_JET)
            return sp_map_color  # (H, W, 3) RGB 이미지 반환

        return ig_map  # 1채널 맵 반환

    def set_config_dict(self, config_dict):
        """설정 업데이트"""
        self.steps = config_dict.get('steps', self.steps)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.type = config_dict.get('type', self.type)  # type 추가
    