import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom
from explainers.utils.normalize_heatmap import process_heatmap_by_type

class GradCAM(nn.Module):
    def __init__(self, model, config_dict):
        super(GradCAM, self).__init__()
        self.model = model.eval()
        
        # config_dict에서 설정 가져오기
        self.target_layer_name = config_dict.get('target_layer', 'layer4')
        self.input_size = tuple(config_dict.get('input_size', (96, 96)))
        self.type = config_dict.get('type', 'both')  # type 추가
        
        self.feature_maps = None
        self.gradients = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        if hasattr(self.model, "backbone"):
            target_layer = dict([*self.model.backbone.named_modules()])[self.target_layer_name]
        else:
            target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        
        self.model.zero_grad()
        self.gradients = None
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[:, class_idx]    
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        if gradients is None:
            raise RuntimeError("Gradients가 None입니다. register_full_backward_hook이 제대로 동작하는지 확인하세요.")
            
        feature_maps = self.feature_maps
        if feature_maps is None:
            raise RuntimeError("feature_maps가 None입니다. forward hook이 제대로 동작하는지 확인하세요.")
            
        # 피쳐맵 해상도 출력
        print(f"feature_maps.shape: {feature_maps.shape}")
        print(f"gradients.shape: {gradients.shape}")
            
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1)  # [1, H, W]
        
        print(f"GradCAM shape: {cam.shape}")
        
        # numpy로 변환 후 바로 리사이즈
        cam_np = cam.detach().cpu().numpy().squeeze()
        
        # 원본 이미지 크기로 업샘플링
        target_size = (input_tensor.shape[2], input_tensor.shape[3])  # (96, 96)
        cam_resized = zoom(cam_np, (target_size[0]/cam_np.shape[0], target_size[1]/cam_np.shape[1]))
        
        print(f"Resized GradCAM shape: {cam_resized.shape}")
        
        # utils 함수로 처리
        cam = process_heatmap_by_type(cam_resized, self.type)
            
        return cam
        
    def set_config_dict(self, config_dict):
        self.target_layer_name = config_dict.get('target_layer', self.target_layer_name)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.type = config_dict.get('type', self.type)  # type 추가
        # target_layer_name이 바뀌면 훅 재등록 필요
        self._register_hook()