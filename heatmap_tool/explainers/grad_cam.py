import torch
import torch.nn.functional as F
import torch.nn as nn

class GradCAM(nn.Module):
    def __init__(self, model, config_dict):
        super(GradCAM, self).__init__()
        self.model = model.eval()
        
        # config_dict에서 설정 가져오기
        self.target_layer_name = config_dict.get('target_layer', 'layer4')
        self.input_size = config_dict.get('input_size', (96, 96))
        
        self.feature_maps = None
        self.gradients = None
        self._register_hook()

    def _register_hook(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
            print(f"[GradCAM] 🔍 feature_maps.shape: {output.shape}")  # ✅ 여기에 찍어야 shape 확인됨
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        target_layer = dict([*self.model.backbone.named_modules()])[self.target_layer_name]
        
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
        cam = torch.sum(weights * feature_maps, dim=1)  # [1, 6, 6]
        cam = F.relu(cam)
        cam = cam.unsqueeze(1)  # [1, 1, 6, 6] - 채널 차원 추가
        
        # 원본 이미지 크기로 업샘플링
        input_size = input_tensor.shape[2:]  # (H, W)
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam
