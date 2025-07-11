from configparser import InterpolationError
import torch
import torch.nn as nn
import torch.nn.functional as F

class IG(nn.Module):
    def __init__(self,model,config_dict):
        super(IG,self).__init__()
        self.model = model.eval()
        self.input_size = config_dict.get('input_size',(96,96))
        self.steps = config_dict.get('steps',10)
        self.baseline = None
        self.feature_maps = None
        self.gradients = None
        # IG는 입력에 대한 attribution이므로 target_layer는 사용하지 않음
        # self.target_layer_name = config_dict.get('target_layer','layer4')
        # self._register_hook()  # 훅 등록 제거

    # 훅 등록 메서드 제거 - IG는 입력에 직접 적용
    # def _register_hook(self):
    #     def forward_hook(module,input,output):
    #         self.feature_maps = output
    #         print(f"[IG] 🔍 feature_maps.shape: {output.shape}")
    #     def backward_hook(module,grad_input,grad_output):
    #         self.gradients = grad_output[0]
    #         
    #     target_layer = dict([*self.model.backbone.named_modules()])[self.target_layer_name]
    #     target_layer.register_forward_hook(forward_hook)
    #     target_layer.register_full_backward_hook(backward_hook)
            
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
    
    def generate(self,input_tensor,class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        self.baseline = self.make_baseline(input_tensor)
        interpolated_inputs = self._interpolate_input(input_tensor,self.baseline)
        interpolated_inputs = interpolated_inputs.to(input_tensor.device)
        
        gradients_sum = torch.zeros_like(input_tensor)
        
        for i,interpolated_input in enumerate(interpolated_inputs):
            # 각 보간된 입력에 대해 그래디언트 계산
            interpolated_input = interpolated_input.clone().requires_grad_(True)
            
            output = self.model(interpolated_input)
            if class_idx is None:
                # 네, output에서 가장 확률(로짓)이 높은 클래스의 인덱스를 가져오는 코드입니다.
                class_idx = torch.argmax(output, dim=1).item()
            # output이 2D (batch, class)라고 가정하고, 항상 첫 번째 배치만 사용하지 말고, 배치 전체에 대해 gather 사용
            score = output[:, class_idx]
            score.backward(retain_graph=True)
            
            # ∂F(αx + (1-α)x')/∂x 계산
            if interpolated_input.grad is not None:
                gradients = interpolated_input.grad.clone()
                gradients_sum += gradients
            else:
                raise RuntimeError("Gradients are not available")
            
            interpolated_input.grad.zero_()
        
        # 평균 그래디언트 계산 (적분의 근사)
        avg_gradients = gradients_sum / self.steps
        
        # IG 계산: (x - x') × 평균 그래디언트
        ig_attribution = (input_tensor - self.baseline) * avg_gradients
        
         # 채널별로 평균하여 2D 히트맵 생성 (업샘플링 제거)
        ig_map = torch.mean(ig_attribution, dim=1)  # [1, H, W]
        
        # 업샘플링 제거 - IG는 입력 픽셀 크기 그대로 사용
        ig_map = ig_map.squeeze().detach().cpu().numpy()
        
        # 정규화
        ig_map = ig_map - ig_map.min()
        if ig_map.max() > 0:
            ig_map = ig_map / ig_map.max()
            
        return ig_map

    def set_config_dict(self, config_dict):
        """설정 업데이트"""
        self.steps = config_dict.get('steps', self.steps)
        self.input_size = config_dict.get('input_size', self.input_size)
    