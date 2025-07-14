from configparser import InterpolationError
import torch
import torch.nn as nn
import torch.nn.functional as F

class IG(nn.Module):
    def __init__(self,model,config_dict):
        super(IG,self).__init__()
        self.model = model.eval()
        self.input_size = tuple(config_dict.get('input_size',(96,96)))
        self.steps = config_dict.get('steps',10)
        print("steps : ",self.steps)
        self.baseline = None
        self.feature_maps = None
        self.gradients = None
        
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
        ig_map = ig_map.squeeze().detach().cpu().numpy()
        ig_map = ig_map - ig_map.min()
        if ig_map.max() > 0:
            ig_map = ig_map / ig_map.max()
        return ig_map

    def set_config_dict(self, config_dict):
        """설정 업데이트"""
        self.steps = config_dict.get('steps', self.steps)
        self.input_size = config_dict.get('input_size', self.input_size)
    