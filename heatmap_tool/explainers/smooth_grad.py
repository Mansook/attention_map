import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type


class SmoothGrad(nn.Module):
    def __init__(self, model, config_dict):
        super(SmoothGrad, self).__init__()
        self.model = model.eval()
        
        # config_dict에서 설정 가져오기
        self.input_size = tuple(config_dict.get('input_size', (96, 96)))
        self.num_sample = config_dict.get('num_sample',10)
        self.var = config_dict.get('var',0.125)
        self.type = config_dict.get('type', 'both')  # type 추가

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
        
        # type에 따라 처리
        if self.type == 'abs':
            saliency = avg_gradients.abs().mean(dim=1)
        elif self.type == 'positive':
            saliency = F.relu(avg_gradients).mean(dim=1)
        elif self.type == 'negative':
            saliency = -F.relu(-avg_gradients).mean(dim=1)
        elif self.type == 'both':
            saliency = avg_gradients.mean(dim=1)
        else:
            saliency = avg_gradients.mean(dim=1)
        
        # utils의 통합 처리 함수 사용
        saliency = process_heatmap_by_type(saliency.squeeze(), self.type)

        return saliency
        
    def set_config_dict(self, config_dict):
        self.input_size = config_dict.get('input_size', self.input_size)
        self.num_sample = config_dict.get('num_sample', self.num_sample)
        self.var = config_dict.get('var',self.var)
        self.type = config_dict.get('type', self.type)  # type 추가

