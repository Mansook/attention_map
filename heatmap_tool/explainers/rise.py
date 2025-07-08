import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resize

class RISE:
    def __init__(self, model,config_dict):
        """
        model: 학습된 PyTorch 모델 (eval 상태 권장)
        N: 마스크 수
        s: 마스크 해상도 (s x s → 업샘플링 됨)
        p1: 마스크 내에서 1(보이는 영역)일 확률
        input_size: 입력 이미지 크기 (H, W)
        """
        self.model = model.eval()
        self.N = config_dict.get('N', 4000)
        self.s = config_dict.get('s', 8)
        self.p1 = config_dict.get('p1', 0.1)
        self.input_size = config_dict.get('input_size', (96, 96))

        self.device = next(model.parameters()).device
        self.masks = self.generate_masks()  # [N, H, W]

    def generate_masks(self):
        """
        무작위 마스크 생성 → 해상도 업샘플링
        """
        masks = []
        for _ in range(self.N):
            small_mask = (torch.rand((self.s, self.s)) < self.p1).float()  # [s, s]
            mask = resize(small_mask.unsqueeze(0), self.input_size, interpolation=InterpolationMode.BILINEAR)
            masks.append(mask.squeeze(0))  # [H, W]
        masks = torch.stack(masks)  # [N, H, W]
        return masks.to(self.device)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W)
        class_idx: 예측 대상 클래스 인덱스
        return: saliency map (H, W)
        """
        B, C, H, W = input_tensor.shape
        masks = self.masks.unsqueeze(1)  # [N, 1, H, W]
        masked_inputs = input_tensor * masks  # Broadcasting → [N, C, H, W]

        with torch.no_grad():
            outputs = self.model(masked_inputs)  # [N, num_classes]
            if class_idx is None:
                class_idx = torch.argmax(outputs.mean(dim=0)).item()
            scores = outputs[:, class_idx]  # [N]

        saliency = torch.sum(scores.view(-1, 1, 1) * self.masks, dim=0)
        saliency = saliency / saliency.max()
        return saliency.detach().cpu()
    def set_config_dict(self, config_dict):
        self.N = config_dict.get('N', self.N)
        self.s = config_dict.get('s', self.s)
        self.p1 = config_dict.get('p1', self.p1)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.masks = self.generate_masks()