import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from explainers.utils.get_layer_name import get_layer_by_name
from explainers.utils.normalize_heatmap import process_heatmap_by_type

class CAM(nn.Module):
    def __init__(self, model, config_dict):
        """
        CAM (Class Activation Mapping) 초기화
        
        Args:
            model: 분석할 모델 (ResNetClassifier 등)
            config_dict: 설정 딕셔너리 (target_layer_name, input_size, type 등 포함)
        """
        super(CAM, self).__init__()
        self.model = model.eval()
        
        # config_dict에서 설정 가져오기
        self.target_layer_name = config_dict.get('target_layer', 'layer4')
        self.input_size = tuple(config_dict.get('input_size', (96, 96)))
        self.type = config_dict.get('type', 'both')  # 'abs', 'positive', 'negative', 'both'
    
        # 타겟 레이어의 출력을 저장할 변수 초기화
        self.feature_maps = None
        
        # 타겟 레이어에 훅을 등록하여 출력을 캡처
        self._register_hook()
    
    def _register_hook(self):
        """타겟 레이어에 훅을 등록하여 feature map을 캡처"""
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        # backbone이 있으면 backbone에서, 없으면 model 전체에서 찾기
        if hasattr(self.model, "backbone"):
            modules = dict([*self.model.backbone.named_modules()])
        else:
            modules = dict([*self.model.named_modules()])

        if self.target_layer_name not in modules:
            raise ValueError(f"타겟 레이어 '{self.target_layer_name}'을(를) 찾을 수 없습니다.")
        target_layer = modules[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
    
    def forward(self, x):
        """모델의 forward pass (훅 실행을 위해 필요)"""
        return self.model(x)
        
    def generate(self, input_tensor, class_idx=None):
        """
        CAM 맵 생성

        Args:
            input_tensor: 입력 텐서 (예: 이미지) [3, H, W] 또는 [1, 3, H, W] (무조건 단일 이미지만 지원)
            class_idx: 특정 클래스의 CAM 맵 생성 (기본값: None, 예측된 클래스 사용)

        Returns:
            cam: CAM 히트맵 [H, W]
        """
        with torch.no_grad():
           
            # 모델 forward pass (훅이 실행되어 feature_maps가 저장됨)
            output = self.model(input_tensor)
 
            # class_idx가 None이면 예측된 클래스 사용
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()

            # FC 레이어의 가중치 가져오기
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "fc"):
                # torchvision resnet 계열
                fc_layer = get_layer_by_name(self.model, "fc") or self.model.backbone.fc
                fc_weights = fc_layer.weight
            else:
                # CustomResNet34
                fc_layer = get_layer_by_name(self.model, "classifier.3")
                if fc_layer is None:
                    raise RuntimeError("FC 레이어(classifier.4)를 찾을 수 없습니다.")
                fc_weights = fc_layer.weight
            target_weight = fc_weights[class_idx]  # 해당 클래스의 가중치
            # feature_maps가 None인지 확인
            if self.feature_maps is None:
                raise RuntimeError("feature_maps가 None입니다. 훅이 제대로 등록되었는지 확인하세요.")
            
            # feature map에서 배치 차원 제거
            feature_map = self.feature_maps.squeeze(0)  # [C, H, W]
            
            # CAM 계산: feature map과 가중치의 가중합
            cam = torch.zeros(feature_map.shape[1], feature_map.shape[2], device=input_tensor.device)
            for i, weight in enumerate(target_weight):
                cam += weight * feature_map[i]
            
            # 원본 이미지 크기로 리사이즈
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            # utils의 통합 처리 함수 사용
            cam = process_heatmap_by_type(cam.squeeze(), self.type)
            
            return cam
            
    def set_config_dict(self, config_dict):
        self.target_layer_name = config_dict.get('target_layer', self.target_layer_name)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.type = config_dict.get('type', self.type)
        # target_layer_name이 바뀌면 훅 재등록 필요
        self._register_hook()
                
                