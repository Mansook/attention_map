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
        
        # 모델 구조에 따라 타겟 레이어 찾기
        if hasattr(self.model, "backbone"):
            target_layer = dict([*self.model.backbone.named_modules()])[self.target_layer_name]
        else:
            target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
            
        target_layer.register_forward_hook(forward_hook)
    
    def _get_fc_weights(self, class_idx):
        """FC 레이어의 가중치를 안전하게 가져오기"""
        try:
            # ResNetClassifier의 경우
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'fc'):
                fc_weights = self.model.backbone.fc.weight
            # CustomResNet34의 경우  
            elif hasattr(self.model, 'classifier'):
                fc_weights = self.model.classifier[-1].weight
            else:
                # 마지막 FC 레이어 찾기
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        fc_weights = module.weight
                        break
                else:
                    raise RuntimeError("FC 레이어를 찾을 수 없습니다.")
            
            return fc_weights[class_idx]
            
        except Exception as e:
            print(f"FC 레이어 가중치 가져오기 실패: {e}")
            raise
    
    def generate(self, input_tensor, class_idx=None):
        """
        CAM 맵 생성

        Args:
            input_tensor: 입력 텐서 [1, 3, H, W]
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
            target_weight = self._get_fc_weights(class_idx)
            
            # feature_maps가 None인지 확인
            if self.feature_maps is None:
                raise RuntimeError("feature_maps가 None입니다. 훅이 제대로 등록되었는지 확인하세요.")
            
            # feature map에서 배치 차원 제거
            feature_map = self.feature_maps.squeeze(0)  # [C, H, W]
            
            print(f"Feature map shape: {feature_map.shape}")
            print(f"Target weight shape: {target_weight.shape}")
            
            # CAM 계산
            cam = torch.zeros(feature_map.shape[1], feature_map.shape[2], device=input_tensor.device)
            for i, weight in enumerate(target_weight):
                cam += weight * feature_map[i]

            print(f"CAM shape: {cam.shape}")

            # numpy로 변환 후 바로 리사이즈
            cam_np = cam.detach().cpu().numpy()
            from scipy.ndimage import zoom

            # 원본 이미지 크기로 업샘플링
            target_size = (input_tensor.shape[2], input_tensor.shape[3])  # (96, 96)
            cam_resized = zoom(cam_np, (target_size[0]/cam_np.shape[0], target_size[1]/cam_np.shape[1]))

            print(f"Resized CAM shape: {cam_resized.shape}")

            # utils 함수로 처리
            cam = process_heatmap_by_type(cam_resized, self.type)

            return cam
            
    def set_config_dict(self, config_dict):
        self.target_layer_name = config_dict.get('target_layer', self.target_layer_name)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.type = config_dict.get('type', self.type)
        # target_layer_name이 바뀌면 훅 재등록 필요
        self._register_hook()
                
                