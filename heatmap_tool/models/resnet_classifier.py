import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10, weights=True, dataset_config=None):
        """
        ResNet 분류기 초기화
        
        Args:
            num_classes: 분류할 클래스 수
            weights: 가중치 사용 여부
            dataset_config: 데이터셋별 모델 구조 설정
        """
        super(ResNetClassifier, self).__init__()
        
        # torchvision에서 제공하는 resnet18 모델을 불러옵니다
        self.backbone = models.resnet18(weights=weights)
        
        # 데이터셋별 모델 구조 조정
        if dataset_config:
            self._apply_dataset_config(dataset_config)
        
        # FC 레이어 조정 (클래스 수에 맞게)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def _apply_dataset_config(self, dataset_config):
        """config에서 설정을 적용하여 모델 구조 조정"""
        
        # conv1 설정 적용
        if 'conv1' in dataset_config:
            conv1_config = dataset_config['conv1']
            self.backbone.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=conv1_config.get('kernel_size', 7),
                stride=conv1_config.get('stride', 2),
                padding=conv1_config.get('padding', 3),
                bias=conv1_config.get('bias', False)
            )
        
        # maxpool 설정 적용
        if 'maxpool' in dataset_config:
            maxpool_type = dataset_config['maxpool']
            if maxpool_type == "identity":
                self.backbone.maxpool = nn.Identity()  # type: ignore
            elif maxpool_type == 'default':
                # 기본값 유지 (변경 안함)
                pass
            else:
                # 커스텀 maxpool 설정
                print("Custom maxpool")
                self.backbone.maxpool = nn.MaxPool2d(
                    kernel_size=maxpool_type.get('kernel_size', 3),
                    stride=maxpool_type.get('stride', 2),
                    padding=maxpool_type.get('padding', 1)
                )

    def forward(self, x):
        return self.backbone(x)