#!/usr/bin/env python3
"""
체크포인트 중심의 config 로더
각 체크포인트마다 고유한 모델 구조와 설정을 관리
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

class CheckpointConfigLoader:
    def __init__(self, checkpoint_config_dir: str = "./configs/checkpoints"):
        """체크포인트 config 로더 초기화"""
        self.checkpoint_config_dir = Path(checkpoint_config_dir)
        
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """YAML 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_checkpoint_config(self, checkpoint_name: str) -> Dict[str, Any]:
        """체크포인트별 설정 로드"""
        # 체크포인트 파일명에서 config 파일명 생성
        config_filename = checkpoint_name.replace('.pt', '.yaml')
        config_path = self.checkpoint_config_dir / config_filename
        
        if config_path.exists():
            return self._load_yaml(str(config_path))
        else:
            # config 파일이 없으면 자동 생성
            return self._create_auto_checkpoint_config(checkpoint_name)
    
    def _create_auto_checkpoint_config(self, checkpoint_name: str) -> Dict[str, Any]:
        """체크포인트 파일명에서 자동으로 config 생성"""
        # 파일명 패턴 매칭
        patterns = [
            (r'resnet_(\w+)\.pt', 'ResNetClassifier'),
            (r'vit_(\w+)\.pt', 'ViT'),
            (r'efficientnet_(\w+)\.pt', 'EfficientNet'),
            (r'(\w+)_(\w+)\.pt', None),
        ]
        
        for pattern, model_name in patterns:
            match = re.match(pattern, checkpoint_name)
            if match:
                if model_name is None:
                    detected_model = match.group(1).title()
                    detected_dataset = match.group(2).upper()
                else:
                    detected_model = model_name
                    detected_dataset = match.group(1).upper()
                
                # 클래스 수 추정
                if detected_dataset in ['STL10', 'CIFAR10']:
                    num_classes = 10
                elif detected_dataset == 'CIFAR100':
                    num_classes = 100
                elif detected_dataset == 'IMAGENET':
                    num_classes = 1000
                else:
                    num_classes = 10  # 기본값
                
                # 기본 모델 구조 (데이터셋별)
                model_structure = self._get_default_model_structure(detected_model, detected_dataset)
                
                return {
                    'checkpoint_name': checkpoint_name,
                    'model': detected_model,
                    'dataset': detected_dataset,
                    'num_classes': num_classes,
                    'checkpoint_path': f"./checkpoints/{checkpoint_name}",
                    'description': f"{detected_model} trained on {detected_dataset} dataset (auto-detected)",
                    'model_structure': model_structure,
                    'dataset_config': self._get_default_dataset_config(detected_dataset),
                    'auto_detected': True
                }
        
        raise ValueError(f"체크포인트 파일명을 파싱할 수 없습니다: {checkpoint_name}")
    
    def _get_default_model_structure(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """기본 모델 구조 반환"""
        if model_name == "ResNetClassifier":
            if dataset_name in ['STL10', 'CIFAR10', 'CIFAR100']:
                return {
                    'conv1': {
                        'kernel_size': 3,
                        'stride': 1,
                        'padding': 1,
                        'bias': False
                    },
                    'maxpool': 'identity',
                    'target_layers': {
                        'CAM': 'layer4',
                        'GradCAM': 'layer4'
                    }
                }
            elif dataset_name == 'IMAGENET':
                return {
                    'conv1': {
                        'kernel_size': 7,
                        'stride': 2,
                        'padding': 3,
                        'bias': False
                    },
                    'maxpool': 'default',
                    'target_layers': {
                        'CAM': 'layer4',
                        'GradCAM': 'layer4'
                    }
                }
        
        # 기본 구조
        return {
            'conv1': {
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'bias': False
            },
            'maxpool': 'identity',
            'target_layers': {
                'CAM': 'layer4',
                'GradCAM': 'layer4'
            }
        }
    
    def _get_default_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """기본 데이터셋 설정 반환"""
        if dataset_name == 'STL10':
            return {
                'name': 'STL10',
                'root': './data',
                'batch_size': 1,
                'num_workers': 0,
                'class_num': 10,
                'download': False,
                'samples_per_class': 100,
                'classes': ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'],
                'transforms': {
                    'img_size': [96, 96],
                    'normalize': {
                        'mean': [0.5, 0.5, 0.5],
                        'std': [0.5, 0.5, 0.5]
                    }
                }
            }
        elif dataset_name == 'CIFAR10':
            return {
                'name': 'CIFAR10',
                'root': './data',
                'batch_size': 1,
                'num_workers': 0,
                'class_num': 10,
                'download': False,
                'samples_per_class': 100,
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                'transforms': {
                    'img_size': [32, 32],
                    'normalize': {
                        'mean': [0.4914, 0.4822, 0.4465],
                        'std': [0.2023, 0.1994, 0.2010]
                    }
                }
            }
        elif dataset_name == 'CIFAR100':
            return {
                'name': 'CIFAR100',
                'root': './data',
                'batch_size': 1,
                'num_workers': 0,
                'class_num': 100,
                'download': False,
                'samples_per_class': 50,
                'classes': [f'class_{i}' for i in range(100)],
                'transforms': {
                    'img_size': [32, 32],
                    'normalize': {
                        'mean': [0.5071, 0.4867, 0.4408],
                        'std': [0.2675, 0.2565, 0.2761]
                    }
                }
            }
        
        # 기본 설정
        return {
            'name': dataset_name,
            'root': './data',
            'batch_size': 1,
            'num_workers': 0,
            'class_num': 10,
            'download': False,
            'samples_per_class': 100,
            'classes': [f'class_{i}' for i in range(10)],
            'transforms': {
                'img_size': [64, 64],
                'normalize': {
                    'mean': [0.5, 0.5, 0.5],
                    'std': [0.5, 0.5, 0.5]
                }
            }
        }
    
    def list_available_checkpoints(self) -> List[str]:
        """사용 가능한 체크포인트 목록 반환"""
        checkpoints = []
        checkpoint_dir = Path("./checkpoints")
        
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("*.pt"):
                checkpoints.append(checkpoint_file.name)
        
        return sorted(checkpoints)
    
    def get_checkpoint_info(self, checkpoint_name: str) -> Dict[str, Any]:
        """체크포인트 정보 반환"""
        config = self.get_checkpoint_config(checkpoint_name)
        
        return {
            'checkpoint_name': config['checkpoint_name'],
            'model': config['model'],
            'dataset': config['dataset'],
            'num_classes': config['num_classes'],
            'description': config['description'],
            'auto_detected': config.get('auto_detected', False)
        }
    
    def create_checkpoint_config_file(self, checkpoint_name: str, config: Dict[str, Any]) -> str:
        """체크포인트 config 파일 생성"""
        config_filename = checkpoint_name.replace('.pt', '.yaml')
        config_path = self.checkpoint_config_dir / config_filename
        
        # 디렉토리 생성
        self.checkpoint_config_dir.mkdir(parents=True, exist_ok=True)
        
        # config 파일 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return str(config_path) 