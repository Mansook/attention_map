#!/usr/bin/env python3
"""
학습 완료 시 체크포인트와 함께 config를 자동으로 저장하는 유틸리티
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def save_checkpoint_config(
    checkpoint_name: str,
    checkpoint_path: str,
    model_name: str,
    dataset_name: str,
    num_classes: int,
    model_structure: Dict[str, Any],
    dataset_config: Dict[str, Any],
    training_info: Any = None,
    config_dir: str = "./configs/checkpoints"
) -> str:
    """
    체크포인트와 함께 config 파일 저장
    
    Args:
        checkpoint_name: 체크포인트 파일명 (예: resnet_stl10.pt)
        model_name: 모델 이름 (예: ResNetClassifier)
        dataset_name: 데이터셋 이름 (예: STL10)
        num_classes: 클래스 수
        model_structure: 모델 구조 설정
        dataset_config: 데이터셋 설정
        training_info: 학습 정보 (선택사항)
        config_dir: config 저장 디렉토리
        
    Returns:
        str: 저장된 config 파일 경로
    """
    
    # config 파일명 생성
    config_filename = checkpoint_name.replace('.pt', '.yaml')
    config_path = Path(config_dir) / config_filename
    
    # 디렉토리 생성 (모든 상위 디렉토리 포함)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # config 데이터 구성
    config_data = {
        'checkpoint_name': checkpoint_name,
        'model': model_name,
        'dataset': dataset_name,
        'num_classes': num_classes,
        'checkpoint_path': checkpoint_path,
        'description': f"{model_name} trained on {dataset_name} dataset",
        'model_structure': model_structure,
        'dataset_config': dataset_config
    }
    
    # 학습 정보 추가 (있는 경우)
    if training_info:
        config_data['training_info'] = training_info
    
    # config 파일 저장
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, 
                  default_flow_style=False, 
                  allow_unicode=True, 
                  sort_keys=False,
                  default_style=None)  # 앵커/별칭 비활성화
    
    print(f"체크포인트 config 저장 완료: {config_path}")
    return str(config_path)

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """데이터셋 설정 반환"""
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
