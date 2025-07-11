import os
import yaml

# utils/load_config.py 상단에 추가
HEATMAP_TOOL_ROOT = os.path.dirname(os.path.dirname(__file__))

def load_dataset_config(dataset_name):
    """configs/dataset/{dataset_name}.yaml 파일에서 설정 로드"""
    config_path = os.path.join(HEATMAP_TOOL_ROOT, f"configs/dataset/{dataset_name.lower()}.yaml")
    
    print(f"🔍 경로 확인: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config 파일이 비어있거나 잘못된 형식입니다: {config_path}")
    
    # 필수 필드 검증
    required_fields = ['name', 'root', 'num_classes', 'batch_size']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Config 파일에 필수 필드가 누락되었습니다: {missing_fields}")
    
    return config