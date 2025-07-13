import re

def make_explainer_tooltip(name, explainers, explainer_dict, config, current_image_tensor):
    name_no_number = re.sub(r'\d+$', '', name)
    expl_config = explainer_dict.get(name_no_number, {})
    
    # 실제 생성된 explainer의 파라미터 가져오기
    actual_params = {}
    if name in explainers:
        actual_explainer = explainers[name]
        if hasattr(actual_explainer, 'params'):
            actual_params = actual_explainer.params
        elif hasattr(actual_explainer, 'target_layer'):
            actual_params = {'target_layer': actual_explainer.target_layer}

    # 1. explainer config 기본 정보
    info_items = []
    for key in ['class', 'description', 'cmap']:
        if key in expl_config:
            info_items.append(f"{key}: {expl_config[key]}")

    # 2. 실제 적용된 파라미터 정보
    param_items = []
    if actual_params:
        for k, v in actual_params.items():
            param_items.append(f"실제 {k}: {v}")
    
    # 3. 모델별 기본 정보 (실제 파라미터가 없을 때)
    model_info_items = []
    model_name = None
    if config is not None and 'model' in config:
        model_name = config['model']
        if not actual_params:  # 실제 파라미터가 없을 때만 기본값 표시
            model_dict = expl_config.get('model', {})
            model_specific = model_dict.get(model_name, {})
            if isinstance(model_specific, dict) and model_specific:
                for k, v in model_specific.items():
                    model_info_items.append(f"기본 {model_name} {k}: {v}")

    # 4. 데이터 정보
    data_items = []
    if current_image_tensor is not None:
        shape = tuple(current_image_tensor.shape)
        data_items.append(f"이미지 shape: {shape}")
    if config is not None:
        if 'num_classes' in config:
            data_items.append(f"클래스 수: {config['num_classes']}")
        if model_name:
            data_items.append(f"모델명: {model_name}")

    # 5. 합쳐서 출력 (실제 파라미터 우선)
    all_items = info_items + param_items + model_info_items + data_items
    msg = "\n".join(all_items) if all_items else "설명자/데이터 정보 없음"
    return msg