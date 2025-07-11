import re

def make_explainer_tooltip(name, explainer_dict, config, current_image_tensor):
    name_no_number = re.sub(r'\d+$', '', name)
    expl_config = explainer_dict.get(name_no_number, {})

    # 1. explainer config 기본 정보
    info_items = []
    for key in ['class', 'description', 'cmap']:
        if key in expl_config:
            info_items.append(f"{key}: {expl_config[key]}")

    # 2. 모델별 추가 정보 (예: target_layer 등)
    model_info_items = []
    model_name = None
    if config is not None and 'model' in config:
        model_name = config['model']
        model_dict = expl_config.get('model', {})
        model_specific = model_dict.get(model_name, {})
        if isinstance(model_specific, dict) and model_specific:
            for k, v in model_specific.items():
                model_info_items.append(f"{model_name} {k}: {v}")

    # 3. 데이터 정보
    data_items = []
    if current_image_tensor is not None:
        shape = tuple(current_image_tensor.shape)
        data_items.append(f"이미지 shape: {shape}")
    if config is not None:
        if 'num_classes' in config:
            data_items.append(f"클래스 수: {config['num_classes']}")
        if model_name:
            data_items.append(f"모델명: {model_name}")

    # 4. 합쳐서 출력
    msg = "\n".join(info_items + model_info_items + data_items) if (info_items or model_info_items or data_items) else "설명자/데이터 정보 없음"
    return msg