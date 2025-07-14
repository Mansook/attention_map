
def get_layer_by_name(model, layer_name):
    # backbone이 있으면 backbone에서, 없으면 model 전체에서 찾기
    if hasattr(model, "backbone"):
        modules = dict([*model.backbone.named_modules()])
    else:
        modules = dict([*model.named_modules()])
    return modules.get(layer_name, None)