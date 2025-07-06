import torch
import argparse
import os
import yaml
from explainers import CAM, GradCAM
from models import ResNetClassifier
from dataset.datasetLoader import get_dataloaders
from utils.visualization import visualize_xai
import random
import numpy as np

def load_checkpoint_yaml(checkpoint_path):
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_path(base_yaml_path, relative_path):
    # base_yaml_path: yaml 파일의 실제 경로
    # relative_path: yaml에 저장된 상대경로
    if os.path.isabs(relative_path):
        return relative_path
    base_dir = os.path.dirname(os.path.abspath(base_yaml_path))
    return os.path.normpath(os.path.join(base_dir, relative_path))

def resolve_path_from_project_root(relative_path):
    # heatmap_tool 폴더 기준으로 경로 변환
    project_root = os.path.dirname(os.path.abspath(__file__))  # use_xai_explainer.py 기준
    while not os.path.isdir(os.path.join(project_root, "checkpoints")) and os.path.dirname(project_root) != project_root:
        project_root = os.path.dirname(project_root)
    return os.path.normpath(os.path.join(project_root, relative_path.lstrip("./")))

def load_explainer_dict(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['explainer_dict']

def main():
    parser = argparse.ArgumentParser(description='XAI Explainer 실행 도구')
    parser.add_argument('-c','--checkpoint', type=str, required=True, help='체크포인트 yaml 파일 경로')
    parser.add_argument('-e','--explainer', type=str, required=True, help='사용할 XAI explainer (예: cam, grad_cam)')
    args = parser.parse_args()

    # 1. checkpoint yaml 로드
    config = load_checkpoint_yaml(args.checkpoint)

    # 2. 데이터셋 정보 추출 및 DataLoader 생성
    dataset_config = config['dataset_config']
    dataset_config['root'] = resolve_path_from_project_root(dataset_config['root'])
    train_loader, val_loader, test_loader = get_dataloaders(dataset_config)
    class_names = dataset_config.get('classes', None)

    # 3. 모델 생성 및 가중치 로드
    model_structure = config['model_structure']
    num_classes = config['num_classes']
    model_name = config['model']
    checkpoint_path = config['checkpoint_path']
    # yaml 기준으로 실제 경로 변환
    checkpoint_path = resolve_path_from_project_root(checkpoint_path)
    # 모델 생성 (예시: ResNetClassifier만 지원)
    if model_name.lower() == "resnet18":
        print("model_structure from yaml:", model_structure)
        model = ResNetClassifier(num_classes=num_classes, dataset_config=model_structure)
        print("실제 모델 conv1:", model.backbone.conv1)
        print("실제 모델 maxpool:", model.backbone.maxpool)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"모델 가중치 로드 완료: {checkpoint_path}")
    else:
        print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    model.eval()

    # 4. explainer 생성
    explainer_dict = load_explainer_dict("explainer_config.yaml")
    explainer_key = args.explainer.lower()
    if explainer_key not in explainer_dict:
        print(f"지원하지 않는 explainer: {args.explainer}")
        return
    explainer_class = eval(explainer_dict[explainer_key]['class'])  # CAM, GradCAM 등
    
    # 모델별 target layer 설정이 있는지 확인
    if 'model' in explainer_dict[explainer_key] and model_name in explainer_dict[explainer_key]['model']:
        target_layer = explainer_dict[explainer_key]['model'][model_name]['target_layer']
    else:
        target_layer = explainer_dict[explainer_key]['default_target_layer']
   
    explainer = explainer_class(model, target_layer_name=target_layer)

    # 5. 설명 실행 (test_loader 기준)
    if test_loader is None:
        print("테스트 데이터셋이 없습니다.")
        return

    # 1. 배치 하나만 랜덤으로 뽑기
    all_batches = list(test_loader)
    images, labels = random.choice(all_batches)  # 랜덤 배치 하나 선택

    images = images.to(device)
    labels = labels.to(device)

    # 2. 배치 안에서 10장만 랜덤으로 뽑기
    num_samples = min(10, images.shape[0])
    rand_indices = random.sample(range(images.shape[0]), num_samples)

    with torch.no_grad():
        output = model(images)  # [batch_size, num_classes]
        predicted_classes = torch.argmax(output, dim=1)  # [batch_size]
        confidences = torch.softmax(output, dim=1).max(dim=1).values  # [batch_size]

    import matplotlib.pyplot as plt
    import numpy as np

    for idx, i in enumerate(rand_indices):
        pred = predicted_classes[i].item()
        conf = confidences[i].item()
        label = labels[i].item()
        print(f"[{idx}] 예측: {pred}, 실제: {label}, 확신도: {conf:.3f}")

        # 히트맵 생성 및 시각화
        input_img = images[i].unsqueeze(0)  # (1, 3, H, W)
        target_class = pred  # 또는 label로 바꿔도 됨
        print("input image shape:", input_img.shape)
        # explainer가 CAM/GradCAM 등일 때
        heatmap = explainer.generate(input_img, class_idx=target_class)
        print("heatmap shape:", heatmap.shape)
        # 이 코드는 각 샘플에 대해 XAI explainer(예: CAM, GradCAM)로 생성한 히트맵을 원본 이미지 위에 시각화하는 부분입니다.
        # 1. 만약 heatmap이 torch.Tensor 타입이면 numpy로 변환합니다.
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().cpu().numpy()
        # 2. 원본 이미지를 복원합니다. (normalize 해제: mean=0.5, std=0.5 기준)
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        # normalize 해제 (기본값: mean=0.5, std=0.5)
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        # 3. 시각화: 왼쪽에는 원본, 오른쪽에는 히트맵을 원본 위에 overlay해서 보여줍니다.
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"원본 (label: {label})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"히트맵 (예측: {pred})")
        plt.axis('off')

        plt.suptitle(f"[{idx}] 예측: {pred}, 실제: {label}, 확신도: {conf:.3f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main() 