import torch
import argparse
import os
import yaml
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms.functional import normalize
from torchcam.methods import CAM, GradCAM
from torchcam.utils import overlay_mask

# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResNetClassifier
from dataset.datasetLoader import get_dataloaders

def load_checkpoint_yaml(checkpoint_path):
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_path_from_project_root(relative_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    while not os.path.isdir(os.path.join(project_root, "checkpoints")) and os.path.dirname(project_root) != project_root:
        project_root = os.path.dirname(project_root)
    return os.path.normpath(os.path.join(project_root, relative_path.lstrip("./")))

def main():
    parser = argparse.ArgumentParser(description='ResNet18 XAI Explainer 실행 도구 (torchcam)')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='체크포인트 yaml 파일 경로')
    parser.add_argument('-e', '--explainer', type=str, default="gradcam", help='사용할 torchcam explainer (cam, gradcam)')
    args = parser.parse_args()

    # 1. checkpoint yaml 로드
    config = load_checkpoint_yaml(args.checkpoint)

    # 2. 데이터셋 정보 추출 및 DataLoader 생성
    dataset_config = config['dataset_config']
    dataset_config['root'] = resolve_path_from_project_root(dataset_config['root'])
    train_loader, val_loader, test_loader = get_dataloaders(dataset_config)
    class_names = dataset_config.get('classes', None)

    # 3. ResNet18 모델 생성 및 가중치 로드
    model_structure = config['model_structure']
    num_classes = config['num_classes']
    checkpoint_path = resolve_path_from_project_root(config['checkpoint_path'])
    
    model = ResNetClassifier(num_classes=num_classes, dataset_config=model_structure)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"모델 가중치 로드 완료: {checkpoint_path}")
    else:
        print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return

    # GradCAM을 위해 gradient 계산 활성화
    for param in model.parameters():
        param.requires_grad = True
    
    model.eval()

    # 4. torchcam explainer 생성
    explainer_type = args.explainer.lower()
    if explainer_type == "gradcam":
        cam_extractor = GradCAM(model, target_layer="backbone.layer4")
    elif explainer_type == "cam":
        cam_extractor = CAM(model, target_layer="backbone.layer4")
    else:
        print(f"지원하지 않는 explainer: {explainer_type}")
        return

    # 5. 설명 실행
    if test_loader is None:
        print("테스트 데이터셋이 없습니다.")
        return

    all_batches = list(test_loader)
    images, labels = random.choice(all_batches)

    images = images.to(device)
    labels = labels.to(device)

    num_samples = min(10, images.shape[0])
    rand_indices = random.sample(range(images.shape[0]), num_samples)

    with torch.no_grad():
        output = model(images)
        predicted_classes = torch.argmax(output, dim=1)
        confidences = torch.softmax(output, dim=1).max(dim=1).values

    for idx, i in enumerate(rand_indices):
        pred = predicted_classes[i].item()
        conf = confidences[i].item()
        label = labels[i].item()
        input_img = images[i].unsqueeze(0)
        
        # GradCAM을 위해 gradient 계산 활성화
        input_img.requires_grad_(True)

        # CAM 생성
        activation_map = cam_extractor(int(pred), output)[0].cpu().numpy()
     
        # 원본 이미지 복원
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        # numpy 배열을 PIL Image로 변환
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        # activation_map을 2차원으로 압축 (첫 번째 차원 사용)
        activation_map_2d = activation_map[0] if activation_map.ndim == 3 else activation_map
        pil_mask = Image.fromarray((activation_map_2d * 255).astype(np.uint8), mode='L')
        result = overlay_mask(pil_img, pil_mask, alpha=0.5)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"원본 (label: {label})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title(f"히트맵 (예측: {pred})")
        plt.axis('off')

        plt.suptitle(f"[{idx}] 예측: {pred}, 실제: {label}, 확신도: {conf:.3f}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
