# XAI Explainer 도구

이 도구는 다양한 데이터셋과 모델에 대해 XAI(설명 가능한 AI) 기법을 적용하여 히트맵을 생성하는 도구입니다.

## 사용법

### 기본 명령어 형식

```bash
python use_xai_explainer.py -d <데이터셋> -m <모델> -x <XAI기법>
```

### 매개변수 설명

- `-d, --dataset`: 사용할 데이터셋 (예: stl10, cifar10, cifar100)
- `-m, --model`: 사용할 모델 (예: resnetclassifier)
- `-x, --explainer`: 사용할 XAI 기법 (예: cam, grad_cam)
- `--config`: 설정 파일 경로 (기본값: config.yaml)

### 사용 예시

#### STL10 데이터셋 + ResNetClassifier + CAM
```bash
python use_xai_explainer.py -d stl10 -m resnetclassifier -x cam
```

#### STL10 데이터셋 + ResNetClassifier + GradCAM
```bash
python use_xai_explainer.py -d stl10 -m resnetclassifier -x grad_cam
```

#### CIFAR10 데이터셋 + ResNetClassifier + CAM
```bash
python use_xai_explainer.py -d cifar10 -m resnetclassifier -x cam
```

#### CIFAR100 데이터셋 + ResNetClassifier + GradCAM
```bash
python use_xai_explainer.py -d cifar100 -m resnetclassifier -x grad_cam
```

## 지원하는 모델-데이터셋 조합

### ResNetClassifier 모델
- **ResNetClassifier-STL10**: STL10 데이터셋으로 학습된 ResNet18
- **ResNetClassifier-CIFAR10**: CIFAR10 데이터셋으로 학습된 ResNet18  
- **ResNetClassifier-CIFAR100**: CIFAR100 데이터셋으로 학습된 ResNet18

### 데이터셋 정보

#### STL10
- **클래스 수**: 10개
- **클래스**: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- **이미지 크기**: 96x96
- **모델 구조**: conv1(3x3, stride=1), maxpool 제거

#### CIFAR10
- **클래스 수**: 10개
- **클래스**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **이미지 크기**: 32x32
- **모델 구조**: conv1(3x3, stride=1), maxpool 제거

#### CIFAR100
- **클래스 수**: 100개
- **이미지 크기**: 32x32
- **모델 구조**: conv1(3x3, stride=1), maxpool 제거

#### ImageNet
- **클래스 수**: 1000개
- **이미지 크기**: 224x224
- **모델 구조**: 표준 ResNet 구조 (conv1: 7x7, stride=2)

## 지원하는 XAI 기법

- **CAM**: Class Activation Mapping
- **GradCAM**: Gradient-weighted Class Activation Mapping

## 결과

생성된 히트맵은 `./xai_results/` 디렉토리에 저장됩니다.

파일명 형식: `{explainer}_{sample_index}_{predicted_class}.png`

예시:
- `cam_0_airplane.png`
- `grad_cam_1_car.png`

## 설정 파일 구조

`config.yaml` 파일은 다음과 같은 구조로 구성됩니다:

### 1. 데이터셋 설정 (`datasets`)
각 데이터셋별 설정 (경로, 배치 크기, 클래스 수, 클래스 이름 등)

### 2. 모델 설정 (`models`)
각 모델별 기본 설정 (이름, pretrained 여부, target layers 등)

### 3. 모델-데이터셋 조합 (`model_dataset_combinations`)
실제 사용 가능한 조합들 (체크포인트 경로, 클래스 수, 설명 등)

### 4. XAI 설정 (`xai`)
XAI 관련 설정 (샘플 수, 저장 디렉토리 등)

## 새로운 모델-데이터셋 조합 추가하기

### 방법 1: 자동 감지 (추천)
체크포인트 파일명을 표준 형식으로 저장하면 자동으로 감지됩니다.

**파일명 규칙**:
- `resnet_stl10.pt` → ResNetClassifier-STL10
- `vit_cifar10.pt` → ViT-CIFAR10
- `efficientnet_imagenet.pt` → EfficientNet-ImageNet

```bash
# 자동 감지 실행
python auto_detect_combinations.py --update
```

### 방법 2: 수동 등록
```bash
python register_checkpoint.py --model ViT --dataset CIFAR10 --checkpoint ./checkpoints/vit_cifar10.pt
```

### 방법 3: config 파일 직접 수정
1. **데이터셋 추가**: `datasets` 섹션에 새로운 데이터셋 설정 추가
2. **모델 구조 설정**: `models` 섹션의 `dataset_configs`에 새로운 데이터셋 구조 추가
3. **조합 추가**: `model_dataset_combinations` 섹션에 `{모델명}-{데이터셋명}` 형식으로 조합 추가

예시:
```yaml
# 모델 구조 설정
models:
  ResNetClassifier:
    dataset_configs:
      NewDataset:
        conv1:
          kernel_size: 5
          stride: 1
          padding: 2
          bias: false
        maxpool: "identity"
        description: "새로운 데이터셋에 최적화"

# 조합 추가
model_dataset_combinations:
  ResNetClassifier-NewDataset:
    model: "ResNetClassifier"
    dataset: "NewDataset"
    num_classes: 20
    checkpoint_path: "./checkpoints/resnet_newdataset.pt"
    description: "ResNet trained on NewDataset"
```

## 자동화 기능

### 1. 체크포인트 자동 감지
```bash
# 체크포인트 디렉토리 스캔하여 자동으로 조합 감지
python auto_detect_combinations.py

# 감지된 조합만 나열 (config 업데이트 안함)
python auto_detect_combinations.py --list_only

# config 파일 자동 업데이트
python auto_detect_combinations.py --update
```

### 2. 체크포인트 수동 등록
```bash
# 학습 완료 후 체크포인트 등록
python register_checkpoint.py --model ResNetClassifier --dataset STL10 --checkpoint ./checkpoints/resnet_stl10.pt
```

### 3. 사용 가능한 조합 확인
```bash
# 모든 정보 확인
python list_combinations.py

# 특정 정보만 확인
python list_combinations.py --type combinations
python list_combinations.py --type datasets
python list_combinations.py --type models
python list_combinations.py --type explainers
```

## 주의사항

1. 해당하는 체크포인트 파일이 존재해야 합니다.
2. 데이터셋이 지정된 경로에 있어야 합니다.
3. GPU가 사용 가능한 경우 자동으로 GPU를 사용합니다.
4. 모델-데이터셋 조합이 config 파일에 정의되어 있어야 합니다. 