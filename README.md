# 🔍 XAI Attention Map Visualization Tool

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Explainable AI (XAI) 기반 딥러닝 모델 해석 도구**

*딥러닝 모델의 의사결정 과정을 시각적으로 이해할 수 있는 확장 가능한 히트맵 생성 도구*

</div>

---

## 📋 목차

- [✨ 주요 기능](#-주요-기능)
- [🚀 빠른 시작](#-빠른-시작)
- [📦 설치](#-설치)
- [🎯 사용법](#-사용법)
- [🔧 현재 지원 모델 & 데이터셋](#-현재-지원-모델--데이터셋)
- [🎨 XAI 기법](#-xai-기법)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [⚙️ 설정](#️-설정)
- [📊 결과 예시](#-결과-예시)
- [🛠️ 고급 사용법](#️-고급-사용법)
- [🔮 확장성 & 유연성](#-확장성--유연성)
- [🤝 기여하기](#-기여하기)
- [📄 라이선스](#-라이선스)

---

## ✨ 주요 기능

- 🎯 **현재 지원 XAI 기법**: CAM, GradCAM
- 🖼️ **현재 지원 데이터셋**: STL10
- 🧠 **현재 지원 모델**: ResNet18
- ⚡ **확장 가능한 아키텍처**: 새로운 모델/데이터셋/XAI 기법 쉽게 추가
- 🔧 **유연한 설정 시스템**: YAML 기반 설정으로 쉬운 커스터마이징
- 📊 **고품질 시각화**: 전문적인 히트맵 생성
- 🎨 **모듈화된 구조**: 각 컴포넌트 독립적 개발 및 테스트 가능
- 🚀 **자동화된 설정**: 체크포인트 자동 감지 및 설정

---

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/Mansook/attention_map.git
cd attention_map
```

### 2. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv xai_env
source xai_env/bin/activate  # Linux/Mac
# 또는
xai_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. 첫 번째 히트맵 생성
```bash
cd heatmap_tool
python use_xai_explainer.py -d stl10 -m resnetclassifier -x cam
```

---

## 📦 설치

### 시스템 요구사항
- **Python**: 3.8 이상
- **CUDA**: 11.0 이상 (GPU 사용 시, 선택사항)
- **RAM**: 최소 8GB (STL10 데이터셋 사용 시 16GB 권장)

### 의존성 설치

#### 기본 설치
```bash
pip install -r requirements.txt
```

#### GPU 지원 설치 (CUDA)
```bash
# CUDA 버전에 맞는 PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 🎯 사용법

### 기본 명령어
```bash
python use_xai_explainer.py -d <데이터셋> -m <모델> -x <XAI기법>
```

### 매개변수 설명
| 매개변수 | 설명 | 현재 지원 값 |
|---------|------|-------------|
| `-d, --dataset` | 사용할 데이터셋 | `stl10` |
| `-m, --model` | 사용할 모델 | `resnetclassifier` |
| `-x, --explainer` | 사용할 XAI 기법 | `cam`, `grad_cam` |
| `--config` | 설정 파일 경로 | `configs/custom.yaml` |
| `--samples` | 생성할 샘플 수 | `10` (기본값: 5) |
| `--output` | 결과 저장 경로 | `./my_results/` |

### 사용 예시

#### STL10 + ResNet18 + CAM
```bash
python use_xai_explainer.py -d stl10 -m resnetclassifier -x cam
```

#### STL10 + ResNet18 + GradCAM
```bash
python use_xai_explainer.py -d stl10 -m resnetclassifier -x grad_cam --samples 10
```

#### 커스텀 설정 파일 사용
```bash
python use_xai_explainer.py -d stl10 -m resnetclassifier -x cam --config configs/my_config.yaml
```

---

## 🔧 현재 지원 모델 & 데이터셋

### 🤖 현재 지원 모델

| 모델 | 아키텍처 | 지원 데이터셋 | 특징 |
|------|----------|--------------|------|
| **ResNet18** | CNN | STL10 | 경량화된 ResNet 구조 (conv1: 3×3, stride=1, maxpool 제거) |

### 📊 현재 지원 데이터셋

#### STL10
- **클래스**: 10개 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **이미지 크기**: 96×96
- **특징**: STL-10 데이터셋의 이진 형식 지원
- **모델 구조**: conv1(3×3, stride=1), maxpool 제거

---

## 🎨 XAI 기법

### CAM (Class Activation Mapping)
- **원리**: 마지막 컨볼루션 레이어의 특성 맵을 가중 평균
- **장점**: 구현이 간단하고 직관적
- **단점**: 모델 구조에 제약 (GAP 레이어 필요)
- **적용**: ResNet18

### GradCAM (Gradient-weighted Class Activation Mapping)
- **원리**: 그래디언트 정보를 활용한 가중 평균
- **장점**: 다양한 모델 구조에 적용 가능
- **단점**: CAM보다 계산 복잡도 높음
- **적용**: ResNet18

### 기법 비교
| 기법 | 모델 호환성 | 계산 속도 | 정확도 | 구현 복잡도 |
|------|-------------|-----------|--------|-------------|
| CAM | 제한적 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| GradCAM | 높음 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📁 프로젝트 구조

```
XAI-study/
├── 📁 heatmap_tool/           # 메인 도구 디렉토리
│   ├── 📁 configs/            # 설정 파일들
│   │   ├── 📁 checkpoints/    # 체크포인트 설정
│   │   └── 📁 dataset/        # 데이터셋 설정
│   ├── 📁 data/               # 데이터 저장소
│   ├── 📁 dataset/            # 데이터셋 처리
│   ├── 📁 explainers/         # XAI 기법 구현
│   │   ├── cam.py            # CAM 구현
│   │   └── grad_cam.py       # GradCAM 구현
│   ├── 📁 models/             # 모델 정의
│   │   └── resnet_classifier.py
│   ├── 📁 predict/            # 예측 도구
│   ├── 📁 train/              # 학습 스크립트
│   ├── 📁 utils/              # 유틸리티 함수
│   ├── 📁 xai_results/        # 결과 저장소
│   ├── explainer_config.yaml  # XAI 설정
│   ├── use_xai_explainer.py   # 메인 실행 파일
│   └── requirements.txt       # 의존성
├── 📁 data/                   # 원본 데이터
├── requirements.txt           # 루트 의존성
└── README.md                  # 프로젝트 문서
```

---

## ⚙️ 설정

### 설정 파일 구조

#### `explainer_config.yaml`
```yaml
explainer_dict:
  cam:
    class: CAM
    description: "Class Activation Mapping"
    model:
      ResNet18: "layer4"
    default_target_layer: "layer4"
  
  grad_cam:
    class: GradCAM
    description: "Gradient-weighted Class Activation Mapping"
    model:
      ResNet18:
        target_layer: "layer4"
    default_target_layer: "layer4"
```

#### `configs/dataset/stl10.yaml`
```yaml
datasets:
  stl10:
    path: "./data/stl10_binary/"
    batch_size: 32
    num_classes: 10
    class_names:
      - airplane
      - bird
      - car
      # ... 기타 클래스들
```

---

## 📊 결과 예시

### 생성되는 파일 구조
```
xai_results/
├── cam_0_airplane.png        # CAM 결과 - 샘플 0, 예측: 비행기
├── cam_1_car.png            # CAM 결과 - 샘플 1, 예측: 자동차
├── grad_cam_0_airplane.png   # GradCAM 결과 - 샘플 0, 예측: 비행기
└── grad_cam_1_car.png       # GradCAM 결과 - 샘플 1, 예측: 자동차
```

### 결과 해석
- **밝은 영역**: 모델이 해당 클래스를 예측할 때 중요하게 고려한 부분
- **어두운 영역**: 모델이 덜 중요하게 고려한 부분
- **색상 강도**: 중요도 수준 (빨강 > 노랑 > 파랑)

---

## 🛠️ 고급 사용법

### 배치 처리
```bash
# 여러 XAI 기법에 대해 일괄 처리
for explainer in cam grad_cam; do
  python use_xai_explainer.py -d stl10 -m resnetclassifier -x $explainer --samples 10
done
```

### 성능 최적화
```bash
# GPU 메모리 최적화
export CUDA_VISIBLE_DEVICES=0
python use_xai_explainer.py --batch_size 1

# 멀티프로세싱
python use_xai_explainer.py --num_workers 4
```

---

## 🔮 확장성 & 유연성

### 🚀 확장성 (Scalability)

#### 1. **모듈화된 아키텍처**
- 각 컴포넌트가 독립적으로 개발/테스트 가능
- 새로운 기능 추가 시 기존 코드 영향 최소화
- 플러그인 방식의 XAI 기법 추가

#### 2. **설정 기반 확장**
```yaml
# 새로운 XAI 기법 추가 예시
explainer_dict:
  my_custom_xai:
    class: MyCustomXAI
    description: "Custom XAI Method"
    model:
      ResNet18: "layer3"
    default_target_layer: "layer3"
```

#### 3. **자동화된 설정 감지**
- 체크포인트 파일명 규칙으로 자동 모델-데이터셋 조합 감지
- 새로운 모델/데이터셋 추가 시 설정 파일 자동 업데이트

#### 4. **대용량 데이터 처리**
- 배치 처리 지원으로 대용량 데이터셋 처리 가능
- 메모리 효율적인 데이터 로딩
- GPU 메모리 최적화 옵션

### 🔧 유연성 (Flexibility)

#### 1. **다양한 설정 옵션**
- 명령줄 인자, 설정 파일, 환경 변수 지원
- 런타임에 설정 변경 가능
- 기본값과 커스텀 설정의 유연한 조합

#### 2. **커스터마이징 가능한 XAI 기법**
```python
# 새로운 XAI 기법 쉽게 추가
class MyCustomXAI:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
    
    def generate_heatmap(self, input_image, target_class):
        # 커스텀 히트맵 생성 로직
        pass
```

#### 3. **유연한 모델 지원**
- 다양한 모델 아키텍처 지원 가능
- 타겟 레이어 자유롭게 설정
- 모델별 최적화된 설정

#### 4. **결과 커스터마이징**
- 다양한 출력 형식 지원 (PNG, JPG, PDF 등)
- 커스텀 시각화 스타일 적용 가능
- 결과 저장 경로 및 파일명 패턴 설정

### 🔄 확장 가이드

#### 새로운 데이터셋 추가
1. **데이터셋 설정 파일 생성**
```yaml
# configs/dataset/new_dataset.yaml
datasets:
  new_dataset:
    path: "./data/new_dataset/"
    batch_size: 32
    num_classes: 20
    class_names: [...]
```

2. **모델 설정 업데이트**
```yaml
# configs/checkpoints/resnet18/new_dataset.yaml
model_dataset_combinations:
  ResNet18-NewDataset:
    model: "ResNet18"
    dataset: "NewDataset"
    num_classes: 20
    checkpoint_path: "./checkpoints/resnet18_newdataset.pt"
```

#### 새로운 XAI 기법 추가
1. **기법 구현**
```python
# explainers/my_xai.py
class MyXAI:
    def __init__(self, model, target_layer):
        # 초기화 로직
        pass
    
    def generate_heatmap(self, input_image, target_class):
        # 히트맵 생성 로직
        pass
```

2. **설정 파일 업데이트**
```yaml
# explainer_config.yaml
explainer_dict:
  my_xai:
    class: MyXAI
    description: "My Custom XAI Method"
    model:
      ResNet18: "layer4"
    default_target_layer: "layer4"
```

#### 새로운 모델 추가
1. **모델 정의**
```python
# models/my_model.py
class MyModel(nn.Module):
    def __init__(self, num_classes):
        # 모델 구조 정의
        pass
```

2. **설정 파일 업데이트**
```yaml
# configs/checkpoints/my_model/stl10.yaml
model_dataset_combinations:
  MyModel-STL10:
    model: "MyModel"
    dataset: "STL10"
    num_classes: 10
    checkpoint_path: "./checkpoints/my_model_stl10.pt"
```

---

## 🤝 기여하기

### 개발 환경 설정
```bash
# 저장소 포크 후 클론
git clone https://github.com/YOUR_USERNAME/attention_map.git
cd attention_map

# 개발 브랜치 생성
git checkout -b feature/new-xai-method

# 개발 후 PR 생성
git add .
git commit -m "feat: Add new XAI method"
git push origin feature/new-xai-method
```

### 기여 가이드라인
1. **코드 스타일**: PEP 8 준수
2. **문서화**: 모든 함수에 docstring 작성
3. **테스트**: 새로운 기능에 대한 테스트 코드 작성
4. **커밋 메시지**: Conventional Commits 형식 사용
5. **확장성 고려**: 새로운 기능이 기존 구조와 호환되도록 설계

### 이슈 리포트
버그 발견 시 다음 정보를 포함하여 이슈를 생성해주세요:
- 운영체제 및 Python 버전
- 에러 메시지 전체
- 재현 가능한 최소 예제
- 예상 동작과 실제 동작

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 감사의 말

- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [STL-10 Dataset](https://cs.stanford.edu/~acoates/stl10/) - 데이터셋 제공
- [GradCAM 논문](https://arxiv.org/abs/1610.02391) - XAI 기법 영감

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!**

Made with ❤️ by [Mansook](https://github.com/Mansook)

</div> 