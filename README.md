# XAI Explainer & Training Tool

이 도구는 PyTorch 기반의 모델 학습과 다양한 XAI(설명 가능한 AI) 기법을 통합적으로 지원하는 데스크탑 어플리케이션입니다. 학습된 체크포인트(YAML) 파일을 기반으로 데이터, 모델, XAI explainer를 자동으로 로드하여, 다양한 데이터셋과 모델에 대해 히트맵 분석 및 슈퍼픽셀 기반 분석까지 한 번에 수행할 수 있습니다.

---

## 주요 특징

- **모델 학습과 XAI 분석의 통합**: 학습된 체크포인트(YAML)만 있으면 데이터, 모델, explainer를 자동으로 로드하여 분석 가능
- **지원 모델**: ResNet18, ResNet34
- **지원 데이터셋**:
  - STL10 (10 클래스)
  - Bird CELF (450종 조류 분류)
  - 참새/솔새 구분 데이터셋 (sparrow-vs-bunting)
- **지원 XAI 기법**: CAM, GradCAM, RISE, SmoothGrad, IG
- **슈퍼픽셀(SLIC) 기반 분석 지원**: SLIC(superpixel)로 세그먼트별 히트맵 집계 및 시각화
- **PyQt5 기반 GUI**: 체크포인트/이미지/XAI 파라미터를 직관적으로 선택, 결과를 PNG로 저장 및 비교
- **스냅샷 저장/비교**: 여러 히트맵 결과를 저장하고, 병렬로 비교 가능

---

## 설치 및 환경설정

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenCV ximgproc 모듈 필요**
   - SLIC 기능을 사용하려면 OpenCV의 ximgproc 모듈이 필요합니다.
   - 설치 예시:
     ```bash
     pip install opencv-contrib-python
     ```

3. **PyQt5 설치**
   ```bash
   pip install PyQt5
   ```

---

## 실행 방법

1. **GUI 실행**
   ```bash
   cd heatmap_tool/app
   python main.py
   ```

2. **기본 사용 흐름**
   - 체크포인트(YAML) 로드 → 이미지 로드 → XAI 기법/파라미터 선택 → 히트맵 생성
   - 히트맵 결과를 PNG로 저장하거나, 여러 스냅샷을 비교할 수 있습니다.
   - "스냅샷 저장" 버튼으로 결과를 저장
   - "스냅샷 미리보기" 버튼으로 여러 PNG를 한 번에 비교
   - "히트맵 폴더 열기"로 결과 폴더를 바로 탐색

3. **명령행 XAI 분석 (스크립트 방식)**
   ```bash
   python use_xai_explainer.py -d <데이터셋> -m <모델> -x <XAI기법>
   ```
   - 예시:
     ```bash
     python use_xai_explainer.py -d stl10 -m resnetclassifier -x cam
     python use_xai_explainer.py -d stl10 -m resnetclassifier -x grad_cam
     python use_xai_explainer.py -d bird_celf -m resnet34 -x ig
     ```

---

## 지원 모델/데이터셋/XAI 기법

- **모델**: ResNet18, ResNet34
- **데이터셋**:
  - STL10 (10 클래스, 96x96)
  - Bird CELF (450 클래스, 조류 이미지)
  - 참새/솔새 구분 데이터셋 (sparrow-vs-bunting)
- **XAI 기법**:
  - CAM
  - GradCAM
  - RISE
  - SmoothGrad
  - IG (Integrated Gradients)
- **슈퍼픽셀(SLIC) 기반 분석**: 각 XAI 기법별로 SLIC 옵션을 켜서 세그먼트별 히트맵 집계/시각화 가능

---

## 체크포인트 및 설정 파일

- 학습된 모델의 체크포인트와 YAML 설정 파일을 기반으로 데이터, 모델, explainer를 자동 로드합니다.
- YAML 파일에는 데이터셋, 모델 구조, 클래스 정보, XAI 파라미터 등이 포함되어야 합니다.

---

## 주요 디렉토리 구조

```
heatmap_tool/
  app/
    main.py                # 메인 GUI 어플리케이션
    use_xai_explainer.py   # 명령행 XAI 분석 스크립트
    utils/                 # 유틸리티 함수/다이얼로그/시각화 등
    explainers/            # XAI 기법별 구현 (CAM, IG, SmoothGrad 등)
    models/                # 모델 구조 정의
    configs/               # 체크포인트/데이터셋 설정
    snapshots/             # 저장된 히트맵/스냅샷 이미지
    xai_gui.ui             # PyQt5 UI 파일
    README.md              # (앱 전용 설명)
  requirements.txt         # 전체 의존성
  README.md                # (본 파일)
```

---

## 참고/문의

- 본 프로젝트는 PyTorch, PyQt5, OpenCV, numpy 등 오픈소스 라이브러리를 적극 활용합니다.
- 문의/이슈는 GitHub 또는 프로젝트 관리자에게 연락해 주세요. 

