# XAI Heatmap Tool

**XAI Heatmap Tool**은 PyTorch 기반 모델의 다양한 XAI(설명가능 인공지능) 기법(CAM, GradCAM, IG, SmoothGrad 등)을
GUI 환경에서 손쉽게 시각화하고, 히트맵 결과를 저장/비교할 수 있는 PyQt5 기반 데스크탑 어플리케이션입니다.

---

## 주요 기능

- 다양한 XAI 기법(CAM, GradCAM, IG, SmoothGrad 등) 지원
- PyQt5 기반 GUI: 체크포인트/이미지/XAI 파라미터를 직관적으로 선택
- 히트맵 결과를 PNG로 저장, 여러 스냅샷을 병렬로 비교
- SLIC(Superpixel) 기반 세그먼트별 히트맵 집계 및 시각화 지원
- 각 Explainer별 파라미터, SLIC 옵션 등 동적 UI 제공

---

## 설치 및 환경설정

1. **의존성 설치**
   ```bash
   pip install -r ../requirements.txt
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

3. **스냅샷/히트맵 비교**
   - "스냅샷 저장" 버튼으로 결과를 저장
   - "스냅샷 미리보기" 버튼으로 여러 PNG를 한 번에 비교
   - "히트맵 폴더 열기"로 결과 폴더를 바로 탐색

---

## 주요 디렉토리 구조

```
heatmap_tool/
  app/
    main.py                # 메인 GUI 어플리케이션
    utils/                 # 유틸리티 함수/다이얼로그/시각화 등
    explainers/            # XAI 기법별 구현 (CAM, IG, SmoothGrad 등)
    models/                # 모델 구조 정의
    configs/               # 체크포인트/데이터셋 설정
    snapshots/             # 저장된 히트맵/스냅샷 이미지
    xai_gui.ui             # PyQt5 UI 파일
    README.md              # (본 파일)
  requirements.txt         # 전체 의존성
```

---

## 참고/문의

- 본 프로젝트는 PyTorch, PyQt5, OpenCV, numpy 등 오픈소스 라이브러리를 적극 활용합니다.
- 문의/이슈는 GitHub 또는 프로젝트 관리자에게 연락해 주세요. 
