# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, 
    QLabel, QListWidget, QMessageBox, QProgressBar, QWidget, QFileDialog
)
from utils.xaiworker import XAIWorker
# 상위 디렉토리 모듈들 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.datasetLoader import get_dataloaders
from explainers import CAM, IG, RISE, GradCAM, SmoothGrad, SHAP, LMAP
from utils.explainer_add_dialog import ExplainerAddDialog
from utils.get_class_name_by_index import get_class_name_by_index
from utils.set_korean import setup_korean_font


class LogitDataWorker(QThread):
    """로짓 데이터 계산을 위한 워커 스레드"""
    progress_update = pyqtSignal(int)
    progress_text_update = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model, target_loader, basis_loader, basis_class, target_class):
        super().__init__()
        self.model = model
        self.target_loader = target_loader
        self.basis_loader = basis_loader
        self.basis_class = basis_class
        self.target_class = target_class
        
    def run(self):
        try:
            device = next(self.model.parameters()).device
            self.model.eval()
            
            basis_logits = []
            target_logits = []
            image_paths = []
            
            # target과 basis 데이터셋 모두 사용
            target_dataset = self.target_loader.dataset
            basis_dataset = self.basis_loader.dataset
            
            # 모든 이미지 처리 (target + basis)
            all_images = []
            all_paths = []
            
            # target 이미지들 추가
            for i in range(len(target_dataset)):
                image, _ = target_dataset[i]
                all_images.append(image)
                if hasattr(target_dataset, 'image_files'):
                    all_paths.append(target_dataset.image_files[i])
                else:
                    all_paths.append(f"target_image_{i}")
            
            # basis 이미지들 추가
            for i in range(len(basis_dataset)):
                image, _ = basis_dataset[i]
                all_images.append(image)
                if hasattr(basis_dataset, 'image_files'):
                    all_paths.append(basis_dataset.image_files[i])
                else:
                    all_paths.append(f"basis_image_{i}")
            
            with torch.no_grad():
                for i in range(len(all_images)):
                    # 이미지 처리
                    image = all_images[i].unsqueeze(0).to(device)
                    output = self.model(image)
                    logit = output.squeeze().cpu().numpy()
                    
                    # dog 로짓과 target 클래스 로짓 추출
                    dog_logit = logit[5]  # dog 클래스 (5)
                    target_logit = logit[self.target_class]  # 선택된 target 클래스
                    
                    basis_logits.append(dog_logit)
                    target_logits.append(target_logit)
                    image_paths.append(all_paths[i])
                    
                    # 진행률 업데이트
                    progress = int((i + 1) / len(all_images) * 100)
                    self.progress_update.emit(progress)
                    self.progress_text_update.emit(f"로짓 계산 중... ({i+1}/{len(all_images)})")
            
            result = {
                'basis_logits': np.array(basis_logits),
                'target_logits': np.array(target_logits),
                'image_paths': image_paths
            }
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class LogitPlotDialog(QDialog):
    """2D 로짓 플롯 다이얼로그"""
    
    def __init__(self, model, config, explainer_config, parent=None):
        super().__init__(parent)
        self.model = model
        self.config = config
        self.explainer_config = explainer_config
        self.explainer_dict = explainer_config.get('explainer_dict', {})
        
        self.logit_data = None
        self.selected_point = None
        self.explainers = {}
        self.current_explainee = None
        self.local_data_path = None
        
        self._init_ui()
        self._load_data()
        self._setup_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        setup_korean_font()
        self.setWindowTitle("2D 로짓 플롯")
        self.resize(3000, 1500)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 왼쪽 패널 (컨트롤)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 가운데 패널 (플롯)
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 2)
        
        # 오른쪽 패널 (이미지 뷰어)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        self.setLayout(main_layout)
        
    def _create_left_panel(self):
        """왼쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 데이터 경로 선택
        layout.addWidget(QLabel("로컬 데이터 경로:"))
        self.data_path_label = QLabel("데이터 경로를 선택하세요")
        layout.addWidget(self.data_path_label)
        
        self.select_data_btn = QPushButton("데이터 경로 선택")
        self.select_data_btn.clicked.connect(self._select_data_path)
        layout.addWidget(self.select_data_btn)
        
        layout.addSpacing(10)
        
        # Basis 클래스 (고정 - dog)
        layout.addWidget(QLabel("Basis 클래스 (X축):"))
        self.basis_label = QLabel("dog (고정)")
        layout.addWidget(self.basis_label)
        
        # Target 클래스 선택
        layout.addWidget(QLabel("Target 클래스 (Y축):"))
        self.target_combo = QComboBox()
        layout.addWidget(self.target_combo)
        
        # 클래스 변경 버튼
        self.update_plot_btn = QPushButton("플롯 업데이트")
        self.update_plot_btn.clicked.connect(self._update_plot)
        layout.addWidget(self.update_plot_btn)
        
        layout.addSpacing(20)
        
        # Explainer 선택
        layout.addWidget(QLabel("Explainer 선택:"))
        self.explainer_combo = QComboBox()
        layout.addWidget(self.explainer_combo)
        
        # Explainer 추가 버튼
        self.add_explainer_btn = QPushButton("Explainer 추가")
        self.add_explainer_btn.clicked.connect(self._add_explainer)
        layout.addWidget(self.add_explainer_btn)
        
        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def _create_center_panel(self):
        """가운데 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 플롯 캔버스 - 크기 확대
        self.fig = Figure(figsize=(20, 12))  # 더 크게
        self.canvas = FigureCanvas(self.fig)
        
        # NavigationToolbar 추가 (줌인/아웃, 팬 기능) - 크기 조정
        try:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            self.toolbar = NavigationToolbar(self.canvas, panel)
            
            # 툴바 크기 조정
            self.toolbar.setMaximumHeight(25)
            self.toolbar.setMinimumHeight(20)
            
            # 툴바 스타일 설정
            self.toolbar.setStyleSheet("""
                QToolBar {
                    spacing: 1px;
                    padding: 1px;
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                }
                QToolButton {
                    max-width: 20px;
                    max-height: 20px;
                    min-width: 18px;
                    min-height: 18px;
                    border: 1px solid #ccc;
                    border-radius: 2px;
                    margin: 1px;
                    background-color: #ffffff;
                }
                QToolButton:hover {
                    background-color: #e0e0e0;
                }
                QToolButton:pressed {
                    background-color: #d0d0d0;
                }
            """)
            
            layout.addWidget(self.toolbar)
        except Exception as e:
            print(f"NavigationToolbar 로드 실패: {e}")
        
        layout.addWidget(self.canvas)
        
        panel.setLayout(layout)
        return panel
        
    def _create_right_panel(self):
        """오른쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("선택된 포인트 정보:"))
        self.point_info_label = QLabel("포인트를 클릭하세요")
        layout.addWidget(self.point_info_label)
        
        # 이미지 표시 영역 - 크기 확대
        self.image_fig = Figure(figsize=(12, 10))
        self.image_canvas = FigureCanvas(self.image_fig)
        layout.addWidget(self.image_canvas)
        
        # Explainer 결과 표시 영역 (2개 나란히) - 크기 확대
        self.explainer_fig = Figure(figsize=(16, 10))
        self.explainer_canvas = FigureCanvas(self.explainer_fig)
        layout.addWidget(self.explainer_canvas)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def _select_data_path(self):
        """데이터 경로 선택"""
        default_path = "C:/Users/orgin/Desktop/ForTest/dog"
        data_path = QFileDialog.getExistingDirectory(
            self, "로컬 데이터 디렉토리 선택", default_path
        )
        
        if data_path:
            self.local_data_path = data_path
            self.data_path_label.setText(f"선택됨: {os.path.basename(data_path)}")
            self._load_local_data()
            
    def _load_data(self):
        """데이터 로드 (기본값으로 로컬 경로 설정)"""
        # 기본 로컬 경로 설정만 하고 데이터는 로드하지 않음
        base_path = "C:/Users/orgin/Desktop/ForTest/dog"  # dog 디렉토리로 변경
        if os.path.exists(base_path):
            self.base_data_path = base_path
            self.data_path_label.setText(f"기본 경로: {os.path.basename(base_path)}")
            # 초기에는 데이터 로드하지 않음
        else:
            self.data_path_label.setText("데이터 경로를 선택하세요")
            
        # 콤보박스 설정 (데이터 로드와 관계없이)
        self._setup_class_combos()
            
    def _setup_class_combos(self):
        """클래스 콤보박스 설정"""
        # STL10 클래스명 사용
        stl10_classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        
        # 콤보박스에 클래스 추가 (dog 제외)
        self.target_combo.clear()
        
        for i, class_name in enumerate(stl10_classes):
            if class_name != "dog":  # dog는 basis로 고정
                self.target_combo.addItem(class_name, i)
                
        # target 클래스 변경 시 데이터 다시 로드
        self.target_combo.currentTextChanged.connect(self._on_target_class_changed)
            
    def _on_target_class_changed(self, class_name):
        """target 클래스 변경 시 호출"""
        if hasattr(self, 'base_data_path') and class_name:
            print(f"선택된 클래스: {class_name}")
            print(f"base_data_path: {self.base_data_path}")
            self._load_local_data_with_class(class_name)
            
    def _load_local_data_with_class(self, class_name):
        """특정 클래스의 데이터 로드"""
        if not hasattr(self, 'base_data_path') or not class_name:
            return
            
        # target: dog 디렉토리 안의 해당 클래스 디렉토리
        target_path = os.path.join(self.base_data_path, class_name)
        target_path = target_path.replace('\\', '/')
        print(f"target 시도하는 경로: {target_path}")
        
        # basis: dog 디렉토리 안의 dog 디렉토리
        basis_path = os.path.join(self.base_data_path, "dog")
        basis_path = basis_path.replace('\\', '/')
        print(f"basis 시도하는 경로: {basis_path}")
        
        if os.path.exists(target_path) and os.path.exists(basis_path):
            print(f"target 디렉토리 내용: {os.listdir(target_path)}")
            print(f"basis 디렉토리 내용: {os.listdir(basis_path)}")
            
            self.target_data_path = target_path
            self.basis_data_path = basis_path
            self.data_path_label.setText(f"선택됨: {class_name}")
            
            # target과 basis 데이터 모두 로드
            self._load_target_and_basis_data()
        else:
            print(f"경로가 존재하지 않음: target={target_path}, basis={basis_path}")
            print(f"base_data_path 내용: {os.listdir(self.base_data_path)}")
            QMessageBox.warning(self, "경고", f"디렉토리를 찾을 수 없습니다")
            
    def _load_target_and_basis_data(self):
        """target과 basis 데이터 로드"""
        try:
            from torch.utils.data import Dataset
            from PIL import Image
            import glob
            
            class DirectImageDataset(Dataset):
                def __init__(self, root_dir, transform=None):
                    self.root_dir = root_dir
                    self.transform = transform
                    self.image_files = []
                    
                    # PNG 파일만 검색
                    self.image_files = sorted(glob.glob(os.path.join(root_dir, "*.png")))
                    
                    print(f"찾은 이미지 파일 수: {len(self.image_files)}")
                    
                def __len__(self):
                    return len(self.image_files)
                    
                def __getitem__(self, idx):
                    img_path = self.image_files[idx]
                    image = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    return image, 0  # 레이블은 0으로 고정
            
            # transform 설정
            transform_config = self.config['dataset_config']['transform']
            img_size = tuple(transform_config['img_size'])
            normalize = transform_config['normalize']
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize['mean'], std=normalize['std'])
            ])
            
            # target 데이터셋 생성
            target_dataset = DirectImageDataset(self.target_data_path, transform=transform)
            from torch.utils.data import DataLoader
            self.target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=0)
            
            # basis 데이터셋 생성
            basis_dataset = DirectImageDataset(self.basis_data_path, transform=transform)
            self.basis_loader = DataLoader(basis_dataset, batch_size=1, shuffle=False, num_workers=0)
            
            print(f"target 데이터 로드 완료: {len(target_dataset)}개 이미지")
            print(f"basis 데이터 로드 완료: {len(basis_dataset)}개 이미지")
            
            QMessageBox.information(self, "성공", f"데이터 로드 완료: target {len(target_dataset)}개, basis {len(basis_dataset)}개")
            
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "오류", f"데이터 로드 실패: {str(e)}")
            
    def _setup_ui(self):
        """UI 설정"""
        # 플롯 클릭 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        
    def _update_plot(self):
        """플롯 업데이트"""
        if not hasattr(self, 'target_loader') or not hasattr(self, 'basis_loader') or self.target_loader is None or self.basis_loader is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 로드해주세요.")
            return
            
        # 모델을 eval 모드로 설정
        self.model.eval()
        
        # dog는 5번 인덱스
        basis_class = 5  # dog
        target_class = self.target_combo.currentData()
        
        if basis_class == target_class:
            QMessageBox.warning(self, "경고", "Basis와 Target 클래스가 같습니다.")
            return
            
        # 워커 스레드로 로짓 데이터 계산
        self.worker = LogitDataWorker(self.model, self.target_loader, self.basis_loader, basis_class, target_class)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.progress_text_update.connect(self.progress_label.setText)
        self.worker.finished.connect(self._on_logit_data_ready)
        self.worker.error.connect(self._on_worker_error)
        
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.update_plot_btn.setEnabled(False)
        self.worker.start()
        
    def _on_logit_data_ready(self, data):
        """로짓 데이터 준비 완료"""
        self.logit_data = data
        self._draw_plot()
        
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.update_plot_btn.setEnabled(True)
        
    def _on_worker_error(self, error_msg):
        """워커 오류 처리"""
        QMessageBox.critical(self, "오류", f"로짓 계산 실패: {error_msg}")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.update_plot_btn.setEnabled(True)
        
    def _draw_plot(self):
        """플롯 그리기"""
        if self.logit_data is None:
            return
            
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        basis_logits = self.logit_data['basis_logits']
        target_logits = self.logit_data['target_logits']
        
        # 각 포인트의 예측 클래스 결정
        all_logits = np.column_stack([basis_logits, target_logits])
        predicted_classes = np.argmax(all_logits, axis=1)
        
        # 색상 매핑
        colors = ['red' if pred == 0 else 'blue' for pred in predicted_classes]
        
        # 스캐터 플롯 - 점 크기 줄임
        scatter = ax.scatter(basis_logits, target_logits, 
                           alpha=0.7, s=30, c=colors,  # 크기를 30으로 줄임
                           edgecolors='black', linewidth=0.3)
        
        # 선택된 포인트가 있으면 노란색으로 강조 표시
        if self.selected_point is not None:
            ax.scatter(basis_logits[self.selected_point], target_logits[self.selected_point],
                      s=80, c='yellow', edgecolors='black', linewidth=1.5, 
                      alpha=1.0, zorder=5, marker='*')
        
        # 축 레이블
        target_name = self.target_combo.currentText()
        ax.set_xlabel("dog 로짓", fontsize=14)
        ax.set_ylabel(f"{target_name} 로짓", fontsize=14)
        ax.set_title("2D 로짓 플롯 (dog vs target)", fontsize=16, fontweight='bold')
        
        # 격자 추가
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 대각선 추가 (y=x)
        min_val = min(basis_logits.min(), target_logits.min())
        max_val = max(basis_logits.max(), target_logits.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.7, 
                linewidth=2, label='y=x')
        
        # 범례 추가
        from matplotlib.lines import Line2D
        # 축 범위 설정
        ax.set_xlim(min_val - 0.1, max_val + 0.1)
        ax.set_ylim(min_val - 0.1, max_val + 0.1)
        
        # 통계 정보 표시
        dog_count = sum(1 for pred in predicted_classes if pred == 0)
        target_count = len(predicted_classes) - dog_count
        
        info_text = f'총 {len(basis_logits)}개 포인트\n'
        info_text += f'dog 예측: {dog_count}개\n'
        info_text += f'{target_name} 예측: {target_count}개'
        
        ax.text(0.02, 0.98, info_text, 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 플롯을 tight하게 조정
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _on_plot_click(self, event):
        """플롯 클릭 이벤트"""
        if not hasattr(self, 'fig') or not self.fig.axes or event.inaxes != self.fig.axes[0] or self.logit_data is None:
            return
            
        # 클릭한 위치에서 가장 가까운 포인트 찾기
        basis_logits = self.logit_data['basis_logits']
        target_logits = self.logit_data['target_logits']
        
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
            
        # 가장 가까운 포인트 찾기
        distances = np.sqrt((basis_logits - click_x)**2 + (target_logits - click_y)**2)
        closest_idx = np.argmin(distances)
        
        # 같은 포인트를 다시 클릭한 경우 무시
        if self.selected_point == closest_idx:
            return
            
        self.selected_point = closest_idx
        self._show_point_info(closest_idx)
        
        # 플롯 다시 그리기 (선택된 포인트 강조)
        self._draw_plot()
        
    def _show_point_info(self, point_idx):
        """포인트 정보 표시"""
        if self.logit_data is None:
            return
            
        basis_logits = self.logit_data['basis_logits']
        target_logits = self.logit_data['target_logits']
        
        # 해당 포인트의 모든 클래스 로짓 계산
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # 모든 이미지를 하나의 리스트로 관리
        all_images = []
        
        # target 이미지들 추가
        target_dataset = self.target_loader.dataset
        for i in range(len(target_dataset)):
            image, _ = target_dataset[i]
            all_images.append(image)
        
        # basis 이미지들 추가
        basis_dataset = self.basis_loader.dataset
        for i in range(len(basis_dataset)):
            image, _ = basis_dataset[i]
            all_images.append(image)
        
        # point_idx가 범위를 벗어나지 않도록 체크
        if point_idx >= len(all_images):
            print(f"point_idx {point_idx}가 범위를 벗어남. 총 이미지 수: {len(all_images)}")
            return
            
        # 해당 인덱스의 이미지로 모든 클래스 로짓 계산
        image = all_images[point_idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.model(image)
            all_logits = output.squeeze().cpu().numpy()
        
        # STL10 클래스명
        stl10_classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        
        # 예측 클래스 결정
        predicted_class = np.argmax(all_logits)
        predicted_name = stl10_classes[predicted_class]
        
        # 포인트 정보 업데이트 - 컴팩트하게 표시
        target_name = self.target_combo.currentText()
        info_text = f"포인트 {point_idx} | 예측: {predicted_name}\n"
        info_text += f"dog: {basis_logits[point_idx]:.2f} | {target_name}: {target_logits[point_idx]:.2f}\n\n"
        
        # 상위 5개 클래스만 표시
        logit_class_pairs = [(all_logits[i], stl10_classes[i]) for i in range(len(all_logits))]
        logit_class_pairs.sort(key=lambda x: x[0], reverse=True)
        
        info_text += "Top 5 로짓:\n"
        for i, (logit, class_name) in enumerate(logit_class_pairs[:5]):
            if i == 0:  # 1위
                info_text += f"★ {class_name}: {logit:.2f}\n"
            elif i == 1:  # 2위
                info_text += f"★ {class_name}: {logit:.2f}\n"
            elif i == 2:  # 3위
                info_text += f"★ {class_name}: {logit:.2f}\n"
            else:
                info_text += f"   {class_name}: {logit:.2f}\n"
        
        self.point_info_label.setText(info_text)
        
        # 이미지 표시
        self._show_image(point_idx)
        
        # Explainer 결과 표시 (자동으로)
        if self.explainer_combo.count() > 0:
            self._show_explainer_result(point_idx)
            
    def _show_image(self, point_idx):
        """이미지 표시"""
        try:
            # 모든 이미지를 하나의 리스트로 관리
            all_images = []
            all_paths = []
            
            # target 이미지들 추가
            target_dataset = self.target_loader.dataset
            for i in range(len(target_dataset)):
                image, _ = target_dataset[i]
                all_images.append(image)
                if hasattr(target_dataset, 'image_files'):
                    all_paths.append(target_dataset.image_files[i])
                else:
                    all_paths.append(f"target_image_{i}")
            
            # basis 이미지들 추가
            basis_dataset = self.basis_loader.dataset
            for i in range(len(basis_dataset)):
                image, _ = basis_dataset[i]
                all_images.append(image)
                if hasattr(basis_dataset, 'image_files'):
                    all_paths.append(basis_dataset.image_files[i])
                else:
                    all_paths.append(f"basis_image_{i}")
            
            # point_idx가 범위를 벗어나지 않도록 체크
            if point_idx >= len(all_images):
                print(f"point_idx {point_idx}가 범위를 벗어남. 총 이미지 수: {len(all_images)}")
                return
                
            # 해당 인덱스의 이미지 가져오기
            image = all_images[point_idx]
            
            # 이미지 역정규화
            normalize = self.config['dataset_config']['transform']['normalize']
            mean = torch.tensor(normalize['mean']).view(3, 1, 1)
            std = torch.tensor(normalize['std']).view(3, 1, 1)
            
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # 이미지 표시
            self.image_fig.clear()
            ax = self.image_fig.add_subplot(111)
            ax.imshow(image_denorm.permute(1, 2, 0).numpy())
            ax.set_title("원본 이미지", fontsize=14, fontweight='bold')
            ax.axis('off')
            self.image_canvas.draw()
            
        except Exception as e:
            print(f"이미지 표시 오류: {str(e)}")
            
    def _add_explainer(self):
        """Explainer 추가"""
        # 클래스 맵을 딕셔너리로 변환
        class_map = {}
        raw_class_map = self.config.get('class_map', [])
        for item in raw_class_map:
            if isinstance(item, dict):
                class_map.update(item)
        
        # Explainer 추가 다이얼로그
        dialog = ExplainerAddDialog(
            self.explainer_config,
            self.config['model'],
            class_map,
            current_predict=0,
            parent=self
        )
        
        if dialog.exec_():
            explainer_name, params, target_class = dialog.get_explainer_info()
            
            # 파라미터 타입 변환
            for k, v in list(params.items()):
                try:
                    val = eval(v)
                    params[k] = val
                except:
                    params[k] = v
            
            # Explainer 생성
            explainer_class = eval(self.explainer_config['explainer_dict'][explainer_name]['class'])
            self.explainers[explainer_name] = explainer_class(self.model, params)
            
            # 콤보박스 업데이트
            self.explainer_combo.addItem(explainer_name)
                    
    def _show_explainer_result(self, point_idx):
        """Explainer 결과 표시"""
        if not self.explainers or self.selected_point is None:
            return
            
        try:
            # 이전 워커가 실행 중이면 중단
            if hasattr(self, 'explainer_worker') and self.explainer_worker.isRunning():
                self.explainer_worker.terminate()
                self.explainer_worker.wait()
                print("이전 Explainer 실행 취소됨")
            
            # 선택된 explainer
            explainer_name = self.explainer_combo.currentText()
            if not explainer_name:
                return
                
            explainer = self.explainers[explainer_name]
            
            # 모든 이미지를 하나의 리스트로 관리
            all_images = []
            
            # target 이미지들 추가
            target_dataset = self.target_loader.dataset
            for i in range(len(target_dataset)):
                image, _ = target_dataset[i]
                all_images.append(image)
            
            # basis 이미지들 추가
            basis_dataset = self.basis_loader.dataset
            for i in range(len(basis_dataset)):
                image, _ = basis_dataset[i]
                all_images.append(image)
            
            # point_idx가 범위를 벗어나지 않도록 체크
            if point_idx >= len(all_images):
                print(f"point_idx {point_idx}가 범위를 벗어남. 총 이미지 수: {len(all_images)}")
                return
                
            # 해당 인덱스의 이미지 가져오기
            image = all_images[point_idx]
            
            # XAIWorker를 사용하여 dog와 target 각각 실행
            basis_class = 5  # dog
            target_class = self.target_combo.currentData()
            
            # dog와 target 각각에 대한 explainer 실행
            explainers_to_run = {
                f"{explainer_name}_dog": (explainer, basis_class),
                f"{explainer_name}_target": (explainer, target_class)
            }
            
            self.explainer_worker = XAIWorker(self.model, explainers_to_run, image.unsqueeze(0))
            self.explainer_worker.progress.connect(self.progress_bar.setValue)
            self.explainer_worker.finished.connect(self._on_explainer_finished)
            self.explainer_worker.error.connect(self._on_explainer_error)
            
            # 진행률 표시
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_label.setText("Explainer 실행 중...")
            
            # Explainer 선택 콤보박스 비활성화
            self.explainer_combo.setEnabled(False)
            
            self.explainer_worker.start()
            
        except Exception as e:
            print(f"Explainer 실행 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _on_explainer_finished(self, results):
        """Explainer 실행 완료"""
        try:
            # 결과 표시
            explainer_name = self.explainer_combo.currentText()
            heatmap_dog = results[f"{explainer_name}_dog"]
            heatmap_target = results[f"{explainer_name}_target"]
            
            # 모든 이미지를 하나의 리스트로 관리
            all_images = []
            
            # target 이미지들 추가
            target_dataset = self.target_loader.dataset
            for i in range(len(target_dataset)):
                image, _ = target_dataset[i]
                all_images.append(image)
            
            # basis 이미지들 추가
            basis_dataset = self.basis_loader.dataset
            for i in range(len(basis_dataset)):
                image, _ = basis_dataset[i]
                all_images.append(image)
            
            # point_idx가 범위를 벗어나지 않도록 체크
            if self.selected_point >= len(all_images):
                print(f"selected_point {self.selected_point}가 범위를 벗어남. 총 이미지 수: {len(all_images)}")
                return
                
            # 해당 인덱스의 이미지 가져오기
            image = all_images[self.selected_point]
            
            normalize = self.config['dataset_config']['transform']['normalize']
            mean = torch.tensor(normalize['mean']).view(3, 1, 1)
            std = torch.tensor(normalize['std']).view(3, 1, 1)
            
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # 결과 표시 (2개 나란히)
            self.explainer_fig.clear()
            
            # dog explainer 결과
            ax1 = self.explainer_fig.add_subplot(1, 2, 1)
            ax1.imshow(image_denorm.permute(1, 2, 0).numpy(), alpha=0.5)
            im1 = ax1.imshow(heatmap_dog, cmap='jet', alpha=0.5)
            ax1.set_title(f"{explainer_name} - dog", fontsize=14, fontweight='bold')
            ax1.axis('off')
            self.explainer_fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
            
            # target explainer 결과
            ax2 = self.explainer_fig.add_subplot(1, 2, 2)
            ax2.imshow(image_denorm.permute(1, 2, 0).numpy(), alpha=0.5)
            im2 = ax2.imshow(heatmap_target, cmap='jet', alpha=0.5)
            target_name = self.target_combo.currentText()
            ax2.set_title(f"{explainer_name} - {target_name}", fontsize=14, fontweight='bold')
            ax2.axis('off')
            self.explainer_fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
            
            self.explainer_canvas.draw()
            
            # 진행률 숨기기
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            
            # Explainer 선택 콤보박스 다시 활성화
            self.explainer_combo.setEnabled(True)
            
        except Exception as e:
            print(f"Explainer 결과 표시 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _on_explainer_error(self, error_msg):
        """Explainer 오류 처리"""
        QMessageBox.critical(self, "오류", f"Explainer 실행 실패: {error_msg}")
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # Explainer 선택 콤보박스 다시 활성화
        self.explainer_combo.setEnabled(True)


def show_logit_plot_dialog(model, config, explainer_config, parent=None):
    """2D 로짓 플롯 다이얼로그 표시"""
    dialog = LogitPlotDialog(model, config, explainer_config, parent)
    dialog.exec_()
