# -*- coding: utf-8 -*-
import os
import re
import sys
import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor, QFont
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QToolTip, QLabel
)

# 상위 디렉토리 모듈들 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.datasetLoader import get_dataloaders
from explainers import CAM, IG, RISE, GradCAM, SmoothGrad, SHAP, LMAP, PERMUTATION_SHAP
from models import CustomResNet34, ResNetClassifier
from utils.explainer_add_dialog import ExplainerAddDialog
from utils.explainer_tooltip import make_explainer_tooltip
from utils.get_class_name_by_index import get_class_name_by_index
from utils.set_korean import setup_korean_font
from utils.xaiworker import XAIWorker
from utils.visualization import save_widget_as_image_auto, save_snapshot_auto
from dialog.logitPlot import show_logit_plot_dialog

class XAIGUI(QMainWindow):
    """XAI GUI 메인 클래스"""
    
    # 진행률 업데이트 시그널
    progress_update = pyqtSignal(int)
    progress_text_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        self._load_explainer_config()
        self._init_variables()
        self._connect_signals()
        self.resize(2600, 1200)

    def _init_ui(self):
        """UI 초기화"""
        setup_korean_font()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'xai_gui.ui'), self)
        self.setup_ui()

    def _load_explainer_config(self):
        """Explainer 설정 로드"""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "explainer_config.yaml"
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                self.explainer_config = yaml.safe_load(f)
            self.explainer_dict = self.explainer_config['explainer_dict']
        except Exception as e:
            print(f"Explainer 설정 로드 실패: {str(e)}")
            self.explainer_config = {}
            self.explainer_dict = {}

    def _init_variables(self):
        """변수 초기화"""
        self.model = None
        self.config = None
        self.current_image = None
        self.current_image_tensor = None
        self.heatmap_results = {}
        self.explainers = {}
        self.worker = None
        self.num_classes = 0
        self.class_map = {}
        self.predicted_class = None
        self.current_explainees = {}
        self.snapshot_filenames = []

    def _connect_signals(self):
        """시그널 연결"""
        self.progress_update.connect(self.progressBar.setValue)
        self.progress_text_update.connect(self.progressInfo.setText)

    def setup_ui(self):
        """UI 설정"""
        # 버튼 연결
        self.loadCheckpointBtn.clicked.connect(self.load_checkpoint)
        self.loadImageBtn.clicked.connect(self.load_image)
        self.runXaiBtn.clicked.connect(self.run_xai)
        self.addExplainerBtn.clicked.connect(self.open_add_explainer_dialog)
        self.saveScreenBtn.clicked.connect(self._on_save_snapshot)
        self.openMultiImageViewerBtn.clicked.connect(self.open_multi_image_viewer)
        
        # 2D 로짓 플롯 버튼 추가
        self.logitPlotBtn.clicked.connect(self.open_logit_plot_dialog)
        
        # 초기 상태 설정
        self.targetClassCombo.setEnabled(False)
        self.runXaiBtn.setEnabled(False)
        self.saveScreenBtn.setEnabled(False)
        self.logitPlotBtn.setEnabled(False)
        
        # 리스트 이벤트 연결
        self.explainerListWidget.itemDoubleClicked.connect(self.remove_selected_explainer)
        self.explainerListWidget.setMouseTracking(True)
        self.explainerListWidget.itemEntered.connect(self.show_explainer_info_tooltip)
        
        # 라벨 바인딩
        self.predictedClassLabel = self.findChild(QLabel, "predictedClassLabel")

    def load_checkpoint(self):
        """체크포인트 로드"""
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "checkpoints"
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 YAML 파일 선택", default_path, "YAML files (*.yaml *.yml)"
        )
        
        if not file_path:
            return

        try:
            # YAML 설정 로드
            import copy
            self.config = copy.deepcopy(yaml.safe_load(open(file_path, 'r', encoding='utf-8')))
            
            # 클래스 맵 설정
            raw_class_map = self.config.get('class_map', [])
            self.class_map = {k: v for d in raw_class_map for k, v in d.items()}
            self.num_classes = self.config.get('num_classes')
            
            # 모델 로드
            self.load_model()

            # UI 업데이트
            self.checkpointLabel.setText(f"로드됨: {os.path.basename(file_path)}")
            self.infoText.append(f"[✓] 체크포인트 로드 완료 - 모델: {self.config.get('model', 'Unknown')}")
            self.infoText.append(f"[✓] 클래스 수: {self.config.get('num_classes', '?')}")
            
            # 2D 로짓 플롯 버튼 활성화
            self.logitPlotBtn.setEnabled(True)

        except Exception as e:
            import traceback
            error_msg = f"체크포인트 로드 실패: {str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "오류", error_msg)
            self.infoText.append(f"[✗] 오류 발생\n{error_msg}")

    def load_model(self):
        """모델 로드"""
        try:
            if self.config is None:
                raise ValueError("설정이 로드되지 않았습니다.")
                
            model_structure = self.config['model_structure']
            num_classes = self.config['num_classes']
            model_name = self.config['model']
            checkpoint_path = self.config['checkpoint_path']
            
            # 절대 경로로 변환
            if not os.path.isabs(checkpoint_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                checkpoint_path = os.path.join(project_root, checkpoint_path)
            
            # 모델 생성
            if model_name.lower() == "resnet18":
                print("Trying resnet18 load") 
                self.model = ResNetClassifier(num_classes=num_classes, dataset_config=model_structure)
            elif model_name.lower() == "resnet34":
                print("Trying resnet34 load") 
                self.model = CustomResNet34(num_classes=num_classes)
            else:
                raise ValueError(f"지원하지 않는 모델: {model_name}")
            
            # 디바이스 설정 및 가중치 로드
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # 체크포인트 처리
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # prefix 제거
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('network.'):
                    new_state_dict[k[8:]] = v
                else:
                    new_state_dict[k] = v

            # 모델에 가중치 로드
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            self.infoText.append(f"모델 가중치 로드 완료: {device}")
            
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")

    def load_image(self):
        """이미지 로드 및 전처리"""
        if self.model is None:
            QMessageBox.warning(self, "경고", "먼저 체크포인트를 로드해주세요.")
            return
        
        # 파일 선택
        default_path = self.config['dataset_config'].get('root', "") if self.config else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", default_path, 
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if not file_path:
            self.saveScreenBtn.setEnabled(False)
            return
        
        try:
            # 이미지 로드 및 전처리
            img_array = cv2.imread(file_path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # transform 설정 가져오기
            transform_config = self.config['dataset_config']['transform']
            img_size = tuple(transform_config['img_size'])
            normalize = transform_config['normalize']
            
            # 리사이즈 및 정규화
            img_array = cv2.resize(img_array, img_size)
            img_array = img_array.astype(np.float32) / 255.0
            mean = np.array(normalize['mean'])
            std = np.array(normalize['std'])
            img_array = (img_array - mean) / std
            
            # 텐서 변환
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.float()
            
            self.current_image_tensor = img_tensor
            
            # UI 업데이트
            self.imageLabel.setText(f"로드됨: {os.path.basename(file_path)}")
            self.infoText.append(f"이미지 로드 완료: {img_tensor.shape}")
            self.current_explainees = {}
            self.setup_target_classes()
            self.runXaiBtn.setEnabled(True)
            self.saveScreenBtn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 로드 실패: {str(e)}")
            self.saveScreenBtn.setEnabled(False)

    def setup_target_classes(self):
        """타겟 클래스 설정"""
        if self.config is None or self.model is None:
            raise ValueError("설정 또는 모델이 로드되지 않았습니다.")
            
        self.targetClassCombo.clear()

        with torch.no_grad():
            output = self.model(self.current_image_tensor)
            logits = output.squeeze().cpu().numpy()
            predicted_class = int(np.argmax(logits))
            confidence = float(torch.softmax(output, dim=1).max(dim=1).values.item())

        # 클래스 목록 생성
        for i in range(self.num_classes):
            class_name = self.class_map.get(i, f"클래스 {i}")
            logit = logits[i]
            if i == predicted_class:
                class_name += f" (예측, logit={logit:.3f}, conf={confidence:.3f})"
            else:
                class_name += f" (logit={logit:.3f})"
            self.targetClassCombo.addItem(class_name, i)

        self.targetClassCombo.setCurrentIndex(predicted_class)
        self.targetClassCombo.setEnabled(True)
        self.current_explainees = {}
        self.predicted_class = predicted_class
        self.predictedClassLabel.setText(
            f"현재 예측 클래스: {self.class_map.get(predicted_class, str(predicted_class))}"
        )

    def open_add_explainer_dialog(self):
        """Explainer 추가 다이얼로그"""
        dialog = ExplainerAddDialog(
            self.explainer_config, 
            self.config['model'], 
            self.class_map, 
            current_predict=self.predicted_class, 
            parent=self
        )
        
        if dialog.exec_():
            explainer_name, params, target_class = dialog.get_explainer_info()
            
            # 파라미터 타입 변환
            vis_type = ""
            for k, v in list(params.items()):
                try:
                    if k == "type":
                        vis_type = v
                    val = eval(v)
                    params[k] = val
                except:
                    params[k] = v
                    if k == "type":
                        vis_type = v
            
            # Explainer 생성
            explainer_class = eval(self.explainer_config['explainer_dict'][explainer_name]['class'])
            class_name = self.class_map.get(target_class, f"class{target_class}")
            base_key = f"[{explainer_name}] target-{target_class}-{vis_type}:{class_name}"
            
            # 중복 처리
            existing = [k for k in self.explainers if k.startswith(base_key)]
            version = len(existing)
            full_name = f"{base_key}{version}"

            # 등록
            self.explainers[full_name] = explainer_class(self.model, params)
            if not hasattr(self, 'explainer_targets'):
                self.explainer_targets = {}
            self.explainer_targets[full_name] = target_class

            self.update_explainer_list()
            self.infoText.append(f"Explainer 추가: {full_name}")

    def update_explainer_list(self):
        """Explainer 리스트 업데이트"""
        self.explainerListWidget.clear()
        for name, explainer in self.explainers.items():
            self.explainerListWidget.addItem(name)

    def show_explainer_info_tooltip(self, item):
        """Explainer 정보 툴팁"""
        name = item.text()
        msg = make_explainer_tooltip(
            name, self.explainers, self.explainer_dict, 
            self.config, self.current_image_tensor
        )
        QToolTip.showText(QCursor.pos(), msg, self.explainerListWidget)

    def remove_selected_explainer(self, item=None):
        """선택된 Explainer 삭제"""
        if item is None:
            item = self.explainerListWidget.currentItem()
        if item:
            name = item.text()
            if name in self.explainers:
                del self.explainers[name]
                self.update_explainer_list()
                if name in self.current_explainees:
                    del self.current_explainees[name]
                    self.infoText.append(f"Explainer 및 Explainee 캐시 삭제: {name}")

    def run_xai(self):
        """XAI 실행"""
        if self.current_image_tensor is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 로드해주세요.")
            return

        self.saveScreenBtn.setEnabled(False)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.progressInfo.setVisible(True)
        
        def progress_callback(percent, name):
            self.progress_update.emit(percent)
            self.progress_text_update.emit(f"진행중인 작업: {name}")
        
        # 실행할 explainer 수집
        explainers_to_run = {}
        for name, explainer in self.explainers.items():
            if not hasattr(self, 'current_explainees'):
                self.current_explainees = {}
            if name not in self.current_explainees:
                try:
                    target_class = int(name.split('-')[1])
                    explainers_to_run[name] = (explainer, target_class)
                except:
                    QMessageBox.warning(self, "이름 오류", f"Explainer 이름에서 TargetClass 파싱 실패: {name}")
                    continue

        # 기존 결과 복사
        results = {}
        for name in self.current_explainees:
            results[name] = self.current_explainees[name]

        # 새로 계산할 explainer가 있으면 worker 실행
        if explainers_to_run:
            self.worker = XAIWorker(self.model, explainers_to_run, self.current_image_tensor)
            self.worker.set_progress_callback(progress_callback)
            self.worker.progress.connect(self.progressBar.setValue)
            self.worker.finished.connect(lambda new_results: self.on_xai_finished_with_cache(new_results, results))
            self.worker.error.connect(self.on_xai_error)
            self.runXaiBtn.setEnabled(False)
            self.worker.start()
        else:
            self.heatmap_results = results
            self.visualize_results()

    def on_xai_finished_with_cache(self, new_results, cached_results):
        """XAI 완료 처리"""
        if not hasattr(self, 'current_explainees'):
            self.current_explainees = {}
        self.current_explainees.update(new_results)
        all_results = {**cached_results, **new_results}
        self.heatmap_results = all_results
        self.runXaiBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.progressInfo.setVisible(False)
        self.progressInfo.setText("진행중인 작업 없음")
        self.visualize_results()

    def on_xai_error(self, error_msg):
        """XAI 오류 처리"""
        QMessageBox.critical(self, "오류", f"히트맵 생성 실패: {error_msg}")
        self.runXaiBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.progressInfo.setText("진행중인 작업 없음")

    def visualize_results(self):
        """결과 시각화"""
        # 기존 위젯 제거
        for i in reversed(range(self.vizLayout.count())):
            item = self.vizLayout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    
        if not self.heatmap_results or self.current_image_tensor is None:
            self.saveScreenBtn.setEnabled(False)
            return
            
        # 정규화 값 가져오기
        if self.config is not None:
            normalize = self.config['dataset_config']['transform']['normalize']
            mean = normalize['mean']
            std = normalize['std']
            print("MEAN", mean)
            print("STD", std)
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # 이미지 역정규화
        img_array = self.current_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_array = img_array * std + mean
        img_array = np.clip(img_array, 0, 1)

        # 원본 이미지 표시
        fig_orig = Figure(figsize=(4, 4))
        ax_orig = fig_orig.add_subplot(111)
        ax_orig.imshow(img_array)
        ax_orig.set_title("원본 이미지")
        ax_orig.axis('off')
        canvas_orig = FigureCanvas(fig_orig)
        self.vizLayout.addWidget(canvas_orig, 0, 0)
        
        # 히트맵 표시
        num_explainers = len(self.heatmap_results)
        cols = min(3, num_explainers + 1)
        rows = (num_explainers + 1 + cols - 1) // cols
        
        for i, (name, heatmap) in enumerate(self.heatmap_results.items()):
            row = (i + 1) // cols
            col = (i + 1) % cols
            fig = Figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            
            match = re.match(r'\[(.*?)\]', name)
            name_no_number = match.group(1) if match else ""
            cmap = self.explainer_dict.get(name_no_number, {}).get('cmap', 'jet')
            
            ax.imshow(img_array, alpha=0.5)
            im = ax.imshow(heatmap, cmap=cmap, alpha=0.5)
            
            target_class_index = self.targetClassCombo.currentData()
            class_name = get_class_name_by_index(self.config, target_class_index)
            
            ax.set_title(f"{name.upper()}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            
            canvas = FigureCanvas(fig)
            self.vizLayout.addWidget(canvas, row, col)
            
        self.infoText.append(f"히트맵 생성 완료: {len(self.heatmap_results)}개")
        self.saveScreenBtn.setEnabled(True)

    def _on_save_snapshot(self):
        """스냅샷 저장"""
        filename = save_snapshot_auto(self.vizWidget, self.infoText)
        if filename:
            self.snapshot_filenames.append(filename)

    def open_multi_image_viewer(self):
        """멀티 이미지 뷰어"""
        from PyQt5.QtWidgets import QFileDialog
        from utils.visualization import show_images_in_dialog
        import os
        
        snapshot_dir = "C:/Users/orgin/XAI-study/heatmap_tool/app/snapshots"
        files, _ = QFileDialog.getOpenFileNames(self, "PNG 이미지 선택", snapshot_dir, "PNG Files (*.png)")
        if files:
            show_images_in_dialog(files, parent=self)

    def open_logit_plot_dialog(self):
        """2D 로짓 플롯 다이얼로그 열기"""
        if self.model is None or self.config is None:
            QMessageBox.warning(self, "경고", "먼저 체크포인트를 로드해주세요.")
            return
            
        try:
            show_logit_plot_dialog(self.model, self.config, self.explainer_config, self)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"2D 로짓 플롯 다이얼로그 열기 실패: {str(e)}")


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    font = QFont("Malgun Gothic", 9)
    app.setFont(font)
    window = XAIGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
