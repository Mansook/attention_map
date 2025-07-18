# -*- coding: utf-8 -*-
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageOps, ImageQt
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCursor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolTip,
)
from utils.explainer_add_dialog import (
    ExplainerAddDialog,  # 다이얼로그는 utils에 구현한다고 가정
)

# 상위 디렉토리 모듈들 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.datasetLoader import get_dataloaders
from explainers import CAM, IG, RISE, GradCAM, SmoothGrad
from models import CustomResNet34, ResNetClassifier
from utils.explainer_tooltip import make_explainer_tooltip
from utils.set_korean import setup_korean_font
from utils.xaiworker import XAIWorker


class XAIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        setup_korean_font()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'xai_gui.ui'), self)
        self.setup_ui()
        
        # explainer config 로드
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explainer_config.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.explainer_config = yaml.safe_load(f)
            self.explainer_dict = self.explainer_config['explainer_dict']
        except Exception as e:
            print(f"Explainer 설정 로드 실패: {str(e)}")
            self.explainer_config = {}
            self.explainer_dict = {}
        
        self.model = None
        self.config = None
        self.current_image = None
        self.current_image_tensor = None
        self.heatmap_results = {}
        self.explainers = {}
        self.worker = None
        
        self.current_explainees = {}
        
        
        self.resize(1800, 1200)  # 또는 원하는 크기로 조정

    def setup_ui(self):
        self.loadCheckpointBtn.clicked.connect(self.load_checkpoint)
        self.loadImageBtn.clicked.connect(self.load_image)
        self.runXaiBtn.clicked.connect(self.run_xai)
        self.targetClassCombo.setEnabled(False)
        self.runXaiBtn.setEnabled(False)
        # Explainer 관리 버튼 바인딩
        self.addExplainerBtn.clicked.connect(self.open_add_explainer_dialog)
        #self.removeExplainerBtn.clicked.connect(self.remove_selected_explainer)
        # 리스트에서 더블클릭 시 삭제
        self.explainerListWidget.itemDoubleClicked.connect(self.remove_selected_explainer)
        # Hover 이벤트 활성화
        self.explainerListWidget.setMouseTracking(True)
        self.explainerListWidget.itemEntered.connect(self.show_explainer_info_tooltip)

    def load_checkpoint(self):
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "checkpoints")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "체크포인트 YAML 파일 선택", default_path, "YAML files (*.yaml *.yml)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.load(f, Loader=yaml.SafeLoader)
                    import copy
                    self.config = copy.deepcopy(loaded_config)
                self.infoText.append(f"Config 로드됨: {type(self.config)}")
                if self.config and 'dataset_config' in self.config:
                    self.infoText.append(f"dataset_config 타입: {type(self.config['dataset_config'])}")
                    if 'root' in self.config['dataset_config']:
                        self.infoText.append(f"root 타입: {type(self.config['dataset_config']['root'])}")
                self.load_model()
                self.checkpointLabel.setText(f"로드됨: {os.path.basename(file_path)}")
                self.infoText.append(f"체크포인트 로드 완료: {self.config['model']}")
                self.infoText.append(f"클래스 수: {self.config['num_classes']}")
            except Exception as e:
                import traceback
                error_msg = f"체크포인트 로드 실패: {str(e)}\n{traceback.format_exc()}"
                QMessageBox.critical(self, "오류", error_msg)
                self.infoText.append(f"오류 상세: {error_msg}")

    def load_model(self):
        try:
            if self.config is None:
                raise ValueError("설정이 로드되지 않았습니다.")
            model_structure = self.config['model_structure']
            num_classes = self.config['num_classes']
            model_name = self.config['model']
            checkpoint_path = self.config['checkpoint_path']
            if not os.path.isabs(checkpoint_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                checkpoint_path = os.path.join(project_root, checkpoint_path)
            if model_name.lower() == "resnet18":
                print("Trying resnet18 load") 
                self.model = ResNetClassifier(num_classes=num_classes, dataset_config=model_structure)
            elif model_name.lower() == "resnet34":
                print("Trying resnet34 load") 
                self.model = CustomResNet34(num_classes=num_classes)
            else:
                
                ##모델 추가##
                
                raise ValueError(f"지원하지 않는 모델: {model_name}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # 혹시 checkpoint가 dict로 감싸져 있으면
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # prefix 'network.' 제거
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('network.'):
                    new_state_dict[k[8:]] = v
                else:
                    new_state_dict[k] = v

            # 모델에 로드
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            self.infoText.append(f"모델 가중치 로드 완료: {device}")
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")

    def open_add_explainer_dialog(self):
        # 이미 로드된 explainer_config 사용
        dialog = ExplainerAddDialog(self.explainer_config, self.config['model'], self)
        if dialog.exec_():
            explainer_name, params = dialog.get_explainer_info()
            # 파라미터 타입 변환
            for k, v in params.items():
                try:
                    params[k] = eval(v)
                except:
                    pass
            explainer_class = eval(self.explainer_config['explainer_dict'][explainer_name]['class'])
            # ---------- 중복 이름 처리 ----------
            base_name = explainer_name
            idx = 0
            while True:
                name = base_name if idx == 0 else f"{base_name}{idx}"
                if name not in self.explainers:
                    break
                idx += 1
            self.explainers[name] = explainer_class(self.model, params)
            self.update_explainer_list()
            self.infoText.append(f"Explainer 추가: {name}")

    def update_explainer_list(self):
        self.explainerListWidget.clear()
        for name, explainer in self.explainers.items():
            self.explainerListWidget.addItem(name)
        # Hover 이벤트 연결 (setMouseTracking 필요)
        self.explainerListWidget.setMouseTracking(True)
        self.explainerListWidget.itemEntered.connect(self.show_explainer_info_tooltip)

    def show_explainer_info_tooltip(self, item):
        name = item.text()
        msg = make_explainer_tooltip(
            name,
            self.explainers,
            self.explainer_dict,
            self.config,
            self.current_image_tensor
        )
        QToolTip.showText(QCursor.pos(), msg, self.explainerListWidget)

    def remove_selected_explainer(self, item=None):
        # 리스트에서 선택된 explainer 삭제
        if item is None:
            item = self.explainerListWidget.currentItem()
        if item:
            name = item.text()
            if name in self.explainers:
                del self.explainers[name]
                self.update_explainer_list()
                self.infoText.append(f"Explainer 삭제: {name}")

    def load_image(self):
        if self.model is None:
            QMessageBox.warning(self, "경고", "먼저 체크포인트를 로드해주세요.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 선택", "", "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            try:
                pil_image = Image.open(file_path).convert('RGB')
                print("pil_image size : ", pil_image.size)
                self.current_image = pil_image
                if self.config is None:
                    raise ValueError("설정이 로드되지 않았습니다.")
                transform_config = self.config['dataset_config']['transform']
                img_size = tuple(transform_config['img_size'])
                print("transform size : ", img_size)
                normalize = transform_config['normalize']
                # 비율 유지하며 가장 짧은 변에 맞춰 resize 후 중앙 crop
                from PIL import ImageOps

                pil_image = ImageOps.pad(pil_image, img_size, method=Image.Resampling.LANCZOS, color=(0,0,0), centering=(0.5,0.5))
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                img_array = (img_array - np.array(normalize['mean'])) / np.array(normalize['std'])
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.float()  # float32로 변환
                self.current_image_tensor = img_tensor
                print("img_tensor size : ", img_tensor.shape)
                self.imageLabel.setText(f"로드됨: {os.path.basename(file_path)}")
                self.infoText.append(f"이미지 로드 완료: {img_tensor.shape}")
                self.current_explainees = {}
                self.setup_target_classes()
                self.runXaiBtn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"이미지 로드 실패: {str(e)}")

    def setup_target_classes(self):
        if self.config is None or self.model is None:
            raise ValueError("설정 또는 모델이 로드되지 않았습니다.")
        num_classes = self.config['num_classes']
        self.targetClassCombo.clear()
        
        # class_map 로드
        class_map = {}
        if 'class_map' in self.config:
            for item in self.config['class_map']:
                for key, value in item.items():
                    class_map[int(key)] = value
        
        with torch.no_grad():
            output = self.model(self.current_image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max(dim=1).values.item()
        
        for i in range(num_classes):
            # class_map에서 클래스 이름 가져오기
            class_name = class_map.get(i, f"클래스 {i}")
            if i == predicted_class:
                class_name += f" (예측, {confidence:.3f})"
            self.targetClassCombo.addItem(class_name, i)
        
        self.targetClassCombo.setCurrentIndex(int(predicted_class))
        self.targetClassCombo.setEnabled(True)
        self.current_explainees = {}

    def get_class_name_by_index(self, index):
        """
        인덱스를 클래스 이름으로 변환
        """
        if self.config is None or 'class_map' not in self.config:
            return f"클래스 {index}"
        
        class_map = {}
        for item in self.config['class_map']:
            for key, value in item.items():
                class_map[int(key)] = value
        
        return class_map.get(index, f"클래스 {index}")

    def run_xai(self):
        if self.current_image_tensor is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 로드해주세요.")
            return
        target_class = self.targetClassCombo.currentData()
        
        # 이미 계산된 explainer는 제외
        explainers_to_run = {}
        for name, explainer in self.explainers.items():
            if not hasattr(self, 'current_explainees'):
                self.current_explainees = {}
            if name not in self.current_explainees:
                explainers_to_run[name] = explainer

        # 이미 계산된 heatmap은 바로 결과에 추가
        results = {}
        for name in self.current_explainees:
            results[name] = self.current_explainees[name]

        # 새로 계산할 explainer가 있으면 worker 실행
        if explainers_to_run:
            self.worker = XAIWorker(
                self.model, explainers_to_run,
                self.current_image_tensor, target_class
            )
            self.worker.progress.connect(self.progressBar.setValue)
            self.worker.finished.connect(lambda new_results: self.on_xai_finished_with_cache(new_results, results))
            self.worker.error.connect(self.on_xai_error)
            self.runXaiBtn.setEnabled(False)
            self.worker.start()
        else:
            # 모두 캐시된 경우 바로 plot
            self.heatmap_results = results
            self.visualize_results()

    def on_xai_finished_with_cache(self, new_results, cached_results):
        # 새로 계산된 결과를 캐시에 추가
        if not hasattr(self, 'current_explainees'):
            self.current_explainees = {}
        self.current_explainees.update(new_results)
        # 전체 결과 합치기
        all_results = {**cached_results, **new_results}
        self.heatmap_results = all_results
        self.runXaiBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self.visualize_results()

    def on_xai_error(self, error_msg):
        QMessageBox.critical(self, "오류", f"히트맵 생성 실패: {error_msg}")
        self.runXaiBtn.setEnabled(True)
        self.progressBar.setValue(0)

    def visualize_results(self):
        # 기존 위젯들 제거
        for i in reversed(range(self.vizLayout.count())):
            item = self.vizLayout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        if not self.heatmap_results:
            return
        if self.current_image_tensor is None:
            return  # 이미지가 없으면 함수 종료
        # 예시: mean, std 값 (dataset config에서 확인)
        if self.config is not None:
            normalize = self.config['dataset_config']['transform']['normalize']
            mean = normalize['mean']
            std = normalize['std']
        else:
            mean = [0.485, 0.456, 0.406]  # 예시 (ImageNet)
            std = [0.229, 0.224, 0.225]   # 예시 (ImageNet)

        # 텐서 → numpy 변환
        img_array = self.current_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        # 역정규화
        img_array = img_array * std + mean
        img_array = np.clip(img_array, 0, 1)  # 값 범위 0~1로 제한

        # 시각화
        fig_orig = Figure(figsize=(4, 4))
        ax_orig = fig_orig.add_subplot(111)
        ax_orig.imshow(img_array)
        ax_orig.set_title("원본 이미지")
        ax_orig.axis('off')
        canvas_orig = FigureCanvas(fig_orig)
        self.vizLayout.addWidget(canvas_orig, 0, 0)
        num_explainers = len(self.heatmap_results)
        cols = min(3, num_explainers + 1)
        rows = (num_explainers + 1 + cols - 1) // cols
        for i, (name, heatmap) in enumerate(self.heatmap_results.items()):
            print("name : ", name)
            row = (i + 1) // cols
            col = (i + 1) % cols
            fig = Figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            name_no_number = re.sub(r'\d+$', '', name)
            cmap = self.explainer_dict.get(name_no_number, {}).get('cmap', 'jet')
            print("cmap : ", cmap)
            
            if cmap == "gray":
                # IG는 원본과 heatmap을 겹치지 않음
                ax.imshow(heatmap, cmap=cmap)
            else:
                # 나머지는 원본 위에 heatmap을 겹침
                ax.imshow(img_array, alpha=0.4)
                ax.imshow(heatmap, cmap=cmap, alpha=0.8)
            
            # 현재 선택된 클래스 이름 가져오기
            target_class_index = self.targetClassCombo.currentData()
            class_name = self.get_class_name_by_index(target_class_index)
            
            ax.set_title(f"{name.upper()} - {class_name}")
            ax.axis('off')
            canvas = FigureCanvas(fig)
            self.vizLayout.addWidget(canvas, row, col)
        self.infoText.append(f"히트맵 생성 완료: {len(self.heatmap_results)}개")

def main():
    app = QApplication(sys.argv)
    font = QFont("Malgun Gothic", 9)
    app.setFont(font)
    window = XAIGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
