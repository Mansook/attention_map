from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QCheckBox,
    QWidget,
    QGroupBox
)


class ExplainerAddDialog(QDialog):
    def __init__(self, explainer_config, model_name, class_map,  current_predict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Explainer 추가")
        self.setFixedWidth(500)  # 가로만 고정
        # self.setFixedHeight(원하는값)  # 세로도 고정하고 싶으면 사용
        self.explainer_config = explainer_config
        self.model_name = model_name
        self.class_map = class_map  # ✅ class_map 저장
        self.current_predict = current_predict  # current_predict 저장
        self.selected_explainer = None
        self.param_inputs = {}
        self.slic_checkbox = None
        self.slic_size_input = None
        self.slic_ruler_input = None
        self.slic_group = QGroupBox("Slic Option")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Explainer 종류 선택
        layout.addWidget(QLabel("Explainer 종류 선택"))
        self.combo = QComboBox()
        self.combo.addItems(self.explainer_config['explainer_dict'].keys())
        self.combo.currentTextChanged.connect(self.update_form)
        layout.addWidget(self.combo)

        # 클래스 선택 추가
        layout.addWidget(QLabel("Target 클래스 선택"))
        self.class_combo = QComboBox()
        for idx, name in self.class_map.items():
            self.class_combo.addItem(f"{idx}: {name}", idx)
        layout.addWidget(self.class_combo)

        # Type 선택 추가
        layout.addWidget(QLabel("Type 선택"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["both", "abs", "positive", "negative"])
        layout.addWidget(self.type_combo)

        # SLIC 토글 체크박스 (초기화는 update_form에서)
        self.slic_checkbox = QCheckBox("SLIC (Superpixel) 적용")
        self.slic_checkbox.setVisible(False)
        layout.addWidget(self.slic_checkbox)

        # SLIC size/ruler 입력란 그룹 추가
        slic_layout = QVBoxLayout()
        slic_layout.addWidget(QLabel("Slic size"))
        self.slic_size_input = QLineEdit()
        self.slic_size_input.setPlaceholderText("SLIC size")
        slic_layout.addWidget(self.slic_size_input)
        slic_layout.addWidget(QLabel("Slic ruler"))
        self.slic_ruler_input = QLineEdit()
        self.slic_ruler_input.setPlaceholderText("SLIC ruler")
        slic_layout.addWidget(self.slic_ruler_input)
        
        self.slic_group.setVisible(False)
        layout.addWidget(self.slic_group)
        self.slic_group.setLayout(slic_layout)  # ★ 이 줄이 반드시 필요!

        # 파라미터 입력 폼
        self.form = QFormLayout()
        layout.addLayout(self.form)
        
        # 확인 버튼
        self.ok_btn = QPushButton("확인")
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)

        self.setLayout(layout)
        
        if self.slic_checkbox is not None:
            self.slic_checkbox.stateChanged.connect(self._on_slic_checkbox_changed)
            
        self.update_form(self.combo.currentText())
        
        # current_predict가 있으면 해당 클래스 선택
        if hasattr(self, 'current_predict') and self.current_predict is not None:
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == self.current_predict:
                    self.class_combo.setCurrentIndex(i)
                    break

    def update_form(self, explainer_name):
        # 폼 초기화
        while self.form.rowCount():
            self.form.removeRow(0)
        self.param_inputs.clear()

        config = self.explainer_config['explainer_dict'][explainer_name]['model'].get(self.model_name, None)

        if isinstance(config, str):
            config = {'target_layer': config}
        for key, value in config.items():
            line = QLineEdit(str(value))
            self.form.addRow(key, line)
            self.param_inputs[key] = line

        # SLIC 옵션이 explainer config에 있으면 체크박스 표시
        slic_option = self.explainer_config['explainer_dict'][explainer_name].get('slic', None)
         # 체크박스 토글에 따라 입력란 표시/숨김
        if self.slic_checkbox is not None:
            if slic_option is not None:
                self.slic_checkbox.setVisible(True)
                # SLIC dict에서 size/ruler 값 추출
                size_val = slic_option.get('size', '') if isinstance(slic_option, dict) else ''
                ruler_val = slic_option.get('ruler', '') if isinstance(slic_option, dict) else ''
                if self.slic_size_input is not None:
                    self.slic_size_input.setText(str(size_val))
                if self.slic_ruler_input is not None:
                    self.slic_ruler_input.setText(str(ruler_val))
               
            else:
                self.slic_checkbox.setVisible(False)


    def _on_slic_checkbox_changed(self, state):
        visible = self.slic_checkbox.isChecked() if self.slic_checkbox is not None else False
        print('visible',visible)
        self.slic_group.setVisible(visible)
        self.slic_group.updateGeometry()
        self.updateGeometry()

    def get_explainer_info(self):
        explainer_name = self.combo.currentText()
        params = {k: v.text() for k, v in self.param_inputs.items()}
        # type 파라미터 추가
        params['type'] = self.type_combo.currentText()
        # slic 파라미터 추가
        if self.slic_checkbox is not None:
     
            params['slic'] = self.slic_checkbox.isChecked()
            if self.slic_checkbox.isChecked():
                if self.slic_size_input is not None:
                    params['slic_size'] = self.slic_size_input.text()
                if self.slic_ruler_input is not None:
                    params['slic_ruler'] = self.slic_ruler_input.text()
        target_class = self.class_combo.currentData()  # ✅ 선택된 클래스 index
        return explainer_name, params, target_class
