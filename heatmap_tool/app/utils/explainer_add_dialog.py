from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class ExplainerAddDialog(QDialog):
    def __init__(self, explainer_config, model_name, class_map,  current_predict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Explainer 추가")
        self.setFixedSize(500, 500)  # 다이얼로그 크기 고정
        self.explainer_config = explainer_config
        self.model_name = model_name
        self.class_map = class_map  # ✅ class_map 저장
        self.current_predict = current_predict  # current_predict 저장
        self.selected_explainer = None
        self.param_inputs = {}
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

        # 파라미터 입력 폼
        self.form = QFormLayout()
        layout.addLayout(self.form)

        # 확인 버튼
        self.ok_btn = QPushButton("확인")
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)

        self.setLayout(layout)
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
        if config is None:
            return
        if isinstance(config, str):
            config = {'target_layer': config}
        for key, value in config.items():
            line = QLineEdit(str(value))
            self.form.addRow(key, line)
            self.param_inputs[key] = line

    def get_explainer_info(self):
        explainer_name = self.combo.currentText()
        params = {k: v.text() for k, v in self.param_inputs.items()}
        # type 파라미터 추가
        params['type'] = self.type_combo.currentText()
        target_class = self.class_combo.currentData()  # ✅ 선택된 클래스 index
        return explainer_name, params, target_class
