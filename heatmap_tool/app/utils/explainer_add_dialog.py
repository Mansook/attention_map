from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QFormLayout, QLineEdit, QPushButton, QLabel

class ExplainerAddDialog(QDialog):
    def __init__(self, explainer_config, model_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Explainer 추가")
        self.explainer_config = explainer_config
        self.model_name = model_name
        self.selected_explainer = None
        self.param_inputs = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(self.explainer_config['explainer_dict'].keys())
        self.combo.currentTextChanged.connect(self.update_form)
        layout.addWidget(QLabel("Explainer 종류 선택"))
        layout.addWidget(self.combo)
        self.form = QFormLayout()
        layout.addLayout(self.form)
        self.ok_btn = QPushButton("확인")
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)
        self.setLayout(layout)
        self.update_form(self.combo.currentText())

    def update_form(self, explainer_name):
        # 폼 초기화
        while self.form.rowCount():
            self.form.removeRow(0)
        self.param_inputs.clear()
        config = self.explainer_config['explainer_dict'][explainer_name]['model'].get(self.model_name, None)
        if config is None:
            # 값이 없으면 빈 dict로 처리 (혹은 안내 메시지)
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
        return explainer_name, params