import torch
from PyQt5.QtCore import QThread, pyqtSignal


class XAIWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, model, explainers, image):
        super().__init__()
        self.model = model
        self.explainers = explainers
        self.image = image
        self.progress_callback = None  # 진행률 콜백 함수
        print("image size : ", self.image.shape)
    
    def set_progress_callback(self, callback):
        """진행률 콜백 함수 설정"""
        self.progress_callback = callback
    
    def run(self):
        """
        QThread의 run 메서드 오버라이드.
        이 메서드는 별도의 스레드에서 실행되므로, 메인 GUI 스레드를 블로킹하지 않음.
        즉, 이 클래스(XAIWorker)는 PyQt의 QThread를 상속받아 동작하므로,
        run() 내부 코드는 메인 스레드와 '병렬'로 실행됨(멀티스레딩).
        하지만 for문 내부의 각 explainer.generate() 호출은 순차적으로 실행됨.
        즉, 여러 explainer가 동시에 병렬로 실행되는 것은 아니고,
        전체적으로는 "백그라운드 스레드에서 순차 실행"임.
        """
        try:
            results = {}
            total_explainers = len(self.explainers)
            
            for i, (name, (explainer, target_class)) in enumerate(self.explainers.items()):
                # 전체 진행률 업데이트
                overall_progress = int((i / total_explainers) * 100)
                self.progress.emit(overall_progress)
                
                # explainer에 진행률 콜백 설정 (있는 경우)
                if hasattr(explainer, 'progress_callback') and self.progress_callback:
                    explainer.progress_callback = self.progress_callback
                
                # 각 explainer의 generate 메서드 실행 (여기서 연산 발생, 순차적임)
                heatmap = explainer.generate(self.image, class_idx=target_class)
                
                # 결과가 torch.Tensor면 numpy로 변환
                if isinstance(heatmap, torch.Tensor):
                    heatmap = heatmap.squeeze().cpu().numpy()
                results[name] = heatmap
            
            # 100% 완료 신호
            self.progress.emit(100)
            # 결과 딕셔너리를 finished 시그널로 emit
            self.finished.emit(results)
        except Exception as e:
            # 에러 발생 시 error 시그널 emit
            self.error.emit(str(e))