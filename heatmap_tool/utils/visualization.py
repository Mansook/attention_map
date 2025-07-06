import numpy as np
import matplotlib.pyplot as plt
import cv2

# STL10 클래스 이름 (필요시 외부에서 import 가능)
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def visualize_xai(image, heatmap, predicted_class, true_class, save_path=None, method_name="CAM", class_names=None):
    """
    XAI heatmap 시각화 (원본, 히트맵, 오버레이)
    Args:
        image: [3, H, W] 형태의 원본 이미지 텐서 (정규화 해제 필요)
        heatmap: [H, W] 형태의 XAI 기법(CAM 등)으로 생성된 히트맵 (numpy array, 값 범위 0~1)
        # 히트맵 파라미터는 모델이 예측한 결과에 대해 "어디를 집중해서 판단했는지" 시각적으로 보여주기 위해 필요함.
        # 즉, image는 실제 입력 이미지이고, heatmap은 해당 이미지에서 모델이 주목한 영역을 나타냄.
        predicted_class: int
        true_class: int
        save_path: 저장 경로
        method_name: 기법명 (ex: 'CAM', 'GradCAM')
        class_names: 클래스 이름 리스트 (None이면 기본 STL10 클래스 사용)
    """
    # 클래스 이름 설정
    if class_names is None:
        class_names = STL10_CLASSES
    
    img_np = image.permute(1, 2, 0).numpy()  # [96, 96, 3] 형태로 변환
    # 이미지 정규화 해제: [-1, 1] 범위를 [0, 1] 범위로 변환
    img_np = (img_np + 1) / 2
    img_np = np.clip(img_np, 0, 1)  # 값 범위를 0~1로 제한
    # CAM 히트맵을 원본 이미지 크기(96x96)로 리사이즈
    cam_resized = cv2.resize(heatmap, (96, 96))

    # CAM 히트맵에 색상 맵 적용 (JET 컬러맵 사용)
    cam_uint8 = np.uint8(255 * cam_resized)  # 0~1 범위를 0~255 범위로 변환
    heatmap = cv2.applyColorMap(np.array(cam_uint8, dtype=np.uint8), cv2.COLORMAP_JET)  # JET 컬러맵 적용
    heatmap = np.float32(heatmap) / 255  # 다시 0~1 범위로 정규화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original\nTrue: {class_names[true_class]}')
    axes[0].axis('off')
    axes[1].imshow(cam_resized,cmap='jet')
    axes[1].set_title(f'{method_name} Heatmap\nPred: {class_names[predicted_class]}')
    axes[1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"히트맵 저장: {save_path}")
    plt.show() 