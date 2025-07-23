import cv2
import numpy as np

def superpixel_mean_map(input_tensor, region_size=30, ruler=10.0, algorithm=cv2.ximgproc.SLIC):
    # tensor를 numpy 배열로 변환
    if hasattr(input_tensor, 'cpu'):  # PyTorch tensor인 경우
        input_np = input_tensor.squeeze().cpu().numpy()
    elif hasattr(input_tensor, 'squeeze'):  # numpy array인 경우
        input_np = input_tensor.squeeze()
    else:  # 이미 numpy array인 경우
        input_np = input_tensor
    
    print(f"[SLIC] input_tensor shape: {input_tensor.shape}")
    print(f"[SLIC] input_np shape: {input_np.shape}")
    print(f"[SLIC] input_np ndim: {input_np.ndim}")
    
    # 형태 변환 (C, H, W) -> (H, W, C)
    if input_np.ndim == 3 and input_np.shape[0] == 3:  # (C, H, W)
        input_np = np.transpose(input_np, (1, 2, 0))  # (H, W, C)
        print(f"[SLIC] transpose 후 shape: {input_np.shape}")
    elif input_np.ndim == 2:  # (H, W) - 1채널
        print(f"[SLIC] 1채널 감지! 원본이 1채널입니다.")
    else:
        print(f"[SLIC] 예상치 못한 shape: {input_np.shape}")
    
    # OpenCV 추천 방식: Gaussian blur + CIELAB 변환
    # 1. 정규화 (0-255 범위)
    input_norm = ((input_np - input_np.min()) / (input_np.max() - input_np.min()) * 255).astype(np.uint8)
    
    # 2. Gaussian blur (3x3 kernel)
    input_blur = cv2.GaussianBlur(input_norm, (3, 3), 0)
    
    # 3. 채널 수에 따른 처리
    if input_blur.ndim == 2:  # 1채널 (H, W)
        # 1채널을 3채널로 변환 후 CIELAB 변환
        input_3ch = cv2.cvtColor(input_blur, cv2.COLOR_GRAY2BGR)
        slic_input = cv2.cvtColor(input_3ch, cv2.COLOR_BGR2LAB)
    elif input_blur.ndim == 3 and input_blur.shape[2] == 3:  # 3채널 (H, W, C)
        # RGB를 CIELAB로 변환
        slic_input = cv2.cvtColor(input_blur, cv2.COLOR_RGB2LAB)
    else:
        # 기타 경우는 그레이스케일로 처리
        input_3ch = cv2.cvtColor(input_blur, cv2.COLOR_GRAY2BGR)
        slic_input = cv2.cvtColor(input_3ch, cv2.COLOR_BGR2LAB)
    
    # SLIC 알고리즘 실행
    slic = cv2.ximgproc.createSuperpixelSLIC(slic_input, algorithm, region_size, ruler)
    slic.iterate(10)
    
    # boundary 추출
    boundary = slic.getLabelContourMask()
    
    labels = slic.getLabels()
    
    return labels, boundary