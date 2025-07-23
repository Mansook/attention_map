import cv2
import numpy as np

def superpixel_mean_map(ig_map, region_size=30, ruler=10.0, algorithm=cv2.ximgproc.SLIC):
    # tensor를 numpy 배열로 변환
    if hasattr(ig_map, 'cpu'):  # tensor인 경우
        ig_map = ig_map.cpu().numpy()
    
    # 정규화
    ig_map_norm = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    
    # uint8로 변환
    if hasattr(ig_map_norm, 'cpu'):  # tensor인 경우
        ig_map_norm = ig_map_norm.cpu().numpy()
    
    ig_map_uint8 = (ig_map_norm * 255).astype(np.uint8)
    
    # BGR로 변환
    slic_input = cv2.cvtColor(ig_map_uint8, cv2.COLOR_GRAY2BGR)
    
    # SLIC 알고리즘 실행
    slic = cv2.ximgproc.createSuperpixelSLIC(slic_input, algorithm, region_size, ruler)
    slic.iterate(10)
    
    # boundary 추출
    boundary = slic.getLabelContourMask()
    
    labels = slic.getLabels()
    
    return labels, boundary