import torch
import torch.nn.functional as F
import numpy as np

def process_heatmap_by_type(heatmap_tensor, type_name):
    """
    히트맵을 type에 따라 처리하고 정규화
    
    Args:
        heatmap_tensor: 처리할 히트맵 텐서
        type_name: 처리 타입 ("abs", "positive", "negative", "both")
    
    Returns:
        normalized_heatmap: 처리되고 정규화된 히트맵 (numpy array)
    """
    # type에 따라 텐서 처리
    if type_name == 'abs':
        # 절댓값 모드
        heatmap_tensor = torch.abs(heatmap_tensor)
    elif type_name == 'positive':
        # 양수만 모드
        heatmap_tensor = F.relu(heatmap_tensor)
    elif type_name == 'negative':
        # 음수만 모드
        heatmap_tensor = -F.relu(-heatmap_tensor)
    elif type_name == 'both':
        # 양수/음수 모두 모드 (기본값)
        pass  # 원본 값 그대로 사용
    else:
        # 기본값: 양수/음수 모두
        pass
    
    # numpy로 변환
    heatmap = heatmap_tensor.cpu().numpy()
    
    # 정규화 처리
    if type_name == 'abs':
        # 절댓값 모드: 절댓값으로 정규화
        if heatmap.max() > 0:
            vmax = np.percentile(heatmap, 99)
            vmin = np.percentile(heatmap, 1)
            heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    elif type_name == 'positive':
        # 양수만 모드: 양수만 정규화
        heatmap = np.maximum(heatmap, 0) #양수만 남김. 음수는 0으로 만듬
        if heatmap.max() > 0:
            vmax = np.percentile(heatmap, 99) #최댓값 
            vmin = np.percentile(heatmap, 1)  #최솟값 
            heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1) #정규화 하여, 0보다 작으면 0, 1보다 크면 1로 만듬
    elif type_name == 'negative':
        # 음수만 모드: 음수만 정규화
        heatmap = np.minimum(heatmap, 0)
        if heatmap.min() < 0:
            vmax = np.percentile(-heatmap, 99)
            vmin = np.percentile(-heatmap, 1)
            heatmap = -np.clip((-heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    elif type_name == 'both':
        # 양수/음수 모두 모드: 전체 범위로 정규화
        if heatmap.max() > 0 or heatmap.min() < 0:
            vmax = np.percentile(heatmap, 99)
            vmin = np.percentile(heatmap, 1)
            heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), -1, 1)
    
    return heatmap 