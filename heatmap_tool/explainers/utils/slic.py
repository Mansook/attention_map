import cv2
import numpy as np

def superpixel_mean_map(ig_map, region_size=30, ruler=10.0, algorithm=cv2.ximgproc.SLICO):
    # ig_map: (H, W) numpy array
    ig_map_norm = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    ig_map_uint8 = (ig_map_norm * 255).astype(np.uint8)
    slic_input = cv2.cvtColor(ig_map_uint8, cv2.COLOR_GRAY2BGR)
    slic = cv2.ximgproc.createSuperpixelSLIC(slic_input, algorithm, region_size, ruler)
    slic.iterate(10)
    labels = slic.getLabels()
    num_superpixels = slic.getNumberOfSuperpixels()
    sp_means = np.zeros(num_superpixels)
    for i in range(num_superpixels):
        sp_means[i] = ig_map[labels == i].mean() if np.any(labels == i) else 0
    sp_map = np.zeros_like(ig_map, dtype=np.float32)
    for i in range(num_superpixels):
        sp_map[labels == i] = sp_means[i]
    return sp_map, labels, sp_means