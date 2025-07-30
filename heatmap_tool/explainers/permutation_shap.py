import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import numpy as np
import cv2
from itertools import combinations
import random
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import hashlib
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 대신 Agg 백엔드 사용
import matplotlib.pyplot as plt
import time

class PERMUTATION_SHAP(nn.Module):
    def __init__(self,model,config_dict):
        super(PERMUTATION_SHAP,self).__init__()
        self.model = model.eval()
        self.type = config_dict.get('type', 'absolute')
        self.input_size = config_dict.get('input_size',(96,96))
        self.sampling_size = config_dict.get('sampling_size', 100)
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 5)
        self.progress_callback = None
        self.samples_dir = "C:/Users/orgin/XAI-study/heatmap_tool/"
        self.save_samples = True
        # 이미지 정보와 시각화 라벨을 저장할 변수 추가
        #self.image_info = config_dict.get('image_info', 'unknown_image')
        self.visualization_label = 'PERMUTATION_SHAP'
        self.class_map=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    def generate(self,input_tensor,class_idx=None):
        start_time = time.time()
        print(f"[PERMUTATION_SHAP] 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        
        # print(f"[DEBUG] input_tensor shape: {input_tensor.shape}")
        
        # input_tensor를 해시화하여 image_info 생성
        input_hash = hashlib.md5(input_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
        self.image_info = f"image_{input_hash}"
        # print(f"[DEBUG] 생성된 image_info: {self.image_info}")
        
        input_3d = input_tensor.squeeze()
        # print(f"[DEBUG] input_3d shape after squeeze: {input_3d.shape}")
        
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)
            # print(f"[DEBUG] input_3d shape after unsqueeze: {input_3d.shape}")
        
        # SLIC로 슈퍼픽셀 라벨 추출
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        # print(f"[DEBUG] labels shape: {labels.shape}")
        # print(f"[DEBUG] unique labels: {np.unique(labels)}")
        
        unique_labels = np.unique(labels)
        num_superpixels = len(unique_labels)
        # print(f"[DEBUG] num_superpixels: {num_superpixels}")
        
        # 미리 마스크 생성 (메모리 효율성)
        masks = {}
        for label in unique_labels:
            mask = (labels == label)
            masks[label] = mask
            # print(f"[DEBUG] mask for label {label} shape: {mask.shape}")
        
        # 배치 처리를 위한 마스크 텐서 미리 생성
        mask_tensors = {}
        for label, mask in masks.items():
            mask_3d = np.stack([mask] * input_3d.shape[0], axis=0)
            # print(f"[DEBUG] mask_3d for label {label} shape: {mask_3d.shape}")
            mask_tensors[label] = torch.from_numpy(mask_3d).bool()
            # print(f"[DEBUG] mask_tensor for label {label} shape: {mask_tensors[label].shape}")
        
        shap_values = np.zeros(num_superpixels)
        
        # 순열 생성
        permutation_list = [np.random.permutation(unique_labels) for _ in range(self.sampling_size)]
        basis = torch.zeros_like(input_3d.unsqueeze(0))
        # print(f"[DEBUG] basis shape: {basis.shape}")
        # print(f"[DEBUG] basis[0] shape: {basis[0].shape}")
        basis_score = 0
        
        with torch.no_grad():
            # 베이스 점수 계산
            output = self.model(basis)
            if class_idx is not None:
                store_basis_score = output[0, class_idx].item()
            else:
                store_basis_score = output.max(1)[1].item()
            # print(f"[DEBUG] 베이스 점수: {store_basis_score}")

        # 디버깅용 저장 디렉토리 생성
        debug_dir = os.path.join(self.samples_dir, "cache", "debug_basis")
        os.makedirs(debug_dir, exist_ok=True)
        
        # 첫 번째 순열만 디버깅용으로 저장
        first_permutation = permutation_list[0]
        # print(f"[DEBUG] 첫 번째 순열: {first_permutation}")
        
        # 빈 이미지 저장
        self._save_debug_image(basis[0], debug_dir, "00_basis_empty.png", "빈 이미지 (베이스)")
        
        for perm_idx, permut in enumerate(permutation_list):
            # 각 순열마다 빈 이미지로 시작
            basis = torch.zeros_like(input_3d.unsqueeze(0))
            basis_score = store_basis_score  # 빈 이미지의 점수
            
            # print(f"[DEBUG] 순열 {perm_idx} 시작 - 베이스 점수: {basis_score}")
    
            # 순열 순서대로 슈퍼픽셀을 하나씩 누적해서 추가
            for idx, val in enumerate(permut):
                progress_percent = int((perm_idx * num_superpixels + idx) / (self.sampling_size * num_superpixels) * 100)
                if self.progress_callback:
                    self.progress_callback(progress_percent, "PERMUT_SHAP")
                
                # 현재 슈퍼픽셀을 basis에 추가
                mask_tensor = mask_tensors[val]
                # print(f"[DEBUG] 순열 {perm_idx}, 단계 {idx}: 슈퍼픽셀 {val} 추가")
                # print(f"[DEBUG] mask_tensor shape: {mask_tensor.shape}")
                # print(f"[DEBUG] input_3d shape: {input_3d.shape}")
                # print(f"[DEBUG] basis[0] shape: {basis[0].shape}")
                
                # 슈퍼픽셀 추가 전 basis 상태 출력
                # print(f"[DEBUG] 슈퍼픽셀 추가 전 basis - min: {basis[0].min()}, max: {basis[0].max()}, mean: {basis[0].mean()}")
                
                basis[0][mask_tensor] = input_3d[mask_tensor]
                
                # 슈퍼픽셀 추가 후 basis 상태 출력
                # print(f"[DEBUG] 슈퍼픽셀 추가 후 basis - min: {basis[0].min()}, max: {basis[0].max()}, mean: {basis[0].mean()}")
                
                # 추가된 이미지로 모델 추론
                with torch.no_grad():
                    output = self.model(basis)
                    if class_idx is not None:
                        new_score = output[0, class_idx].item()
                    else:
                        new_score = output.max(1)[1].item()
                
                # print(f"[DEBUG] 순열 {perm_idx}, 단계 {idx}: 점수 변화 {basis_score} -> {new_score} (차이: {new_score - basis_score})")
                
                # 점수 변화를 해당 슈퍼픽셀의 SHAP 값에 누적
                shap_values[val] += new_score - basis_score
                basis_score = new_score  # 다음 단계를 위한 점수 업데이트
        
        shap_values /= self.sampling_size
        # print(f"[DEBUG] 최종 SHAP 값들: {shap_values}")
        
        heatmap = shap_values[labels]
        heatmap = process_heatmap_by_type(heatmap, self.type)
        
        # 최종 히트맵 결과 저장
        self._save_heatmap_result(heatmap, class_idx)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[PERMUTATION_SHAP] 종료 시간: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[PERMUTATION_SHAP] 총 소요 시간: {elapsed_time:.2f}초")
        
        return heatmap
    
    def _save_debug_image(self, image_tensor, debug_dir, filename, description):
        """디버깅용 이미지 저장"""
        try:
            # 텐서를 numpy로 변환
            if isinstance(image_tensor, torch.Tensor):
                image_np = image_tensor.detach().cpu().numpy()
            else:
                image_np = image_tensor
            
            # 채널이 3개인 경우 RGB로 처리, 1개인 경우 그레이스케일로 처리
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))
                
                # 고정된 정규화 범위 사용 (ImageNet 정규화 기준)
                # ImageNet은 보통 [-1, 1] 또는 [0, 1] 범위를 사용
                # 여기서는 [-3, 3] 범위로 정규화 (더 넓은 범위)
                image_np = np.clip(image_np, -3, 3)  # 범위 제한
                image_np = (image_np + 3) / 6  # [-3, 3] -> [0, 1]로 변환
                    
            elif image_np.ndim == 3 and image_np.shape[0] == 1:
                # (1, H, W) -> (H, W)
                image_np = image_np.squeeze(0)
                # 고정된 정규화 범위 사용
                image_np = np.clip(image_np, -3, 3)
                image_np = (image_np + 3) / 6
                    
            elif image_np.ndim == 2:
                # 이미 (H, W) 형태
                # 고정된 정규화 범위 사용
                image_np = np.clip(image_np, -3, 3)
                image_np = (image_np + 3) / 6
            
            plt.figure(figsize=(8, 6))
            if image_np.ndim == 3:
                plt.imshow(image_np)
            else:
                plt.imshow(image_np, cmap='gray')
            
            plt.axis('off')
            
            # 원본 텐서의 통계 정보도 표시
            if isinstance(image_tensor, torch.Tensor):
                orig_min = image_tensor.min().item()
                orig_max = image_tensor.max().item()
                orig_mean = image_tensor.mean().item()
                plt.title(f"{description}\n원본 - Min: {orig_min:.4f}, Max: {orig_max:.4f}, Mean: {orig_mean:.4f}\n정규화 후 - Min: {image_np.min():.4f}, Max: {image_np.max():.4f}, Mean: {image_np.mean():.4f}")
            else:
                plt.title(f"{description}\n정규화 후 - Min: {image_np.min():.4f}, Max: {image_np.max():.4f}, Mean: {image_np.mean():.4f}")
            
            filepath = os.path.join(debug_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            #print(f"[DEBUG] 디버깅 이미지 저장됨: {filepath}")
            #print(f"[DEBUG] 원본 텐서 - Min: {orig_min:.4f}, Max: {orig_max:.4f}, Mean: {orig_mean:.4f}")
            #print(f"[DEBUG] 정규화 후 - Min: {image_np.min():.4f}, Max: {image_np.max():.4f}, Mean: {image_np.mean():.4f}")
            
        except Exception as e:
            #print(f"[DEBUG] 디버깅 이미지 저장 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_heatmap_result(self, heatmap, class_idx=None):
        """최종 히트맵 결과를 numpy array로 저장"""
        try:
            # 클래스 라벨 가져오기 (class_idx가 있는 경우)
            class_label = self.class_map[class_idx]
            
            # 이미지명으로 디렉토리 생성
            dir_name = self.image_info
            
            # permute_cache 디렉토리 생성 및 저장 경로 설정
            cache_dir = os.path.join(self.samples_dir, "permute_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 최종 저장 디렉토리 생성
            result_dir = os.path.join(cache_dir, dir_name)
            os.makedirs(result_dir, exist_ok=True)
            
            # 파일명 생성 (target label로 저장)
            filename = f"{class_label}.npy" if class_label else "unknown.npy"
            filepath = os.path.join(result_dir, filename)
            
            # 원본 heatmap을 그대로 저장 (float64 타입 유지)
            np.save(filepath, heatmap)
            
            print(f"[PERMUTATION_SHAP] 히트맵 저장됨: {filepath}")
            print(f"[PERMUTATION_SHAP] 디렉토리 구조: {dir_name}/{filename}")
            print(f"[PERMUTATION_SHAP] 히트맵 통계 - Min: {heatmap.min():.4f}, Max: {heatmap.max():.4f}, Mean: {heatmap.mean():.4f}")
            print(f"[PERMUTATION_SHAP] 저장된 배열 shape: {heatmap.shape}, dtype: {heatmap.dtype}")
            
        except Exception as e:
            print(f"[PERMUTATION_SHAP] 히트맵 저장 실패: {e}")
            import traceback
            traceback.print_exc()
        
        
