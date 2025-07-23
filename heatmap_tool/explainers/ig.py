from configparser import InterpolationError
import torch
import torch.nn as nn
import torch.nn.functional as F
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import numpy as np
import os
from torchvision.utils import save_image

class IG(nn.Module):
    def __init__(self,model,config_dict):
        super(IG,self).__init__()
        self.model = model.eval()
        self.input_size = tuple(config_dict.get('input_size',(96,96)))
        self.steps = config_dict.get('steps',10)
        self.type = config_dict.get('type', 'both')  # type 추가
        # print("steps : ",self.steps)
        self.baseline = None
        self.feature_maps = None
        self.gradients = None
        self.slic = config_dict.get('slic',False)
        self.slic_size = config_dict.get('slic_size',10)
        self.slic_ruler = config_dict.get('slic_ruler',10)
        
    def make_baseline(self, input_tensor):
        # input_tensor: [1, C, H, W] 또는 [C, H, W]
        mean_val = input_tensor.mean()
        return torch.ones_like(input_tensor) * mean_val
    
    def _interpolate_input(self, input_tensor, baseline):
        """입력과 베이스라인 사이를 보간"""
        # 배치 차원 제거 (N=1일 때만)
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor[0]
            baseline = baseline[0]
        alphas = torch.linspace(0, 1, self.steps, device=input_tensor.device)
        interpolated_inputs = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated_inputs.append(interpolated)
        # [steps, C, H, W]로 반환
        return torch.stack(interpolated_inputs, dim=0)
    
    def generate(self, input_tensor, class_idx=None, save_debug=False, debug_dir="ig_debug"):
        
        # print(f"[IG] generate 시작")
        self.model.eval()
        self.model.zero_grad()
        self.baseline = self.make_baseline(input_tensor)
        # print(f"[IG] baseline shape: {self.baseline.shape}, min: {self.baseline.min()}, max: {self.baseline.max()}")

        # 디버그 폴더 생성
        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            # baseline 저장
            save_image(self.baseline, os.path.join(debug_dir, "baseline.png"), normalize=True)
            # print(f"[IG] baseline 저장 완료: {os.path.join(debug_dir, 'baseline.png')}")

        interpolated_inputs = self._interpolate_input(input_tensor, self.baseline)
        # print(f"[IG] interpolated_inputs shape: {interpolated_inputs.shape}")
        interpolated_inputs = interpolated_inputs.to(input_tensor.device)

        if save_debug:
            for idx, img in enumerate(interpolated_inputs):
                save_image(img, os.path.join(debug_dir, f"interpolated_{idx:02d}.png"), normalize=True)
            # print(f"[IG] interpolated_inputs 저장 완료")

        gradients_sum = torch.zeros_like(input_tensor)
        # print(f"[IG] gradients_sum 초기화 shape: {gradients_sum.shape}")
        for i in range(self.steps):
            # print(f"[IG] step {i+1}/{self.steps}")
            interpolated_input = interpolated_inputs[i].unsqueeze(0)  # [1, 3, 96, 96]
            interpolated_input = interpolated_input.clone().requires_grad_(True)
            output = self.model(interpolated_input)
            # print(f"[IG] output shape: {output.shape}, output 값: {output}")
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
                # print(f"[IG] class_idx 자동 선택: {class_idx}")
            score = output[:, class_idx]
            # print(f"[IG] score 값: {score}")
            # self.model.zero_grad()  # <-- 이 부분은 아래에서 설명
            score.backward()
            gradients = interpolated_input.grad
            if gradients is not None:
                # print(f"[IG] gradients shape: {gradients.shape}, min: {gradients.min()}, max: {gradients.max()}")
                gradients_sum += gradients
            else:
                # print("[IG] gradients가 None입니다!")
                raise RuntimeError("Gradients are not available")

        avg_gradients = gradients_sum / self.steps
        #print(f"[IG] avg_gradients shape: {avg_gradients.shape}, min: {avg_gradients.min()}, max: {avg_gradients.max()}")
        ig_attribution = (input_tensor - self.baseline) * avg_gradients
        #print(f"[IG] ig_attribution shape: {ig_attribution.shape}, min: {ig_attribution.min()}, max: {ig_attribution.max()}")
        ig_map = torch.mean(ig_attribution, dim=1)  # [1, H, W]
        #print(f"[IG] ig_map shape: {ig_map.shape}, min: {ig_map.min()}, max: {ig_map.max()}")

        ig_map = process_heatmap_by_type(ig_map.squeeze(), self.type)
        #print(f"[IG] process_heatmap_by_type 적용 후 ig_map shape: {ig_map.shape}, min: {ig_map.min()}, max: {ig_map.max()}")
        if self.slic is True:
            print("[IG] Slic Lets go")
            # input_tensor를 squeeze해서 2D 배열로 변환
            input_2d = input_tensor.squeeze()  # [C, H, W] 또는 [H, W]로 변환
            if input_2d.dim() == 3:  # [C, H, W]인 경우
                input_2d = input_2d.mean(dim=0)  # 채널 평균을 취해서 [H, W]로 변환
            print(f"[IG] input_tensor 원본 shape: {input_tensor.shape}")
            print(f"[IG] input_2d 변환 후 shape: {input_2d.shape}")
            
            # SLIC로 슈퍼픽셀 라벨과 경계선 추출
            labels, boundary = superpixel_mean_map(input_2d, region_size=self.slic_size, ruler=self.slic_ruler)
            print(f"[IG] labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
            print(f"[IG] boundary shape: {boundary.shape}")
            
            # ig_map을 numpy로 변환 (tensor인 경우)
            if hasattr(ig_map, 'cpu'):
                ig_map_np = ig_map.cpu().numpy()
            else:
                ig_map_np = ig_map
            print(f"[IG] ig_map_np shape: {ig_map_np.shape}, min: {ig_map_np.min()}, max: {ig_map_np.max()}")
            
            # 각 슈퍼픽셀별로 IG 기여도 평균 계산
            unique_labels = np.unique(labels)  # 모든 고유한 슈퍼픽셀 라벨들 (예: [0, 1, 2, 3, ...])
            num_superpixels = len(unique_labels)
            print(f"[IG] 슈퍼픽셀 개수: {num_superpixels}")
            
            # 슈퍼픽셀별 평균 기여도 계산
            superpixel_contributions = np.zeros(num_superpixels)  # 각 슈퍼픽셀의 평균 기여도를 저장할 배열
            for i, label in enumerate(unique_labels):
                # mask = labels == label: 현재 보고 있는 라벨과 같은 픽셀들만 True(1), 나머지는 False(0)
                # 예: label=2일 때, labels[y,x]==2인 픽셀들만 mask[y,x]=True
                mask = labels == label
                if np.any(mask):  # 해당 라벨의 픽셀이 하나라도 있는지 확인
                    # ig_map_np[mask]: mask가 True인 픽셀들의 IG 기여도만 추출
                    # .mean(): 추출된 기여도들의 평균을 계산
                    superpixel_contributions[i] = ig_map_np[mask].mean()
                    print(f"[IG] 슈퍼픽셀 {label}: 평균 기여도 = {superpixel_contributions[i]:.4f}, 픽셀 수 = {np.sum(mask)}")
            
            # 슈퍼픽셀별 기여도로 시각화 맵 생성
            slic_visualization = np.zeros_like(ig_map_np)  # 원본 IG 맵과 같은 크기의 빈 배열
            for i, label in enumerate(unique_labels):
                # 다시 같은 마스크 생성: 현재 라벨에 해당하는 픽셀들만 True
                mask = labels == label
                # slic_visualization[mask] = superpixel_contributions[i]: 
                # mask가 True인 모든 픽셀에 해당 슈퍼픽셀의 평균 기여도를 할당
                # 결과적으로 같은 슈퍼픽셀 내의 모든 픽셀이 동일한 평균 기여도 값을 가지게 됨
                slic_visualization[mask] = superpixel_contributions[i]
            
            print(f"[IG] slic_visualization shape: {slic_visualization.shape}, min: {slic_visualization.min()}, max: {slic_visualization.max()}")
            
            return slic_visualization  # 슈퍼픽셀별 평균 기여도 맵 반환
     
        return ig_map  # 1채널 맵 반환

    def set_config_dict(self, config_dict):
        """설정 업데이트"""
        self.steps = config_dict.get('steps', self.steps)
        self.input_size = config_dict.get('input_size', self.input_size)
        self.type = config_dict.get('type', self.type)  # type 추가
    