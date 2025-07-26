import torch
import torch.nn as nn
import numpy as np
from explainers.utils.normalize_heatmap import process_heatmap_by_type
from explainers.utils.slic import superpixel_mean_map
import random
import matplotlib
matplotlib.use('TkAgg')  # GUI 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns

class LMAP(nn.Module):
    def __init__(self, model, config_dict):
        super(LMAP, self).__init__()
        self.model = model.eval()
        self.type = config_dict.get('type', 'absolute')
        self.input_size = config_dict.get('input_size', (96, 96))
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 5)
        self.sampling_size = config_dict.get('sampling_size', 100)
        
        self.progress_callback = None
        
    def analyze_superpixel_logit_distributions(self, input_tensor, num_superpixels=10, class_idx=None):
        """
        몇 개 슈퍼픽셀에 대해 모든 마스킹 조합의 로짓 분포를 분석
        
        Args:
            input_tensor: 입력 텐서
            num_superpixels: 분석할 슈퍼픽셀 개수
            class_idx: 대상 클래스 인덱스
            
        Returns:
            분포 플롯
        """
        print(f"[LMAP] 슈퍼픽셀별 로짓 분포 분석 시작...")
        
        input_3d = input_tensor.squeeze()
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)
        
        # SLIC로 슈퍼픽셀 분할
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        unique_labels = np.unique(labels)
        num_total_superpixels = len(unique_labels)
        
        # 원본 로짓 추출
        with torch.no_grad():
            original_logits = self.model(input_3d.unsqueeze(0)).cpu().numpy().squeeze()
        
        if class_idx is None:
            class_idx = np.argmax(original_logits)
        
        # 정답 클래스와 오답 클래스 분리
        true_class = class_idx
        false_classes = [i for i in range(len(original_logits)) if i != true_class]
        
        print(f"[LMAP] 분석 대상 클래스: {class_idx}")
        print(f"[LMAP] 전체 슈퍼픽셀 개수: {num_total_superpixels}")
        print(f"[LMAP] 정답 클래스: {true_class}, 오답 클래스: {false_classes}")
        
        # 랜덤하게 슈퍼픽셀 선택
        selected_superpixels = random.sample(list(unique_labels), min(num_superpixels, num_total_superpixels))
        print(f"[LMAP] 선택된 슈퍼픽셀: {selected_superpixels}")
        
        # 각 선택된 슈퍼픽셀에 대해 모든 마스킹 조합의 로짓 분포 분석
        superpixel_logit_distributions = {}
        
        for sp_idx, superpixel_id in enumerate(selected_superpixels):
            print(f"[LMAP] 슈퍼픽셀 {sp_idx+1}/{len(selected_superpixels)} 분석 중... (ID: {superpixel_id})")
            
            # 모든 가능한 마스킹 조합 생성
            all_combinations = []
            all_logits = []
            
            # 샘플링 수를 늘려서 더 많은 조합 테스트
            sampling_size = min(1000, 2**num_total_superpixels)
            
            for i in range(sampling_size):
                # 랜덤 마스킹 조합 생성
                mask_combination = np.random.choice([0, 1], size=num_total_superpixels, p=[0.5, 0.5])
                
                # 마스킹된 이미지 생성
                masked_image = np.zeros_like(input_3d)
                for sp_idx_val, val in enumerate(mask_combination):
                    mask = labels == sp_idx_val
                    if val == 1:  # on인 경우
                        for c in range(masked_image.shape[0]):
                            masked_image[c][mask] = input_3d[c][mask]
                
                # 로짓 추출
                masked_tensor = torch.tensor(masked_image).unsqueeze(0).float().to(input_tensor.device)
                with torch.no_grad():
                    logit_vector = self.model(masked_tensor).cpu().numpy().squeeze()
                
                all_combinations.append(mask_combination)
                all_logits.append(logit_vector)
            
            # 해당 슈퍼픽셀이 on/off인 경우 분리
            sp_on_logits = []
            sp_off_logits = []
            
            for i, combination in enumerate(all_combinations):
                if combination[superpixel_id] == 1:  # 해당 슈퍼픽셀이 on
                    sp_on_logits.append(all_logits[i])
                else:  # off
                    sp_off_logits.append(all_logits[i])
            
            # 정답/오답 로짓 분리
            sp_on_true_logits = [logits[true_class] for logits in sp_on_logits]
            sp_off_true_logits = [logits[true_class] for logits in sp_off_logits]
            sp_on_false_logits = [logits[false_classes] for logits in sp_on_logits]
            sp_off_false_logits = [logits[false_classes] for logits in sp_off_logits]
            
            superpixel_logit_distributions[f"superpixel_{superpixel_id}"] = {
                'superpixel_id': superpixel_id,
                'on_true_logits': sp_on_true_logits,
                'off_true_logits': sp_off_true_logits,
                'on_false_logits': sp_on_false_logits,
                'off_false_logits': sp_off_false_logits,
                'on_true_mean': np.mean(sp_on_true_logits),
                'off_true_mean': np.mean(sp_off_true_logits),
                'on_false_mean': np.mean([np.mean(logits) for logits in sp_on_false_logits]),
                'off_false_mean': np.mean([np.mean(logits) for logits in sp_off_false_logits])
            }
        
        # 분포 시각화
        self._plot_superpixel_logit_distributions(superpixel_logit_distributions, class_idx)
        
        return superpixel_logit_distributions
    
    def _plot_superpixel_logit_distributions(self, superpixel_distributions, class_idx):
        """슈퍼픽셀별 로짓 분포를 시각화"""
        try:
            # matplotlib 설정
            plt.rcParams['figure.figsize'] = (20, 12)
            plt.rcParams['font.size'] = 10
            
            num_superpixels = len(superpixel_distributions)
            fig, axes = plt.subplots(2, num_superpixels, figsize=(4*num_superpixels, 8))
            
            if num_superpixels == 1:
                axes = axes.reshape(2, 1)
            
            for idx, (sp_name, data) in enumerate(superpixel_distributions.items()):
                sp_id = data['superpixel_id']
                
                # 히스토그램을 선으로 그리기
                # 정답 클래스 On 분포 (파란색 실선)
                on_true_hist, on_true_bins, _ = axes[0, idx].hist(data['on_true_logits'], bins=20, alpha=0, density=True)
                on_true_bin_centers = (on_true_bins[:-1] + on_true_bins[1:]) / 2
                axes[0, idx].plot(on_true_bin_centers, on_true_hist, 'b-', linewidth=1, label=f'정답 On (μ={data["on_true_mean"]:.3f})')
                
                # 정답 클래스 Off 분포 (파란색 점선)
                off_true_hist, off_true_bins, _ = axes[0, idx].hist(data['off_true_logits'], bins=20, alpha=0, density=True)
                off_true_bin_centers = (off_true_bins[:-1] + off_true_bins[1:]) / 2
                axes[0, idx].plot(off_true_bin_centers, off_true_hist, 'b--', linewidth=1, label=f'정답 Off (μ={data["off_true_mean"]:.3f})')
                
                # 오답 클래스 On 분포 (주황색 실선)
                on_false_means = [np.mean(logits) for logits in data['on_false_logits']]
                on_false_hist, on_false_bins, _ = axes[0, idx].hist(on_false_means, bins=20, alpha=0, density=True)
                on_false_bin_centers = (on_false_bins[:-1] + on_false_bins[1:]) / 2
                axes[0, idx].plot(on_false_bin_centers, on_false_hist, 'orange', linewidth=1, label=f'오답 On (μ={data["on_false_mean"]:.3f})')
                
                # 오답 클래스 Off 분포 (주황색 점선)
                off_false_means = [np.mean(logits) for logits in data['off_false_logits']]
                off_false_hist, off_false_bins, _ = axes[0, idx].hist(off_false_means, bins=20, alpha=0, density=True)
                off_false_bin_centers = (off_false_bins[:-1] + off_false_bins[1:]) / 2
                axes[0, idx].plot(off_false_bin_centers, off_false_hist, 'orange', linestyle='--', linewidth=1, label=f'오답 Off (μ={data["off_false_mean"]:.3f})')
                
                axes[0, idx].set_title(f'슈퍼픽셀 {sp_id}')
                axes[0, idx].set_xlabel('Logit 값')
                axes[0, idx].set_ylabel('밀도')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
                
                # 박스플롯
                box_data = [data['on_true_logits'], data['off_true_logits'], 
                           on_false_means, off_false_means]
                bp = axes[1, idx].boxplot(box_data, labels=['정답 On', '정답 Off', '오답 On', '오답 Off'], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightblue')
                bp['boxes'][2].set_facecolor('lightcoral')
                bp['boxes'][3].set_facecolor('lightcoral')
                axes[1, idx].set_title(f'슈퍼픽셀 {sp_id} 분포 비교')
                axes[1, idx].set_ylabel('Logit 값')
                axes[1, idx].grid(True, alpha=0.3)
            
            plt.suptitle(f'클래스 {class_idx}에 대한 슈퍼픽셀별 로짓 분포 분석', fontsize=16)
            plt.tight_layout()
            
            # 현재 작업 디렉토리에 저장
            import os
            save_dir = r"C:\Users\orgin\XAI-study\heatmap_tool\cache"
            save_path = os.path.join(save_dir, 'superpixel_logit_distributions.png')
            
            # 디렉토리가 없으면 생성
            try:
                os.makedirs(save_dir, exist_ok=True)
                print(f"[LMAP] 디렉토리 생성/확인 완료: {save_dir}")
                
                # 파일 저장 시도
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[LMAP] 플롯이 저장되었습니다: {save_path}")
                print(f"[LMAP] 저장 디렉토리: {save_dir}")
                
                # 파일이 실제로 생성되었는지 확인
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"[LMAP] 파일 생성 확인: {save_path} (크기: {file_size} bytes)")
                else:
                    print(f"[LMAP] 경고: 파일이 생성되지 않았습니다: {save_path}")
                    
            except Exception as e:
                print(f"[LMAP] 파일 저장 중 오류 발생: {e}")
                print(f"[LMAP] 현재 작업 디렉토리: {os.getcwd()}")
                print(f"[LMAP] 저장 시도 경로: {save_path}")
                
                # 대안: 현재 디렉토리에 저장
                try:
                    alt_path = 'superpixel_logit_distributions.png'
                    plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                    print(f"[LMAP] 대안 경로에 저장 완료: {alt_path}")
                except Exception as e2:
                    print(f"[LMAP] 대안 저장도 실패: {e2}")
            
            # 플롯 표시
            plt.show()
            
        except Exception as e:
            print(f"[LMAP] 플롯 생성 중 오류 발생: {e}")
            print(f"[LMAP] 백엔드 정보: {matplotlib.get_backend()}")
            
            # 대안: 텍스트 기반 출력
            self._print_superpixel_text_summary(superpixel_distributions, class_idx)
    
    def _print_superpixel_text_summary(self, superpixel_distributions, class_idx):
        """슈퍼픽셀별 텍스트 기반 요약 출력"""
        print(f"\n[LMAP] 클래스 {class_idx}에 대한 슈퍼픽셀별 로짓 분포 분석 (텍스트 요약)")
        print("=" * 80)
        
        for sp_name, data in superpixel_distributions.items():
            sp_id = data['superpixel_id']
            
            print(f"\n{sp_name} - 슈퍼픽셀 ID: {sp_id}")
            print(f"  정답 On:   평균={data['on_true_mean']:.4f}, 샘플수={len(data['on_true_logits'])}")
            print(f"  정답 Off:  평균={data['off_true_mean']:.4f}, 샘플수={len(data['off_true_logits'])}")
            print(f"  오답 On:   평균={data['on_false_mean']:.4f}, 샘플수={len(data['on_false_logits'])}")
            print(f"  오답 Off:  평균={data['off_false_mean']:.4f}, 샘플수={len(data['off_false_logits'])}")
            
            # 정답 기여도
            true_contribution = data['on_true_mean'] - data['off_true_mean']
            false_contribution = data['on_false_mean'] - data['off_false_mean']
            print(f"  정답 기여도: {true_contribution:.4f}")
            print(f"  오답 기여도: {false_contribution:.4f}")
        
        # 통계 요약 테이블
        print(f"\n[LMAP] 슈퍼픽셀별 로짓 분포 통계 요약:")
        print(f"{'슈퍼픽셀':<15} {'정답 On':<10} {'정답 Off':<10} {'오답 On':<10} {'오답 Off':<10} {'정답 기여':<10}")
        print("-" * 70)
        
        for sp_name, data in superpixel_distributions.items():
            true_contribution = data['on_true_mean'] - data['off_true_mean']
            print(f"{sp_name:<15} {data['on_true_mean']:<10.3f} {data['off_true_mean']:<10.3f} "
                  f"{data['on_false_mean']:<10.3f} {data['off_false_mean']:<10.3f} {true_contribution:<10.3f}")
        
        return superpixel_distributions
        
    # LMAP generate 함수 수정본 - Likelihood & Prior 기반 MAP 히트맵

    def generate(self, input_tensor, class_idx=None):
        """
        LMAP (Logit-based MAP) 히트맵 생성
        - 각 슈퍼픽셀에 대해: Likelihood * Prior 기반 기여도 계산
        - SHAP 합산 불변성 정규화 적용
        """
        input_3d = input_tensor.squeeze()
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)

        labels, _ = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        unique_labels = np.unique(labels)
        num_superpixels = len(unique_labels)

        with torch.no_grad():
            original_logits = self.model(input_3d.unsqueeze(0)).cpu().numpy().squeeze()

        if class_idx is None:
            class_idx = np.argmax(original_logits)
        true_class = class_idx
        false_classes = [i for i in range(len(original_logits)) if i != true_class]

        logits_list = []
        mask_combinations = []

        for i in range(self.sampling_size):
            
            progress_percent = int((i / self.sampling_size) * 100)
            if self.progress_callback:
                self.progress_callback(progress_percent, "SHAP")
            index = np.zeros(num_superpixels)
            random_integer = random.randint(0, num_superpixels-1)
            if random_integer > 0:
                ones_indices = random.sample(range(num_superpixels), random_integer)
                index[ones_indices] = 1
            mask_combinations.append(index.copy())

            masked_image = np.zeros_like(input_3d)
            for idx, val in enumerate(index):
                mask = labels == idx
                if val == 1:
                    for c in range(masked_image.shape[0]):
                        masked_image[c][mask] = input_3d[c][mask]

            masked_tensor = torch.tensor(masked_image).unsqueeze(0).float().to(input_tensor.device)
            with torch.no_grad():
                logit_vector = self.model(masked_tensor)
            logits_list.append(logit_vector.cpu().numpy())

        logits_array = np.concatenate(logits_list, axis=0)
        mask_combinations = np.array(mask_combinations)

        superpixel_stats = []
        for sp_idx in range(num_superpixels):
            on_mask = mask_combinations[:, sp_idx] == 1
            off_mask = ~on_mask
            on_logits = logits_array[on_mask]
            off_logits = logits_array[off_mask]

            stats = {}

            # True / False 클래스별 로짓 분포
            stats['on_true'] = on_logits[:, true_class] if len(on_logits) > 0 else np.array([])
            stats['off_true'] = off_logits[:, true_class] if len(off_logits) > 0 else np.array([])
            stats['on_false'] = on_logits[:, false_classes] if len(on_logits) > 0 else np.array([])
            stats['off_false'] = off_logits[:, false_classes] if len(off_logits) > 0 else np.array([])

            # Gap 계산
            stats['on_gap'] = stats['on_true'] - np.max(stats['on_false'], axis=1) if len(stats['on_true']) > 0 else np.array([])
            stats['off_gap'] = stats['off_true'] - np.max(stats['off_false'], axis=1) if len(stats['off_true']) > 0 else np.array([])

            superpixel_stats.append(stats)

        raw_contributions = []

        for stats in superpixel_stats:
            # Likelihood (1): 정답 로짓 차이
            L1 = np.mean(stats['on_true']) - np.mean(stats['off_true']) if len(stats['on_true']) > 0 and len(stats['off_true']) > 0 else 0

            # Likelihood (2): on_true - on_false 평균 차이
            L2 = np.mean(stats['on_true']) - np.mean(stats['on_false']) if len(stats['on_true']) > 0 and len(stats['on_false']) > 0 else 0

            # Likelihood (3): off_true - off_false 평균 차이 (작을수록 좋음)
            L3 = - (np.mean(stats['off_true']) - np.mean(stats['off_false'])) if len(stats['off_true']) > 0 and len(stats['off_false']) > 0 else 0

            # Likelihood (4): on_gap - off_gap
            L4 = np.mean(stats['on_gap']) - np.mean(stats['off_gap']) if len(stats['on_gap']) > 0 and len(stats['off_gap']) > 0 else 0

            # Prior (confidence): 분산 기반 신뢰도
            var_sum = np.var(stats['on_true']) + np.var(stats['off_true']) if len(stats['on_true']) > 1 and len(stats['off_true']) > 1 else 1
            prior = 1 / (1 + var_sum)  # 분산이 작을수록 prior 증가

            # MAP 추정 (prior * likelihood의 soft voting)
            contribution = prior * (0.35 * L1 + 0.25 * L2 + 0.25 * L3 + 0.15 * L4)
            raw_contributions.append(contribution)

        raw_contributions = np.array(raw_contributions)

        # SHAP 스타일 정규화
        baseline_tensor = torch.zeros_like(input_3d).unsqueeze(0).float().to(input_tensor.device)
        with torch.no_grad():
            baseline_logits = self.model(baseline_tensor).cpu().numpy().squeeze()
        baseline_prediction = baseline_logits[true_class]
        original_prediction = original_logits[true_class]
        target_sum = original_prediction - baseline_prediction

        if np.sum(raw_contributions) != 0:
            normalized_contributions = raw_contributions * (target_sum / np.sum(raw_contributions))
        else:
            normalized_contributions = raw_contributions

        heatmap = np.zeros_like(labels, dtype=np.float32)
        for idx, sp_idx in enumerate(unique_labels):
            heatmap[labels == sp_idx] = normalized_contributions[idx]

        heatmap = process_heatmap_by_type(heatmap, self.type)

        if self.progress_callback:
            self.progress_callback(100, "LMAP")

        # (Optional) VISUALIZATION: 선택된 슈퍼픽셀들의 로짓 분포 시각화 (원본 유지)
        # self._plot_superpixel_logit_distributions(...)

        return heatmap

