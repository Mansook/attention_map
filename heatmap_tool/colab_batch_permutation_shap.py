import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from datetime import datetime
import json
from tqdm import tqdm
import hashlib
import hashlib

# 구글 드라이브 마운트 (코랩에서 실행 시)
from google.colab import drive
drive.mount('/content/drive')

# ResNet 모델 클래스 정의
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # 🔧 [1] conv1 stride=1로 변경 (기존 2 → 1)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=1,        # 👈 다운샘플링 완화
            padding=3,
            bias=False
        )
        
        # 🔧 [2] maxpool 제거
        self.backbone.maxpool = nn.Identity()
        
        # 🔧 [3] fc 레이어 조정
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# ===== 사용자 설정 =====

# 1. 이미지 폴더 경로 설정 (구글 드라이브 경로)
IMAGE_FOLDER_PATH = "/content/drive/MyDrive/data/test_images"  # 여기에 이미지 폴더 경로 입력

# 2. 타겟 클래스 설정 (STL-10 클래스)
TARGET_CLASSES = ["dog", "cat"]  # 분석할 클래스들
CLASS_MAP = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

# 3. 모델 파일 경로
MODEL_PATH = "/content/drive/MyDrive/data/stl10.pt"

# 4. Permutation SHAP 파라미터 설정
PERMUTATION_SHAP_CONFIG = {
    'type': 'both',  # 'absolute', 'relative', 또는 'both'
    'input_size': (96, 96),  # 입력 이미지 크기
    'sampling_size': 50,  # 샘플링 크기 (코랩에서는 작게 설정)
    'slic_size': 10,  # SLIC 슈퍼픽셀 크기
    'slic_ruler': 5   # SLIC ruler 파라미터
}

# 5. 이미지 전처리 설정
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. 결과 저장 설정
SAVE_RESULTS = True
RESULTS_DIR = "/content/batch_permutation_results"

print("=== 설정 확인 ===")
print(f"이미지 폴더: {IMAGE_FOLDER_PATH}")
print(f"타겟 클래스: {TARGET_CLASSES}")
print(f"모델 파일: {MODEL_PATH}")
print(f"Permutation SHAP 설정: {PERMUTATION_SHAP_CONFIG}")

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 모델 로드
print("모델 로드 중...")
model = ResNetClassifier(num_classes=10)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"모델 로드 완료: {MODEL_PATH}")
else:
    print(f"경고: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

model = model.to(device)
model.eval()

# SLIC 라이브러리 import
from explainers.utils.slic import superpixel_mean_map

# Permutation SHAP 클래스 (기존과 동일한 로직)
class PERMUTATION_SHAP(nn.Module):
    def __init__(self, model, config_dict):
        super(PERMUTATION_SHAP, self).__init__()
        self.model = model.eval()
        self.type = config_dict.get('type', 'absolute')
        self.input_size = config_dict.get('input_size', (96, 96))
        self.sampling_size = config_dict.get('sampling_size', 100)
        self.slic_size = config_dict.get('slic_size', 10)
        self.slic_ruler = config_dict.get('slic_ruler', 5)
        self.progress_callback = None
        self.samples_dir = "/content/"
        self.save_samples = True
        self.visualization_label = 'PERMUTATION_SHAP'
        self.class_map = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    
    def generate(self, input_tensor, class_idx=None):
        start_time = time.time()
        print(f"[PERMUTATION_SHAP] 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
        
        # input_tensor를 해시화하여 image_info 생성
        input_hash = hashlib.md5(input_tensor.cpu().numpy().tobytes()).hexdigest()[:8]
        self.image_info = f"image_{input_hash}"
        
        input_3d = input_tensor.squeeze()
        
        if input_3d.dim() == 2:
            input_3d = input_3d.unsqueeze(0)
        
        # SLIC로 슈퍼픽셀 라벨 추출
        labels, boundary = superpixel_mean_map(input_tensor, region_size=self.slic_size, ruler=self.slic_ruler)
        
        unique_labels = np.unique(labels)
        num_superpixels = len(unique_labels)
        
        # 미리 마스크 생성 (메모리 효율성)
        masks = {}
        for label in unique_labels:
            mask = (labels == label)
            masks[label] = mask
        
        # 배치 처리를 위한 마스크 텐서 미리 생성
        mask_tensors = {}
        for label, mask in masks.items():
            mask_3d = np.stack([mask] * input_3d.shape[0], axis=0)
            mask_tensors[label] = torch.from_numpy(mask_3d).bool()
        
        shap_values = np.zeros(num_superpixels)
        
        # 순열 생성
        permutation_list = [np.random.permutation(unique_labels) for _ in range(self.sampling_size)]
        basis = torch.zeros_like(input_3d.unsqueeze(0))
        basis_score = 0
        
        with torch.no_grad():
            # 베이스 점수 계산
            output = self.model(basis)
            if class_idx is not None:
                store_basis_score = output[0, class_idx].item()
            else:
                store_basis_score = output.max(1)[1].item()
        
        for perm_idx, permut in enumerate(permutation_list):
            # 각 순열마다 빈 이미지로 시작
            basis = torch.zeros_like(input_3d.unsqueeze(0))
            basis_score = store_basis_score  # 빈 이미지의 점수
            
            # 순열 순서대로 슈퍼픽셀을 하나씩 누적해서 추가
            for idx, val in enumerate(permut):
                progress_percent = int((perm_idx * num_superpixels + idx) / (self.sampling_size * num_superpixels) * 100)
                if self.progress_callback:
                    self.progress_callback(progress_percent, "PERMUT_SHAP")
                
                # 현재 슈퍼픽셀을 basis에 추가
                mask_tensor = mask_tensors[val]
                basis[0][mask_tensor] = input_3d[mask_tensor]
                
                # 추가된 이미지로 모델 추론
                with torch.no_grad():
                    output = self.model(basis)
                    if class_idx is not None:
                        new_score = output[0, class_idx].item()
                    else:
                        new_score = output.max(1)[1].item()
                
                # 점수 변화를 해당 슈퍼픽셀의 SHAP 값에 누적
                shap_values[val] += new_score - basis_score
                basis_score = new_score  # 다음 단계를 위한 점수 업데이트
        
        shap_values /= self.sampling_size
        
        heatmap = shap_values[labels]
        
        # 히트맵 정규화 (기존과 동일)
        if self.type == 'absolute':
            heatmap = np.abs(heatmap)
        elif self.type == 'relative':
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        elif self.type == 'both':
            # both는 아무 처리도 안함 (원본 값 그대로)
            pass
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[PERMUTATION_SHAP] 종료 시간: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[PERMUTATION_SHAP] 총 소요 시간: {elapsed_time:.2f}초")
        
        return heatmap

def scan_image_folders(root_path):
    """이미지 폴더를 스캔하여 처리할 폴더 목록을 반환"""
    folders = []
    
    if not os.path.exists(root_path):
        print(f"경고: 경로가 존재하지 않습니다: {root_path}")
        return folders
    
    # 먼저 root_path 자체에 이미지 파일이 있는지 확인
    root_image_files = [f for f in os.listdir(root_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if root_image_files:
        # root_path 자체가 이미지 폴더인 경우
        folders.append({
            'folder_name': os.path.basename(root_path),
            'folder_path': root_path,
            'image_files': root_image_files
        })
        print(f"루트 경로에서 {len(root_image_files)}개 이미지 발견")
    
    # 하위 폴더들도 확인
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            # 폴더 내 이미지 파일 확인
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if image_files:
                folders.append({
                    'folder_name': item,
                    'folder_path': item_path,
                    'image_files': image_files
                })
    
    return folders

def load_and_preprocess_image(image_path):
    """이미지를 로드하고 전처리"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 전처리
        image_tensor = IMAGE_TRANSFORMS(image)
        
        return image_tensor.unsqueeze(0)  # 배치 차원 추가
    except Exception as e:
        print(f"이미지 로드 실패 {image_path}: {e}")
        return None

def predict_class(image_tensor):
    """이미지의 클래스 예측"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

def run_permutation_shap_analysis(image_tensor, target_class_name, image_name):
    """Permutation SHAP 분석 수행"""
    try:
        # 타겟 클래스 인덱스 찾기
        target_class_idx = CLASS_MAP.index(target_class_name)
        
        # Permutation SHAP explainer 생성
        explainer = PERMUTATION_SHAP(model, PERMUTATION_SHAP_CONFIG)
        
        # 진행률 콜백 함수
        def progress_callback(percent, method):
            if percent % 20 == 0:  # 20%마다 출력
                print(f"  {method}: {percent}% 완료")
        
        explainer.progress_callback = progress_callback
        
        # SHAP 분석 수행
        print(f"    {target_class_name} 클래스에 대한 Permutation SHAP 분석 시작...")
        heatmap = explainer.generate(image_tensor, class_idx=target_class_idx)
        
        return heatmap
        
    except Exception as e:
        print(f"Permutation SHAP 분석 실패: {e}")
        return None

def save_analysis_results(folder_name, image_name, prediction_results, shap_results):
    """분석 결과 저장"""
    if not SAVE_RESULTS:
        return
    
    # 결과 디렉토리 생성
    results_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 데이터 준비
    result_data = {
        'image_name': image_name,
        'folder_name': folder_name,
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction_results,
        'shap_analysis': {}
    }
    
    # SHAP 결과 추가
    for class_name, heatmap in shap_results.items():
        if heatmap is not None:
            # 히트맵 저장
            heatmap_filename = f"{image_name}_{class_name}_heatmap.npy"
            heatmap_path = os.path.join(results_dir, heatmap_filename)
            np.save(heatmap_path, heatmap)
            
            result_data['shap_analysis'][class_name] = {
                'heatmap_file': heatmap_filename,
                'heatmap_stats': {
                    'min': float(heatmap.min()),
                    'max': float(heatmap.max()),
                    'mean': float(heatmap.mean()),
                    'std': float(heatmap.std())
                }
            }
    
    # JSON 결과 저장
    json_filename = f"{image_name}_results.json"
    json_path = os.path.join(results_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"    결과 저장됨: {json_path}")

# 이미지 폴더 스캔
image_folders = scan_image_folders(IMAGE_FOLDER_PATH)

print(f"=== 발견된 이미지 폴더 ({len(image_folders)}개) ===")
for folder in image_folders:
    print(f"- {folder['folder_name']}: {len(folder['image_files'])}개 이미지")

if not image_folders:
    print("처리할 이미지 폴더가 없습니다. 경로를 확인해주세요.")
else:
    # 전체 분석 시작
    print(f"\n=== Batch Permutation SHAP 분석 시작 ===")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 {len(image_folders)}개 폴더, {sum(len(f['image_files']) for f in image_folders)}개 이미지 처리 예정")
    print(f"타겟 클래스: {TARGET_CLASSES}")

    total_start_time = time.time()
    processed_count = 0
    success_count = 0

    # 각 폴더별로 처리
    for folder_idx, folder in enumerate(image_folders, 1):
        print(f"\n[{folder_idx}/{len(image_folders)}] 폴더 처리: {folder['folder_name']}")
        print(f"  이미지 개수: {len(folder['image_files'])}")
        
        folder_start_time = time.time()
        
        # 각 이미지별로 처리
        for img_idx, image_file in enumerate(folder['image_files'], 1):
            image_path = os.path.join(folder['folder_path'], image_file)
            image_name = os.path.splitext(image_file)[0]  # 확장자 제거
            
            print(f"\n  [{img_idx}/{len(folder['image_files'])}] 이미지 처리: {image_file}")
            
            # 이미지 로드 및 전처리
            image_tensor = load_and_preprocess_image(image_path)
            if image_tensor is None:
                print(f"    이미지 로드 실패, 건너뜀")
                continue
            
            # GPU로 이동
            image_tensor = image_tensor.to(device)
            
            # 클래스 예측
            predicted_class, confidence, all_probabilities = predict_class(image_tensor)
            predicted_class_name = CLASS_MAP[predicted_class]
            
            print(f"    예측 결과: {predicted_class_name} (확률: {confidence:.4f})")
            
            # 예측 결과 저장
            prediction_results = {
                'predicted_class': predicted_class_name,
                'predicted_class_idx': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities.tolist()
            }
            
            # 각 타겟 클래스에 대해 Permutation SHAP 분석
            shap_results = {}
            for target_class in TARGET_CLASSES:
                print(f"    {target_class} 클래스 SHAP 분석 중...")
                heatmap = run_permutation_shap_analysis(image_tensor, target_class, image_name)
                shap_results[target_class] = heatmap
            
            # 결과 저장
            save_analysis_results(folder['folder_name'], image_name, prediction_results, shap_results)
            
            processed_count += 1
            success_count += 1
            
            print(f"    완료: {image_file}")
        
        folder_end_time = time.time()
        folder_elapsed = folder_end_time - folder_start_time
        print(f"  폴더 완료: {folder['folder_name']} (소요시간: {folder_elapsed:.2f}초)")

    # 전체 완료
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    print(f"\n=== 분석 완료 ===")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {total_elapsed:.2f}초")
    print(f"처리된 이미지: {processed_count}개")
    print(f"성공한 이미지: {success_count}개")
    print(f"성공률: {(success_count/processed_count*100):.1f}%" if processed_count > 0 else "성공률: 0%")

    if SAVE_RESULTS:
        print(f"결과 저장 위치: {os.path.abspath(RESULTS_DIR)}")

# 결과 확인
if SAVE_RESULTS and os.path.exists(RESULTS_DIR):
    print("\n=== 저장된 결과 확인 ===")
    
    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if os.path.isdir(folder_path):
            print(f"\n폴더: {folder_name}")
            
            # JSON 파일들 확인
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            print(f"  JSON 결과 파일: {len(json_files)}개")
            
            # 히트맵 파일들 확인
            heatmap_files = [f for f in os.listdir(folder_path) if f.endswith('_heatmap.npy')]
            print(f"  히트맵 파일: {len(heatmap_files)}개")
            
            # 첫 번째 JSON 파일 내용 확인
            if json_files:
                first_json = os.path.join(folder_path, json_files[0])
                with open(first_json, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                print(f"  예시 결과:")
                print(f"    이미지: {result_data['image_name']}")
                print(f"    예측 클래스: {result_data['prediction']['predicted_class']}")
                print(f"    확률: {result_data['prediction']['confidence']:.4f}")
                print(f"    SHAP 분석 클래스: {list(result_data['shap_analysis'].keys())}")
else:
    print("결과가 저장되지 않았습니다.")

# 히트맵 시각화 함수 (선택사항)
def visualize_heatmap(folder_name, image_name, target_class):
    """저장된 히트맵을 시각화"""
    if not SAVE_RESULTS:
        print("결과가 저장되지 않았습니다.")
        return
    
    heatmap_path = os.path.join(RESULTS_DIR, folder_name, f"{image_name}_{target_class}_heatmap.npy")
    
    if not os.path.exists(heatmap_path):
        print(f"히트맵 파일을 찾을 수 없습니다: {heatmap_path}")
        return
    
    # 히트맵 로드
    heatmap = np.load(heatmap_path)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='RdBu_r')
    plt.colorbar(label='SHAP Value')
    plt.title(f'{image_name} - {target_class} 클래스 SHAP 히트맵')
    plt.axis('off')
    plt.show()
    
    print(f"히트맵 통계:")
    print(f"  Min: {heatmap.min():.4f}")
    print(f"  Max: {heatmap.max():.4f}")
    print(f"  Mean: {heatmap.mean():.4f}")
    print(f"  Std: {heatmap.std():.4f}")

# 예시: 첫 번째 폴더의 첫 번째 이미지 히트맵 시각화
if image_folders and SAVE_RESULTS:
    first_folder = image_folders[0]['folder_name']
    first_image = os.path.splitext(image_folders[0]['image_files'][0])[0]
    
    print(f"\n=== 히트맵 시각화 예시 ===")
    print(f"폴더: {first_folder}")
    print(f"이미지: {first_image}")
    
    for target_class in TARGET_CLASSES:
        visualize_heatmap(first_folder, first_image, target_class) 