import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from PIL import Image
import cv2

def organize_heatmap_results(results_dir, output_dir):
    """
    히트맵 결과를 이미지별로 정리하고 시각화
    
    Args:
        results_dir: 원본 결과 디렉토리 경로
        output_dir: 정리된 결과를 저장할 디렉토리 경로
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 npy 파일 찾기
    npy_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_heatmap.npy'):
                npy_files.append(os.path.join(root, file))
    
    print(f"총 {len(npy_files)}개의 히트맵 파일 발견")
    
    # 파일명에서 이미지 번호와 클래스 추출
    image_data = {}
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        
        # img_{숫자}_{클래스}_heatmap.npy 패턴 매칭
        match = re.match(r'img_(\d+)_(dog|cat)_heatmap\.npy', filename)
        if match:
            img_num = match.group(1)
            class_name = match.group(2)
            
            if img_num not in image_data:
                image_data[img_num] = {}
            
            image_data[img_num][class_name] = npy_file
    
    print(f"총 {len(image_data)}개의 이미지 발견")
    
    # 각 이미지별로 처리
    for img_num, class_files in image_data.items():
        print(f"\n이미지 {img_num} 처리 중...")
        
        # 이미지별 디렉토리 생성
        img_dir = os.path.join(output_dir, f"img_{img_num}")
        os.makedirs(img_dir, exist_ok=True)
        
        # cat과 dog 히트맵 로드
        cat_heatmap = None
        dog_heatmap = None
        
        if 'cat' in class_files:
            cat_heatmap = np.load(class_files['cat'])
            print(f"  cat 히트맵 로드: {cat_heatmap.shape}")
        
        if 'dog' in class_files:
            dog_heatmap = np.load(class_files['dog'])
            print(f"  dog 히트맵 로드: {dog_heatmap.shape}")
        
        # 1. cat 히트맵 저장
        if cat_heatmap is not None:
            np.save(os.path.join(img_dir, f"img_{img_num}_cat_heatmap.npy"), cat_heatmap)
            
            # 2. cat 히트맵 흑백 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(cat_heatmap, cmap='gray')
            plt.title(f'Cat Heatmap - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_cat_gray.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # 3. cat 히트맵 jet 색상 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(cat_heatmap, cmap='jet')
            plt.colorbar(label='SHAP Value')
            plt.title(f'Cat Heatmap (Jet) - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_cat_jet.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 4. dog 히트맵 저장
        if dog_heatmap is not None:
            np.save(os.path.join(img_dir, f"img_{img_num}_dog_heatmap.npy"), dog_heatmap)
            
            # 5. dog 히트맵 흑백 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(dog_heatmap, cmap='gray')
            plt.title(f'Dog Heatmap - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_dog_gray.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # 6. dog 히트맵 jet 색상 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(dog_heatmap, cmap='jet')
            plt.colorbar(label='SHAP Value')
            plt.title(f'Dog Heatmap (Jet) - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_dog_jet.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 7. cat - dog 차이 히트맵 계산 및 저장
        if cat_heatmap is not None and dog_heatmap is not None:
            diff_heatmap = cat_heatmap - dog_heatmap
            np.save(os.path.join(img_dir, f"img_{img_num}_diff_heatmap.npy"), diff_heatmap)
            
            # 8. 차이 히트맵 흑백 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(diff_heatmap, cmap='gray')
            plt.title(f'Difference Heatmap (Cat-Dog) - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_diff_gray.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # 9. 차이 히트맵 jet 색상 시각화
            plt.figure(figsize=(8, 6))
            plt.imshow(diff_heatmap, cmap='jet')
            plt.colorbar(label='SHAP Difference')
            plt.title(f'Difference Heatmap (Cat-Dog, Jet) - Image {img_num}')
            plt.axis('off')
            plt.savefig(os.path.join(img_dir, f"img_{img_num}_diff_jet.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"  차이 히트맵 생성 완료")
        
        # JSON 파일도 복사 (있는 경우)
        json_file = os.path.join(results_dir, f"img_{img_num}_results.json")
        if os.path.exists(json_file):
            import shutil
            shutil.copy2(json_file, os.path.join(img_dir, f"img_{img_num}_results.json"))
            print(f"  JSON 파일 복사 완료")
        
        print(f"  이미지 {img_num} 처리 완료 - 총 9개 파일 생성")
    
    print(f"\n=== 정리 완료 ===")
    print(f"출력 디렉토리: {output_dir}")
    print(f"처리된 이미지: {len(image_data)}개")
    
    # 결과 요약
    total_files = 0
    for img_num in image_data.keys():
        img_dir = os.path.join(output_dir, f"img_{img_num}")
        if os.path.exists(img_dir):
            files_in_dir = len([f for f in os.listdir(img_dir) if f.endswith(('.npy', '.png', '.json'))])
            total_files += files_in_dir
            print(f"  img_{img_num}: {files_in_dir}개 파일")
    
    print(f"총 생성된 파일: {total_files}개")

def visualize_sample_results(output_dir, sample_count=5):
    """
    샘플 결과 시각화
    
    Args:
        output_dir: 정리된 결과 디렉토리
        sample_count: 시각화할 샘플 개수
    """
    
    # 이미지 디렉토리들 찾기
    img_dirs = [d for d in os.listdir(output_dir) if d.startswith('img_')]
    img_dirs.sort(key=lambda x: int(x.split('_')[1]))  # 숫자 순서로 정렬
    
    if not img_dirs:
        print("시각화할 이미지가 없습니다.")
        return
    
    # 샘플 개수만큼 선택
    sample_dirs = img_dirs[:sample_count]
    
    for img_dir_name in sample_dirs:
        img_dir = os.path.join(output_dir, img_dir_name)
        img_num = img_dir_name.split('_')[1]
        
        print(f"\n=== 이미지 {img_num} 시각화 ===")
        
        # 파일 목록 확인
        files = os.listdir(img_dir)
        print(f"파일 목록: {files}")
        
        # 히트맵 통계 출력
        cat_heatmap_file = os.path.join(img_dir, f"img_{img_num}_cat_heatmap.npy")
        dog_heatmap_file = os.path.join(img_dir, f"img_{img_num}_dog_heatmap.npy")
        diff_heatmap_file = os.path.join(img_dir, f"img_{img_num}_diff_heatmap.npy")
        
        if os.path.exists(cat_heatmap_file):
            cat_heatmap = np.load(cat_heatmap_file)
            print(f"Cat 히트맵 - Min: {cat_heatmap.min():.4f}, Max: {cat_heatmap.max():.4f}, Mean: {cat_heatmap.mean():.4f}")
        
        if os.path.exists(dog_heatmap_file):
            dog_heatmap = np.load(dog_heatmap_file)
            print(f"Dog 히트맵 - Min: {dog_heatmap.min():.4f}, Max: {dog_heatmap.max():.4f}, Mean: {dog_heatmap.mean():.4f}")
        
        if os.path.exists(diff_heatmap_file):
            diff_heatmap = np.load(diff_heatmap_file)
            print(f"Diff 히트맵 - Min: {diff_heatmap.min():.4f}, Max: {diff_heatmap.max():.4f}, Mean: {diff_heatmap.mean():.4f}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    RESULTS_DIR = "/content/batch_permutation_results"  # 원본 결과 디렉토리
    OUTPUT_DIR = "/content/organized_heatmap_results"   # 정리된 결과 디렉토리
    
    print("=== 히트맵 결과 정리 시작 ===")
    
    # 결과 정리
    organize_heatmap_results(RESULTS_DIR, OUTPUT_DIR)
    
    # 샘플 시각화
    visualize_sample_results(OUTPUT_DIR, sample_count=3)
    
    print("\n=== 완료 ===")
    print(f"정리된 결과 위치: {OUTPUT_DIR}") 