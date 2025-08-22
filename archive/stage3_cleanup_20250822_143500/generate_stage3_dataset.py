#!/usr/bin/env python3
"""
Stage 3 클래스 선택 및 샘플링 스크립트
자동 생성됨 - stage3_optimized_strategy.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def sample_classes_for_stage3():
    """Stage 3용 클래스 선택 및 샘플링"""
    
    # 데이터 경로
    source_paths = [
        "/home/max16/pillsnap_data/train/images/single",
        "/mnt/windows/pillsnap_data/train/images/single"
    ]
    
    output_path = "/home/max16/pillsnap_data/stage3"
    os.makedirs(output_path, exist_ok=True)
    
    # 클래스별 이미지 수 수집
    class_images = defaultdict(list)
    
    for source_path in source_paths:
        if not os.path.exists(source_path):
            continue
            
        for ts_folder in os.listdir(source_path):
            ts_path = os.path.join(source_path, ts_folder)
            if not os.path.isdir(ts_path):
                continue
                
            for class_folder in os.listdir(ts_path):
                if class_folder.startswith('K-'):
                    class_path = os.path.join(ts_path, class_folder)
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(class_path, img_file)
                                class_images[class_folder].append(img_path)
    
    # 클래스별 이미지 수로 정렬
    sorted_classes = sorted(class_images.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 적응적 샘플링 전략
    tier1_classes = sorted_classes[:400]  # 상위 400개
    tier2_classes = sorted_classes[400:800]  # 중위 400개
    tier3_classes = sorted_classes[800:1000]  # 하위 200개
    
    sampling_plan = [
        (tier1_classes, 150, "tier1"),
        (tier2_classes, 75, "tier2"), 
        (tier3_classes, 50, "tier3")
    ]
    
    total_sampled = 0
    
    for tier_classes, samples_per_class, tier_name in sampling_plan:
        tier_path = os.path.join(output_path, tier_name)
        os.makedirs(tier_path, exist_ok=True)
        
        for class_name, image_paths in tier_classes:
            class_output_path = os.path.join(tier_path, class_name)
            os.makedirs(class_output_path, exist_ok=True)
            
            # 샘플링
            if len(image_paths) >= samples_per_class:
                sampled_paths = random.sample(image_paths, samples_per_class)
            else:
                sampled_paths = image_paths  # 모든 이미지 사용
            
            # 복사
            for i, src_path in enumerate(sampled_paths):
                dst_path = os.path.join(class_output_path, f"{class_name}_{i:04d}.png")
                shutil.copy2(src_path, dst_path)
            
            total_sampled += len(sampled_paths)
            print(f"{tier_name}/{class_name}: {len(sampled_paths)}장 샘플링")
    
    print(f"\n총 {total_sampled:,}장 샘플링 완료")
    print(f"출력 경로: {output_path}")
    
    return total_sampled

if __name__ == "__main__":
    sample_classes_for_stage3()
