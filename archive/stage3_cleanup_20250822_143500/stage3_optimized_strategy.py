#!/usr/bin/env python3
"""
PillSnap Stage 3 최적화된 구성 전략
목표: 100,000 이미지, 1000 클래스 - 디스크 공간 최적화
"""

import json
import numpy as np
from collections import defaultdict

def load_analysis_results():
    """이전 분석 결과 로드"""
    with open('/home/max16/pillsnap/stage3_analysis_results.json', 'r') as f:
        return json.load(f)

def create_balanced_strategy(analysis_results, target_images=100000, target_classes=1000):
    """균형잡힌 Stage 3 구성 전략"""
    
    # 전체 클래스 정보
    all_classes = analysis_results['top_20_classes'] + analysis_results['bottom_20_classes']
    
    # 클래스별 이미지 수 재구성
    class_image_counts = {}
    
    # 상위 클래스들 (2592장씩)
    top_classes_count = 0
    current_images = 0
    
    print("=== Stage 3 최적화 전략 ===\n")
    
    # 전략 1: 균등 분배 (클래스당 100장)
    print("🎯 전략 1: 균등 분배")
    print(f"   - 1000개 클래스 × 100장 = 100,000장")
    print(f"   - 예상 용량: {(100000 * 0.15 / 1024):.1f}GB")
    print(f"   - 장점: 클래스 균형, 최소 용량")
    print(f"   - 단점: 상위 클래스 데이터 손실\n")
    
    # 전략 2: 가중 분배 (상위 클래스 더 많이)
    print("🎯 전략 2: 가중 분배")
    
    # 상위 500개 클래스: 평균 120장
    # 중위 300개 클래스: 평균 80장  
    # 하위 200개 클래스: 평균 60장
    weighted_images = (500 * 120) + (300 * 80) + (200 * 60)
    weighted_gb = (weighted_images * 0.15 / 1024)
    
    print(f"   - 상위 500개: 120장씩 = 60,000장")
    print(f"   - 중위 300개: 80장씩 = 24,000장") 
    print(f"   - 하위 200개: 60장씩 = 12,000장")
    print(f"   - 총합: {weighted_images:,}장")
    print(f"   - 예상 용량: {weighted_gb:.1f}GB")
    print(f"   - 장점: 중요 클래스 강조, 합리적 용량")
    print(f"   - 단점: 클래스 불균형\n")
    
    # 전략 3: 적응적 샘플링 (현재 디스크 고려)
    print("🎯 전략 3: 적응적 샘플링 (권장)")
    
    available_gb = 139  # 현재 사용 가능 공간
    safe_usage_gb = available_gb * 0.7  # 안전 사용량 (70%)
    max_images = int((safe_usage_gb * 1024) / 0.15)
    
    print(f"   - 사용 가능 공간: {available_gb}GB")
    print(f"   - 안전 사용량: {safe_usage_gb:.1f}GB")
    print(f"   - 최대 이미지: {max_images:,}장")
    
    if max_images >= 100000:
        adaptive_images = 100000
        adaptive_gb = (adaptive_images * 0.15 / 1024)
        print(f"   ✅ 목표 달성 가능: 100,000장")
    else:
        adaptive_images = max_images
        adaptive_gb = safe_usage_gb
        print(f"   ⚠️  목표 조정 필요: {adaptive_images:,}장")
    
    print(f"   - 실제 사용량: {adaptive_gb:.1f}GB")
    print(f"   - 클래스당 평균: {adaptive_images/1000:.0f}장")
    
    # 적응적 클래스 분배
    if adaptive_images >= 100000:
        tier1_classes = 400  # 상위 400개
        tier2_classes = 400  # 중위 400개  
        tier3_classes = 200  # 하위 200개
        
        tier1_per_class = 150  # 상위: 150장
        tier2_per_class = 75   # 중위: 75장
        tier3_per_class = 50   # 하위: 50장
        
        total_adaptive = (tier1_classes * tier1_per_class + 
                         tier2_classes * tier2_per_class + 
                         tier3_classes * tier3_per_class)
        
        print(f"\n   📊 적응적 분배:")
        print(f"      상위 {tier1_classes}개: {tier1_per_class}장씩 = {tier1_classes * tier1_per_class:,}장")
        print(f"      중위 {tier2_classes}개: {tier2_per_class}장씩 = {tier2_classes * tier2_per_class:,}장")
        print(f"      하위 {tier3_classes}개: {tier3_per_class}장씩 = {tier3_classes * tier3_per_class:,}장")
        print(f"      총합: {total_adaptive:,}장")
        
    print(f"\n   - 훈련/검증 분할: 8:2")
    print(f"   - 훈련 이미지: {int(adaptive_images * 0.8):,}장") 
    print(f"   - 검증 이미지: {int(adaptive_images * 0.2):,}장")
    
    # 전략 4: 단계별 확장
    print(f"\n🎯 전략 4: 단계별 확장")
    print(f"   Phase 1: 500 클래스, 50,000 이미지 (40GB)")
    print(f"   Phase 2: 750 클래스, 75,000 이미지 (60GB)")  
    print(f"   Phase 3: 1000 클래스, 100,000 이미지 (80GB)")
    print(f"   - 장점: 점진적 확장, 위험 최소화")
    print(f"   - 단점: 여러 단계 필요\n")
    
    # 최종 권장사항
    print("🏆 최종 권장사항:")
    
    if available_gb > 100:
        recommendation = "전략 3: 적응적 샘플링"
        print(f"   ✅ {recommendation}")
        print(f"   - 현재 디스크 여유 공간 충분 ({available_gb}GB)")
        print(f"   - Stage 3 목표 달성 가능")
        print(f"   - 클래스별 차등 분배로 효율 극대화")
        
        # 구체적 실행 계획
        print(f"\n📋 실행 계획:")
        print(f"   1. 상위 400개 클래스 선별 (이미지 수 기준)")
        print(f"   2. 각 클래스당 150장 샘플링")
        print(f"   3. 중위 400개 클래스 선별")  
        print(f"   4. 각 클래스당 75장 샘플링")
        print(f"   5. 하위 200개 클래스 선별")
        print(f"   6. 각 클래스당 50장 샘플링")
        print(f"   7. 8:2 훈련/검증 분할")
        print(f"   8. 데이터 증강 설정 (Albumentations)")
        
    else:
        recommendation = "전략 4: 단계적 확장"
        print(f"   ⚠️  {recommendation}")
        print(f"   - 디스크 여유 공간 부족 ({available_gb}GB)")
        print(f"   - 단계별 확장 권장")
        print(f"   - M.2 SSD 업그레이드 고려")
    
    return {
        'recommendation': recommendation,
        'available_space_gb': available_gb,
        'strategies': {
            'equal_distribution': {'images': 100000, 'size_gb': 14.6},
            'weighted_distribution': {'images': weighted_images, 'size_gb': weighted_gb},
            'adaptive_sampling': {'images': adaptive_images, 'size_gb': adaptive_gb},
            'phased_expansion': {'phase1': 50000, 'phase2': 75000, 'phase3': 100000}
        }
    }

def generate_class_selection_script(strategy_result):
    """클래스 선택 및 샘플링 스크립트 생성"""
    
    script_content = '''#!/usr/bin/env python3
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
    
    print(f"\\n총 {total_sampled:,}장 샘플링 완료")
    print(f"출력 경로: {output_path}")
    
    return total_sampled

if __name__ == "__main__":
    sample_classes_for_stage3()
'''
    
    script_path = "/home/max16/pillsnap/generate_stage3_dataset.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n📝 클래스 선택 스크립트 생성: {script_path}")
    print(f"실행 명령어: python {script_path}")

def main():
    """메인 함수"""
    # 분석 결과 로드
    analysis_results = load_analysis_results()
    
    # 최적화 전략 생성
    strategy_result = create_balanced_strategy(analysis_results)
    
    # 클래스 선택 스크립트 생성
    generate_class_selection_script(strategy_result)
    
    # 결과 저장
    output_file = "/home/max16/pillsnap/stage3_strategy_recommendation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(strategy_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 전략 결과 저장: {output_file}")

if __name__ == "__main__":
    main()