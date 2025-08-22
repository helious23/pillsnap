#!/usr/bin/env python3
"""
PillSnap Stage 3 데이터셋 분석 스크립트
목표: 100,000 이미지, 1000 클래스 구성을 위한 데이터 현황 분석
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd

def analyze_class_distribution(data_paths):
    """클래스별 이미지 수 분포 분석"""
    class_stats = defaultdict(int)
    
    for data_path in data_paths:
        single_path = os.path.join(data_path, 'train/images/single')
        
        if not os.path.exists(single_path):
            print(f"경로 없음: {single_path}")
            continue
            
        print(f"분석 중: {single_path}")
        
        # TS_* 폴더들을 탐색
        for ts_folder in os.listdir(single_path):
            ts_path = os.path.join(single_path, ts_folder)
            if not os.path.isdir(ts_path):
                continue
            
            # K-* 클래스 폴더들을 탐색
            for class_folder in os.listdir(ts_path):
                if class_folder.startswith('K-'):
                    class_path = os.path.join(ts_path, class_folder)
                    if os.path.isdir(class_path):
                        image_count = len([f for f in os.listdir(class_path) 
                                         if f.endswith(('.png', '.jpg', '.jpeg'))])
                        class_stats[class_folder] += image_count
    
    return dict(class_stats)

def get_stage_1_2_classes():
    """Stage 1-2에서 검증된 클래스 목록 가져오기"""
    stage_1_2_classes = set()
    
    # Stage 1-2 보고서 확인
    report_paths = [
        "/home/max16/pillsnap_data/exp/exp01/reports/stage_1_evaluation.json",
        "/home/max16/pillsnap_data/exp/exp01/reports/stage_2_evaluation.json"
    ]
    
    for report_path in report_paths:
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    if 'classes' in report:
                        stage_1_2_classes.update(report['classes'])
                    print(f"Stage 보고서 로드: {report_path}")
            except Exception as e:
                print(f"보고서 로드 실패 {report_path}: {e}")
    
    return stage_1_2_classes

def recommend_stage3_classes(class_stats, target_classes=1000):
    """Stage 3용 1000개 클래스 선별 권장사항"""
    
    # 클래스별 이미지 수로 정렬 (내림차순)
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
    
    # Stage 1-2 검증 클래스
    validated_classes = get_stage_1_2_classes()
    
    # 선별 기준
    recommendations = {
        'by_image_count': [],
        'by_medical_priority': [],
        'by_validation_status': []
    }
    
    # 1. 이미지 수 기준 상위 1000개
    recommendations['by_image_count'] = [cls[0] for cls in sorted_classes[:target_classes]]
    
    # 2. 검증된 클래스 우선
    validated_in_data = [(cls, count) for cls, count in sorted_classes if cls in validated_classes]
    unvalidated_in_data = [(cls, count) for cls, count in sorted_classes if cls not in validated_classes]
    
    # 검증된 클래스 + 상위 미검증 클래스
    validation_priority = validated_in_data + unvalidated_in_data[:target_classes-len(validated_in_data)]
    recommendations['by_validation_status'] = [cls[0] for cls in validation_priority[:target_classes]]
    
    # 3. 의학적 중요도 (K-코드 기준 추정)
    # K-코드가 낮을수록 더 기본적/중요한 약물로 가정
    medical_priority = sorted(sorted_classes, key=lambda x: int(x[0].split('-')[1]))
    recommendations['by_medical_priority'] = [cls[0] for cls in medical_priority[:target_classes]]
    
    return recommendations, sorted_classes, validated_classes

def estimate_storage_requirements(selected_classes, class_stats):
    """선택된 클래스들의 스토리지 요구사항 추정"""
    
    total_images = sum(class_stats[cls] for cls in selected_classes if cls in class_stats)
    
    # PNG 이미지 평균 크기 추정 (200x200px 기준)
    avg_image_size_mb = 0.15  # 약 150KB per image
    total_size_gb = (total_images * avg_image_size_mb) / 1024
    
    # 훈련/검증 분할 고려 (8:2)
    train_images = int(total_images * 0.8)
    val_images = total_images - train_images
    
    return {
        'total_images': total_images,
        'train_images': train_images,
        'val_images': val_images,
        'estimated_size_gb': round(total_size_gb, 2),
        'classes': len(selected_classes)
    }

def main():
    """메인 분석 함수"""
    
    print("=== PillSnap Stage 3 데이터셋 분석 ===\n")
    
    # 데이터 경로 설정
    data_paths = [
        "/home/max16/pillsnap_data",  # Linux SSD
        "/mnt/windows/pillsnap_data"  # Windows SSD
    ]
    
    # 1. 전체 클래스 분포 분석
    print("1. 전체 클래스 분포 분석 중...")
    class_stats = analyze_class_distribution(data_paths)
    
    if not class_stats:
        print("❌ 클래스 데이터를 찾을 수 없습니다.")
        return
    
    total_classes = len(class_stats)
    total_images = sum(class_stats.values())
    
    print(f"✅ 전체 클래스 수: {total_classes:,}개")
    print(f"✅ 전체 이미지 수: {total_images:,}장\n")
    
    # 2. 통계 요약
    image_counts = list(class_stats.values())
    stats_summary = {
        'mean': np.mean(image_counts),
        'median': np.median(image_counts),
        'std': np.std(image_counts),
        'min': min(image_counts),
        'max': max(image_counts)
    }
    
    print("2. 클래스별 이미지 수 통계:")
    print(f"   평균: {stats_summary['mean']:.1f}장")
    print(f"   중앙값: {stats_summary['median']:.1f}장")
    print(f"   표준편차: {stats_summary['std']:.1f}")
    print(f"   최소: {stats_summary['min']}장")
    print(f"   최대: {stats_summary['max']}장\n")
    
    # 3. 상위/하위 클래스
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("3. 상위 20개 클래스 (이미지 수 기준):")
    for i, (cls, count) in enumerate(sorted_classes[:20], 1):
        print(f"   {i:2d}. {cls}: {count:,}장")
    
    print("\n하위 20개 클래스:")
    for i, (cls, count) in enumerate(sorted_classes[-20:], 1):
        print(f"   {i:2d}. {cls}: {count:,}장")
    
    # 4. Stage 3 권장사항
    print("\n4. Stage 3 구성 권장사항 (1000 클래스, 100K 이미지):")
    recommendations, _, validated_classes = recommend_stage3_classes(class_stats, 1000)
    
    print(f"   검증된 클래스 수: {len(validated_classes)}개")
    
    for strategy, classes in recommendations.items():
        storage_req = estimate_storage_requirements(classes, class_stats)
        print(f"\n   📋 {strategy} 전략:")
        print(f"      선택 클래스: {storage_req['classes']}개")
        print(f"      총 이미지: {storage_req['total_images']:,}장")
        print(f"      훈련용: {storage_req['train_images']:,}장")
        print(f"      검증용: {storage_req['val_images']:,}장")
        print(f"      예상 용량: {storage_req['estimated_size_gb']}GB")
    
    # 5. 권장 전략 선택
    print("\n5. 최종 권장 전략:")
    
    # 이미지 수와 검증 상태를 균형있게 고려한 전략
    final_strategy = recommendations['by_validation_status'][:1000]
    final_req = estimate_storage_requirements(final_strategy, class_stats)
    
    print(f"   🎯 검증 우선 + 이미지 수 기준")
    print(f"      - 검증된 클래스 우선 선택")
    print(f"      - 상위 이미지 수 클래스로 보완")
    print(f"      - 총 {final_req['classes']}개 클래스")
    print(f"      - 총 {final_req['total_images']:,}장 이미지")
    print(f"      - 예상 용량: {final_req['estimated_size_gb']}GB")
    
    # 6. 클래스당 평균 이미지 수
    avg_images_per_class = final_req['total_images'] / 1000
    print(f"      - 클래스당 평균: {avg_images_per_class:.0f}장")
    
    if avg_images_per_class < 80:
        print(f"      ⚠️  클래스당 이미지가 부족할 수 있음 (권장: 100장 이상)")
    elif avg_images_per_class >= 100:
        print(f"      ✅ 클래스당 이미지 수 충분")
    
    # 7. 디스크 공간 체크
    print(f"\n6. 스토리지 요구사항:")
    print(f"   현재 Linux SSD 여유 공간 확인 필요")
    print(f"   Stage 3 예상 용량: {final_req['estimated_size_gb']}GB")
    print(f"   여유 공간 권장: {final_req['estimated_size_gb'] * 1.5:.1f}GB (1.5배)")
    
    # 결과를 JSON으로 저장
    result = {
        'analysis_summary': {
            'total_classes': total_classes,
            'total_images': total_images,
            'statistics': stats_summary
        },
        'top_20_classes': sorted_classes[:20],
        'bottom_20_classes': sorted_classes[-20:],
        'recommendations': {
            strategy: {
                'classes': classes[:10],  # 상위 10개만 저장
                'storage_requirements': estimate_storage_requirements(classes, class_stats)
            }
            for strategy, classes in recommendations.items()
        },
        'final_recommendation': {
            'strategy': 'validation_priority_with_image_count',
            'selected_classes': final_strategy[:50],  # 상위 50개만 저장
            'requirements': final_req
        }
    }
    
    output_file = "/home/max16/pillsnap/stage3_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 상세 결과 저장: {output_file}")
    print(f"✅ Stage 3 데이터셋 분석 완료")

if __name__ == "__main__":
    main()