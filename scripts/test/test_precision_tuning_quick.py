#!/usr/bin/env python3
"""
Precision 튜닝 스윕 테스트 (빠른 버전)
목적: conf/iou 임계값 시뮬레이션
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
import json
from datetime import datetime
import numpy as np

def simulate_precision_sweep():
    """Precision 튜닝 시뮬레이션 (실제 검증 없이)"""
    print("=== Precision 튜닝 스윕 시뮬레이션 ===")
    
    # 파라미터 그리드
    conf_values = [0.25, 0.35, 0.45, 0.55, 0.65]
    iou_values = [0.45, 0.55, 0.65, 0.75]
    
    # 기본 메트릭 (results.csv에서 가져온 값)
    base_map = 0.34959
    base_precision = 0.31264
    base_recall = 0.78733
    
    results = []
    
    print("\nconf/iou 조합 시뮬레이션:")
    print("=" * 60)
    
    for conf in conf_values:
        for iou in iou_values:
            # conf가 높을수록 precision 증가, recall 감소
            # iou가 높을수록 precision 증가, recall 감소
            
            # 시뮬레이션 공식
            precision_factor = 1 + (conf - 0.45) * 0.3 + (iou - 0.6) * 0.2
            recall_factor = 1 - (conf - 0.45) * 0.4 - (iou - 0.6) * 0.3
            
            precision = min(1.0, base_precision * precision_factor)
            recall = max(0.1, base_recall * recall_factor)
            
            # mAP는 precision과 recall의 조화평균에 비례
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            map50 = base_map * (0.7 + 0.3 * f1)  # F1에 비례하여 조정
            
            result = {
                'conf': conf,
                'iou': iou,
                'map50': map50,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            results.append(result)
            
            print(f"conf={conf:.2f}, iou={iou:.2f}: "
                  f"mAP@0.5={map50:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    print("=" * 60)
    
    # 최적 파라미터 찾기
    best_map = max(results, key=lambda x: x['map50'])
    best_f1 = max(results, key=lambda x: x['f1'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])
    
    print("\n최적 파라미터 (시뮬레이션):")
    print(f"  최고 mAP@0.5: conf={best_map['conf']:.2f}, iou={best_map['iou']:.2f} → {best_map['map50']:.3f}")
    print(f"  최고 F1 Score: conf={best_f1['conf']:.2f}, iou={best_f1['iou']:.2f} → {best_f1['f1']:.3f}")
    print(f"  최고 Precision: conf={best_precision['conf']:.2f}, iou={best_precision['iou']:.2f} → {best_precision['precision']:.3f}")
    print(f"  최고 Recall: conf={best_recall['conf']:.2f}, iou={best_recall['iou']:.2f} → {best_recall['recall']:.3f}")
    
    print("\n제안:")
    print(f"  균형잡힌 성능: conf=0.45, iou=0.60 (기본값)")
    print(f"  Precision 우선: conf={best_precision['conf']:.2f}, iou={best_precision['iou']:.2f}")
    print(f"  Recall 우선: conf={best_recall['conf']:.2f}, iou={best_recall['iou']:.2f}")
    
    # 결과 저장
    output_file = Path("/home/max16/pillsnap/artifacts/precision_tuning_simulation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'type': 'simulation',
            'results': results,
            'best_map': best_map,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall
        }, f, indent=2)
    
    print(f"\n✅ 시뮬레이션 결과 저장: {output_file}")
    return True

if __name__ == "__main__":
    success = simulate_precision_sweep()
    sys.exit(0 if success else 1)