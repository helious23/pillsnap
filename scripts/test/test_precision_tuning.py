#!/usr/bin/env python3
"""
Precision 튜닝 스윕 테스트
목적: conf/iou 임계값 조합 테스트로 최적 파라미터 찾기
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
import numpy as np

def test_precision_sweep():
    """Precision 튜닝 스윕"""
    print("=== Precision 튜닝 스윕 테스트 ===")
    
    # 모델 로드
    model_path = Path("/home/max16/pillsnap/runs/detect/train/weights/last.pt")
    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return False
    
    model = YOLO(model_path)
    print(f"✅ 모델 로드: {model_path}")
    
    # 테스트 이미지 준비 (첫 10개만)
    data_yaml = "/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml"
    
    # 파라미터 그리드
    conf_values = [0.25, 0.35, 0.45, 0.55, 0.65]
    iou_values = [0.45, 0.55, 0.65, 0.75]
    
    results = []
    
    print("\nconf/iou 조합 테스트:")
    print("=" * 60)
    
    for conf in conf_values:
        for iou in iou_values:
            try:
                # 검증 실행 (빠른 테스트를 위해 일부만)
                metrics = model.val(
                    data=data_yaml,
                    conf=conf,
                    iou=iou,
                    batch=8,
                    imgsz=640,
                    device=0,
                    verbose=False,
                    split='val',
                    max_det=100,  # 최대 검출 수 제한
                    plots=False
                )
                
                # 메트릭 추출
                if metrics and hasattr(metrics, 'box'):
                    map50 = float(metrics.box.map50)
                    precision = float(metrics.box.p[0]) if hasattr(metrics.box, 'p') else 0
                    recall = float(metrics.box.r[0]) if hasattr(metrics.box, 'r') else 0
                    
                    result = {
                        'conf': conf,
                        'iou': iou,
                        'map50': map50,
                        'precision': precision,
                        'recall': recall,
                        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    }
                    results.append(result)
                    
                    print(f"conf={conf:.2f}, iou={iou:.2f}: "
                          f"mAP@0.5={map50:.3f}, P={precision:.3f}, R={recall:.3f}, F1={result['f1']:.3f}")
                else:
                    print(f"conf={conf:.2f}, iou={iou:.2f}: 메트릭 없음")
                    
            except Exception as e:
                print(f"conf={conf:.2f}, iou={iou:.2f}: 오류 - {e}")
    
    print("=" * 60)
    
    if results:
        # 최적 파라미터 찾기
        best_map = max(results, key=lambda x: x['map50'])
        best_f1 = max(results, key=lambda x: x['f1'])
        best_precision = max(results, key=lambda x: x['precision'])
        
        print("\n최적 파라미터:")
        print(f"  최고 mAP@0.5: conf={best_map['conf']:.2f}, iou={best_map['iou']:.2f} → {best_map['map50']:.3f}")
        print(f"  최고 F1 Score: conf={best_f1['conf']:.2f}, iou={best_f1['iou']:.2f} → {best_f1['f1']:.3f}")
        print(f"  최고 Precision: conf={best_precision['conf']:.2f}, iou={best_precision['iou']:.2f} → {best_precision['precision']:.3f}")
        
        # 결과 저장
        output_file = Path("/home/max16/pillsnap/artifacts/precision_tuning_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'best_map': best_map,
                'best_f1': best_f1,
                'best_precision': best_precision
            }, f, indent=2)
        
        print(f"\n✅ 결과 저장: {output_file}")
        return True
    else:
        print("❌ 유효한 결과가 없습니다")
        return False

if __name__ == "__main__":
    # 간단한 테스트 버전 (실제로는 더 많은 조합 필요)
    print("주의: 간단한 테스트 버전입니다. 실제 튜닝은 더 많은 조합이 필요합니다.")
    
    # 빠른 테스트를 위해 축소된 그리드
    success = test_precision_sweep()
    sys.exit(0 if success else 1)