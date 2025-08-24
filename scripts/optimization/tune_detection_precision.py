#!/usr/bin/env python3
"""
Detection Precision 튜닝 유틸리티
conf/iou 파라미터 스윕을 통한 최적 값 탐색
"""

import sys
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
import subprocess
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class DetectionPrecisionTuner:
    """Detection Precision 튜너"""
    
    def __init__(self):
        self.yolo_dir = Path("/home/max16/pillsnap/artifacts/yolo/stage3")
        self.best_pt = self.yolo_dir / "weights" / "best.pt"
        self.test_manifest = Path("/home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv")
        
    def sweep_confidence_threshold(self, conf_values: List[float] = None) -> Dict[float, Dict]:
        """Confidence threshold 스윕"""
        
        if conf_values is None:
            # 기본값: 0.1 ~ 0.9 (0.1 간격)
            conf_values = np.arange(0.1, 1.0, 0.1)
        
        results = {}
        
        for conf in conf_values:
            logger.info(f"\n테스트 중: conf={conf:.1f}")
            metrics = self.evaluate_with_params(conf=conf, iou=0.5)
            results[conf] = metrics
            
            # 결과 출력
            logger.info(f"  mAP@0.5: {metrics['map50']:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall: {metrics['recall']:.3f}")
            
            time.sleep(1)  # GPU 쿨다운
        
        return results
    
    def sweep_iou_threshold(self, iou_values: List[float] = None) -> Dict[float, Dict]:
        """IoU threshold 스윕"""
        
        if iou_values is None:
            # 기본값: 0.3 ~ 0.7 (0.1 간격)
            iou_values = np.arange(0.3, 0.8, 0.1)
        
        results = {}
        
        for iou in iou_values:
            logger.info(f"\n테스트 중: iou={iou:.1f}")
            metrics = self.evaluate_with_params(conf=0.25, iou=iou)
            results[iou] = metrics
            
            # 결과 출력
            logger.info(f"  mAP@0.5: {metrics['map50']:.3f}")
            logger.info(f"  Precision: {metrics['precision']:.3f}")
            logger.info(f"  Recall: {metrics['recall']:.3f}")
            
            time.sleep(1)  # GPU 쿨다운
        
        return results
    
    def evaluate_with_params(self, conf: float, iou: float) -> Dict[str, float]:
        """특정 파라미터로 평가"""
        
        if not self.best_pt.exists():
            logger.error(f"best.pt 파일이 없습니다: {self.best_pt}")
            return {'map50': 0, 'precision': 0, 'recall': 0}
        
        # YOLO 평가 실행
        cmd = [
            sys.executable, '-c',
            f"""
import sys
sys.path.insert(0, '/home/max16/pillsnap')
from ultralytics import YOLO

model = YOLO('{self.best_pt}')

# 검증 실행
results = model.val(
    data='/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml',
    batch=8,
    imgsz=640,
    conf={conf},
    iou={iou},
    device='cuda',
    verbose=False,
    plots=False
)

# 메트릭 출력
if hasattr(results, 'results_dict'):
    metrics = results.results_dict
    print(f"MAP50: {{metrics.get('metrics/mAP50(B)', 0):.4f}}")
    print(f"PRECISION: {{metrics.get('metrics/precision(B)', 0):.4f}}")
    print(f"RECALL: {{metrics.get('metrics/recall(B)', 0):.4f}}")
else:
    print("MAP50: 0.0000")
    print("PRECISION: 0.0000")
    print("RECALL: 0.0000")
"""
        ]
        
        try:
            # subprocess 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 출력 파싱
            output = result.stdout
            
            map50 = 0.0
            precision = 0.0
            recall = 0.0
            
            for line in output.split('\n'):
                if 'MAP50:' in line:
                    map50 = float(line.split(':')[1])
                elif 'PRECISION:' in line:
                    precision = float(line.split(':')[1])
                elif 'RECALL:' in line:
                    recall = float(line.split(':')[1])
            
            return {
                'map50': map50,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            logger.error(f"평가 실패: {e}")
            return {'map50': 0, 'precision': 0, 'recall': 0}
    
    def find_optimal_params(self) -> Tuple[float, float]:
        """최적 파라미터 찾기"""
        
        logger.info("=" * 60)
        logger.info("Precision 튜닝 시작")
        logger.info("=" * 60)
        
        # 1단계: Confidence threshold 스윕
        logger.info("\n[1단계] Confidence Threshold 스윕")
        conf_results = self.sweep_confidence_threshold([0.1, 0.25, 0.4, 0.5, 0.6, 0.7])
        
        # Precision이 가장 높은 conf 찾기
        best_conf = max(conf_results.keys(), key=lambda k: conf_results[k]['precision'])
        logger.info(f"\n최적 Confidence: {best_conf:.2f} (Precision: {conf_results[best_conf]['precision']:.3f})")
        
        # 2단계: IoU threshold 스윕
        logger.info("\n[2단계] IoU Threshold 스윕")
        iou_results = self.sweep_iou_threshold([0.3, 0.4, 0.5, 0.6, 0.7])
        
        # mAP가 가장 높은 iou 찾기
        best_iou = max(iou_results.keys(), key=lambda k: iou_results[k]['map50'])
        logger.info(f"\n최적 IoU: {best_iou:.2f} (mAP@0.5: {iou_results[best_iou]['map50']:.3f})")
        
        # 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("튜닝 결과 요약")
        logger.info("=" * 60)
        logger.info(f"최적 Confidence Threshold: {best_conf:.2f}")
        logger.info(f"최적 IoU Threshold: {best_iou:.2f}")
        
        # 최적 파라미터로 최종 평가
        logger.info("\n최적 파라미터로 최종 평가...")
        final_metrics = self.evaluate_with_params(conf=best_conf, iou=best_iou)
        
        logger.info(f"\n최종 성능:")
        logger.info(f"  mAP@0.5: {final_metrics['map50']:.3f}")
        logger.info(f"  Precision: {final_metrics['precision']:.3f}")
        logger.info(f"  Recall: {final_metrics['recall']:.3f}")
        
        # 결과를 파일로 저장
        self.save_tuning_results(best_conf, best_iou, final_metrics)
        
        return best_conf, best_iou
    
    def save_tuning_results(self, best_conf: float, best_iou: float, metrics: Dict):
        """튜닝 결과 저장"""
        
        result_file = self.yolo_dir / "precision_tuning_results.txt"
        
        with open(result_file, 'w') as f:
            f.write("Detection Precision 튜닝 결과\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"최적 Confidence Threshold: {best_conf:.2f}\n")
            f.write(f"최적 IoU Threshold: {best_iou:.2f}\n\n")
            f.write("최종 성능:\n")
            f.write(f"  mAP@0.5: {metrics['map50']:.3f}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n\n")
            f.write(f"생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"\n✅ 결과 저장: {result_file}")

def main():
    """메인 함수"""
    
    tuner = DetectionPrecisionTuner()
    
    # best.pt 존재 확인
    if not tuner.best_pt.exists():
        logger.error(f"best.pt 파일이 없습니다. Detection 학습을 먼저 실행하세요.")
        logger.error(f"확인 경로: {tuner.best_pt}")
        return
    
    # 최적 파라미터 찾기
    best_conf, best_iou = tuner.find_optimal_params()
    
    logger.info("\n" + "=" * 60)
    logger.info("튜닝 완료!")
    logger.info("=" * 60)
    logger.info(f"다음 학습에서 사용할 파라미터:")
    logger.info(f"  --conf {best_conf:.2f}")
    logger.info(f"  --iou {best_iou:.2f}")

if __name__ == "__main__":
    main()