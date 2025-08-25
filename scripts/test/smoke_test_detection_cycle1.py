#!/usr/bin/env python3
"""
Detection 스모크 테스트 - 사이클 #1
목적: Detection만 1 epoch 추가 학습하여 누적 확인
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

import torch
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO

# Detection State Manager
from src.utils.detection_state_manager import DetectionStateManager
from src.utils.core import PillSnapLogger

def run_detection_cycle():
    logger = PillSnapLogger(__name__)
    logger.info("=== 스모크 테스트 사이클 #1 시작 ===")
    
    # State Manager 초기화
    state_manager = DetectionStateManager()
    state = state_manager.load_state()
    
    # 현재 에폭 확인
    current_epochs = state.get("det_epochs_done", 0)
    target_epochs = current_epochs + 1  # 누적값 +1
    
    logger.info(f"현재 완료된 에폭: {current_epochs}")
    logger.info(f"목표 에폭 (누적): {target_epochs}")
    
    # YOLO 모델 로드 (기존 weights에서 resume)
    model_path = "/home/max16/pillsnap/runs/detect/train/weights/last.pt"
    if not Path(model_path).exists():
        logger.error(f"모델 파일이 없습니다: {model_path}")
        return False
    
    model = YOLO(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 학습 데이터 경로 (combination만 사용 - 빠른 테스트)
    data_yaml = """
path: /home/max16/pillsnap_data
train: train/images/combination
val: val/images/combination
nc: 1
names: ['pill']
"""
    
    # 임시 YAML 파일 생성
    yaml_path = Path("/tmp/smoke_test_data.yaml")
    yaml_path.write_text(data_yaml)
    
    # YOLO 학습 실행 (Detection만)
    logger.info(f"YOLO 학습 시작 - epochs={target_epochs}, resume=True")
    
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=target_epochs,  # 누적 총량
            batch=8,
            imgsz=640,
            device=0,
            project="/home/max16/pillsnap/runs/detect",
            name="train",
            exist_ok=True,  # 기존 디렉토리 사용
            resume=True,     # 이전 학습에서 이어서
            save=True,
            save_period=1,
            val=True,        # 검증 켜기
            verbose=True,
            patience=100,    # 조기종료 방지
            workers=4,
            amp=True,
            seed=42
        )
        
        logger.info("YOLO 학습 완료")
        
        # State 업데이트
        state["det_epochs_done"] = target_epochs
        
        # 메트릭 업데이트 (results에서 추출)
        if results and hasattr(results, 'metrics'):
            metrics = {
                "map50": float(results.metrics.get('metrics/mAP50(B)', 0)),
                "precision": float(results.metrics.get('metrics/precision(B)', 0)),
                "recall": float(results.metrics.get('metrics/recall(B)', 0))
            }
            state_manager.update_metrics(state, metrics)
            logger.info(f"메트릭 업데이트: mAP@0.5={metrics['map50']:.4f}")
        
        # State 저장
        state_manager.save_state(state)
        logger.info(f"State 저장 완료: det_epochs_done={target_epochs}")
        
        return True
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = run_detection_cycle()
    if success:
        print("✅ 사이클 #1 완료")
    else:
        print("❌ 사이클 #1 실패")
    sys.exit(0 if success else 1)