#!/usr/bin/env python3
"""
Detection 스모크 테스트 - 사이클 #1 (수정본)
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
    logger.info("=== 스모크 테스트 사이클 #1 시작 (수정본) ===")
    
    # State Manager 초기화
    state_manager = DetectionStateManager()
    state = state_manager.load_state()
    
    # 현재 에폭 확인
    current_epochs = state.get("det_epochs_done", 0)
    target_epochs = current_epochs + 1  # 누적값 +1
    
    logger.info(f"현재 완료된 에폭: {current_epochs}")
    logger.info(f"목표 에폭 (누적): {target_epochs}")
    
    # YOLO 모델 로드 (기존 weights에서 시작)
    model_path = "/home/max16/pillsnap/runs/detect/train/weights/last.pt"
    if not Path(model_path).exists():
        logger.error(f"모델 파일이 없습니다: {model_path}")
        return False
    
    model = YOLO(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 데이터 YAML 경로 (기존 Stage 3 데이터 사용)
    data_yaml = "/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml"
    
    if not Path(data_yaml).exists():
        # YAML이 없으면 생성
        logger.info("데이터 YAML 생성 중...")
        yaml_content = """
path: /home/max16/pillsnap_data/yolo_configs/yolo_dataset
train: images
val: images
test: images

nc: 1
names: ['pill']
"""
        Path(data_yaml).parent.mkdir(parents=True, exist_ok=True)
        Path(data_yaml).write_text(yaml_content)
    
    # YOLO 학습 실행 - resume 없이 새로운 1 epoch만
    logger.info(f"YOLO 학습 시작 - 1 epoch 추가")
    
    try:
        # resume=False로 하되, 기존 weights에서 시작
        results = model.train(
            data=data_yaml,
            epochs=1,  # 단일 epoch만
            batch=8,
            imgsz=640,
            device=0,
            project="/home/max16/pillsnap/runs/detect",
            name="train",
            exist_ok=True,  # 기존 디렉토리 사용
            resume=False,    # resume 비활성화
            save=True,
            save_period=1,
            val=True,        # 검증 켜기
            verbose=True,
            patience=100,    # 조기종료 방지
            workers=4,
            amp=True,
            seed=42,
            pretrained=False  # pretrained 비활성화
        )
        
        logger.info("YOLO 학습 완료")
        
        # State 업데이트
        state["det_epochs_done"] = target_epochs
        
        # results.csv에서 마지막 메트릭 읽기
        import pandas as pd
        csv_path = Path("/home/max16/pillsnap/runs/detect/train/results.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                last_row = df.iloc[-1]
                metrics = {
                    "map50": float(last_row.get('metrics/mAP50(B)', 0)),
                    "precision": float(last_row.get('metrics/precision(B)', 0)),
                    "recall": float(last_row.get('metrics/recall(B)', 0)),
                    "box_loss": float(last_row.get('train/box_loss', 0)),
                    "cls_loss": float(last_row.get('train/cls_loss', 0)),
                    "dfl_loss": float(last_row.get('train/dfl_loss', 0))
                }
                state_manager.update_metrics(state, metrics)
                logger.info(f"메트릭 업데이트: mAP@0.5={metrics['map50']:.4f}")
        
        # State 저장
        state_manager.save_state(state)
        logger.info(f"State 저장 완료: det_epochs_done={target_epochs}")
        
        return True
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = run_detection_cycle()
    if success:
        print("✅ 사이클 #1 완료")
    else:
        print("❌ 사이클 #1 실패")
    sys.exit(0 if success else 1)