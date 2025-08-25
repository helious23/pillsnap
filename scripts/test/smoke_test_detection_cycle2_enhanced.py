#!/usr/bin/env python3
"""
Detection 스모크 테스트 - 사이클 #2 (개선 버전)
목적: TensorBoard 로깅 + metrics_history 추가
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

import torch
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

# Detection State Manager
from src.utils.detection_state_manager import DetectionStateManager
from src.utils.core import PillSnapLogger

def run_detection_cycle():
    logger = PillSnapLogger(__name__)
    logger.info("=== 스모크 테스트 사이클 #2 (개선 버전) ===")
    
    # State Manager 초기화
    state_manager = DetectionStateManager()
    state = state_manager.load_state()
    
    # 현재 에폭 확인
    current_epochs = state.get("det_epochs_done", 0)
    target_epochs = current_epochs + 1  # 누적값 +1
    
    logger.info(f"현재 완료된 에폭: {current_epochs}")
    logger.info(f"목표 에폭 (누적): {target_epochs}")
    
    # TensorBoard Writer 초기화
    tb_dir = Path("/home/max16/pillsnap_data/exp/exp01/tensorboard/smoke_test")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    logger.info(f"TensorBoard 로깅 시작: {tb_dir}")
    
    # YOLO 모델 로드
    model_path = "/home/max16/pillsnap/runs/detect/train/weights/last.pt"
    if not Path(model_path).exists():
        logger.error(f"모델 파일이 없습니다: {model_path}")
        return False
    
    model = YOLO(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 데이터 YAML 경로
    data_yaml = "/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml"
    
    if not Path(data_yaml).exists():
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
    
    # YOLO 학습 실행
    logger.info(f"YOLO 학습 시작 - 1 epoch 추가")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=1,  # 단일 epoch만
            batch=8,
            imgsz=640,
            device=0,
            project="/home/max16/pillsnap/runs/detect",
            name="train",
            exist_ok=True,
            resume=False,
            save=True,
            save_period=1,
            val=True,
            verbose=True,
            patience=100,
            workers=4,
            amp=True,
            seed=42,
            pretrained=False,
            plots=True  # TensorBoard를 위한 plots 활성화
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
                
                # 메트릭 업데이트
                state_manager.update_metrics(state, metrics)
                logger.info(f"메트릭 업데이트: mAP@0.5={metrics['map50']:.4f}")
                
                # TensorBoard 로깅
                writer.add_scalar('Detection/mAP50', metrics['map50'], target_epochs)
                writer.add_scalar('Detection/Precision', metrics['precision'], target_epochs)
                writer.add_scalar('Detection/Recall', metrics['recall'], target_epochs)
                writer.add_scalar('Loss/Box', metrics['box_loss'], target_epochs)
                writer.add_scalar('Loss/Class', metrics['cls_loss'], target_epochs)
                writer.add_scalar('Loss/DFL', metrics['dfl_loss'], target_epochs)
                logger.info(f"TensorBoard 로깅 완료")
                
                # metrics_history 업데이트
                if "metrics_history" not in state:
                    state["metrics_history"] = []
                
                history_entry = {
                    "epoch": target_epochs,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
                state["metrics_history"].append(history_entry)
                
                # 최근 10개만 유지
                if len(state["metrics_history"]) > 10:
                    state["metrics_history"] = state["metrics_history"][-10:]
                
                logger.info(f"metrics_history 업데이트: {len(state['metrics_history'])}개 에폭")
        
        # State 저장
        state_manager.save_state(state)
        logger.info(f"State 저장 완료: det_epochs_done={target_epochs}")
        
        # TensorBoard 종료
        writer.close()
        
        return True
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        writer.close()
        return False

if __name__ == "__main__":
    success = run_detection_cycle()
    if success:
        print("✅ 사이클 #2 완료")
    else:
        print("❌ 사이클 #2 실패")
    sys.exit(0 if success else 1)