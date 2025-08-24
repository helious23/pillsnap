#!/usr/bin/env python
"""YOLO 로그 캡처 테스트"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 로깅 설정
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/yolo_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# YOLO 모델 테스트
from ultralytics import YOLO
import torch

logger.info("="*60)
logger.info("🧪 YOLO 로그 캡처 테스트 시작")
logger.info("="*60)

# 모델 로드
model = YOLO('yolo11x.pt')
logger.info("✅ YOLO 모델 로드 완료")

# 콜백 함수 정의
def on_train_batch_end(trainer):
    """각 배치 끝날 때 호출되는 콜백"""
    try:
        batch_idx = trainer.epoch_progress.current if hasattr(trainer, 'epoch_progress') else 0
        total_batches = trainer.epoch_progress.total if hasattr(trainer, 'epoch_progress') else 100
        
        # 매 배치마다 로그
        if hasattr(trainer, 'loss_items'):
            losses = trainer.loss_items
            if len(losses) >= 3:
                logger.info(f"[YOLO Batch {batch_idx}/{total_batches}] box_loss: {losses[0]:.4f} | cls_loss: {losses[1]:.4f} | dfl_loss: {losses[2]:.4f}")
    except Exception as e:
        logger.warning(f"콜백 에러: {e}")

def on_train_epoch_end(trainer):
    """에포크 끝날 때 호출되는 콜백"""
    logger.info("="*40)
    logger.info("✅ YOLO Epoch 완료!")
    if hasattr(trainer, 'metrics'):
        metrics = trainer.metrics
        logger.info(f"최종 메트릭: {metrics}")
    logger.info("="*40)

# 학습 실행 (아주 작은 설정)
logger.info("🚀 YOLO 학습 시작...")
try:
    results = model.train(
        data='/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml',
        epochs=1,
        batch=2,  # 아주 작은 배치
        imgsz=320,  # 작은 이미지
        device=0,
        save=False,
        verbose=False,  # verbose는 False로
        workers=2,
        patience=0,
        val=False,
        exist_ok=True,
        project='/tmp/yolo_test',
        name='test_run',
        # 콜백 등록
        callbacks={
            'on_train_batch_end': on_train_batch_end,
            'on_train_epoch_end': on_train_epoch_end
        }
    )
    
    logger.info("✅ YOLO 학습 완료!")
    
    # 결과 확인
    if hasattr(results, 'results_dict'):
        logger.info(f"Results dict: {results.results_dict}")
    if hasattr(results, 'metrics'):
        logger.info(f"Metrics: {results.metrics}")
        
except Exception as e:
    logger.error(f"❌ YOLO 학습 에러: {e}")
    import traceback
    traceback.print_exc()

logger.info("="*60)
logger.info("🏁 테스트 종료")