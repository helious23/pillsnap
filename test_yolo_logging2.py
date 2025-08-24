#!/usr/bin/env python
"""YOLO 로그 캡처 테스트 - 커스텀 트레이너 사용"""

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

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import torch

logger.info("="*60)
logger.info("🧪 YOLO 로그 캡처 테스트 시작 (커스텀 트레이너)")
logger.info("="*60)

# 커스텀 트레이너 클래스
class CustomTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_logger = logger
        
    def on_train_batch_end(self):
        """배치 학습 끝날 때 호출"""
        super().on_train_batch_end()
        
        # 로그 출력
        if hasattr(self, 'loss_items') and self.loss_items is not None:
            losses = self.loss_items.cpu().numpy() if torch.is_tensor(self.loss_items) else self.loss_items
            batch_idx = self.epoch_progress.current if hasattr(self, 'epoch_progress') else 0
            total_batches = self.epoch_progress.total if hasattr(self, 'epoch_progress') else 100
            
            if len(losses) >= 3:
                log_msg = f"[YOLO Batch {batch_idx}/{total_batches}] box_loss: {losses[0]:.4f} | cls_loss: {losses[1]:.4f} | dfl_loss: {losses[2]:.4f}"
                self.custom_logger.info(log_msg)
                
                # 파일에도 직접 쓰기
                with open('/tmp/mini_test.log', 'a', encoding='utf-8') as f:
                    f.write(f"{log_msg}\n")
                    f.flush()
    
    def on_train_epoch_end(self):
        """에포크 끝날 때 호출"""
        super().on_train_epoch_end()
        
        self.custom_logger.info("="*40)
        self.custom_logger.info("✅ YOLO Epoch 완료!")
        if hasattr(self, 'metrics') and self.metrics:
            self.custom_logger.info(f"최종 메트릭: {self.metrics}")
        self.custom_logger.info("="*40)

# 모델 로드
model = YOLO('yolo11x.pt')
logger.info("✅ YOLO 모델 로드 완료")

# 학습 실행 - 커스텀 트레이너 사용
logger.info("🚀 YOLO 학습 시작 (커스텀 트레이너)...")
try:
    # 트레이너 설정
    args = dict(
        model=model,
        data='/home/max16/pillsnap_data/yolo_configs/stage3_detection.yaml',
        epochs=1,
        batch=2,  # 아주 작은 배치
        imgsz=320,  # 작은 이미지
        device=0,
        save=False,
        verbose=True,  # verbose 활성화
        workers=2,
        patience=0,
        val=False,
        exist_ok=True,
        project='/tmp/yolo_test',
        name='test_run',
    )
    
    # 커스텀 트레이너로 학습
    trainer = CustomTrainer(overrides=args)
    trainer.train()
    
    logger.info("✅ YOLO 학습 완료!")
    
except Exception as e:
    logger.error(f"❌ YOLO 학습 에러: {e}")
    import traceback
    traceback.print_exc()

logger.info("="*60)
logger.info("🏁 테스트 종료")