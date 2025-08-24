#!/usr/bin/env python3
"""
TensorBoard Integration Patch for Stage 3 Training
기존 학습 코드에 TensorBoard를 추가하는 헬퍼 모듈
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tensorboard_logger import TensorBoardLogger


class TensorBoardIntegration:
    """기존 학습 코드에 TensorBoard 통합을 위한 래퍼"""
    
    def __init__(self, trainer_instance):
        """
        Args:
            trainer_instance: TwoStageTrainer 인스턴스
        """
        self.trainer = trainer_instance
        
        # TensorBoard 로거 초기화
        exp_dir = self.trainer.exp_dir if hasattr(self.trainer, 'exp_dir') else '/home/max16/pillsnap_data/exp/exp01'
        tb_dir = os.path.join(exp_dir, 'tensorboard')
        
        self.tb_logger = TensorBoardLogger(
            log_dir=tb_dir,
            experiment_name=f"stage3_resume",
            comment="two_stage",
            flush_secs=30
        )
        
        # 하이퍼파라미터 로깅
        self._log_hyperparameters()
        
        print(f"✅ TensorBoard 통합 완료")
        print(f"   📊 실행: tensorboard --logdir {tb_dir}")
        print(f"   🌐 브라우저: http://localhost:6006")
    
    def _log_hyperparameters(self):
        """하이퍼파라미터 로깅"""
        hparams = {}
        
        # Training config
        if hasattr(self.trainer, 'training_config'):
            cfg = self.trainer.training_config
            hparams['epochs'] = cfg.epochs
            hparams['batch_size'] = cfg.batch_size
            hparams['lr_classifier'] = cfg.lr_classifier
            hparams['lr_detector'] = cfg.lr_detector
            hparams['mixed_precision'] = cfg.mixed_precision
            hparams['num_classes'] = cfg.num_classes
        
        # 텍스트로 로깅
        hparam_text = "\n".join([f"- {k}: {v}" for k, v in hparams.items()])
        self.tb_logger.log_text("hyperparameters", hparam_text, 0)
    
    def log_classification_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ):
        """Classification 배치 메트릭 로깅"""
        step = epoch * total_batches + batch_idx
        
        self.tb_logger.log_scalar('train/classification/batch_loss', loss, step)
        
        if accuracy is not None:
            self.tb_logger.log_scalar('train/classification/batch_accuracy', accuracy, step)
        
        if learning_rate is not None:
            self.tb_logger.log_scalar('train/learning_rate', learning_rate, step)
    
    def log_classification_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        val_top5_accuracy: Optional[float] = None
    ):
        """Classification 에포크 메트릭 로깅"""
        # Train metrics
        self.tb_logger.log_classification_metrics(
            loss=train_loss,
            accuracy=train_accuracy,
            step=epoch,
            phase="train"
        )
        
        # Validation metrics
        if val_loss is not None and val_accuracy is not None:
            self.tb_logger.log_classification_metrics(
                loss=val_loss,
                accuracy=val_accuracy,
                top5_accuracy=val_top5_accuracy,
                step=epoch,
                phase="val"
            )
    
    def log_detection_epoch(
        self,
        epoch: int,
        box_loss: float,
        cls_loss: float,
        dfl_loss: float,
        map50: float,
        map50_95: Optional[float] = None
    ):
        """Detection 에포크 메트릭 로깅"""
        self.tb_logger.log_detection_metrics(
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss,
            map50=map50,
            map50_95=map50_95,
            step=epoch,
            phase="train"
        )
    
    def log_system_metrics(self, epoch: int):
        """시스템 메트릭 로깅"""
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024**3)
            
            self.tb_logger.log_system_metrics(
                gpu_memory_used=gpu_memory_used,
                gpu_memory_peak=gpu_memory_peak,
                step=epoch
            )
    
    def close(self):
        """TensorBoard 로거 종료"""
        self.tb_logger.close()


def patch_trainer_with_tensorboard(trainer_class):
    """기존 Trainer 클래스에 TensorBoard 메소드 패치"""
    
    # 중복 패치 방지 (idempotent guard)
    if hasattr(trainer_class, '_tb_patched'):
        print("⚠️ TensorBoard patch already applied, skipping...")
        return trainer_class
    
    original_init = trainer_class.__init__
    original_train = trainer_class.train if hasattr(trainer_class, 'train') else None
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # TensorBoard 통합 추가
        self.tb_integration = TensorBoardIntegration(self)
    
    def new_train(self, *args, **kwargs):
        try:
            # original_train이 없으면 기본 train 메소드 호출
            if original_train:
                result = original_train(self, *args, **kwargs)
            else:
                # train 메소드가 없는 경우 처리
                result = None
            return result
        finally:
            # 학습 종료 시 TensorBoard 닫기
            if hasattr(self, 'tb_integration'):
                self.tb_integration.close()
    
    # 메소드 패치
    trainer_class.__init__ = new_init
    trainer_class.train = new_train
    
    # TensorBoard 로깅 메소드 추가
    def log_tb_classification_batch(self, epoch, batch_idx, total_batches, loss, accuracy=None, lr=None):
        if hasattr(self, 'tb_integration'):
            self.tb_integration.log_classification_batch(
                epoch, batch_idx, total_batches, loss, accuracy, lr
            )
    
    def log_tb_classification_epoch(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, val_top5=None):
        if hasattr(self, 'tb_integration'):
            self.tb_integration.log_classification_epoch(
                epoch, train_loss, train_acc, val_loss, val_acc, val_top5
            )
    
    def log_tb_detection_epoch(self, epoch, box_loss, cls_loss, dfl_loss, map50, map50_95=None):
        if hasattr(self, 'tb_integration'):
            self.tb_integration.log_detection_epoch(
                epoch, box_loss, cls_loss, dfl_loss, map50, map50_95
            )
    
    def log_tb_system_metrics(self, epoch):
        if hasattr(self, 'tb_integration'):
            self.tb_integration.log_system_metrics(epoch)
    
    trainer_class.log_tb_classification_batch = log_tb_classification_batch
    trainer_class.log_tb_classification_epoch = log_tb_classification_epoch
    trainer_class.log_tb_detection_epoch = log_tb_detection_epoch
    trainer_class.log_tb_system_metrics = log_tb_system_metrics
    
    # 패치 완료 마커
    trainer_class._tb_patched = True
    
    return trainer_class


def log_tb_smoke(experiment="stage3", step=0):
    """TensorBoard 스모크 테스트 - 더미 값 로깅"""
    from src.utils.tensorboard_logger import TensorBoardLogger
    import random
    
    tb_logger = TensorBoardLogger(
        log_dir="artifacts/tensorboard",
        experiment_name=experiment,
        comment="smoke_test"
    )
    
    # 더미 스칼라 로깅
    tb_logger.log_scalar('smoke/test_metric_1', random.random(), step)
    tb_logger.log_scalar('smoke/test_metric_2', random.random() * 100, step)
    tb_logger.log_scalar('smoke/test_metric_3', random.random() * 0.01, step)
    
    # 분류 메트릭 예제
    tb_logger.log_scalar('train/loss', 2.3 - step * 0.1, step)
    tb_logger.log_scalar('train/lr', 1e-4 * (0.95 ** step), step)
    tb_logger.log_scalar('val/top1', 0.3 + step * 0.02, step)
    tb_logger.log_scalar('val/top5', 0.5 + step * 0.03, step)
    
    # 검출 메트릭 예제
    tb_logger.log_scalar('det/map50', 0.25 + step * 0.01, step)
    tb_logger.log_scalar('det/box_loss', 0.5 - step * 0.01, step)
    
    # 시스템 메트릭 예제
    tb_logger.log_scalar('sys/vram_used', 8000 + random.randint(-500, 500), step)
    tb_logger.log_scalar('latency/total', 50 + random.randint(-10, 10), step)
    
    print(f"✅ Smoke test logged at step {step}")
    print(f"📊 View at: tensorboard --logdir artifacts/tensorboard")
    
    tb_logger.close()
    return True


# 간단한 사용 예제
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    
    if args.smoke:
        print("🔥 Running TensorBoard smoke test...")
        for step in range(5):
            log_tb_smoke("stage3_smoke", step)
        print("✅ Smoke test complete!")
    else:
        print("TensorBoard Integration Module")
        print("이 모듈을 import하고 patch_trainer_with_tensorboard()를 사용하세요.")
        print("\n예제:")
        print("from src.training.tensorboard_integration import patch_trainer_with_tensorboard")
        print("patch_trainer_with_tensorboard(TwoStageTrainer)")
        print("\n스모크 테스트:")
        print("python -m src.training.tensorboard_integration --smoke")