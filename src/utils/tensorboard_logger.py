#!/usr/bin/env python3
"""
TensorBoard Logger for PillSnap ML Training
실시간 학습 모니터링을 위한 통합 로거
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorBoardLogger:
    """TensorBoard 통합 로거"""
    
    def __init__(
        self,
        log_dir: str = None,
        experiment_name: str = None,
        comment: str = "",
        flush_secs: int = 30
    ):
        """
        Args:
            log_dir: TensorBoard 로그 디렉토리 (기본값: runs/)
            experiment_name: 실험 이름
            comment: 추가 코멘트
            flush_secs: 디스크 플러시 주기 (초)
        """
        # 로그 디렉토리 설정
        if log_dir is None:
            log_dir = os.environ.get('TENSORBOARD_DIR', 'runs')
        
        # 실험 이름 설정
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"stage3_{timestamp}"
        
        # 전체 경로 생성
        if comment:
            experiment_name = f"{experiment_name}_{comment}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # SummaryWriter 초기화
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs
        )
        
        self.global_step = 0
        self.epoch = 0
        print(f"📊 TensorBoard 로거 초기화: {self.log_dir}")
        print(f"   실행: tensorboard --logdir {log_dir}")
    
    def set_epoch(self, epoch: int):
        """현재 에포크 설정"""
        self.epoch = epoch
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """스칼라 값 로깅"""
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """여러 스칼라 값 동시 로깅"""
        if step is None:
            step = self.global_step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_classification_metrics(
        self,
        loss: float,
        accuracy: float,
        top5_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        step: Optional[int] = None,
        phase: str = "train"
    ):
        """Classification 메트릭 로깅"""
        if step is None:
            step = self.global_step
        
        # Loss와 Accuracy
        self.writer.add_scalar(f'{phase}/classification/loss', loss, step)
        self.writer.add_scalar(f'{phase}/classification/accuracy', accuracy, step)
        
        # Top-5 Accuracy (있으면)
        if top5_accuracy is not None:
            self.writer.add_scalar(f'{phase}/classification/top5_accuracy', top5_accuracy, step)
        
        # Learning Rate (있으면)
        if learning_rate is not None:
            self.writer.add_scalar(f'{phase}/learning_rate', learning_rate, step)
        
        # 통합 메트릭
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        if top5_accuracy is not None:
            metrics['top5_accuracy'] = top5_accuracy
        
        self.writer.add_scalars(f'{phase}/classification/all_metrics', metrics, step)
    
    def log_detection_metrics(
        self,
        box_loss: float,
        cls_loss: float,
        dfl_loss: float,
        map50: float,
        map50_95: Optional[float] = None,
        step: Optional[int] = None,
        phase: str = "train"
    ):
        """Detection 메트릭 로깅"""
        if step is None:
            step = self.global_step
        
        # 개별 Loss
        self.writer.add_scalar(f'{phase}/detection/box_loss', box_loss, step)
        self.writer.add_scalar(f'{phase}/detection/cls_loss', cls_loss, step)
        self.writer.add_scalar(f'{phase}/detection/dfl_loss', dfl_loss, step)
        
        # mAP
        self.writer.add_scalar(f'{phase}/detection/mAP50', map50, step)
        if map50_95 is not None:
            self.writer.add_scalar(f'{phase}/detection/mAP50_95', map50_95, step)
        
        # 통합 Loss
        total_loss = box_loss + cls_loss + dfl_loss
        self.writer.add_scalar(f'{phase}/detection/total_loss', total_loss, step)
        
        # 모든 메트릭 한번에
        metrics = {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss,
            'total_loss': total_loss,
            'mAP50': map50
        }
        if map50_95 is not None:
            metrics['mAP50_95'] = map50_95
        
        self.writer.add_scalars(f'{phase}/detection/all_metrics', metrics, step)
    
    def log_system_metrics(
        self,
        gpu_memory_used: float,
        gpu_memory_peak: float,
        gpu_utilization: Optional[float] = None,
        step: Optional[int] = None
    ):
        """시스템 메트릭 로깅"""
        if step is None:
            step = self.global_step
        
        self.writer.add_scalar('system/gpu_memory_used_gb', gpu_memory_used, step)
        self.writer.add_scalar('system/gpu_memory_peak_gb', gpu_memory_peak, step)
        
        if gpu_utilization is not None:
            self.writer.add_scalar('system/gpu_utilization', gpu_utilization, step)
    
    def log_batch_metrics(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ):
        """배치별 메트릭 로깅 (빠른 업데이트)"""
        # Global step 계산
        step = epoch * total_batches + batch
        
        # 기본 메트릭
        self.writer.add_scalar('batch/loss', loss, step)
        
        if accuracy is not None:
            self.writer.add_scalar('batch/accuracy', accuracy, step)
        
        if learning_rate is not None:
            self.writer.add_scalar('batch/learning_rate', learning_rate, step)
        
        # 진행률
        progress = (batch / total_batches) * 100
        self.writer.add_scalar('batch/progress_percent', progress, step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """히스토그램 로깅 (가중치 분포 등)"""
        if step is None:
            step = self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """모델 구조 그래프 로깅"""
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"⚠️ 모델 그래프 로깅 실패: {e}")
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """텍스트 로깅 (설정, 하이퍼파라미터 등)"""
        if step is None:
            step = self.global_step
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """하이퍼파라미터와 최종 메트릭 로깅"""
        # 하이퍼파라미터를 플랫한 딕셔너리로 변환
        flat_hparams = self._flatten_dict(hparams)
        self.writer.add_hparams(flat_hparams, metrics)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        """중첩된 딕셔너리를 플랫하게 변환"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def step(self):
        """Global step 증가"""
        self.global_step += 1
    
    def flush(self):
        """버퍼 플러시 (즉시 디스크 쓰기)"""
        self.writer.flush()
    
    def close(self):
        """TensorBoard writer 종료"""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 간편 사용을 위한 전역 로거
_global_logger: Optional[TensorBoardLogger] = None


def get_tensorboard_logger(
    log_dir: str = None,
    experiment_name: str = None,
    reset: bool = False
) -> TensorBoardLogger:
    """전역 TensorBoard 로거 가져오기"""
    global _global_logger
    
    if _global_logger is None or reset:
        _global_logger = TensorBoardLogger(log_dir, experiment_name)
    
    return _global_logger


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
    """간편 메트릭 로깅"""
    logger = get_tensorboard_logger()
    for key, value in metrics.items():
        tag = f"{prefix}/{key}" if prefix else key
        logger.log_scalar(tag, value, step)