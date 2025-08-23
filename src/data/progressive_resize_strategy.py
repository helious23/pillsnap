"""
Progressive Resize 전략 구현

RTX 5080 16GB 환경을 위한 점진적 해상도 증가 전략:
- Stage 3-4 대용량 데이터셋 안정적 처리
- 메모리 효율성과 성능 최적화 동시 달성
- 훈련 초기 빠른 수렴, 후반부 고화질 학습

Features:
- Epoch 기반 해상도 점진적 증가 (224px → 384px)  
- GPU 메모리 사용량 실시간 모니터링
- 배치 크기 동적 조정
- Two-Stage Pipeline 호환

Author: Claude Code - PillSnap ML Team
Date: 2025-08-23
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger
from src.utils.memory_state_manager import MemoryStateManager, create_rtx5080_manager


class ResizeSchedule(Enum):
    """해상도 증가 스케줄"""
    LINEAR = "linear"           # 선형 증가
    EXPONENTIAL = "exponential"  # 지수적 증가  
    COSINE = "cosine"           # 코사인 증가
    STEP = "step"              # 단계별 증가


@dataclass
class ProgressiveResizeConfig:
    """Progressive Resize 설정"""
    
    # 해상도 범위
    initial_size: int = 224      # 시작 해상도
    target_size: int = 384       # 목표 해상도 (Classification)
    detection_size: int = 640    # 검출용 해상도 (고정)
    
    # 스케줄링
    schedule: ResizeSchedule = ResizeSchedule.COSINE
    warmup_epochs: int = 10      # 최소 해상도 유지 기간
    transition_epochs: int = 20  # 해상도 변화 기간
    stable_epochs: int = 20      # 최대 해상도 유지 기간
    
    # 메모리 관리
    max_memory_gb: float = 14.0  # RTX 5080 안전 한계
    memory_check_interval: int = 100  # 배치 단위 메모리 체크
    auto_batch_adjustment: bool = True  # 배치 크기 자동 조정
    
    # 성능 최적화
    interpolation_mode: str = "bilinear"  # 빠른 보간
    antialias: bool = False      # 속도 우선
    memory_format: str = "channels_last"  # RTX 5080 최적화
    
    # 품질 보존
    preserve_aspect_ratio: bool = True
    center_crop: bool = True
    quality_threshold: float = 0.95  # 품질 임계값


@dataclass
class ResizeState:
    """현재 해상도 상태"""
    current_size: int
    epoch: int
    batch_idx: int
    memory_usage_gb: float
    batch_size: int
    throughput_samples_per_sec: float
    last_update_time: float


class ProgressiveResizeScheduler:
    """점진적 해상도 증가 스케줄러"""
    
    def __init__(self, config: ProgressiveResizeConfig):
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 메모리 관리
        self.memory_manager = create_rtx5080_manager()
        
        # 상태 추적
        self.current_state = ResizeState(
            current_size=config.initial_size,
            epoch=0,
            batch_idx=0,
            memory_usage_gb=0.0,
            batch_size=16,  # 기본값
            throughput_samples_per_sec=0.0,
            last_update_time=time.time()
        )
        
        # 이력 추적
        self.resize_history: List[ResizeState] = []
        
        # 성능 통계
        self.performance_stats = {
            'total_resizes': 0,
            'memory_peaks': [],
            'throughput_improvements': [],
            'batch_adjustments': 0
        }
        
        self.logger.info(f"Progressive Resize 스케줄러 초기화")
        self.logger.info(f"해상도 범위: {config.initial_size}px → {config.target_size}px")
        self.logger.info(f"스케줄: {config.schedule.value}, 전환 기간: {config.transition_epochs} epochs")
    
    def get_current_size(self, epoch: int, batch_idx: int = 0) -> int:
        """현재 에포크에 대한 해상도 계산"""
        
        # Warmup 기간: 최소 해상도 유지
        if epoch < self.config.warmup_epochs:
            return self.config.initial_size
        
        # Stable 기간: 최대 해상도 유지
        transition_end = self.config.warmup_epochs + self.config.transition_epochs
        if epoch >= transition_end:
            return self.config.target_size
        
        # Transition 기간: 점진적 증가
        transition_progress = (epoch - self.config.warmup_epochs) / self.config.transition_epochs
        
        if self.config.schedule == ResizeSchedule.LINEAR:
            size_progress = transition_progress
        elif self.config.schedule == ResizeSchedule.EXPONENTIAL:
            size_progress = transition_progress ** 2
        elif self.config.schedule == ResizeSchedule.COSINE:
            size_progress = 0.5 * (1 - math.cos(math.pi * transition_progress))
        elif self.config.schedule == ResizeSchedule.STEP:
            # 4단계로 나누어 증가
            step_size = 1.0 / 4
            step_idx = int(transition_progress / step_size)
            size_progress = min(1.0, (step_idx + 1) * step_size)
        else:
            size_progress = transition_progress
        
        # 해상도 계산
        size_diff = self.config.target_size - self.config.initial_size
        current_size = int(self.config.initial_size + size_diff * size_progress)
        
        # 8의 배수로 정렬 (GPU 최적화)
        current_size = ((current_size + 7) // 8) * 8
        
        return max(self.config.initial_size, min(self.config.target_size, current_size))
    
    def update_state(self, epoch: int, batch_idx: int, batch_size: int) -> ResizeState:
        """상태 업데이트 및 메모리 모니터링"""
        
        current_time = time.time()
        new_size = self.get_current_size(epoch, batch_idx)
        
        # 메모리 사용량 확인
        memory_usage_gb = 0.0
        if torch.cuda.is_available():
            memory_usage_gb = torch.cuda.memory_allocated() / (1024**3)
        
        # 처리량 계산 (samples/sec)
        time_elapsed = current_time - self.current_state.last_update_time
        throughput = 0.0
        if time_elapsed > 0:
            samples_processed = batch_size
            throughput = samples_processed / time_elapsed
        
        # 상태 업데이트
        old_size = self.current_state.current_size
        self.current_state = ResizeState(
            current_size=new_size,
            epoch=epoch,
            batch_idx=batch_idx,
            memory_usage_gb=memory_usage_gb,
            batch_size=batch_size,
            throughput_samples_per_sec=throughput,
            last_update_time=current_time
        )
        
        # 해상도 변경 감지
        if new_size != old_size:
            self.performance_stats['total_resizes'] += 1
            self.logger.info(f"해상도 변경: {old_size}px → {new_size}px (Epoch {epoch})")
            
            # 이력 저장
            self.resize_history.append(self.current_state)
        
        # 메모리 위험 감지
        if memory_usage_gb > self.config.max_memory_gb * 0.9:
            self.logger.warning(f"높은 메모리 사용: {memory_usage_gb:.1f}GB / {self.config.max_memory_gb}GB")
        
        return self.current_state
    
    def get_optimal_batch_size(self, base_batch_size: int, current_size: int) -> int:
        """현재 해상도에 최적화된 배치 크기 계산"""
        if not self.config.auto_batch_adjustment:
            return base_batch_size
        
        # 해상도 기반 메모리 사용량 추정
        size_ratio = (current_size / self.config.initial_size) ** 2
        adjusted_batch_size = int(base_batch_size / math.sqrt(size_ratio))
        
        # RTX 5080 메모리 한계 고려
        max_batch_for_memory = self._calculate_max_batch_size(current_size)
        optimal_batch_size = min(adjusted_batch_size, max_batch_for_memory)
        
        # 최소/최대 제한
        return max(1, min(64, optimal_batch_size))
    
    def _calculate_max_batch_size(self, image_size: int) -> int:
        """이미지 크기 기반 최대 배치 크기 계산"""
        # 대략적인 메모리 사용량 추정 (MB per sample)
        base_memory_per_sample = 2.0  # 기본 메모리 사용량
        size_factor = (image_size / 224) ** 2
        memory_per_sample_mb = base_memory_per_sample * size_factor
        
        # 사용 가능한 메모리 (모델 메모리 제외)
        available_memory_gb = self.config.max_memory_gb * 0.7  # 70% 활용
        available_memory_mb = available_memory_gb * 1024
        
        max_batch_size = int(available_memory_mb / memory_per_sample_mb)
        return max(1, min(64, max_batch_size))
    
    def create_transform(self, mode: str = 'train') -> A.Compose:
        """현재 해상도에 맞는 변환 파이프라인 생성"""
        current_size = self.current_state.current_size
        
        transforms = []
        
        # 해상도 조정
        if self.config.preserve_aspect_ratio:
            transforms.append(A.LongestMaxSize(max_size=current_size))
            if self.config.center_crop:
                transforms.append(A.CenterCrop(height=current_size, width=current_size))
            else:
                transforms.append(A.PadIfNeeded(min_height=current_size, min_width=current_size, 
                                              border_mode=0, value=(0, 0, 0)))
        else:
            transforms.append(A.Resize(height=current_size, width=current_size, 
                                     interpolation=getattr(cv2, f'INTER_{self.config.interpolation_mode.upper()}')))
        
        # 훈련시 데이터 증강
        if mode == 'train':
            # 해상도에 따른 증강 강도 조절
            aug_strength = min(1.0, current_size / self.config.target_size)
            
            transforms.extend([
                A.HorizontalFlip(p=0.5 * aug_strength),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1 * aug_strength,
                    contrast_limit=0.1 * aug_strength,
                    p=0.3 * aug_strength
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05 * aug_strength,
                    scale_limit=0.05 * aug_strength,
                    rotate_limit=5 * aug_strength,
                    p=0.3 * aug_strength
                ),
            ])
        
        # 정규화 및 텐서 변환
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def should_update_transform(self, epoch: int, batch_idx: int) -> bool:
        """변환 파이프라인 업데이트 필요 여부 확인"""
        new_size = self.get_current_size(epoch, batch_idx)
        return new_size != self.current_state.current_size
    
    def get_training_schedule_summary(self) -> Dict[str, Any]:
        """훈련 스케줄 요약"""
        total_epochs = self.config.warmup_epochs + self.config.transition_epochs + self.config.stable_epochs
        
        return {
            'total_epochs': total_epochs,
            'phases': {
                'warmup': {
                    'epochs': f"0-{self.config.warmup_epochs-1}",
                    'size': f"{self.config.initial_size}px",
                    'purpose': "빠른 초기 수렴"
                },
                'transition': {
                    'epochs': f"{self.config.warmup_epochs}-{self.config.warmup_epochs + self.config.transition_epochs - 1}",
                    'size': f"{self.config.initial_size}px → {self.config.target_size}px",
                    'purpose': "점진적 품질 향상"
                },
                'stable': {
                    'epochs': f"{self.config.warmup_epochs + self.config.transition_epochs}+",
                    'size': f"{self.config.target_size}px",
                    'purpose': "고품질 최종 학습"
                }
            },
            'memory_optimization': {
                'max_memory_gb': self.config.max_memory_gb,
                'auto_batch_adjustment': self.config.auto_batch_adjustment,
                'memory_format': self.config.memory_format
            }
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 성능 통계"""
        return {
            'current_state': {
                'size': f"{self.current_state.current_size}px",
                'epoch': self.current_state.epoch,
                'memory_usage_gb': round(self.current_state.memory_usage_gb, 2),
                'batch_size': self.current_state.batch_size,
                'throughput_sps': round(self.current_state.throughput_samples_per_sec, 1)
            },
            'performance_stats': self.performance_stats,
            'resize_history_count': len(self.resize_history)
        }
    
    def export_resize_history(self, file_path: Path):
        """해상도 변경 이력 내보내기"""
        try:
            import json
            
            history_data = {
                'config': {
                    'initial_size': self.config.initial_size,
                    'target_size': self.config.target_size,
                    'schedule': self.config.schedule.value,
                    'warmup_epochs': self.config.warmup_epochs,
                    'transition_epochs': self.config.transition_epochs
                },
                'resize_history': [
                    {
                        'size': state.current_size,
                        'epoch': state.epoch,
                        'batch_idx': state.batch_idx,
                        'memory_usage_gb': state.memory_usage_gb,
                        'batch_size': state.batch_size,
                        'throughput_sps': state.throughput_samples_per_sec,
                        'timestamp': state.last_update_time
                    }
                    for state in self.resize_history
                ],
                'performance_stats': self.performance_stats
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Progressive Resize 이력 저장: {file_path}")
            
        except Exception as e:
            self.logger.error(f"이력 저장 실패: {e}")


class ProgressiveDataLoader:
    """Progressive Resize 지원 데이터로더"""
    
    def __init__(self, 
                 dataset, 
                 scheduler: ProgressiveResizeScheduler,
                 base_batch_size: int = 16,
                 num_workers: int = 8,
                 pin_memory: bool = True):
        
        self.dataset = dataset
        self.scheduler = scheduler
        self.base_batch_size = base_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.logger = PillSnapLogger(__name__)
        self.current_dataloader = None
        self.current_transform = None
        
        self.logger.info(f"Progressive DataLoader 초기화")
        self.logger.info(f"기본 배치 크기: {base_batch_size}, 워커: {num_workers}")
    
    def update_for_epoch(self, epoch: int):
        """에포크별 데이터로더 업데이트"""
        # 해상도 상태 업데이트
        state = self.scheduler.update_state(epoch, 0, self.base_batch_size)
        
        # 최적 배치 크기 계산
        optimal_batch_size = self.scheduler.get_optimal_batch_size(
            self.base_batch_size, state.current_size
        )
        
        # 변환 파이프라인 업데이트
        if self.scheduler.should_update_transform(epoch, 0):
            new_transform = self.scheduler.create_transform('train')
            
            # 데이터셋 변환 업데이트
            if hasattr(self.dataset, 'transform'):
                self.dataset.transform = new_transform
            elif hasattr(self.dataset, 'transforms'):
                self.dataset.transforms = new_transform
            
            self.logger.info(f"변환 파이프라인 업데이트: {state.current_size}px")
        
        # 데이터로더 재생성 (배치 크기 변경시)
        if self.current_dataloader is None or optimal_batch_size != state.batch_size:
            self.current_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=optimal_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory and torch.cuda.is_available(),
                drop_last=True,  # 배치 크기 일관성
                persistent_workers=True if self.num_workers > 0 else False
            )
            
            self.logger.info(f"데이터로더 업데이트: 배치 크기 {optimal_batch_size}")
    
    def __iter__(self):
        """데이터로더 반복자"""
        if self.current_dataloader is None:
            raise RuntimeError("update_for_epoch()을 먼저 호출하세요")
        
        return iter(self.current_dataloader)
    
    def __len__(self):
        """데이터로더 길이"""
        if self.current_dataloader is None:
            # 추정값 반환
            return len(self.dataset) // self.base_batch_size
        
        return len(self.current_dataloader)


# 편의 함수들
def create_progressive_scheduler(initial_size: int = 224,
                               target_size: int = 384,
                               warmup_epochs: int = 10,
                               transition_epochs: int = 20,
                               schedule: ResizeSchedule = ResizeSchedule.COSINE) -> ProgressiveResizeScheduler:
    """Progressive Resize 스케줄러 생성"""
    config = ProgressiveResizeConfig(
        initial_size=initial_size,
        target_size=target_size,
        warmup_epochs=warmup_epochs,
        transition_epochs=transition_epochs,
        schedule=schedule,
        max_memory_gb=14.0,  # RTX 5080 최적화
        auto_batch_adjustment=True
    )
    
    return ProgressiveResizeScheduler(config)


def create_progressive_dataloader(dataset, 
                                 scheduler: ProgressiveResizeScheduler,
                                 batch_size: int = 16,
                                 num_workers: int = 8) -> ProgressiveDataLoader:
    """Progressive 데이터로더 생성"""
    return ProgressiveDataLoader(
        dataset=dataset,
        scheduler=scheduler,
        base_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )


def quick_progressive_demo(total_epochs: int = 50):
    """Progressive Resize 데모"""
    scheduler = create_progressive_scheduler()
    
    print("=" * 60)
    print("Progressive Resize 전략 데모")
    print("=" * 60)
    
    summary = scheduler.get_training_schedule_summary()
    print(f"총 에포크: {summary['total_epochs']}")
    print()
    
    for phase, info in summary['phases'].items():
        print(f"{phase.upper()} 단계:")
        print(f"  에포크: {info['epochs']}")
        print(f"  해상도: {info['size']}")
        print(f"  목적: {info['purpose']}")
        print()
    
    print("에포크별 해상도 변화:")
    for epoch in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
        if epoch <= total_epochs:
            size = scheduler.get_current_size(epoch)
            print(f"  Epoch {epoch:2d}: {size}px")
    
    print()
    print("✅ Progressive Resize 전략 준비 완료!")
    

if __name__ == "__main__":
    quick_progressive_demo()