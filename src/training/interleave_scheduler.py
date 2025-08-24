"""
PillSnap ML 인터리브 학습 스케줄러 (1단계 필수)

det:cls=1:2 확률 스케줄링:
- 엄격한 교차 대신 확률 기반 스케줄링
- Detection:Classification = 1:2 비율 유지
- 개발 비용 최소화, 기대값 동일한 효과
- 옵티마이저 step 호출 일관성 보장

RTX 5080 최적화
"""

import random
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.core import PillSnapLogger


class TaskType(Enum):
    """학습 태스크 타입"""
    DETECTION = "detection"
    CLASSIFICATION = "classification"


@dataclass
class InterleaveConfig:
    """인터리브 학습 설정 (1단계 필수)"""
    
    # 비율 설정
    detection_probability: float = 1/3    # det:cls = 1:2
    classification_probability: float = 2/3
    
    # 확률 조정 설정
    rebalance_every_n_steps: int = 10    # 10 step마다 비율 재조정
    target_tolerance: float = 0.1        # 10% 허용 오차
    
    # 옵티마이저 스케줄링
    sync_optimizer_steps: bool = True    # 옵티마이저 step 동기화
    min_steps_before_sync: int = 5       # 최소 5 step 후 동기화
    sync_grace_steps: int = 20           # 동기화 보정 주기 (드리프트 방지)
    
    # 그래디언트 누적 설정
    grad_accum_det: int = 2              # Detection 그래디언트 누적 스텝
    grad_accum_cls: int = 2              # Classification 그래디언트 누적 스텝
    
    # 로깅
    log_scheduling_stats: bool = True


class InterleaveScheduler:
    """인터리브 학습 스케줄러 (1단계 필수)"""
    
    def __init__(self, config: InterleaveConfig):
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 상태 추적
        self.total_steps = 0
        self.detection_steps = 0
        self.classification_steps = 0
        self.last_rebalance_step = 0
        
        # 확률 조정
        self.current_det_prob = config.detection_probability
        self.current_cls_prob = config.classification_probability
        
        # 옵티마이저 동기화
        self.pending_det_steps = 0
        self.pending_cls_steps = 0
        self.last_sync_step = 0
        self.force_sync_next = False  # 리밸런스 후 강제 동기화
        
        # 통계
        self.scheduling_history = []
        
        self.logger.info(
            f"📋 인터리브 스케줄러 초기화 - det:cls = {config.detection_probability:.2f}:{config.classification_probability:.2f}"
        )
    
    def should_train_detection(self) -> bool:
        """
        Detection 학습 여부 결정 (확률 기반)
        
        Returns:
            bool: Detection 학습해야 하는지 여부
        """
        # 확률 기반 선택
        choice = random.random()
        should_det = choice < self.current_det_prob
        
        # 상태 업데이트
        self.total_steps += 1
        
        if should_det:
            self.detection_steps += 1
            self.pending_det_steps += 1
            task_type = TaskType.DETECTION
        else:
            self.classification_steps += 1
            self.pending_cls_steps += 1
            task_type = TaskType.CLASSIFICATION
        
        # 히스토리 기록
        self.scheduling_history.append({
            "step": self.total_steps,
            "task": task_type.value,
            "det_prob": self.current_det_prob,
            "cls_prob": self.current_cls_prob
        })
        
        # 주기적 재조정
        if (self.total_steps - self.last_rebalance_step) >= self.config.rebalance_every_n_steps:
            self._rebalance_probabilities()
        
        # 로깅
        if self.config.log_scheduling_stats and self.total_steps % 50 == 0:
            self._log_scheduling_stats()
        
        return should_det
    
    def get_next_task(self) -> TaskType:
        """
        다음 태스크 타입 반환
        
        Returns:
            TaskType: 다음에 학습할 태스크
        """
        if self.should_train_detection():
            return TaskType.DETECTION
        else:
            return TaskType.CLASSIFICATION
    
    def should_sync_optimizers(self, pending_det_steps: int = None, pending_cls_steps: int = None, 
                              grad_accum_det: int = None, grad_accum_cls: int = None) -> bool:
        """
        태스크 무관 글로벌 동기화 판단
        
        Args:
            pending_det_steps: 대기 중인 Detection 스텝 수 (기본값: self.pending_det_steps)
            pending_cls_steps: 대기 중인 Classification 스텝 수 (기본값: self.pending_cls_steps)
            grad_accum_det: Detection 그래디언트 누적 스텝 (기본값: config.grad_accum_det)
            grad_accum_cls: Classification 그래디언트 누적 스텝 (기본값: config.grad_accum_cls)
            
        Returns:
            bool: 양쪽 옵티마이저를 동기화해야 하는지 여부
        """
        if not self.config.sync_optimizer_steps:
            return True  # 동기화 비활성화시 항상 step
        
        # 기본값 설정
        if pending_det_steps is None:
            pending_det_steps = self.pending_det_steps
        if pending_cls_steps is None:
            pending_cls_steps = self.pending_cls_steps
        if grad_accum_det is None:
            grad_accum_det = self.config.grad_accum_det
        if grad_accum_cls is None:
            grad_accum_cls = self.config.grad_accum_cls
        
        # 강제 동기화 플래그 체크
        if self.force_sync_next:
            self.force_sync_next = False
            return True
        
        # 조건 a: 양쪽 모두 grad_accum 조건 만족
        both_ready = (pending_det_steps >= grad_accum_det and 
                     pending_cls_steps >= grad_accum_cls)
        
        if both_ready:
            return True
        
        # 조건 b: 동기화 보정 주기 경과 시 드리프트 방지
        steps_since_sync = self.total_steps - self.last_sync_step
        grace_period_exceeded = steps_since_sync >= self.config.sync_grace_steps
        
        if grace_period_exceeded:
            # 한쪽만 계속 미달일 때도 동기화 강제
            either_ready = (pending_det_steps >= grad_accum_det or 
                           pending_cls_steps >= grad_accum_cls)
            if either_ready:
                self.logger.warning(
                    f"⚠️ 동기화 보정 주기 경과 - 강제 동기화 실행 "
                    f"(det: {pending_det_steps}/{grad_accum_det}, cls: {pending_cls_steps}/{grad_accum_cls})"
                )
                return True
        
        return False
    
    def sync_and_step_optimizers(self, det_opt: Any, cls_opt: Any) -> None:
        """
        동기화 시점에서 양쪽 옵티마이저 모두 step 및 리셋
        
        Args:
            det_opt: Detection 옵티마이저
            cls_opt: Classification 옵티마이저
        """
        # 양쪽 옵티마이저 모두 step
        if det_opt:
            det_opt.step()
            det_opt.zero_grad()
        
        if cls_opt:
            cls_opt.step()
            cls_opt.zero_grad()
        
        # 카운터 리셋
        det_steps = self.pending_det_steps
        cls_steps = self.pending_cls_steps
        
        self.pending_det_steps = 0
        self.pending_cls_steps = 0
        self.last_sync_step = self.total_steps
        
        # 동기화 로그
        det_cls_ratio = f"1:{cls_steps/max(1, det_steps):.2f}" if det_steps > 0 else f"0:{cls_steps}"
        self.logger.info(
            f"🔄 SYNC(det:cls={det_cls_ratio}, det_steps={det_steps}, cls_steps={cls_steps})"
        )
    
    def _rebalance_probabilities(self) -> None:
        """확률 재조정 (비율 유지)"""
        if self.total_steps <= 1:
            return
        
        # 현재 실제 비율 계산
        actual_det_ratio = self.detection_steps / self.total_steps
        actual_cls_ratio = self.classification_steps / self.total_steps
        
        # 목표 비율과 비교
        target_det_ratio = self.config.detection_probability
        target_cls_ratio = self.config.classification_probability
        
        det_diff = actual_det_ratio - target_det_ratio
        cls_diff = actual_cls_ratio - target_cls_ratio
        
        # 허용 오차 내에 있는지 확인
        if abs(det_diff) <= self.config.target_tolerance:
            self.last_rebalance_step = self.total_steps
            return  # 재조정 불필요
        
        # 확률 조정 (부족한 쪽의 확률 증가)
        adjustment_factor = 0.1  # 10% 조정
        
        if det_diff < 0:  # Detection이 부족
            self.current_det_prob = min(0.9, self.current_det_prob + adjustment_factor)
            self.current_cls_prob = 1.0 - self.current_det_prob
        elif cls_diff < 0:  # Classification이 부족
            self.current_cls_prob = min(0.9, self.current_cls_prob + adjustment_factor)
            self.current_det_prob = 1.0 - self.current_cls_prob
        
        self.last_rebalance_step = self.total_steps
        
        # 보정 직후 첫 동기화에서 양쪽 동시 step 유도
        self.force_sync_next = True
        
        self.logger.info(
            f"📊 확률 재조정 - det: {target_det_ratio:.2f}→{self.current_det_prob:.2f}, "
            f"cls: {target_cls_ratio:.2f}→{self.current_cls_prob:.2f} "
            f"(실제 비율: det {actual_det_ratio:.3f}, cls {actual_cls_ratio:.3f}) - 다음 동기화 강제"
        )
    
    def _log_scheduling_stats(self) -> None:
        """스케줄링 통계 로깅"""
        if self.total_steps == 0:
            return
        
        actual_det_ratio = self.detection_steps / self.total_steps
        actual_cls_ratio = self.classification_steps / self.total_steps
        
        target_det_ratio = self.config.detection_probability
        target_cls_ratio = self.config.classification_probability
        
        det_deviation = abs(actual_det_ratio - target_det_ratio)
        cls_deviation = abs(actual_cls_ratio - target_cls_ratio)
        
        self.logger.info(
            f"📈 스케줄링 통계 (step {self.total_steps}) - "
            f"det: {actual_det_ratio:.3f} (목표 {target_det_ratio:.3f}, 편차 {det_deviation:.3f}), "
            f"cls: {actual_cls_ratio:.3f} (목표 {target_cls_ratio:.3f}, 편차 {cls_deviation:.3f}), "
            f"대기: det {self.pending_det_steps}, cls {self.pending_cls_steps}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """스케줄링 통계 반환"""
        if self.total_steps == 0:
            return {
                "total_steps": 0,
                "detection_steps": 0,
                "classification_steps": 0,
                "actual_ratios": {"detection": 0.0, "classification": 0.0},
                "target_ratios": {"detection": self.config.detection_probability, "classification": self.config.classification_probability},
                "deviations": {"detection": 0.0, "classification": 0.0},
                "current_probabilities": {"detection": self.current_det_prob, "classification": self.current_cls_prob}
            }
        
        actual_det_ratio = self.detection_steps / self.total_steps
        actual_cls_ratio = self.classification_steps / self.total_steps
        
        return {
            "total_steps": self.total_steps,
            "detection_steps": self.detection_steps,
            "classification_steps": self.classification_steps,
            "actual_ratios": {
                "detection": actual_det_ratio,
                "classification": actual_cls_ratio
            },
            "target_ratios": {
                "detection": self.config.detection_probability,
                "classification": self.config.classification_probability
            },
            "deviations": {
                "detection": abs(actual_det_ratio - self.config.detection_probability),
                "classification": abs(actual_cls_ratio - self.config.classification_probability)
            },
            "current_probabilities": {
                "detection": self.current_det_prob,
                "classification": self.current_cls_prob
            },
            "pending_steps": {
                "detection": self.pending_det_steps,
                "classification": self.pending_cls_steps
            }
        }
    
    def reset_statistics(self) -> None:
        """통계 리셋 (에포크 시작 시 호출)"""
        self.total_steps = 0
        self.detection_steps = 0
        self.classification_steps = 0
        self.last_rebalance_step = 0
        self.pending_det_steps = 0
        self.pending_cls_steps = 0
        self.last_sync_step = 0
        self.force_sync_next = False
        self.scheduling_history.clear()
        
        # 확률 초기화
        self.current_det_prob = self.config.detection_probability
        self.current_cls_prob = self.config.classification_probability
        
        self.logger.info("🔄 인터리브 스케줄러 통계 리셋")


class InterleaveTrainingWrapper:
    """인터리브 학습 래퍼 (편의 클래스)"""
    
    def __init__(
        self,
        detection_train_fn: Callable,
        classification_train_fn: Callable,
        scheduler: InterleaveScheduler,
        detection_optimizer: Optional[Any] = None,
        classification_optimizer: Optional[Any] = None
    ):
        """
        Args:
            detection_train_fn: Detection 학습 함수
            classification_train_fn: Classification 학습 함수
            scheduler: 인터리브 스케줄러
            detection_optimizer: Detection 옵티마이저 (선택적)
            classification_optimizer: Classification 옵티마이저 (선택적)
        """
        self.detection_train_fn = detection_train_fn
        self.classification_train_fn = classification_train_fn
        self.scheduler = scheduler
        self.detection_optimizer = detection_optimizer
        self.classification_optimizer = classification_optimizer
        
        self.logger = PillSnapLogger(__name__)
    
    def training_step(self, **kwargs) -> Tuple[TaskType, Any]:
        """
        인터리브 학습 스텝 실행
        
        Returns:
            Tuple[TaskType, Any]: (실행된 태스크 타입, 학습 결과)
        """
        # 다음 태스크 결정
        task_type = self.scheduler.get_next_task()
        
        # 해당 태스크 학습 실행
        if task_type == TaskType.DETECTION:
            result = self.detection_train_fn(**kwargs)
            optimizer = self.detection_optimizer
        else:
            result = self.classification_train_fn(**kwargs)
            optimizer = self.classification_optimizer
        
        # 옵티마이저 동기화 처리 (새로운 방식)
        if self.scheduler.should_sync_optimizers():
            self.scheduler.sync_and_step_optimizers(
                self.detection_optimizer, 
                self.classification_optimizer
            )
        
        return task_type, result
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """스케줄러 통계 반환"""
        return self.scheduler.get_statistics()


if __name__ == "__main__":
    print("🧪 인터리브 학습 스케줄러 테스트 (1단계 필수)")
    print("=" * 60)
    
    # 설정 테스트
    config = InterleaveConfig(
        detection_probability=1/3,
        classification_probability=2/3,
        rebalance_every_n_steps=10
    )
    scheduler = InterleaveScheduler(config)
    
    print(f"✅ 초기 설정: det={config.detection_probability:.2f}, cls={config.classification_probability:.2f}")
    
    # 100 step 시뮬레이션
    for step in range(100):
        task = scheduler.get_next_task()
        should_step = scheduler.should_step_optimizer(task)
        
        if step % 25 == 24:  # 25 step마다 통계 출력
            stats = scheduler.get_statistics()
            print(f"Step {step+1}: det={stats['actual_ratios']['detection']:.3f}, cls={stats['actual_ratios']['classification']:.3f}")
    
    # 최종 통계
    final_stats = scheduler.get_statistics()
    print(f"\n📊 최종 통계:")
    print(f"  Detection: {final_stats['actual_ratios']['detection']:.3f} (목표: {final_stats['target_ratios']['detection']:.3f})")
    print(f"  Classification: {final_stats['actual_ratios']['classification']:.3f} (목표: {final_stats['target_ratios']['classification']:.3f})")
    print(f"  편차: det={final_stats['deviations']['detection']:.3f}, cls={final_stats['deviations']['classification']:.3f}")
    
    print("🎉 인터리브 학습 스케줄러 테스트 완료!")