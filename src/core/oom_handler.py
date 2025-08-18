"""
OOM Recovery State Machine - Finite State Machine for Training Consistency
OOM 복구를 위한 유한 상태 머신 - 학습 일관성 보장
"""

import torch
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

logger = logging.getLogger(__name__)


class OOMAction(Enum):
    """OOM 복구 액션"""
    EMPTY_CACHE = "empty_cache"
    FORCE_FP16 = "force_fp16"
    MICROBATCHING = "microbatching"
    GLOBAL_BATCH_CHANGE = "global_batch_change"
    EMERGENCY_EXIT = "emergency_exit"
    CONTINUE = "continue"


@dataclass
class OOMState:
    """OOM 복구 상태"""
    retries: int = 0
    did_empty_cache: bool = False
    did_amp_escalate: bool = False
    batch_size: int = 16
    grad_accum: int = 1
    global_batch: int = 16
    lr_scale: float = 1.0
    wd_scale: float = 1.0
    scheduler_mode: str = "by_steps"
    
    def get_effective_batch(self) -> int:
        """유효 배치 크기 계산"""
        return self.batch_size * self.grad_accum


@dataclass
class OOMConfig:
    """OOM 복구 설정"""
    max_retries: int = 4
    max_grad_accum: int = 8
    min_batch: int = 1
    cooldown_sec: float = 2.0
    escalate_to_fp16: bool = True
    preserve_global_batch: bool = True
    lr_rescaling: bool = True
    wd_rescaling: bool = True
    ema_per_sample: bool = True
    bn_handling: str = "freeze_after_warmup"  # freeze_after_warmup | groupnorm
    replay_failed_batch: bool = True
    audit_logging: bool = True


class OOMRecoveryStateMachine:
    """
    OOM 복구 상태 머신
    학습 일관성을 보장하면서 OOM 복구
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: OOM 설정 딕셔너리 (train.oom 섹션)
        """
        self.config = OOMConfig(
            max_retries=config.get("max_retries", 4),
            max_grad_accum=config.get("max_grad_accum", 8),
            min_batch=config.get("min_batch", 1),
            cooldown_sec=config.get("cooldown_sec", 2.0),
            escalate_to_fp16=config.get("escalate_to_fp16", True),
            preserve_global_batch=config.get("consistency", {}).get("preserve_global_batch", True),
            lr_rescaling=config.get("consistency", {}).get("lr_rescaling", True),
            wd_rescaling=config.get("consistency", {}).get("wd_rescaling", True),
            ema_per_sample=config.get("consistency", {}).get("ema_per_sample", True),
            bn_handling=config.get("consistency", {}).get("bn_handling", "freeze_after_warmup"),
            replay_failed_batch=config.get("consistency", {}).get("replay_failed_batch", True),
            audit_logging=config.get("consistency", {}).get("audit_logging", True)
        )
        
        self.state = OOMState()
        self.recovery_history = []
        
        logger.info(f"OOM Recovery initialized: max_retries={self.config.max_retries}, "
                   f"max_grad_accum={self.config.max_grad_accum}, min_batch={self.config.min_batch}")
    
    def handle_oom(
        self, 
        current_state: Dict[str, Any],
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """
        OOM 예외 처리 및 복구 전략 결정
        
        Args:
            current_state: 현재 학습 상태
            exception: 발생한 예외
            
        Returns:
            복구 액션 딕셔너리
        """
        # 재시도 횟수 증가
        self.state.retries += 1
        
        # 최대 재시도 초과
        if self.state.retries > self.config.max_retries:
            logger.error(f"OOM recovery failed after {self.config.max_retries} retries")
            return self._emergency_exit()
        
        # 쿨다운
        time.sleep(self.config.cooldown_sec)
        
        # 현재 상태 업데이트
        self.state.batch_size = current_state.get("batch_size", self.state.batch_size)
        self.state.grad_accum = current_state.get("grad_accum", self.state.grad_accum)
        self.state.global_batch = self.state.get_effective_batch()
        
        # 복구 전략 결정
        action = self._determine_action()
        
        # 감사 로깅
        if self.config.audit_logging:
            self._audit_log(action)
        
        return action
    
    def _determine_action(self) -> Dict[str, Any]:
        """복구 액션 결정"""
        
        # S1: 캐시 정리 (1회만)
        if not self.state.did_empty_cache:
            return self._empty_cache_action()
        
        # S2: AMP fp16 강제 (1회만)
        if self.config.escalate_to_fp16 and not self.state.did_amp_escalate:
            return self._force_fp16_action()
        
        # S3: 마이크로배칭 (글로벌 배치 유지)
        if self.config.preserve_global_batch:
            action = self._try_microbatching()
            if action:
                return action
        
        # S4: 글로벌 배치 변경 (최후 수단)
        return self._global_batch_change()
    
    def _empty_cache_action(self) -> Dict[str, Any]:
        """GPU 캐시 정리 액션"""
        logger.warning("OOM Recovery S1: Emptying CUDA cache")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.state.did_empty_cache = True
        
        return {
            "action": OOMAction.EMPTY_CACHE,
            "continue_training": True,
            "replay_batch": self.config.replay_failed_batch,
            "state_update": {"did_empty_cache": True}
        }
    
    def _force_fp16_action(self) -> Dict[str, Any]:
        """AMP fp16 강제 액션"""
        logger.warning("OOM Recovery S2: Forcing AMP to fp16")
        
        self.state.did_amp_escalate = True
        
        return {
            "action": OOMAction.FORCE_FP16,
            "amp_dtype": torch.float16,
            "continue_training": True,
            "preserve_schedules": True,
            "replay_batch": self.config.replay_failed_batch,
            "state_update": {"did_amp_escalate": True}
        }
    
    def _try_microbatching(self) -> Optional[Dict[str, Any]]:
        """마이크로배칭 시도"""
        G_old = self.state.global_batch
        
        # 배치 크기 절반으로 감소
        bs_new = max(self.state.batch_size // 2, self.config.min_batch)
        
        # 글로벌 배치 유지를 위한 accumulation 계산
        accum_new = min(
            math.ceil(G_old / bs_new),
            self.config.max_grad_accum
        )
        
        G_new = bs_new * accum_new
        
        # 글로벌 배치 유지 가능한 경우
        if G_new == G_old and accum_new <= self.config.max_grad_accum:
            logger.warning(f"OOM Recovery S3: Microbatching bs={bs_new}, accum={accum_new} "
                          f"(global batch preserved: {G_old})")
            
            self.state.batch_size = bs_new
            self.state.grad_accum = accum_new
            
            return {
                "action": OOMAction.MICROBATCHING,
                "batch_size": bs_new,
                "grad_accum": accum_new,
                "preserve_schedules": True,
                "continue_training": True,
                "replay_batch": self.config.replay_failed_batch,
                "state_update": {
                    "batch_size": bs_new,
                    "grad_accum": accum_new
                }
            }
        
        return None
    
    def _global_batch_change(self) -> Dict[str, Any]:
        """글로벌 배치 변경 (최후 수단)"""
        G_old = self.state.global_batch
        
        # 배치 크기 감소
        bs_new = max(self.state.batch_size // 2, self.config.min_batch)
        accum_new = min(self.config.max_grad_accum, math.ceil(G_old / bs_new))
        G_new = bs_new * accum_new
        
        # 더 이상 감소 불가능
        if bs_new <= self.config.min_batch and accum_new >= self.config.max_grad_accum:
            return self._emergency_exit()
        
        # 학습 일관성 보장을 위한 스케일링
        lr_scale = 1.0
        wd_scale = 1.0
        
        if self.config.lr_rescaling:
            lr_scale = G_new / G_old  # Linear Scaling Rule
        
        if self.config.wd_rescaling:
            wd_scale = G_old / G_new  # Weight Decay 총량 유지
        
        logger.warning(f"OOM Recovery S4: Global batch change G_old={G_old}→G_new={G_new}, "
                      f"lr_scale={lr_scale:.4f}, by_samples scheduling")
        
        self.state.batch_size = bs_new
        self.state.grad_accum = accum_new
        self.state.global_batch = G_new
        self.state.lr_scale = lr_scale
        self.state.wd_scale = wd_scale
        self.state.scheduler_mode = "by_samples"
        
        return {
            "action": OOMAction.GLOBAL_BATCH_CHANGE,
            "batch_size": bs_new,
            "grad_accum": accum_new,
            "lr_scale": lr_scale,
            "wd_scale": wd_scale,
            "scheduler_mode": "by_samples",
            "continue_training": True,
            "replay_batch": self.config.replay_failed_batch,
            "rebuild_dataloader": True,
            "state_update": {
                "batch_size": bs_new,
                "grad_accum": accum_new,
                "global_batch": G_new,
                "lr_scale": lr_scale,
                "wd_scale": wd_scale
            },
            "audit_log": {
                "G_old": G_old,
                "G_new": G_new,
                "lr_scale": lr_scale,
                "wd_scale": wd_scale
            }
        }
    
    def _emergency_exit(self) -> Dict[str, Any]:
        """긴급 종료"""
        logger.error("OOM Recovery FAILED: Emergency exit")
        
        return {
            "action": OOMAction.EMERGENCY_EXIT,
            "continue_training": False,
            "save_checkpoint": True,
            "exit_code": 1,
            "error_message": f"OOM recovery exhausted after {self.state.retries} retries"
        }
    
    def _audit_log(self, action: Dict[str, Any]):
        """감사 로그 기록"""
        log_entry = {
            "timestamp": time.time(),
            "retry": self.state.retries,
            "action": action.get("action", "unknown"),
            "batch_size": self.state.batch_size,
            "grad_accum": self.state.grad_accum,
            "global_batch": self.state.global_batch,
            "details": action.get("audit_log", {})
        }
        
        self.recovery_history.append(log_entry)
        
        logger.info(f"[OOM_AUDIT] {log_entry}")
    
    def reset(self):
        """상태 리셋"""
        self.state = OOMState()
        logger.info("OOM recovery state reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            "total_retries": self.state.retries,
            "current_batch_size": self.state.batch_size,
            "current_grad_accum": self.state.grad_accum,
            "current_global_batch": self.state.global_batch,
            "did_empty_cache": self.state.did_empty_cache,
            "did_amp_escalate": self.state.did_amp_escalate,
            "lr_scale": self.state.lr_scale,
            "wd_scale": self.state.wd_scale,
            "recovery_history": self.recovery_history
        }
    
    def apply_batch_size_update(
        self,
        dataloader_fn,
        optimizer,
        scheduler=None
    ) -> Tuple[Any, Any, Any]:
        """
        배치 크기 업데이트 적용
        
        Args:
            dataloader_fn: 데이터로더 생성 함수
            optimizer: 옵티마이저
            scheduler: 스케줄러 (옵션)
            
        Returns:
            (새 데이터로더, 업데이트된 옵티마이저, 업데이트된 스케줄러)
        """
        # 데이터로더 재생성
        new_dataloader = dataloader_fn(batch_size=self.state.batch_size)
        
        # Learning Rate 스케일링
        if self.state.lr_scale != 1.0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.state.lr_scale
        
        # Weight Decay 스케일링
        if self.state.wd_scale != 1.0:
            for param_group in optimizer.param_groups:
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] *= self.state.wd_scale
        
        # 스케줄러 모드 변경
        if scheduler and self.state.scheduler_mode == "by_samples":
            # 샘플 기반 스케줄링으로 전환 (구현 필요)
            logger.info("Switching to sample-based scheduling")
        
        return new_dataloader, optimizer, scheduler


def handle_training_oom(
    exception: Exception,
    state_machine: OOMRecoveryStateMachine,
    current_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    학습 중 OOM 처리 헬퍼 함수
    
    Args:
        exception: 발생한 OOM 예외
        state_machine: OOM 복구 상태 머신
        current_state: 현재 학습 상태
        
    Returns:
        복구 액션 딕셔너리
    """
    logger.error(f"OOM Exception caught: {exception}")
    
    # GPU 메모리 상태 로깅
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # 복구 액션 결정
    action = state_machine.handle_oom(current_state, exception)
    
    return action