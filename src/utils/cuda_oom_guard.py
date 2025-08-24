"""
PillSnap ML CUDA OOM 가드 시스템 (1단계 필수)

CUDA OOM 복구 및 동적 배치 크기 조정:
- 학습/검증 CUDAMemoryError 캐치 → per-GPU batch ↓ → grad_accum ↑ → LR 재보정
- torch.compile 초기 완충 기간 지원
- 검증 배치 크기 강제 제한

RTX 5080 16GB 최적화
"""

import torch
import time
import gc
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

from src.utils.core import PillSnapLogger


@dataclass
class OOMGuardConfig:
    """OOM 가드 설정"""
    
    # 배치 크기 조정 설정
    min_batch_size: int = 1
    batch_reduction_factor: float = 0.5  # 50%씩 감소
    max_oom_recoveries: int = 3          # 최대 3회 복구 시도
    
    # Gradient Accumulation 보정
    maintain_effective_batch: bool = True
    lr_scale_with_effective: bool = True  # Effective batch 변화시 LR 보정
    
    # torch.compile 완충 설정
    compile_warmup_steps: int = 200      # 초기 200 step 완충
    compile_batch_reduction: int = 1     # 완충 기간 배치 -1 단계
    
    # 검증 배치 크기 제한
    max_validation_batch: int = 4        # 검증 배치 최대 4
    
    # 메모리 정리 설정
    force_gc_on_oom: bool = True
    empty_cache_on_oom: bool = True


class CUDAOOMGuard:
    """CUDA OOM 복구 및 동적 배치 크기 관리"""
    
    def __init__(self, config: OOMGuardConfig):
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 상태 추적
        self.oom_count = 0
        self.current_batch_size = None
        self.original_batch_size = None
        self.current_grad_accum = None
        self.original_grad_accum = None
        self.original_lr = None
        
        # torch.compile 완충 상태
        self.compile_warmup_active = False
        self.compile_step_count = 0
        self.compile_original_batch = None
        
        # 복구 히스토리
        self.recovery_history = []
        
    def setup_training_params(
        self, 
        batch_size: int, 
        grad_accum_steps: int, 
        learning_rate: float
    ) -> None:
        """학습 파라미터 초기 설정"""
        self.current_batch_size = batch_size
        self.original_batch_size = batch_size
        self.current_grad_accum = grad_accum_steps
        self.original_grad_accum = grad_accum_steps
        self.original_lr = learning_rate
        
        self.logger.info(f"🛡️ OOM Guard 설정완료 - Batch: {batch_size}, GradAccum: {grad_accum_steps}, LR: {learning_rate}")
    
    def enable_compile_warmup(self) -> Tuple[int, int]:
        """
        torch.compile 완충 기간 활성화
        
        Returns:
            Tuple[int, int]: (완충 배치 크기, 완충 grad_accum)
        """
        if not self.compile_warmup_active:
            self.compile_warmup_active = True
            self.compile_step_count = 0
            self.compile_original_batch = self.current_batch_size
            
            # 완충 기간 배치 크기 감소
            warmup_batch = max(
                self.config.min_batch_size, 
                self.current_batch_size - self.config.compile_batch_reduction
            )
            
            # Effective batch 유지를 위한 grad_accum 증가
            if self.config.maintain_effective_batch:
                effective_batch = self.current_batch_size * self.current_grad_accum
                new_grad_accum = max(1, effective_batch // warmup_batch)
            else:
                new_grad_accum = self.current_grad_accum
            
            self.current_batch_size = warmup_batch
            self.current_grad_accum = new_grad_accum
            
            self.logger.warning(
                f"🔧 torch.compile 완충 활성화 ({self.config.compile_warmup_steps} steps) - "
                f"Batch: {self.compile_original_batch}→{warmup_batch}, "
                f"GradAccum: {self.original_grad_accum}→{new_grad_accum}"
            )
            
            return warmup_batch, new_grad_accum
        
        return self.current_batch_size, self.current_grad_accum
    
    def step_compile_warmup(self) -> bool:
        """
        torch.compile 완충 step 업데이트
        
        Returns:
            bool: 완충 기간이 끝났는지 여부
        """
        if not self.compile_warmup_active:
            return False
            
        self.compile_step_count += 1
        
        if self.compile_step_count >= self.config.compile_warmup_steps:
            # 완충 기간 종료 - 원래 설정으로 복구
            self.current_batch_size = self.compile_original_batch
            self.current_grad_accum = self.original_grad_accum
            self.compile_warmup_active = False
            
            self.logger.info(
                f"✅ torch.compile 완충 종료 - "
                f"Batch: {self.current_batch_size}, GradAccum: {self.current_grad_accum}"
            )
            return True
        
        return False
    
    def handle_oom_error(
        self, 
        error: Exception, 
        context: str = "training"
    ) -> Tuple[bool, int, int, Optional[float]]:
        """
        CUDA OOM 에러 처리 및 복구
        
        Args:
            error: CUDA OOM 에러
            context: 에러 발생 컨텍스트 ("training" 또는 "validation")
            
        Returns:
            Tuple[bool, int, int, Optional[float]]: (복구 가능 여부, 새 배치 크기, 새 grad_accum, 새 LR)
        """
        self.oom_count += 1
        
        self.logger.error(f"🚨 CUDA OOM 발생 ({self.oom_count}/{self.config.max_oom_recoveries}회) - {context}: {error}")
        
        # 최대 복구 횟수 초과 시 실패
        if self.oom_count > self.config.max_oom_recoveries:
            self.logger.critical(f"❌ 최대 OOM 복구 횟수 초과 ({self.config.max_oom_recoveries}회) - 학습 중단")
            return False, self.current_batch_size, self.current_grad_accum, None
        
        # 메모리 정리
        if self.config.force_gc_on_oom:
            gc.collect()
        
        if self.config.empty_cache_on_oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(1)  # 메모리 정리 대기
        
        # 배치 크기 감소
        old_batch = self.current_batch_size
        new_batch = max(
            self.config.min_batch_size,
            int(self.current_batch_size * self.config.batch_reduction_factor)
        )
        
        # 최소 배치 크기에 도달한 경우
        if new_batch == self.current_batch_size:
            self.logger.critical(f"❌ 최소 배치 크기({self.config.min_batch_size})에 도달 - 더 이상 복구 불가")
            return False, self.current_batch_size, self.current_grad_accum, None
        
        # Effective batch 유지를 위한 grad_accum 증가
        new_grad_accum = self.current_grad_accum
        new_lr = None
        
        if self.config.maintain_effective_batch:
            original_effective = self.original_batch_size * self.original_grad_accum
            new_grad_accum = max(1, original_effective // new_batch)
            
            # LR 재보정 (effective batch 기준)
            if self.config.lr_scale_with_effective and self.original_lr:
                new_effective = new_batch * new_grad_accum
                lr_scale_factor = new_effective / original_effective
                new_lr = self.original_lr * lr_scale_factor
        
        # 검증 컨텍스트의 경우 더 강한 제한 적용
        if context == "validation":
            new_batch = min(new_batch, self.config.max_validation_batch)
            new_grad_accum = 1  # 검증에서는 grad_accum 사용하지 않음
            new_lr = None       # 검증에서는 LR 조정 없음
        
        # 상태 업데이트
        self.current_batch_size = new_batch
        self.current_grad_accum = new_grad_accum
        
        # 복구 히스토리 기록
        recovery_record = {
            "timestamp": time.time(),
            "context": context,
            "oom_count": self.oom_count,
            "batch_change": f"{old_batch}→{new_batch}",
            "grad_accum": new_grad_accum,
            "new_lr": new_lr
        }
        self.recovery_history.append(recovery_record)
        
        self.logger.warning(
            f"🔧 OOM 복구 시도 - "
            f"Batch: {old_batch}→{new_batch}, "
            f"GradAccum: {self.current_grad_accum}, "
            f"새 LR: {new_lr}"
        )
        
        return True, new_batch, new_grad_accum, new_lr
    
    def get_current_params(self) -> Tuple[int, int]:
        """현재 배치 크기와 grad_accum 반환"""
        return self.current_batch_size, self.current_grad_accum
    
    def get_validation_batch_size(self) -> int:
        """검증용 배치 크기 반환 (강제 제한 적용)"""
        return min(self.current_batch_size, self.config.max_validation_batch)
    
    def reset_oom_count(self) -> None:
        """OOM 카운터 리셋 (에포크 시작 시 호출)"""
        self.oom_count = 0
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """복구 히스토리 요약 반환"""
        return {
            "total_oom_recoveries": len(self.recovery_history),
            "current_batch_size": self.current_batch_size,
            "current_grad_accum": self.current_grad_accum,
            "batch_size_reduction": f"{self.original_batch_size}→{self.current_batch_size}",
            "recovery_history": self.recovery_history[-5:]  # 최근 5개만
        }


def oom_safe_training_step(
    oom_guard: CUDAOOMGuard,
    training_step_fn: Callable,
    *args,
    context: str = "training",
    **kwargs
) -> Tuple[bool, Any]:
    """
    OOM 안전 학습 스텝 래퍼
    
    Args:
        oom_guard: OOM 가드 인스턴스
        training_step_fn: 학습 스텝 함수
        context: 실행 컨텍스트
        *args, **kwargs: 학습 스텝 함수 인자
        
    Returns:
        Tuple[bool, Any]: (성공 여부, 결과)
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # torch.compile 완충 처리
            if oom_guard.compile_warmup_active:
                oom_guard.step_compile_warmup()
            
            # 학습 스텝 실행
            result = training_step_fn(*args, **kwargs)
            return True, result
            
        except torch.cuda.OutOfMemoryError as e:
            # OOM 복구 시도
            can_recover, new_batch, new_grad_accum, new_lr = oom_guard.handle_oom_error(e, context)
            
            if not can_recover:
                return False, None
            
            # 새로운 파라미터로 재시도를 위해 kwargs 업데이트
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = new_batch
            if 'grad_accum_steps' in kwargs:
                kwargs['grad_accum_steps'] = new_grad_accum
            if 'learning_rate' in kwargs and new_lr:
                kwargs['learning_rate'] = new_lr
            
            # 잠시 대기 후 재시도
            time.sleep(2)
            continue
            
        except Exception as e:
            # 다른 에러는 그대로 전파
            raise e
    
    # 최대 재시도 횟수 초과
    return False, None


if __name__ == "__main__":
    print("🧪 CUDA OOM Guard 시스템 테스트 (1단계 필수)")
    print("=" * 60)
    
    # 설정 테스트
    config = OOMGuardConfig()
    guard = CUDAOOMGuard(config)
    
    # 초기 설정
    guard.setup_training_params(batch_size=16, grad_accum_steps=4, learning_rate=2e-4)
    
    # torch.compile 완충 테스트
    warmup_batch, warmup_grad_accum = guard.enable_compile_warmup()
    print(f"✅ Compile 완충: Batch={warmup_batch}, GradAccum={warmup_grad_accum}")
    
    # 가상 OOM 복구 테스트
    fake_error = torch.cuda.OutOfMemoryError("CUDA out of memory")
    can_recover, new_batch, new_grad_accum, new_lr = guard.handle_oom_error(fake_error, "training")
    
    if can_recover:
        print(f"✅ OOM 복구 성공: Batch={new_batch}, GradAccum={new_grad_accum}, LR={new_lr}")
    
    # 복구 요약
    summary = guard.get_recovery_summary()
    print(f"📊 복구 요약: {summary['total_oom_recoveries']}회 복구")
    
    print("🎉 CUDA OOM Guard 테스트 완료!")