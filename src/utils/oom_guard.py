"""
OOM 가드 유틸리티

목적: 학습 중 Out of Memory 에러 복구 및 배치 크기 자동 조정
핵심 기능:
- OOM 에러 감지 및 자동 복구
- 배치 크기 동적 조정
- GPU 메모리 정리 및 최적화
- 재시도 로직 및 가드레일

사용법:
    from src.utils.oom_guard import OOMGuard, handle_oom_error
    
    # OOM 가드 초기화
    oom_guard = OOMGuard(initial_batch_size=64, min_batch_size=1)
    
    # 학습 루프에서 사용
    try:
        loss.backward()
    except RuntimeError as e:
        if "out of memory" in str(e):
            success = handle_oom_error(e, oom_guard)
            if success:
                # 새로운 배치 크기로 재시도
                continue
"""

import gc
import logging
import psutil
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class OOMGuardState:
    """OOM 가드 상태 정보"""
    current_batch_size: int
    retry_count: int = 0
    total_oom_events: int = 0
    successful_recoveries: int = 0
    last_oom_epoch: int = -1
    last_oom_batch: int = -1
    memory_cleared_count: int = 0
    amp_enabled_forced: bool = False


class OOMGuard:
    """
    Out of Memory 복구 가드
    
    Features:
    - 배치 크기 자동 감소
    - GPU 메모리 캐시 정리
    - AMP 강제 활성화
    - 재시도 횟수 제한
    - 복구 통계 추적
    """
    
    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_retries: int = 4,
        reduction_factor: float = 0.5,
        memory_threshold: float = 0.9
    ):
        """
        Args:
            initial_batch_size: 초기 배치 크기
            min_batch_size: 최소 배치 크기 (이하로 줄이지 않음)
            max_retries: 최대 재시도 횟수
            reduction_factor: 배치 크기 감소 비율 (0.5 = 50% 감소)
            memory_threshold: 메모리 사용량 임계값 (90%)
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_retries = max_retries
        self.reduction_factor = reduction_factor
        self.memory_threshold = memory_threshold
        
        # 상태 초기화
        self.state = OOMGuardState(current_batch_size=initial_batch_size)
        
        logger.info(f"Initialized OOMGuard: batch_size={initial_batch_size}, "
                   f"min_batch_size={min_batch_size}, max_retries={max_retries}")
    
    @property
    def current_batch_size(self) -> int:
        """현재 배치 크기 반환"""
        return self.state.current_batch_size
    
    def can_retry(self) -> bool:
        """재시도 가능 여부 확인"""
        return (self.state.retry_count < self.max_retries and 
                self.state.current_batch_size >= self.min_batch_size)
    
    def reduce_batch_size(self) -> int:
        """배치 크기 감소"""
        old_size = self.state.current_batch_size
        new_size = max(
            self.min_batch_size,
            int(self.state.current_batch_size * self.reduction_factor)
        )
        
        self.state.current_batch_size = new_size
        logger.info(f"Reduced batch size: {old_size} → {new_size}")
        
        return new_size
    
    def clear_gpu_memory(self) -> Dict[str, float]:
        """GPU 메모리 정리"""
        if not torch.cuda.is_available():
            return {"cleared_mb": 0, "total_mb": 0, "free_mb": 0}
        
        # 메모리 정리 전 상태
        before_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        
        # 캐시 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        # 메모리 정리 후 상태
        after_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        cleared_mb = (after_free - before_free) / 1024 / 1024
        total_mb = total_memory / 1024 / 1024
        free_mb = after_free / 1024 / 1024
        
        self.state.memory_cleared_count += 1
        
        logger.info(f"GPU memory cleared: {cleared_mb:.1f}MB freed, "
                   f"{free_mb:.1f}MB/{total_mb:.1f}MB available")
        
        return {
            "cleared_mb": cleared_mb,
            "total_mb": total_mb,
            "free_mb": free_mb,
            "utilization": (total_mb - free_mb) / total_mb
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """시스템 메모리 정보 조회"""
        info = {"gpu": {}, "cpu": {}}
        
        # GPU 메모리
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            info["gpu"] = {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "total_mb": total,
                "utilization": reserved / total,
                "free_mb": total - reserved
            }
        
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        info["cpu"] = {
            "total_mb": cpu_memory.total / 1024 / 1024,
            "available_mb": cpu_memory.available / 1024 / 1024,
            "percent": cpu_memory.percent,
            "used_mb": cpu_memory.used / 1024 / 1024
        }
        
        return info
    
    def handle_oom(self, error: Exception, epoch: int = -1, batch_idx: int = -1) -> bool:
        """
        OOM 에러 처리
        
        Args:
            error: OOM 에러 객체
            epoch: 현재 에포크
            batch_idx: 현재 배치 인덱스
            
        Returns:
            bool: 복구 성공 여부
        """
        # OOM 이벤트 기록
        self.state.total_oom_events += 1
        self.state.last_oom_epoch = epoch
        self.state.last_oom_batch = batch_idx
        
        logger.warning(f"OOM detected at epoch {epoch}, batch {batch_idx}: {str(error)[:100]}")
        
        # 재시도 가능성 확인
        if not self.can_retry():
            logger.error(f"Cannot retry: retry_count={self.state.retry_count}, "
                        f"batch_size={self.state.current_batch_size}")
            return False
        
        # 메모리 정리
        memory_info = self.clear_gpu_memory()
        
        # 배치 크기 감소
        old_batch_size = self.state.current_batch_size
        new_batch_size = self.reduce_batch_size()
        
        # 재시도 카운트 증가
        self.state.retry_count += 1
        self.state.successful_recoveries += 1
        
        logger.info(f"OOM recovery #{self.state.retry_count}: "
                   f"batch_size {old_batch_size}→{new_batch_size}, "
                   f"freed {memory_info['cleared_mb']:.1f}MB")
        
        return True
    
    def reset_retry_count(self):
        """재시도 카운트 리셋 (에포크 완료 시 호출)"""
        if self.state.retry_count > 0:
            logger.info(f"Resetting retry count (was {self.state.retry_count})")
            self.state.retry_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """OOM 가드 통계 반환"""
        return {
            "current_batch_size": self.state.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "retry_count": self.state.retry_count,
            "total_oom_events": self.state.total_oom_events,
            "successful_recoveries": self.state.successful_recoveries,
            "memory_cleared_count": self.state.memory_cleared_count,
            "last_oom_epoch": self.state.last_oom_epoch,
            "last_oom_batch": self.state.last_oom_batch,
            "batch_reduction_ratio": self.state.current_batch_size / self.initial_batch_size,
            "amp_forced": self.state.amp_enabled_forced
        }
    
    def get_state(self) -> Dict[str, Any]:
        """상태 딕셔너리 반환 (체크포인트 저장용)"""
        return {
            "current_batch_size": self.state.current_batch_size,
            "retry_count": self.state.retry_count,
            "total_oom_events": self.state.total_oom_events,
            "successful_recoveries": self.state.successful_recoveries,
            "last_oom_epoch": self.state.last_oom_epoch,
            "last_oom_batch": self.state.last_oom_batch,
            "memory_cleared_count": self.state.memory_cleared_count,
            "amp_enabled_forced": self.state.amp_enabled_forced
        }
    
    def load_state(self, state_dict: Dict[str, Any]):
        """상태 복원 (체크포인트 로드 시)"""
        self.state.current_batch_size = state_dict.get("current_batch_size", self.initial_batch_size)
        self.state.retry_count = state_dict.get("retry_count", 0)
        self.state.total_oom_events = state_dict.get("total_oom_events", 0)
        self.state.successful_recoveries = state_dict.get("successful_recoveries", 0)
        self.state.last_oom_epoch = state_dict.get("last_oom_epoch", -1)
        self.state.last_oom_batch = state_dict.get("last_oom_batch", -1)
        self.state.memory_cleared_count = state_dict.get("memory_cleared_count", 0)
        self.state.amp_enabled_forced = state_dict.get("amp_enabled_forced", False)
        
        logger.info(f"Loaded OOMGuard state: batch_size={self.state.current_batch_size}, "
                   f"total_oom_events={self.state.total_oom_events}")


def handle_oom_error(error: Exception, oom_guard: OOMGuard, 
                    epoch: int = -1, batch_idx: int = -1) -> bool:
    """
    OOM 에러 처리 헬퍼 함수
    
    Args:
        error: RuntimeError 객체
        oom_guard: OOMGuard 인스턴스
        epoch: 현재 에포크
        batch_idx: 현재 배치 인덱스
        
    Returns:
        bool: 복구 성공 여부
    """
    if "out of memory" not in str(error).lower():
        logger.warning("Error is not OOM related, cannot handle")
        return False
    
    return oom_guard.handle_oom(error, epoch, batch_idx)


def get_optimal_batch_size(model: torch.nn.Module, 
                          input_shape: Tuple[int, ...],
                          device: torch.device,
                          max_batch_size: int = 128,
                          safety_factor: float = 0.8) -> int:
    """
    모델과 입력 크기에 따른 최적 배치 크기 추정
    
    Args:
        model: PyTorch 모델
        input_shape: 입력 텐서 크기 (C, H, W)
        device: 디바이스
        max_batch_size: 최대 배치 크기
        safety_factor: 안전 계수 (80% 메모리만 사용)
        
    Returns:
        int: 추정된 최적 배치 크기
    """
    if not torch.cuda.is_available():
        return min(32, max_batch_size)  # CPU 기본값
    
    model.eval()
    optimal_batch = 1
    
    try:
        # 이진 탐색으로 최적 배치 크기 찾기
        low, high = 1, max_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # 테스트 텐서 생성
                test_input = torch.randn(mid, *input_shape).to(device)
                
                # 메모리 정리
                torch.cuda.empty_cache()
                
                # Forward 패스 테스트
                with torch.no_grad():
                    _ = model(test_input)
                
                # 성공하면 더 큰 배치 시도
                optimal_batch = mid
                low = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM 발생하면 더 작은 배치 시도
                    high = mid - 1
                else:
                    # 다른 에러는 무시하고 계속
                    high = mid - 1
            
            finally:
                # 테스트 텐서 정리
                if 'test_input' in locals():
                    del test_input
                torch.cuda.empty_cache()
    
    except Exception as e:
        logger.warning(f"Failed to estimate optimal batch size: {e}")
        optimal_batch = 16  # 안전한 기본값
    
    # 안전 계수 적용
    optimal_batch = int(optimal_batch * safety_factor)
    optimal_batch = max(1, min(optimal_batch, max_batch_size))
    
    logger.info(f"Estimated optimal batch size: {optimal_batch}")
    return optimal_batch


def monitor_memory_usage() -> Dict[str, Any]:
    """
    메모리 사용량 모니터링
    
    Returns:
        Dict: 메모리 사용량 정보
    """
    info = {
        "timestamp": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
        "gpu": {},
        "cpu": {}
    }
    
    # GPU 메모리
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        info["gpu"] = {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "total_bytes": total,
            "allocated_mb": allocated / 1024 / 1024,
            "reserved_mb": reserved / 1024 / 1024,
            "total_mb": total / 1024 / 1024,
            "utilization": reserved / total,
            "free_mb": (total - reserved) / 1024 / 1024
        }
    
    # CPU 메모리
    cpu_memory = psutil.virtual_memory()
    info["cpu"] = {
        "total_bytes": cpu_memory.total,
        "available_bytes": cpu_memory.available,
        "used_bytes": cpu_memory.used,
        "percent": cpu_memory.percent,
        "total_mb": cpu_memory.total / 1024 / 1024,
        "available_mb": cpu_memory.available / 1024 / 1024,
        "used_mb": cpu_memory.used / 1024 / 1024
    }
    
    return info


if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)
    
    # OOMGuard 테스트
    oom_guard = OOMGuard(initial_batch_size=64, min_batch_size=4)
    print(f"Initial state: {oom_guard.get_stats()}")
    
    # 메모리 정보
    memory_info = monitor_memory_usage()
    print(f"Memory info: {memory_info}")
    
    # OOM 시뮬레이션
    fake_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    success = handle_oom_error(fake_error, oom_guard, epoch=1, batch_idx=10)
    print(f"OOM recovery success: {success}")
    print(f"After OOM: {oom_guard.get_stats()}")
    
    print("OOMGuard test completed")