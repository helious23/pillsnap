"""
OOM 방지 상태 머신 (Memory State Manager)

RTX 5080 16GB GPU 메모리를 안전하게 관리하는 상태 기반 시스템.
실시간 메모리 모니터링과 동적 배치 크기 조정을 통해 OOM 방지.

State Transitions:
SAFE (0-10GB) → WARNING (10-12GB) → CRITICAL (12-14GB) → EMERGENCY (14GB+)

Author: Claude Code - PillSnap ML Team
Date: 2025-08-23
"""

import gc
import time
import torch
import psutil
import threading
import sys
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.core import PillSnapLogger


class MemoryState(Enum):
    """메모리 상태 정의"""
    SAFE = "safe"           # 0-10GB: 정상 운영
    WARNING = "warning"     # 10-12GB: 주의 필요
    CRITICAL = "critical"   # 12-14GB: 위험 상태
    EMERGENCY = "emergency" # 14GB+: 긴급 상태


@dataclass
class MemoryStats:
    """메모리 통계 정보"""
    gpu_allocated: float      # GB
    gpu_reserved: float       # GB
    gpu_max_allocated: float  # GB
    gpu_total: float         # GB
    system_used: float       # GB
    system_available: float   # GB
    current_state: MemoryState
    timestamp: float


@dataclass  
class MemoryThresholds:
    """메모리 임계값 설정"""
    safe_threshold: float = 10.0      # GB
    warning_threshold: float = 12.0   # GB
    critical_threshold: float = 14.0  # GB
    emergency_threshold: float = 15.0 # GB


class MemoryStateManager:
    """
    OOM 방지 상태 머신 클래스
    
    실시간 GPU/시스템 메모리 모니터링과 동적 배치 크기 조정을 통해
    RTX 5080 16GB 환경에서 안정적인 훈련 보장.
    """
    
    def __init__(self, 
                 thresholds: Optional[MemoryThresholds] = None,
                 monitoring_interval: float = 1.0,
                 enable_auto_cleanup: bool = True):
        """
        초기화
        
        Args:
            thresholds: 메모리 임계값 설정
            monitoring_interval: 모니터링 간격 (초)
            enable_auto_cleanup: 자동 메모리 정리 활성화
        """
        self.logger = PillSnapLogger(__name__)
        self.thresholds = thresholds or MemoryThresholds()
        self.monitoring_interval = monitoring_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # 상태 관리
        self.current_state = MemoryState.SAFE
        self.previous_state = MemoryState.SAFE
        self.state_history: list[MemoryStats] = []
        
        # 콜백 함수들
        self.state_change_callbacks: Dict[MemoryState, list[Callable]] = {
            state: [] for state in MemoryState
        }
        
        # 모니터링 스레드
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 통계
        self.stats = {
            'total_cleanups': 0,
            'oom_prevented': 0,
            'max_memory_seen': 0.0,
            'state_transitions': 0
        }
        
        # GPU 가용성 확인
        if not torch.cuda.is_available():
            self.logger.warning("CUDA 미사용 가능. CPU 모드로 동작합니다.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            self.logger.info(f"GPU 감지: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def get_current_memory_stats(self) -> MemoryStats:
        """현재 메모리 상태 조회"""
        if self.device.type == 'cuda':
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_total = 0.0
            
        # 시스템 메모리
        memory = psutil.virtual_memory()
        system_used = memory.used / 1024**3
        system_available = memory.available / 1024**3
        
        # 상태 결정 (GPU 메모리 기준)
        current_state = self._determine_memory_state(gpu_allocated)
        
        return MemoryStats(
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved, 
            gpu_max_allocated=gpu_max_allocated,
            gpu_total=gpu_total,
            system_used=system_used,
            system_available=system_available,
            current_state=current_state,
            timestamp=time.time()
        )
    
    def _determine_memory_state(self, gpu_memory_gb: float) -> MemoryState:
        """GPU 메모리 사용량에 따른 상태 결정"""
        if gpu_memory_gb >= self.thresholds.emergency_threshold:    # >= 15.0GB
            return MemoryState.EMERGENCY
        elif gpu_memory_gb >= self.thresholds.critical_threshold:  # >= 14.0GB  
            return MemoryState.CRITICAL
        elif gpu_memory_gb >= self.thresholds.warning_threshold:   # >= 12.0GB
            return MemoryState.WARNING
        else:  # < 12.0GB (including safe threshold 10.0GB)
            return MemoryState.SAFE
    
    def register_state_callback(self, state: MemoryState, callback: Callable[[MemoryStats], None]):
        """상태 변경 시 호출될 콜백 함수 등록"""
        self.state_change_callbacks[state].append(callback)
    
    def start_monitoring(self):
        """백그라운드 메모리 모니터링 시작"""
        if self._monitoring_active:
            self.logger.warning("모니터링이 이미 실행 중입니다.")
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("메모리 모니터링 시작됨")
    
    def stop_monitoring(self):
        """백그라운드 메모리 모니터링 중지"""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2.0)
        self.logger.info("메모리 모니터링 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프 (백그라운드 스레드)"""
        while self._monitoring_active:
            try:
                with self._lock:
                    stats = self.get_current_memory_stats()
                    self._update_state(stats)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_state(self, stats: MemoryStats):
        """상태 업데이트 및 콜백 실행"""
        self.previous_state = self.current_state
        self.current_state = stats.current_state
        self.state_history.append(stats)
        
        # 통계 업데이트
        self.stats['max_memory_seen'] = max(self.stats['max_memory_seen'], stats.gpu_allocated)
        
        # 상태 변경 감지
        if self.current_state != self.previous_state:
            self.stats['state_transitions'] += 1
            self.logger.info(f"메모리 상태 변경: {self.previous_state.value} → {self.current_state.value}")
            self.logger.info(f"GPU 메모리: {stats.gpu_allocated:.1f}GB / {stats.gpu_total:.1f}GB")
            
            # 콜백 실행
            for callback in self.state_change_callbacks[self.current_state]:
                try:
                    callback(stats)
                except Exception as e:
                    self.logger.error(f"콜백 실행 오류: {e}")
        
        # 자동 정리 실행
        if self.enable_auto_cleanup:
            if self.current_state in [MemoryState.CRITICAL, MemoryState.EMERGENCY]:
                self._perform_cleanup(stats)
    
    def _perform_cleanup(self, stats: MemoryStats):
        """메모리 정리 수행"""
        self.logger.warning(f"메모리 정리 시작 - 현재: {stats.gpu_allocated:.1f}GB")
        
        if self.device.type == 'cuda':
            # GPU 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 가비지 컬렉션
        collected = gc.collect()
        
        self.stats['total_cleanups'] += 1
        
        # 정리 후 메모리 재확인
        new_stats = self.get_current_memory_stats()
        freed_memory = stats.gpu_allocated - new_stats.gpu_allocated
        
        if freed_memory > 0.1:  # 100MB 이상 해제
            self.logger.info(f"메모리 정리 완료: {freed_memory:.1f}GB 해제")
        else:
            self.logger.warning("메모리 정리 효과 제한적")
    
    def get_safe_batch_size(self, base_batch_size: int) -> int:
        """현재 메모리 상태에 따른 안전한 배치 크기 계산"""
        current_stats = self.get_current_memory_stats()
        
        if current_stats.current_state == MemoryState.SAFE:
            return base_batch_size
        elif current_stats.current_state == MemoryState.WARNING:
            return max(1, int(base_batch_size * 0.75))
        elif current_stats.current_state == MemoryState.CRITICAL:
            return max(1, int(base_batch_size * 0.5))
        else:  # EMERGENCY
            return max(1, int(base_batch_size * 0.25))
    
    def force_cleanup(self):
        """강제 메모리 정리"""
        stats = self.get_current_memory_stats()
        self._perform_cleanup(stats)
        self.stats['oom_prevented'] += 1
        self.logger.info("강제 메모리 정리 완료")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """상세 메모리 보고서 생성"""
        current_stats = self.get_current_memory_stats()
        
        return {
            'current_state': current_stats.current_state.value,
            'gpu_memory': {
                'allocated': f"{current_stats.gpu_allocated:.1f}GB",
                'reserved': f"{current_stats.gpu_reserved:.1f}GB", 
                'max_allocated': f"{current_stats.gpu_max_allocated:.1f}GB",
                'total': f"{current_stats.gpu_total:.1f}GB",
                'utilization': f"{(current_stats.gpu_allocated / current_stats.gpu_total * 100):.1f}%"
            },
            'system_memory': {
                'used': f"{current_stats.system_used:.1f}GB",
                'available': f"{current_stats.system_available:.1f}GB"
            },
            'statistics': self.stats,
            'monitoring_active': self._monitoring_active,
            'state_history_length': len(self.state_history)
        }
    
    def __enter__(self):
        """Context Manager 진입"""
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 종료"""
        self.stop_monitoring()
        
        # 예외 발생 시 강제 정리
        if exc_type is not None:
            self.logger.error(f"예외 발생으로 인한 강제 정리: {exc_type.__name__}")
            self.force_cleanup()


# 편의 함수들
def create_rtx5080_manager() -> MemoryStateManager:
    """RTX 5080 16GB 환경에 최적화된 메모리 매니저 생성"""
    thresholds = MemoryThresholds(
        safe_threshold=10.0,     # RTX 5080 16GB의 62.5%
        warning_threshold=12.0,  # 75%
        critical_threshold=14.0, # 87.5%
        emergency_threshold=15.0 # 93.7%
    )
    
    return MemoryStateManager(
        thresholds=thresholds,
        monitoring_interval=0.5,  # 빠른 모니터링
        enable_auto_cleanup=True
    )


def memory_safe_operation(func: Callable) -> Callable:
    """메모리 안전 데코레이터"""
    def wrapper(*args, **kwargs):
        manager = create_rtx5080_manager()
        with manager:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    manager.logger.error("OOM 감지 - 강제 정리 수행")
                    manager.force_cleanup()
                    raise
                else:
                    raise
    return wrapper