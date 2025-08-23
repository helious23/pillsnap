"""
Memory State Manager 단위 테스트
"""

import pytest
import time
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.memory_state_manager import (
    MemoryStateManager, 
    MemoryState, 
    MemoryThresholds,
    create_rtx5080_manager,
    memory_safe_operation
)


class TestMemoryStateManager:
    """Memory State Manager 테스트 클래스"""
    
    def test_memory_state_determination(self):
        """메모리 상태 결정 로직 테스트"""
        manager = MemoryStateManager()
        
        # 각 임계값 테스트 (기본값: safe=10, warning=12, critical=14, emergency=15)
        assert manager._determine_memory_state(5.0) == MemoryState.SAFE
        assert manager._determine_memory_state(11.0) == MemoryState.SAFE    # < 12GB
        assert manager._determine_memory_state(12.0) == MemoryState.WARNING # >= 12GB
        assert manager._determine_memory_state(13.0) == MemoryState.WARNING # < 14GB
        assert manager._determine_memory_state(14.0) == MemoryState.CRITICAL # >= 14GB
        assert manager._determine_memory_state(15.0) == MemoryState.EMERGENCY # >= 15GB
    
    def test_safe_batch_size_calculation(self):
        """안전한 배치 크기 계산 테스트"""
        manager = MemoryStateManager()
        base_batch_size = 16
        
        # Mock 메모리 상태
        with patch.object(manager, 'get_current_memory_stats') as mock_stats:
            # SAFE 상태
            mock_stats.return_value.current_state = MemoryState.SAFE
            assert manager.get_safe_batch_size(base_batch_size) == 16
            
            # WARNING 상태 (75% 감소)
            mock_stats.return_value.current_state = MemoryState.WARNING
            assert manager.get_safe_batch_size(base_batch_size) == 12
            
            # CRITICAL 상태 (50% 감소)
            mock_stats.return_value.current_state = MemoryState.CRITICAL
            assert manager.get_safe_batch_size(base_batch_size) == 8
            
            # EMERGENCY 상태 (25% 감소)
            mock_stats.return_value.current_state = MemoryState.EMERGENCY
            assert manager.get_safe_batch_size(base_batch_size) == 4
    
    def test_callback_registration(self):
        """콜백 함수 등록 테스트"""
        manager = MemoryStateManager()
        callback_called = False
        
        def test_callback(stats):
            nonlocal callback_called
            callback_called = True
        
        # 콜백 등록
        manager.register_state_callback(MemoryState.WARNING, test_callback)
        
        # 콜백이 등록되었는지 확인
        assert len(manager.state_change_callbacks[MemoryState.WARNING]) == 1
        assert manager.state_change_callbacks[MemoryState.WARNING][0] == test_callback
    
    def test_memory_report_generation(self):
        """메모리 보고서 생성 테스트"""
        manager = MemoryStateManager()
        report = manager.get_memory_report()
        
        # 보고서 구조 검증
        assert 'current_state' in report
        assert 'gpu_memory' in report
        assert 'system_memory' in report
        assert 'statistics' in report
        
        # GPU 메모리 정보 검증
        gpu_info = report['gpu_memory']
        assert 'allocated' in gpu_info
        assert 'total' in gpu_info
        assert 'utilization' in gpu_info
    
    def test_context_manager(self):
        """Context Manager 기능 테스트"""
        manager = MemoryStateManager()
        
        with manager:
            assert manager._monitoring_active == True
        
        # Context 종료 후 모니터링 중지 확인
        time.sleep(0.1)  # 스레드 종료 대기
        assert manager._monitoring_active == False
    
    @patch('torch.cuda.is_available')
    def test_cpu_mode_fallback(self, mock_cuda_available):
        """CUDA 미사용 환경에서 CPU 모드 fallback 테스트"""
        mock_cuda_available.return_value = False
        
        manager = MemoryStateManager()
        assert manager.device.type == 'cpu'
        
        # CPU 모드에서도 메모리 통계 조회 가능
        stats = manager.get_current_memory_stats()
        assert stats.gpu_allocated == 0.0
        assert stats.gpu_total == 0.0
    
    def test_rtx5080_manager_creation(self):
        """RTX 5080 전용 매니저 생성 테스트"""
        manager = create_rtx5080_manager()
        
        # RTX 5080 최적화 임계값 확인
        assert manager.thresholds.safe_threshold == 10.0
        assert manager.thresholds.warning_threshold == 12.0
        assert manager.thresholds.critical_threshold == 14.0
        assert manager.thresholds.emergency_threshold == 15.0
        assert manager.monitoring_interval == 0.5
    
    def test_memory_safe_decorator(self):
        """메모리 안전 데코레이터 테스트"""
        @memory_safe_operation
        def test_function():
            return "success"
        
        # 정상 실행 테스트
        result = test_function()
        assert result == "success"
    
    def test_force_cleanup(self):
        """강제 메모리 정리 테스트"""
        manager = MemoryStateManager()
        initial_cleanup_count = manager.stats['total_cleanups']
        
        manager.force_cleanup()
        
        # 정리 횟수 증가 확인
        assert manager.stats['total_cleanups'] == initial_cleanup_count + 1
        assert manager.stats['oom_prevented'] == 1


class TestMemoryThresholds:
    """Memory Thresholds 테스트 클래스"""
    
    def test_default_thresholds(self):
        """기본 임계값 설정 테스트"""
        thresholds = MemoryThresholds()
        
        assert thresholds.safe_threshold == 10.0
        assert thresholds.warning_threshold == 12.0
        assert thresholds.critical_threshold == 14.0
        assert thresholds.emergency_threshold == 15.0
    
    def test_custom_thresholds(self):
        """사용자 정의 임계값 테스트"""
        thresholds = MemoryThresholds(
            safe_threshold=8.0,
            warning_threshold=10.0,
            critical_threshold=12.0,
            emergency_threshold=14.0
        )
        
        assert thresholds.safe_threshold == 8.0
        assert thresholds.warning_threshold == 10.0
        assert thresholds.critical_threshold == 12.0
        assert thresholds.emergency_threshold == 14.0


@pytest.mark.integration
class TestMemoryStateManagerIntegration:
    """Memory State Manager 통합 테스트"""
    
    def test_real_memory_monitoring(self):
        """실제 메모리 모니터링 테스트 (CUDA 환경에서만)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 환경이 아님")
        
        manager = MemoryStateManager(monitoring_interval=0.1)
        
        # 모니터링 시작
        manager.start_monitoring()
        
        try:
            # 잠시 대기하여 모니터링 동작 확인
            time.sleep(0.5)
            
            # 현재 상태 확인
            stats = manager.get_current_memory_stats()
            assert stats.gpu_total > 0
            assert stats.current_state in MemoryState
            
            # 상태 히스토리 확인
            assert len(manager.state_history) > 0
            
        finally:
            manager.stop_monitoring()
    
    def test_oom_simulation(self):
        """OOM 상황 시뮬레이션 테스트"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 환경이 아님")
        
        manager = MemoryStateManager()
        callback_triggered = False
        
        def emergency_callback(stats):
            nonlocal callback_triggered
            callback_triggered = True
        
        manager.register_state_callback(MemoryState.EMERGENCY, emergency_callback)
        
        try:
            # 대용량 텐서 할당으로 메모리 압박 시뮬레이션
            large_tensor = torch.randn(1000, 1000, 1000, device='cuda')
            
            # 강제 정리 테스트
            manager.force_cleanup()
            
            del large_tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM 발생 시 정리 동작 확인
                manager.force_cleanup()
                assert manager.stats['oom_prevented'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])