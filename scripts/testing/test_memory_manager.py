#!/usr/bin/env python3
"""
Memory State Manager 실제 동작 테스트 스크립트
"""

import sys
import time
import torch
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.memory_state_manager import create_rtx5080_manager, MemoryState


def test_memory_manager():
    """Memory State Manager 실제 동작 테스트"""
    print("=" * 60)
    print("RTX 5080 Memory State Manager 테스트")
    print("=" * 60)
    
    # RTX 5080 최적화 매니저 생성
    manager = create_rtx5080_manager()
    
    # 현재 메모리 상태 확인
    stats = manager.get_current_memory_stats()
    print(f"GPU 총 메모리: {stats.gpu_total:.1f}GB")
    print(f"현재 사용량: {stats.gpu_allocated:.1f}GB")
    print(f"현재 상태: {stats.current_state.value}")
    print()
    
    # 상태 변경 콜백 등록
    def warning_callback(stats):
        print(f"⚠️  WARNING 상태 진입: {stats.gpu_allocated:.1f}GB")
    
    def critical_callback(stats):
        print(f"🚨 CRITICAL 상태 진입: {stats.gpu_allocated:.1f}GB")
    
    def emergency_callback(stats):
        print(f"💥 EMERGENCY 상태 진입: {stats.gpu_allocated:.1f}GB")
    
    manager.register_state_callback(MemoryState.WARNING, warning_callback)
    manager.register_state_callback(MemoryState.CRITICAL, critical_callback) 
    manager.register_state_callback(MemoryState.EMERGENCY, emergency_callback)
    
    # Context Manager로 모니터링 시작
    with manager:
        print("메모리 모니터링 시작됨")
        
        # 배치 크기 조정 테스트
        base_batch_size = 32
        safe_batch_size = manager.get_safe_batch_size(base_batch_size)
        print(f"권장 배치 크기: {base_batch_size} → {safe_batch_size}")
        print()
        
        # 메모리 보고서 출력
        report = manager.get_memory_report()
        print("📊 메모리 상태 보고서:")
        print(f"  현재 상태: {report['current_state']}")
        print(f"  GPU 사용률: {report['gpu_memory']['utilization']}")
        print(f"  시스템 가용 메모리: {report['system_memory']['available']}")
        print()
        
        # 메모리 부하 시뮬레이션 (안전한 수준)
        print("메모리 부하 테스트 중...")
        try:
            # 작은 텐서들로 점진적 메모리 사용
            tensors = []
            for i in range(5):
                tensor = torch.randn(100, 100, 100, device='cuda' if torch.cuda.is_available() else 'cpu')
                tensors.append(tensor)
                
                current_stats = manager.get_current_memory_stats()
                print(f"  텐서 {i+1}: {current_stats.gpu_allocated:.1f}GB / {current_stats.gpu_total:.1f}GB")
                
                time.sleep(0.5)
            
            # 메모리 정리 테스트
            print("\n강제 메모리 정리 테스트...")
            manager.force_cleanup()
            
            final_stats = manager.get_current_memory_stats()
            print(f"정리 후: {final_stats.gpu_allocated:.1f}GB")
            
            # 텐서 해제
            del tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"메모리 테스트 중 오류: {e}")
        
        print("\n3초간 모니터링 지속...")
        time.sleep(3)
    
    print("모니터링 종료됨")
    print()
    
    # 최종 통계 출력
    final_report = manager.get_memory_report()
    print("📈 최종 통계:")
    print(f"  총 정리 횟수: {final_report['statistics']['total_cleanups']}")
    print(f"  OOM 방지 횟수: {final_report['statistics']['oom_prevented']}")
    print(f"  상태 전환 횟수: {final_report['statistics']['state_transitions']}")
    print(f"  최대 메모리 사용: {final_report['statistics']['max_memory_seen']:.1f}GB")
    
    print("\n✅ Memory State Manager 테스트 완료!")


if __name__ == "__main__":
    test_memory_manager()