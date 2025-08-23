#!/usr/bin/env python3
"""
Progressive Resize 전략 실제 동작 데모

RTX 5080 16GB 환경에서 Stage 3-4 대용량 데이터셋
훈련을 위한 Progressive Resize 전략 시연
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.progressive_resize_strategy import (
    create_progressive_scheduler,
    create_progressive_dataloader,
    ResizeSchedule
)
from src.utils.core import PillSnapLogger


def demo_progressive_resize():
    """Progressive Resize 전략 데모"""
    logger = PillSnapLogger(__name__)
    
    print("=" * 70)
    print("🚀 Progressive Resize 전략 실제 동작 데모")
    print("=" * 70)
    print()
    
    # 1. 스케줄러 생성
    print("1️⃣ Progressive Resize 스케줄러 생성")
    scheduler = create_progressive_scheduler(
        initial_size=224,
        target_size=384,
        warmup_epochs=5,
        transition_epochs=15,
        schedule=ResizeSchedule.COSINE
    )
    
    # 훈련 스케줄 요약
    summary = scheduler.get_training_schedule_summary()
    print(f"   총 에포크: {summary['total_epochs']}")
    print(f"   메모리 한계: {summary['memory_optimization']['max_memory_gb']}GB (RTX 5080)")
    print()
    
    for phase_name, phase_info in summary['phases'].items():
        print(f"   📋 {phase_name.upper()} 단계:")
        print(f"      에포크: {phase_info['epochs']}")
        print(f"      해상도: {phase_info['size']}")
        print(f"      목적: {phase_info['purpose']}")
        print()
    
    # 2. 에포크별 해상도 변화 시뮬레이션
    print("2️⃣ 에포크별 해상도 및 배치 크기 변화")
    print("-" * 60)
    print(f"{'Epoch':<6} {'Size(px)':<10} {'Batch Size':<12} {'Memory Est.':<12} {'Phase':<12}")
    print("-" * 60)
    
    test_epochs = [0, 2, 5, 8, 12, 15, 20, 25, 30, 40, 50]
    
    for epoch in test_epochs:
        # 현재 해상도 계산
        current_size = scheduler.get_current_size(epoch)
        
        # 최적 배치 크기 계산
        optimal_batch = scheduler.get_optimal_batch_size(base_batch_size=32, current_size=current_size)
        
        # 메모리 사용량 추정 (MB)
        memory_est = optimal_batch * (current_size / 224) ** 2 * 2.0  # MB
        
        # 단계 판정
        if epoch < 5:
            phase = "Warmup"
        elif epoch < 20:
            phase = "Transition"
        else:
            phase = "Stable"
        
        print(f"{epoch:<6} {current_size:<10} {optimal_batch:<12} {memory_est:.1f}MB{'':<5} {phase:<12}")
    
    print("-" * 60)
    print()
    
    # 3. 실시간 상태 모니터링 시뮬레이션
    print("3️⃣ 실시간 훈련 상태 모니터링 시뮬레이션")
    print("-" * 50)
    
    for epoch in [0, 5, 10, 15, 20]:
        # 상태 업데이트
        batch_size = scheduler.get_optimal_batch_size(32, scheduler.get_current_size(epoch))
        state = scheduler.update_state(epoch=epoch, batch_idx=100, batch_size=batch_size)
        
        print(f"Epoch {epoch:2d}: {state.current_size}px, 배치={state.batch_size}, "
              f"메모리={state.memory_usage_gb:.1f}GB")
        
        time.sleep(0.1)  # 시각적 효과
    
    print()
    
    # 4. 변환 파이프라인 생성 데모
    print("4️⃣ 해상도별 변환 파이프라인")
    print("-" * 40)
    
    for epoch in [0, 10, 20]:
        current_size = scheduler.get_current_size(epoch)
        scheduler.update_state(epoch, 0, 16)  # 상태 업데이트
        
        train_transform = scheduler.create_transform('train')
        val_transform = scheduler.create_transform('val')
        
        print(f"Epoch {epoch:2d} ({current_size}px):")
        print(f"   훈련용 변환: {len(train_transform.transforms)}개 단계")
        print(f"   검증용 변환: {len(val_transform.transforms)}개 단계")
        print()
    
    # 5. 성능 통계
    print("5️⃣ Progressive Resize 성능 통계")
    print("-" * 40)
    
    stats = scheduler.get_current_stats()
    current_state = stats['current_state']
    perf_stats = stats['performance_stats']
    
    print(f"현재 상태:")
    print(f"   해상도: {current_state['size']}")
    print(f"   에포크: {current_state['epoch']}")
    print(f"   메모리 사용량: {current_state['memory_usage_gb']}GB")
    print(f"   배치 크기: {current_state['batch_size']}")
    print()
    
    print(f"성능 통계:")
    print(f"   총 해상도 변경: {perf_stats['total_resizes']}회")
    print(f"   배치 크기 조정: {perf_stats['batch_adjustments']}회")
    print(f"   이력 기록 수: {stats['resize_history_count']}개")
    print()
    
    # 6. RTX 5080 최적화 확인
    print("6️⃣ RTX 5080 16GB 환경 최적화 검증")
    print("-" * 45)
    
    # 다양한 해상도에서 메모리 안전성 확인
    test_sizes = [224, 288, 352, 384, 448, 512]
    
    print(f"{'Size(px)':<10} {'Max Batch':<12} {'Memory Est.':<15} {'Status':<10}")
    print("-" * 45)
    
    for size in test_sizes:
        max_batch = scheduler._calculate_max_batch_size(size)
        memory_est_gb = max_batch * (size / 224) ** 2 * 0.002  # GB 추정
        
        if memory_est_gb <= 14.0:
            status = "✅ 안전"
        elif memory_est_gb <= 15.0:
            status = "⚠️  주의"  
        else:
            status = "❌ 위험"
        
        print(f"{size:<10} {max_batch:<12} {memory_est_gb:.2f}GB{'':<8} {status:<10}")
    
    print("-" * 45)
    print()
    
    # 7. 실전 사용 예제
    print("7️⃣ 실전 사용 예제 코드")
    print("-" * 30)
    
    example_code = '''
# Progressive Resize 훈련 루프 예제
scheduler = create_progressive_scheduler(
    initial_size=224,
    target_size=384,
    warmup_epochs=10,
    transition_epochs=20
)

for epoch in range(total_epochs):
    # 에포크별 해상도 업데이트
    current_size = scheduler.get_current_size(epoch)
    
    # 최적 배치 크기 계산
    optimal_batch = scheduler.get_optimal_batch_size(
        base_batch_size=32, 
        current_size=current_size
    )
    
    # 변환 파이프라인 업데이트 (필요시)
    if scheduler.should_update_transform(epoch, 0):
        new_transform = scheduler.create_transform('train')
        # 데이터셋에 적용...
    
    # 훈련 루프...
    for batch_idx, batch in enumerate(dataloader):
        # 상태 모니터링
        scheduler.update_state(epoch, batch_idx, optimal_batch)
        
        # 실제 훈련 코드...
    '''
    
    print(example_code)
    print()
    
    print("=" * 70)
    print("✅ Progressive Resize 전략 데모 완료!")
    print()
    print("주요 장점:")
    print("  🚀 훈련 초기 빠른 수렴 (224px)")
    print("  💾 메모리 효율성 극대화 (RTX 5080 16GB)")
    print("  📈 점진적 품질 향상 (224px → 384px)")
    print("  🔧 배치 크기 자동 최적화")
    print("  📊 실시간 성능 모니터링")
    print()
    print("Stage 3-4 대용량 훈련 준비 완료! 🎯")
    print("=" * 70)


if __name__ == "__main__":
    demo_progressive_resize()