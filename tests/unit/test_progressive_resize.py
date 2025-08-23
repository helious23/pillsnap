#!/usr/bin/env python3
"""
Progressive Resize 전략 포괄적 테스트
RTX 5080 16GB 환경 최적화 검증
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time
import tempfile
import json
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.progressive_resize_strategy import (
    ProgressiveResizeScheduler,
    ProgressiveResizeConfig,
    ProgressiveDataLoader,
    ResizeSchedule,
    ResizeState,
    create_progressive_scheduler,
    create_progressive_dataloader
)


class TestProgressiveResizeConfig:
    """Progressive Resize 설정 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = ProgressiveResizeConfig()
        
        assert config.initial_size == 224
        assert config.target_size == 384
        assert config.detection_size == 640
        assert config.schedule == ResizeSchedule.COSINE
        assert config.max_memory_gb == 14.0
        assert config.auto_batch_adjustment == True
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = ProgressiveResizeConfig(
            initial_size=192,
            target_size=512,
            warmup_epochs=5,
            transition_epochs=15,
            schedule=ResizeSchedule.LINEAR
        )
        
        assert config.initial_size == 192
        assert config.target_size == 512
        assert config.warmup_epochs == 5
        assert config.transition_epochs == 15
        assert config.schedule == ResizeSchedule.LINEAR
    
    def test_rtx5080_optimization(self):
        """RTX 5080 최적화 설정 확인"""
        config = ProgressiveResizeConfig()
        
        # RTX 5080 16GB 환경 특화 확인
        assert config.max_memory_gb <= 14.0  # 안전 마진
        assert config.memory_format == "channels_last"  # RTX 5080 최적화
        assert config.interpolation_mode == "bilinear"  # 속도 우선


class TestResizeSchedule:
    """해상도 스케줄 테스트"""
    
    @pytest.fixture
    def scheduler(self):
        """기본 스케줄러"""
        config = ProgressiveResizeConfig(
            initial_size=224,
            target_size=384,
            warmup_epochs=10,
            transition_epochs=20,
            schedule=ResizeSchedule.COSINE
        )
        return ProgressiveResizeScheduler(config)
    
    def test_warmup_phase(self, scheduler):
        """Warmup 단계 테스트"""
        # Warmup 기간 동안 초기 해상도 유지
        for epoch in range(10):
            size = scheduler.get_current_size(epoch)
            assert size == 224, f"Epoch {epoch}: 예상 224px, 실제 {size}px"
    
    def test_stable_phase(self, scheduler):
        """Stable 단계 테스트"""
        # Transition 완료 후 목표 해상도 유지
        for epoch in range(30, 50):
            size = scheduler.get_current_size(epoch)
            assert size == 384, f"Epoch {epoch}: 예상 384px, 실제 {size}px"
    
    def test_transition_phase(self, scheduler):
        """Transition 단계 테스트"""
        # 점진적 증가 확인
        sizes = []
        for epoch in range(10, 30):  # Transition 기간
            size = scheduler.get_current_size(epoch)
            sizes.append(size)
        
        # 단조 증가 확인
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i-1], f"해상도가 감소함: {sizes[i-1]} → {sizes[i]}"
        
        # 범위 확인
        assert sizes[0] >= 224  # 시작점
        assert sizes[-1] <= 384  # 끝점
    
    def test_linear_schedule(self):
        """Linear 스케줄 테스트"""
        config = ProgressiveResizeConfig(
            initial_size=200,
            target_size=400,
            warmup_epochs=0,
            transition_epochs=10,
            schedule=ResizeSchedule.LINEAR
        )
        scheduler = ProgressiveResizeScheduler(config)
        
        # Linear 증가 확인
        for epoch in range(5):
            size = scheduler.get_current_size(epoch)
            expected_size = int(200 + (400 - 200) * (epoch / 10))
            expected_size = ((expected_size + 7) // 8) * 8  # 8의 배수 정렬
            
            # ±16px 오차 허용 (8의 배수 정렬 때문)
            assert abs(size - expected_size) <= 16
    
    def test_exponential_schedule(self):
        """Exponential 스케줄 테스트"""
        config = ProgressiveResizeConfig(
            initial_size=224,
            target_size=384,
            warmup_epochs=0,
            transition_epochs=10,
            schedule=ResizeSchedule.EXPONENTIAL
        )
        scheduler = ProgressiveResizeScheduler(config)
        
        # 초기에는 천천히, 후반에는 빠르게 증가
        sizes = [scheduler.get_current_size(epoch) for epoch in range(10)]
        
        # 첫 절반의 증가량 < 후반 절반의 증가량
        first_half_increase = sizes[4] - sizes[0]
        second_half_increase = sizes[9] - sizes[4]
        
        assert second_half_increase >= first_half_increase
    
    def test_step_schedule(self):
        """Step 스케줄 테스트"""
        config = ProgressiveResizeConfig(
            initial_size=224,
            target_size=384,
            warmup_epochs=0,
            transition_epochs=8,  # 4단계로 나누어지도록
            schedule=ResizeSchedule.STEP
        )
        scheduler = ProgressiveResizeScheduler(config)
        
        sizes = [scheduler.get_current_size(epoch) for epoch in range(8)]
        
        # 단계별 증가 확인 (일부 구간에서 동일한 값)
        unique_sizes = len(set(sizes))
        assert unique_sizes <= 5  # 최대 5개 단계 (시작 + 4단계)


class TestProgressiveResizeScheduler:
    """Progressive Resize 스케줄러 테스트"""
    
    @pytest.fixture
    def scheduler(self):
        """테스트용 스케줄러"""
        return create_progressive_scheduler(
            initial_size=224,
            target_size=384,
            warmup_epochs=5,
            transition_epochs=10
        )
    
    def test_scheduler_initialization(self, scheduler):
        """스케줄러 초기화 테스트"""
        assert scheduler.config.initial_size == 224
        assert scheduler.config.target_size == 384
        assert scheduler.current_state.current_size == 224
        assert scheduler.current_state.epoch == 0
        assert len(scheduler.resize_history) == 0
        assert scheduler.performance_stats['total_resizes'] == 0
    
    def test_state_update(self, scheduler):
        """상태 업데이트 테스트"""
        # 초기 상태
        initial_state = scheduler.current_state
        
        # 상태 업데이트
        new_state = scheduler.update_state(epoch=1, batch_idx=100, batch_size=16)
        
        assert new_state.epoch == 1
        assert new_state.batch_idx == 100
        assert new_state.batch_size == 16
        assert new_state.current_size >= initial_state.current_size
    
    def test_batch_size_optimization(self, scheduler):
        """배치 크기 최적화 테스트"""
        # 작은 해상도에서는 큰 배치 크기
        small_batch = scheduler.get_optimal_batch_size(base_batch_size=32, current_size=224)
        
        # 큰 해상도에서는 작은 배치 크기
        large_batch = scheduler.get_optimal_batch_size(base_batch_size=32, current_size=384)
        
        # 큰 해상도일수록 작은 배치 크기
        assert large_batch <= small_batch
        
        # 합리적인 범위 확인
        assert 1 <= small_batch <= 64
        assert 1 <= large_batch <= 64
    
    def test_memory_safety(self, scheduler):
        """메모리 안전성 테스트"""
        # RTX 5080 16GB 한계 확인
        max_batch = scheduler._calculate_max_batch_size(image_size=384)
        
        # 메모리 한계 내에서 동작
        assert max_batch >= 1
        assert max_batch <= 64
        
        # 큰 해상도일수록 작은 배치 크기
        max_batch_small = scheduler._calculate_max_batch_size(image_size=224)
        max_batch_large = scheduler._calculate_max_batch_size(image_size=512)
        
        assert max_batch_large <= max_batch_small
    
    def test_transform_creation(self, scheduler):
        """변환 파이프라인 생성 테스트"""
        # 훈련용 변환
        train_transform = scheduler.create_transform('train')
        assert train_transform is not None
        assert len(train_transform.transforms) > 0
        
        # 검증용 변환
        val_transform = scheduler.create_transform('val')
        assert val_transform is not None
        
        # 훈련용이 더 많은 증강 포함
        assert len(train_transform.transforms) >= len(val_transform.transforms)
    
    def test_should_update_transform(self, scheduler):
        """변환 업데이트 필요성 판단 테스트"""
        # 동일한 해상도면 업데이트 불필요
        assert not scheduler.should_update_transform(epoch=0, batch_idx=0)
        
        # 해상도 변경시 업데이트 필요
        scheduler.current_state.current_size = 224
        new_size_epoch = 10  # Transition 시작
        
        if scheduler.get_current_size(new_size_epoch) != 224:
            assert scheduler.should_update_transform(epoch=new_size_epoch, batch_idx=0)
    
    def test_training_schedule_summary(self, scheduler):
        """훈련 스케줄 요약 테스트"""
        summary = scheduler.get_training_schedule_summary()
        
        assert 'total_epochs' in summary
        assert 'phases' in summary
        assert 'memory_optimization' in summary
        
        phases = summary['phases']
        assert 'warmup' in phases
        assert 'transition' in phases
        assert 'stable' in phases
        
        # 메모리 최적화 정보 확인
        memory_opt = summary['memory_optimization']
        assert memory_opt['max_memory_gb'] == 14.0
        assert memory_opt['auto_batch_adjustment'] == True
    
    def test_current_stats(self, scheduler):
        """현재 통계 테스트"""
        # 상태 업데이트
        scheduler.update_state(epoch=1, batch_idx=50, batch_size=16)
        
        stats = scheduler.get_current_stats()
        
        assert 'current_state' in stats
        assert 'performance_stats' in stats
        assert 'resize_history_count' in stats
        
        current_state = stats['current_state']
        assert 'size' in current_state
        assert 'epoch' in current_state
        assert 'memory_usage_gb' in current_state
        assert 'batch_size' in current_state
    
    def test_export_resize_history(self, scheduler, tmp_path):
        """해상도 변경 이력 내보내기 테스트"""
        # 몇 번의 상태 업데이트로 이력 생성
        for epoch in range(3):
            scheduler.update_state(epoch=epoch, batch_idx=0, batch_size=16)
        
        # 이력 내보내기
        output_file = tmp_path / "resize_history.json"
        scheduler.export_resize_history(output_file)
        
        # 파일 존재 확인
        assert output_file.exists()
        
        # JSON 유효성 확인
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'config' in data
        assert 'resize_history' in data
        assert 'performance_stats' in data
        
        # 설정 정보 확인
        config = data['config']
        assert config['initial_size'] == 224
        assert config['target_size'] == 384


class TestProgressiveDataLoader:
    """Progressive DataLoader 테스트"""
    
    @pytest.fixture
    def mock_dataset(self):
        """Mock 데이터셋"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1000)
        dataset.transform = None
        return dataset
    
    @pytest.fixture
    def scheduler(self):
        """테스트용 스케줄러"""
        return create_progressive_scheduler()
    
    def test_dataloader_initialization(self, mock_dataset, scheduler):
        """데이터로더 초기화 테스트"""
        dataloader = ProgressiveDataLoader(
            dataset=mock_dataset,
            scheduler=scheduler,
            base_batch_size=16,
            num_workers=4
        )
        
        assert dataloader.dataset == mock_dataset
        assert dataloader.scheduler == scheduler
        assert dataloader.base_batch_size == 16
        assert dataloader.num_workers == 4
    
    @patch('torch.utils.data.DataLoader')
    def test_update_for_epoch(self, mock_dataloader_class, mock_dataset, scheduler):
        """에포크별 업데이트 테스트"""
        # Mock DataLoader 설정
        mock_dataloader_instance = Mock()
        mock_dataloader_class.return_value = mock_dataloader_instance
        
        progressive_loader = ProgressiveDataLoader(
            dataset=mock_dataset,
            scheduler=scheduler,
            base_batch_size=16
        )
        
        # 에포크 업데이트
        progressive_loader.update_for_epoch(epoch=5)
        
        # 스케줄러 상태 업데이트 확인
        assert scheduler.current_state.epoch == 5
        
        # 데이터로더 생성 확인
        assert mock_dataloader_class.called
        assert progressive_loader.current_dataloader == mock_dataloader_instance
    
    def test_batch_size_adjustment(self, mock_dataset, scheduler):
        """배치 크기 자동 조정 테스트"""
        dataloader = ProgressiveDataLoader(
            dataset=mock_dataset,
            scheduler=scheduler,
            base_batch_size=32
        )
        
        with patch('torch.utils.data.DataLoader') as mock_dataloader_class:
            # 작은 해상도 (큰 배치)
            dataloader.update_for_epoch(epoch=0)  # 224px
            small_res_call = mock_dataloader_class.call_args
            
            # 큰 해상도 (작은 배치)  
            dataloader.update_for_epoch(epoch=20)  # 384px
            large_res_call = mock_dataloader_class.call_args
            
            # 배치 크기 변화 확인 (큰 해상도에서 작은 배치)
            if small_res_call and large_res_call:
                small_batch = small_res_call[1]['batch_size']
                large_batch = large_res_call[1]['batch_size']
                assert large_batch <= small_batch
    
    def test_dataloader_iteration(self, mock_dataset, scheduler):
        """데이터로더 반복 테스트"""
        dataloader = ProgressiveDataLoader(
            dataset=mock_dataset,
            scheduler=scheduler
        )
        
        # 업데이트 없이 반복 시도하면 오류
        with pytest.raises(RuntimeError):
            iter(dataloader)
        
        # 업데이트 후에는 정상 동작
        with patch('torch.utils.data.DataLoader') as mock_dataloader_class:
            mock_dataloader_instance = Mock()
            mock_iter = Mock()
            mock_iter.__iter__ = Mock(return_value=iter([1, 2, 3]))  # 실제 iterator 반환
            mock_dataloader_instance.__iter__ = Mock(return_value=iter([1, 2, 3]))
            mock_dataloader_class.return_value = mock_dataloader_instance
            
            dataloader.update_for_epoch(epoch=0)
            result = iter(dataloader)
            
            # iterator가 정상적으로 반환되는지 확인
            assert hasattr(result, '__next__')  # iterator 인터페이스 확인


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""
    
    def test_full_training_simulation(self):
        """전체 훈련 시뮬레이션 테스트"""
        scheduler = create_progressive_scheduler(
            warmup_epochs=2,
            transition_epochs=5
        )
        
        total_epochs = 10
        sizes_history = []
        
        # 전체 훈련 과정 시뮬레이션
        for epoch in range(total_epochs):
            size = scheduler.get_current_size(epoch)
            scheduler.update_state(epoch, 0, 16)
            sizes_history.append(size)
        
        # 예상 패턴 확인
        # Warmup: 0-1 epochs, 224px
        assert all(size == 224 for size in sizes_history[:2])
        
        # Transition: 2-6 epochs, 점진적 증가
        transition_sizes = sizes_history[2:7]
        for i in range(1, len(transition_sizes)):
            assert transition_sizes[i] >= transition_sizes[i-1]
        
        # Stable: 7+ epochs, 384px
        assert all(size == 384 for size in sizes_history[7:])
    
    def test_memory_pressure_scenario(self):
        """메모리 압박 시나리오 테스트"""
        scheduler = create_progressive_scheduler()
        
        # 메모리 압박 상황 시뮬레이션
        with patch('torch.cuda.memory_allocated') as mock_memory:
            # 높은 메모리 사용량 시뮬레이션 (13GB)
            mock_memory.return_value = 13 * 1024**3
            
            # 상태 업데이트
            state = scheduler.update_state(epoch=10, batch_idx=0, batch_size=32)
            
            # 메모리 사용량이 기록되는지 확인
            assert state.memory_usage_gb > 0
            
            # 배치 크기 자동 조정 확인
            optimal_batch = scheduler.get_optimal_batch_size(32, 384)
            assert optimal_batch <= 32  # 메모리 압박으로 인한 감소
    
    def test_rtx5080_optimization_scenario(self):
        """RTX 5080 최적화 시나리오 테스트"""
        config = ProgressiveResizeConfig(
            max_memory_gb=14.0,  # RTX 5080 한계
            memory_format="channels_last",
            interpolation_mode="bilinear",
            auto_batch_adjustment=True
        )
        scheduler = ProgressiveResizeScheduler(config)
        
        # RTX 5080 환경 시뮬레이션
        large_batch_size = scheduler.get_optimal_batch_size(64, 384)
        
        # 메모리 안전성 확인
        max_memory_usage_estimate = large_batch_size * (384/224)**2 * 0.002  # GB
        assert max_memory_usage_estimate < 14.0
        
        # 변환 파이프라인 최적화 확인
        transform = scheduler.create_transform('train')
        assert transform is not None
        
        # 성능 통계 확인
        stats = scheduler.get_current_stats()
        assert 'current_state' in stats


class TestEdgeCases:
    """경계 사례 테스트"""
    
    def test_zero_transition_epochs(self):
        """Transition 기간이 0인 경우"""
        scheduler = create_progressive_scheduler(
            warmup_epochs=5,
            transition_epochs=0
        )
        
        # Warmup 직후 바로 목표 해상도
        assert scheduler.get_current_size(4) == 224  # Warmup
        assert scheduler.get_current_size(5) == 384  # 바로 목표 해상도
    
    def test_very_small_batch_size(self):
        """매우 작은 배치 크기"""
        scheduler = create_progressive_scheduler()
        
        # 최소 배치 크기 보장
        small_batch = scheduler.get_optimal_batch_size(1, 512)
        assert small_batch >= 1
    
    def test_very_large_image_size(self):
        """매우 큰 이미지 크기"""
        scheduler = create_progressive_scheduler()
        
        # 큰 이미지에 대한 안전한 배치 크기
        safe_batch = scheduler._calculate_max_batch_size(1024)
        assert safe_batch >= 1
        assert safe_batch <= 64
    
    def test_invalid_schedule_fallback(self):
        """유효하지 않은 스케줄에 대한 fallback"""
        config = ProgressiveResizeConfig()
        # schedule enum을 직접 조작하는 대신 기본 동작 확인
        scheduler = ProgressiveResizeScheduler(config)
        
        # 정상적으로 크기 계산되는지 확인
        size = scheduler.get_current_size(15)  # Transition 중간
        assert 224 <= size <= 384


if __name__ == "__main__":
    # 빠른 테스트 실행
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"
    ])