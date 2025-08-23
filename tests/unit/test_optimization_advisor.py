"""
OptimizationAdvisor 단위 테스트
"""

import pytest
import time
import json
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.optimization_advisor import (
    OptimizationAdvisor,
    OptimizationLevel,
    BottleneckType,
    HardwareProfile,
    TrainingMetrics,
    OptimizationRecommendation,
    OptimizationReport,
    create_rtx5080_advisor,
    quick_performance_check
)


class TestHardwareProfile:
    """HardwareProfile 테스트 클래스"""
    
    def test_hardware_profile_creation(self):
        """HardwareProfile 생성 테스트"""
        profile = HardwareProfile(
            gpu_name="RTX 5080",
            gpu_memory_gb=16.0,
            gpu_compute_capability=(8, 9),
            cpu_cores=16,
            system_memory_gb=128.0,
            storage_type="nvme",
            cuda_version="12.8",
            pytorch_version="2.8.0"
        )
        
        assert profile.gpu_name == "RTX 5080"
        assert profile.gpu_memory_gb == 16.0
        assert profile.gpu_compute_capability == (8, 9)
        assert profile.cpu_cores == 16
        assert profile.system_memory_gb == 128.0
        assert profile.storage_type == "nvme"


class TestTrainingMetrics:
    """TrainingMetrics 테스트 클래스"""
    
    def test_training_metrics_creation(self):
        """TrainingMetrics 생성 테스트"""
        metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=12.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0,
            validation_accuracy=0.85,
            training_loss=0.25
        )
        
        assert metrics.batch_size == 32
        assert metrics.learning_rate == 1e-3
        assert metrics.validation_accuracy == 0.85
        assert metrics.training_loss == 0.25


class TestOptimizationRecommendation:
    """OptimizationRecommendation 테스트 클래스"""
    
    def test_recommendation_creation(self):
        """권고사항 생성 테스트"""
        rec = OptimizationRecommendation(
            category="batch_size",
            current_value=32,
            recommended_value=24,
            expected_improvement=15.0,
            confidence=0.9,
            reasoning="GPU 메모리 부족",
            risk_level="low",
            implementation_priority=1
        )
        
        assert rec.category == "batch_size"
        assert rec.current_value == 32
        assert rec.recommended_value == 24
        assert rec.expected_improvement == 15.0
        assert rec.confidence == 0.9
        assert rec.implementation_priority == 1


class TestOptimizationAdvisor:
    """OptimizationAdvisor 테스트 클래스"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_advisor_initialization(self, mock_props, mock_name, mock_cuda):
        """Advisor 초기화 테스트"""
        # Mock CUDA 환경
        mock_cuda.return_value = True
        mock_name.return_value = "NVIDIA GeForce RTX 5080"
        
        # Mock GPU properties
        mock_device = MagicMock()
        mock_device.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device
        
        advisor = OptimizationAdvisor(
            optimization_level=OptimizationLevel.BALANCED,
            enable_auto_tuning=False
        )
        
        assert advisor.optimization_level == OptimizationLevel.BALANCED
        assert advisor.enable_auto_tuning == False
        assert advisor.hardware_profile.gpu_name == "NVIDIA GeForce RTX 5080"
        assert advisor.hardware_profile.gpu_memory_gb == 16.0
    
    def test_bottleneck_identification(self):
        """병목 지점 식별 테스트"""
        advisor = OptimizationAdvisor()
        
        # GPU 메모리 병목 테스트
        metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=95.0,
            gpu_memory_usage_gb=15.0,  # 16GB 중 15GB 사용
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        bottleneck_type, severity = advisor._identify_bottleneck(metrics)
        assert bottleneck_type == BottleneckType.GPU_MEMORY
        assert severity > 0.0
    
    def test_gpu_compute_bottleneck_identification(self):
        """GPU 연산 병목 식별 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=98.0,  # GPU 사용률 높음
            gpu_memory_usage_gb=10.0,  # 메모리는 여유
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        bottleneck_type, severity = advisor._identify_bottleneck(metrics)
        assert bottleneck_type == BottleneckType.GPU_COMPUTE
        assert severity > 0.0
    
    def test_io_bottleneck_identification(self):
        """I/O 병목 식별 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=60.0,  # GPU 여유
            gpu_memory_usage_gb=8.0,
            cpu_utilization=40.0,  # CPU 여유
            io_wait_percent=30.0   # I/O 대기 높음
        )
        
        bottleneck_type, severity = advisor._identify_bottleneck(metrics)
        assert bottleneck_type == BottleneckType.IO_BOUND
        assert severity > 0.0
    
    def test_batch_size_recommendation_memory_bottleneck(self):
        """메모리 병목 시 배치 크기 권고 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=15.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        rec = advisor._recommend_batch_size(metrics, BottleneckType.GPU_MEMORY)
        
        assert rec is not None
        assert rec.category == "batch_size"
        assert rec.current_value == 32
        assert rec.recommended_value < 32  # 배치 크기 감소 권고
        assert rec.confidence > 0.8
        assert rec.implementation_priority == 1
    
    def test_batch_size_recommendation_memory_available(self):
        """메모리 여유 시 배치 크기 증가 권고 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=60.0,  # GPU 활용도 낮음
            gpu_memory_usage_gb=8.0,  # 메모리 여유 (16GB 중 8GB)
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        rec = advisor._recommend_batch_size(metrics, BottleneckType.NONE)
        
        assert rec is not None
        assert rec.category == "batch_size"
        assert rec.current_value == 16
        assert rec.recommended_value > 16  # 배치 크기 증가 권고
        assert rec.reasoning == "메모리 여유로 배치 크기 증가하여 GPU 활용도 향상"
    
    def test_dataloader_workers_recommendation(self):
        """DataLoader 워커 수 권고 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=70.0,
            gpu_memory_usage_gb=10.0,
            cpu_utilization=40.0,
            io_wait_percent=25.0  # I/O 병목
        )
        
        rec = advisor._recommend_dataloader_workers(metrics, BottleneckType.IO_BOUND)
        
        assert rec is not None
        assert rec.category == "dataloader_workers"
        assert rec.recommended_value <= advisor.hardware_profile.cpu_cores
        assert rec.reasoning == "I/O 병목으로 DataLoader 워커 수 증가 필요"
        assert rec.implementation_priority == 2
    
    def test_memory_optimizations_recommendation(self):
        """메모리 최적화 권고 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=15.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        recs = advisor._recommend_memory_optimizations(metrics, BottleneckType.GPU_MEMORY)
        
        assert len(recs) > 0
        
        # Mixed Precision 권고 확인
        mixed_precision_rec = next((r for r in recs if r.category == "mixed_precision"), None)
        assert mixed_precision_rec is not None
        assert mixed_precision_rec.recommended_value == True
        assert mixed_precision_rec.confidence >= 0.9
        
        # Gradient Checkpointing 권고 확인
        checkpoint_rec = next((r for r in recs if r.category == "gradient_checkpointing"), None)
        assert checkpoint_rec is not None
        assert checkpoint_rec.recommended_value == True
    
    def test_optimization_score_calculation(self):
        """최적화 점수 계산 테스트"""
        advisor = OptimizationAdvisor()
        
        # 이상적인 메트릭
        optimal_metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=100.0,  # 높은 처리량
            gpu_utilization=85.0,      # 적정 GPU 활용도
            gpu_memory_usage_gb=13.6,  # 85% 메모리 사용
            cpu_utilization=50.0,      # 적정 CPU 활용도
            io_wait_percent=5.0
        )
        
        score = advisor._calculate_optimization_score(optimal_metrics)
        assert 0.8 <= score <= 1.0  # 높은 점수 기대
        
        # 비최적 메트릭
        suboptimal_metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=20.0,   # 낮은 처리량
            gpu_utilization=30.0,      # 낮은 GPU 활용도
            gpu_memory_usage_gb=5.0,   # 낮은 메모리 사용
            cpu_utilization=90.0,      # 높은 CPU 활용도
            io_wait_percent=25.0
        )
        
        score = advisor._calculate_optimization_score(suboptimal_metrics)
        assert 0.0 <= score <= 0.6  # 낮은 점수 기대
    
    def test_speedup_estimation(self):
        """속도 향상 추정 테스트"""
        advisor = OptimizationAdvisor()
        
        recommendations = [
            OptimizationRecommendation(
                category="batch_size",
                current_value=32,
                recommended_value=24,
                expected_improvement=15.0,
                confidence=0.9,
                reasoning="메모리 부족",
                risk_level="low",
                implementation_priority=1
            ),
            OptimizationRecommendation(
                category="mixed_precision",
                current_value=False,
                recommended_value=True,
                expected_improvement=30.0,
                confidence=0.95,
                reasoning="메모리 절약",
                risk_level="low",
                implementation_priority=1
            )
        ]
        
        speedup = advisor._estimate_speedup(recommendations)
        assert speedup > 1.0  # 속도 향상 기대
        assert speedup < 2.0  # 합리적인 범위
    
    def test_performance_analysis(self):
        """성능 분석 전체 프로세스 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=32,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=15.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0,
            validation_accuracy=0.85,
            training_loss=0.25
        )
        
        report = advisor.analyze_current_performance(metrics)
        
        assert isinstance(report, OptimizationReport)
        assert report.current_metrics == metrics
        assert report.bottleneck_type in BottleneckType
        assert 0.0 <= report.bottleneck_severity <= 1.0
        assert 0.0 <= report.overall_score <= 1.0
        assert report.estimated_speedup >= 1.0
        assert len(report.recommendations) >= 0
    
    def test_stage_recommendations(self):
        """Stage별 권고사항 테스트"""
        advisor = OptimizationAdvisor()
        
        # Stage 3 권고사항
        stage3_rec = advisor.get_stage_recommendations("stage_3")
        
        assert "max_batch_size" in stage3_rec
        assert "base_learning_rate" in stage3_rec
        assert "memory_target" in stage3_rec
        assert "dataloader_workers" in stage3_rec
        assert "mixed_precision" in stage3_rec
        assert stage3_rec["mixed_precision"] == True
        
        # Stage 4 권고사항
        stage4_rec = advisor.get_stage_recommendations("stage_4")
        
        assert stage4_rec["max_batch_size"] <= stage3_rec["max_batch_size"]
        assert stage4_rec["memory_target"] >= stage3_rec["memory_target"]
    
    def test_report_export(self):
        """보고서 내보내기 테스트"""
        advisor = OptimizationAdvisor()
        
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=12.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        report = advisor.analyze_current_performance(metrics)
        
        # 임시 파일로 저장 테스트
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            advisor.export_report(report, temp_path)
            
            # 저장된 파일 검증
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert "timestamp" in loaded_data
            assert "hardware_profile" in loaded_data
            assert "current_metrics" in loaded_data
            assert "recommendations" in loaded_data
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_optimization_summary(self):
        """최적화 현황 요약 테스트"""
        advisor = OptimizationAdvisor()
        
        # 데이터 없을 때
        summary = advisor.get_optimization_summary()
        assert summary["status"] == "no_data"
        
        # 데이터 있을 때
        metrics = TrainingMetrics(
            batch_size=16,
            learning_rate=1e-3,
            epoch_time_seconds=300.0,
            samples_per_second=50.0,
            gpu_utilization=85.0,
            gpu_memory_usage_gb=12.0,
            cpu_utilization=45.0,
            io_wait_percent=5.0
        )
        
        advisor.analyze_current_performance(metrics)
        summary = advisor.get_optimization_summary()
        
        assert "current_score" in summary
        assert "bottleneck" in summary
        assert "total_recommendations" in summary
        assert "hardware_profile" in summary
        assert "gpu" in summary["hardware_profile"]


class TestConvenienceFunctions:
    """편의 함수들 테스트"""
    
    @patch('torch.cuda.is_available')
    def test_create_rtx5080_advisor(self, mock_cuda):
        """RTX 5080 Advisor 생성 함수 테스트"""
        mock_cuda.return_value = True
        
        advisor = create_rtx5080_advisor(OptimizationLevel.AGGRESSIVE)
        
        assert isinstance(advisor, OptimizationAdvisor)
        assert advisor.optimization_level == OptimizationLevel.AGGRESSIVE
        assert advisor.enable_auto_tuning == False
    
    @patch('torch.cuda.is_available')
    def test_quick_performance_check(self, mock_cuda):
        """빠른 성능 체크 함수 테스트"""
        mock_cuda.return_value = True
        
        report = quick_performance_check(
            batch_size=32,
            learning_rate=1e-3,
            gpu_memory_gb=12.0,
            samples_per_second=75.0
        )
        
        assert isinstance(report, OptimizationReport)
        assert report.current_metrics.batch_size == 32
        assert report.current_metrics.learning_rate == 1e-3
        assert report.current_metrics.gpu_memory_usage_gb == 12.0
        assert report.current_metrics.samples_per_second == 75.0


@pytest.mark.integration
class TestOptimizationAdvisorIntegration:
    """OptimizationAdvisor 통합 테스트"""
    
    def test_full_optimization_workflow(self):
        """전체 최적화 워크플로우 테스트"""
        advisor = OptimizationAdvisor(OptimizationLevel.BALANCED)
        
        # 시뮬레이션된 훈련 메트릭들
        training_sessions = [
            # 초기 성능 (비최적)
            TrainingMetrics(
                batch_size=64, learning_rate=1e-2,
                epoch_time_seconds=600, samples_per_second=30,
                gpu_utilization=95, gpu_memory_usage_gb=15.5,
                cpu_utilization=80, io_wait_percent=15,
                training_loss=0.8, validation_accuracy=0.6
            ),
            # 배치 크기 조정 후
            TrainingMetrics(
                batch_size=32, learning_rate=1e-2,
                epoch_time_seconds=400, samples_per_second=50,
                gpu_utilization=85, gpu_memory_usage_gb=12.0,
                cpu_utilization=60, io_wait_percent=10,
                training_loss=0.6, validation_accuracy=0.75
            ),
            # 추가 최적화 후
            TrainingMetrics(
                batch_size=32, learning_rate=5e-3,
                epoch_time_seconds=350, samples_per_second=65,
                gpu_utilization=90, gpu_memory_usage_gb=13.0,
                cpu_utilization=50, io_wait_percent=5,
                training_loss=0.4, validation_accuracy=0.85
            )
        ]
        
        scores = []
        for i, metrics in enumerate(training_sessions):
            report = advisor.analyze_current_performance(metrics)
            scores.append(report.overall_score)
            
            # 점수가 개선되는지 확인 (첫 번째는 제외)
            if i > 0:
                assert report.overall_score >= scores[i-1] * 0.9  # 약간의 오차 허용
        
        # 최종 요약 확인
        summary = advisor.get_optimization_summary()
        assert summary["current_score"] > 0.5  # 합리적인 최종 점수
        assert len(advisor.recommendations_history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])