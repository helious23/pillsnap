"""
새로운 아키텍처 컴포넌트 통합 테스트
상업용 수준의 새로 추가된 모든 시스템 검증

테스트 대상:
- Training Stage Components
- Evaluation Components  
- Data Loading Components
- Memory Management
- State Management
"""

import pytest
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.train_classification_stage import ClassificationStageTrainer
from src.training.train_detection_stage import DetectionStageTrainer
from src.training.batch_size_auto_tuner import BatchSizeAutoTuner, create_sample_batch_generator
from src.training.training_state_manager import TrainingStateManager, ModelMetadata
from src.evaluation.evaluate_detection_metrics import DetectionMetricsEvaluator, DetectionMetrics
from src.evaluation.evaluate_pipeline_end_to_end import EndToEndPipelineEvaluator
from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor


class TestTrainingComponents:
    """학습 컴포넌트 테스트"""
    
    def test_classification_stage_trainer_initialization(self):
        """분류 Stage 학습기 초기화 테스트"""
        trainer = ClassificationStageTrainer(
            num_classes=10,
            target_accuracy=0.8,
            device="cpu"  # CI에서 GPU 없을 수 있음
        )
        
        assert trainer.num_classes == 10
        assert trainer.target_accuracy == 0.8
        assert trainer.device.type == "cpu"
        assert trainer.model is None  # 아직 설정 안됨
    
    def test_classification_trainer_model_setup(self):
        """분류 학습기 모델 설정 테스트"""
        trainer = ClassificationStageTrainer(num_classes=5, device="cpu")
        
        # 모델 설정
        trainer.setup_model_and_optimizers(
            learning_rate=1e-3,
            mixed_precision=False  # CPU에서는 비활성화
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert len(list(trainer.model.parameters())) > 0
    
    def test_detection_stage_trainer_initialization(self):
        """검출 Stage 학습기 초기화 테스트"""
        trainer = DetectionStageTrainer(target_map=0.5, device="cpu")
        
        assert trainer.target_map == 0.5
        assert trainer.device == "cpu"
        assert trainer.best_map == 0.0
    
    def test_detection_trainer_model_setup(self):
        """검출 학습기 모델 설정 테스트"""
        trainer = DetectionStageTrainer(device="cpu")
        
        # YOLO 모델 설정 (CPU에서는 제한적)
        trainer.setup_model()
        assert trainer.model is not None


class TestBatchSizeAutoTuner:
    """배치 크기 자동 조정 테스트"""
    
    def test_auto_tuner_initialization(self):
        """자동 튜너 초기화 테스트"""
        tuner = BatchSizeAutoTuner(
            initial_batch_size=16,
            max_batch_size=64,
            memory_threshold_gb=8.0
        )
        
        assert tuner.initial_batch_size == 16
        assert tuner.max_batch_size == 64
        assert tuner.memory_threshold_gb == 8.0
    
    def test_batch_size_optimization_cpu(self):
        """CPU에서 배치 크기 최적화 테스트 (강화된 검증)"""
        # 간단한 더미 모델
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # 배치 생성기
        batch_generator = create_sample_batch_generator((100,), 10)
        
        tuner = BatchSizeAutoTuner(
            initial_batch_size=4,
            max_batch_size=16,
            memory_threshold_gb=1.0  # 낮은 임계값
        )
        
        # 최적화 실행
        start_time = time.time()
        result = tuner.find_optimal_batch_size(
            model=model,
            sample_batch_generator=batch_generator,
            device=torch.device('cpu')
        )
        execution_time = time.time() - start_time
        
        # 강화된 검증
        assert result.optimal_batch_size > 0, "최적 배치 크기는 0보다 커야 함"
        assert result.optimal_batch_size <= 16, "최적 배치 크기는 최대값을 초과할 수 없음"
        assert result.optimal_batch_size >= 4, "최적 배치 크기는 초기값보다 작을 수 없음"
        assert result.tuning_time_seconds > 0, "튜닝 시간이 기록되어야 함"
        assert result.tuning_time_seconds < 60.0, "튜닝 시간이 60초를 초과하면 안됨"
        assert result.throughput_samples_per_sec > 0, "처리량이 0보다 커야 함"
        assert execution_time < result.tuning_time_seconds + 5.0, "실행 시간 일관성 검증"
    
    def test_inference_batch_size_suggestion(self):
        """추론용 배치 크기 제안 테스트"""
        model = nn.Linear(50, 5)
        
        def input_generator(batch_size):
            return torch.randn(batch_size, 50)
        
        tuner = BatchSizeAutoTuner()
        
        optimal_batch = tuner.suggest_batch_size_for_inference(
            model=model,
            sample_input_generator=input_generator,
            target_latency_ms=100.0
        )
        
        assert optimal_batch >= 1
        assert optimal_batch <= 64


class TestTrainingStateManager:
    """학습 상태 관리 테스트"""
    
    def setup_method(self):
        """테스트 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
    
    def teardown_method(self):
        """테스트 정리"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_state_manager_initialization(self):
        """상태 관리자 초기화 테스트"""
        manager = TrainingStateManager(
            experiment_name=self.experiment_name,
            stage=1,
            checkpoint_dir=self.temp_dir
        )
        
        assert manager.experiment_name == self.experiment_name
        assert manager.stage == 1
        assert manager.experiment_dir.exists()
    
    def test_checkpoint_save_and_load(self):
        """체크포인트 저장/로드 테스트"""
        manager = TrainingStateManager(
            experiment_name=self.experiment_name,
            checkpoint_dir=self.temp_dir
        )
        
        # 더미 모델 및 옵티마이저
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 체크포인트 저장
        checkpoint_path = manager.save_checkpoint(
            epoch=5,
            model=model,
            optimizer=optimizer,
            metric_value=0.85,
            is_best=True
        )
        
        assert Path(checkpoint_path).exists()
        
        # 새 모델에 로드
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        resume_info = manager.load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            load_best=True
        )
        
        # 강화된 체크포인트 검증
        assert resume_info['epoch'] == 5, f"에포크 불일치: 예상 5, 실제 {resume_info['epoch']}"
        assert resume_info['best_metric'] == 0.85, f"최고 성능 불일치: 예상 0.85, 실제 {resume_info['best_metric']}"
        assert resume_info['best_epoch'] == 5, f"최고 성능 에포크 불일치: 예상 5, 실제 {resume_info['best_epoch']}"
        assert 'training_config' in resume_info, "training_config가 누락됨"
        assert 'timestamp' in resume_info, "timestamp가 누락됨"
        
        # 모델 상태 검증 - 로드된 모델이 실제로 작동하는지 확인
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = new_model(test_input)
            assert output.shape == (1, 5), f"모델 출력 형태 오류: {output.shape}"
            assert torch.isfinite(output).all(), "모델 출력에 무한값이나 NaN이 포함됨"
    
    def test_deployment_model_saving(self):
        """배포용 모델 저장 테스트"""
        manager = TrainingStateManager(
            experiment_name=self.experiment_name,
            checkpoint_dir=self.temp_dir
        )
        
        model = nn.Linear(20, 3)
        
        metadata = ModelMetadata(
            model_name="test_model",
            stage=1,
            num_classes=3,
            input_size=(20,),
            best_metric_value=0.92,
            best_metric_name="accuracy",
            training_samples=1000,
            validation_samples=200,
            training_time_hours=1.5,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        deploy_path = manager.save_final_model_for_deployment(
            model=model,
            model_metadata=metadata,
            include_training_artifacts=True
        )
        
        assert Path(deploy_path).exists()
        assert Path(deploy_path).parent.joinpath("model_metadata.json").exists()
        assert Path(deploy_path).parent.joinpath("deployment_guide.md").exists()


class TestEvaluationComponents:
    """평가 컴포넌트 테스트"""
    
    def test_detection_metrics_evaluator(self):
        """검출 메트릭 평가기 테스트"""
        evaluator = DetectionMetricsEvaluator(num_classes=1)
        
        # 더미 YOLO 결과
        dummy_results = {
            'metrics': {
                'mAP_0.5': 0.75,
                'mAP_0.5:0.95': 0.55,
                'precision': 0.80,
                'recall': 0.70
            },
            'num_images': 500
        }
        
        metrics = evaluator.evaluate_yolo_results(dummy_results)
        
        assert isinstance(metrics, DetectionMetrics)
        assert metrics.map_50 == 0.75
        assert metrics.precision == 0.80
        assert metrics.f1_score > 0  # 계산됨
        assert metrics.num_images == 500
    
    def test_stage_target_achievement(self):
        """Stage 목표 달성 평가 테스트"""
        evaluator = DetectionMetricsEvaluator()
        
        # 좋은 성능 메트릭
        good_metrics = DetectionMetrics(
            map_50=0.85,
            map_50_95=0.65,
            precision=0.90,
            recall=0.80,
            f1_score=0.85,
            ap_per_class=[0.85],
            precision_per_class=[0.90],
            recall_per_class=[0.80],
            num_classes=1,
            num_images=1000
        )
        
        achievement = evaluator.evaluate_stage_target_achievement(good_metrics, stage=1)
        assert achievement['stage1_detection_completed'] == True
        
        # 나쁜 성능 메트릭  
        poor_metrics = DetectionMetrics(
            map_50=0.15,
            map_50_95=0.10,
            precision=0.20,
            recall=0.15,
            f1_score=0.17,
            ap_per_class=[0.15],
            precision_per_class=[0.20],
            recall_per_class=[0.15],
            num_classes=1,
            num_images=1000
        )
        
        achievement = evaluator.evaluate_stage_target_achievement(poor_metrics, stage=1)
        assert achievement['stage1_detection_completed'] == False
    
    def test_end_to_end_evaluator_initialization(self):
        """End-to-End 평가기 초기화 테스트"""
        evaluator = EndToEndPipelineEvaluator(pipeline=None)
        
        assert evaluator.pipeline is None
        assert evaluator.memory_monitor is not None
        assert evaluator.classification_evaluator is not None
        assert evaluator.detection_evaluator is not None


class TestMemoryMonitoring:
    """메모리 모니터링 테스트"""
    
    def test_memory_monitor_initialization(self):
        """메모리 모니터 초기화 테스트"""
        monitor = GPUMemoryMonitor(target_memory_gb=8.0)
        
        assert monitor.target_memory_gb == 8.0
        assert monitor.warning_threshold == 0.9
        assert monitor.critical_threshold == 0.95
    
    def test_memory_usage_reporting(self):
        """메모리 사용량 보고 테스트 (강화된 검증)"""
        monitor = GPUMemoryMonitor()
        
        usage = monitor.get_current_usage()
        
        # 기본 키 존재 검증
        required_keys = ['used_gb', 'total_gb', 'utilization_percent', 'usage_ratio']
        for key in required_keys:
            assert key in usage, f"필수 키 '{key}'가 누락됨"
        
        # 값 범위 검증
        assert usage['used_gb'] >= 0, "사용 메모리는 0 이상이어야 함"
        assert usage['total_gb'] > 0, "전체 메모리는 0보다 커야 함"
        assert 0 <= usage['utilization_percent'] <= 100, "사용률은 0-100% 범위여야 함"
        assert 0 <= usage['usage_ratio'] <= 1.0, "사용 비율은 0-1 범위여야 함"
        assert usage['used_gb'] <= usage['total_gb'], "사용 메모리는 전체 메모리를 초과할 수 없음"
        
        # 계산 일관성 검증
        expected_ratio = usage['used_gb'] / usage['total_gb'] if usage['total_gb'] > 0 else 0
        assert abs(usage['usage_ratio'] - expected_ratio) < 0.01, "사용 비율 계산 오류"
    
    def test_memory_cleanup(self):
        """메모리 정리 테스트"""
        monitor = GPUMemoryMonitor()
        
        cleanup_result = monitor.force_memory_cleanup()
        
        assert 'freed_gb' in cleanup_result
        assert cleanup_result['freed_gb'] >= 0


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""
    
    def test_training_pipeline_integration(self):
        """학습 파이프라인 통합 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 상태 관리자
            state_manager = TrainingStateManager(
                experiment_name="integration_test",
                checkpoint_dir=temp_dir
            )
            
            # 메모리 모니터
            memory_monitor = GPUMemoryMonitor(target_memory_gb=4.0)
            
            # 분류 학습기
            trainer = ClassificationStageTrainer(
                num_classes=5,
                target_accuracy=0.7,
                device="cpu"
            )
            
            # 모델 설정
            trainer.setup_model_and_optimizers(mixed_precision=False)
            
            # 체크포인트 저장
            checkpoint_path = state_manager.save_checkpoint(
                epoch=1,
                model=trainer.model,
                optimizer=trainer.optimizer,
                metric_value=0.75,
                is_best=True
            )
            
            # 메모리 사용량 확인
            memory_usage = memory_monitor.get_current_usage()
            
            # 검증
            assert Path(checkpoint_path).exists()
            assert memory_usage['used_gb'] >= 0
            assert trainer.model is not None
    
    def test_evaluation_pipeline_integration(self):
        """평가 파이프라인 통합 테스트"""
        # 검출 평가기
        detection_evaluator = DetectionMetricsEvaluator(num_classes=1)
        
        # End-to-End 평가기
        e2e_evaluator = EndToEndPipelineEvaluator(pipeline=None)
        
        # 더미 결과로 평가
        dummy_detection_results = {
            'metrics': {
                'mAP_0.5': 0.42,
                'precision': 0.45,
                'recall': 0.40
            },
            'num_images': 100
        }
        
        detection_metrics = detection_evaluator.evaluate_yolo_results(dummy_detection_results)
        
        # Stage 1 목표 달성 확인
        achievement = detection_evaluator.evaluate_stage_target_achievement(detection_metrics, stage=1)
        
        assert isinstance(detection_metrics, DetectionMetrics)
        assert 'stage1_detection_completed' in achievement


class TestStrictValidation:
    """엄격한 검증 테스트 (상업용 수준)"""
    
    def test_performance_requirements_validation(self):
        """성능 요구사항 엄격 검증"""
        from src.training.batch_size_auto_tuner import BatchSizeAutoTuner, create_sample_batch_generator
        import time
        
        # 최소 성능 요구사항
        MIN_THROUGHPUT = 10.0  # samples/sec
        MAX_TUNING_TIME = 30.0  # seconds
        
        model = torch.nn.Linear(50, 10)
        batch_generator = create_sample_batch_generator((50,), 10)
        tuner = BatchSizeAutoTuner(max_batch_size=32)
        
        start_time = time.time()
        result = tuner.find_optimal_batch_size(model, batch_generator)
        actual_time = time.time() - start_time
        
        # 엄격한 성능 검증
        assert result.throughput_samples_per_sec >= MIN_THROUGHPUT, \
            f"처리량 미달: {result.throughput_samples_per_sec:.2f} < {MIN_THROUGHPUT}"
        assert actual_time <= MAX_TUNING_TIME, \
            f"튜닝 시간 초과: {actual_time:.2f}s > {MAX_TUNING_TIME}s"
        assert result.optimal_batch_size >= 4, "최적 배치가 너무 작음"
    
    def test_memory_efficiency_validation(self):
        """메모리 효율성 엄격 검증"""
        from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor
        
        monitor = GPUMemoryMonitor()
        
        # 메모리 사용량 연속 측정
        measurements = []
        for _ in range(3):
            usage = monitor.get_current_usage()
            measurements.append(usage)
            time.sleep(0.1)
        
        # 메모리 사용량 안정성 검증
        used_values = [m['used_gb'] for m in measurements]
        max_variance = max(used_values) - min(used_values)
        assert max_variance < 1.0, f"메모리 사용량 변동이 너무 큼: {max_variance:.2f}GB"
        
        # 메모리 효율성 검증
        for usage in measurements:
            assert usage['usage_ratio'] < 0.95, "메모리 사용률이 95%를 초과함"
    
    def test_error_handling_robustness(self):
        """에러 처리 강건성 테스트"""
        from src.training.training_state_manager import TrainingStateManager
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingStateManager("error_test", checkpoint_dir=temp_dir)
            
            # 잘못된 체크포인트 로드 시도
            fake_checkpoint = Path(temp_dir) / "fake_checkpoint.pt"
            fake_checkpoint.write_text("invalid data")
            
            model = torch.nn.Linear(5, 3)
            
            # 예외가 적절히 발생하는지 확인
            with pytest.raises(Exception):
                manager.load_checkpoint(model, checkpoint_path=str(fake_checkpoint))
    
    def test_stage1_performance_targets(self):
        """Stage 1 성능 목표 엄격 검증"""
        from src.evaluation.evaluate_detection_metrics import DetectionMetricsEvaluator, DetectionMetrics
        
        evaluator = DetectionMetricsEvaluator()
        
        # Stage 1 최소 요구사항보다 약간 낮은 성능
        borderline_metrics = DetectionMetrics(
            map_50=0.29,  # 목표: 0.30
            map_50_95=0.20,
            precision=0.34,  # 목표: 0.35
            recall=0.29,  # 목표: 0.30
            f1_score=0.31,
            ap_per_class=[0.29],
            precision_per_class=[0.34],
            recall_per_class=[0.29],
            num_classes=1,
            num_images=1000
        )
        
        achievement = evaluator.evaluate_stage_target_achievement(borderline_metrics, stage=1)
        
        # 엄격하게 목표 미달성 확인
        assert achievement['stage1_detection_completed'] == False, \
            "경계선 성능에서 목표 달성으로 잘못 판정됨"


# pytest 실행을 위한 main 함수
def main():
    """테스트 실행"""
    print("🧪 새로운 아키텍처 컴포넌트 통합 테스트")
    print("=" * 60)
    
    # pytest 실행
    test_file = __file__
    exit_code = pytest.main([
        test_file,
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    if exit_code == 0:
        print("\n✅ 모든 테스트 통과!")
    else:
        print(f"\n❌ 테스트 실패 (Exit code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    main()