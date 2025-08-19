"""
Two-Stage 파이프라인 단위 테스트

테스트 범위:
- 파이프라인 초기화 및 설정
- Single 모드 직접 분류
- Combo 모드 검출 → 크롭 → 분류
- 모드 전환 및 사용자 제어
- 배치 처리 및 성능 메트릭
- 오류 처리 및 예외 상황
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.pipeline import (
    PillSnapPipeline,
    PipelineConfig,
    PipelineResult,
    create_pillsnap_pipeline
)
from src.models.detector import YOLOConfig, DetectionResult
from src.models.classifier import ClassifierConfig, ClassificationResult


@pytest.fixture
def basic_pipeline_config():
    """기본 파이프라인 설정"""
    return PipelineConfig(
        default_mode="single",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=2,
        enable_optimization=False  # 테스트에서는 비활성화
    )


@pytest.fixture
def pipeline(basic_pipeline_config):
    """기본 파이프라인 인스턴스"""
    # 테스트용 작은 모델 설정
    detector_config = YOLOConfig(
        model_size="yolo11n",  # 작은 모델
        device=basic_pipeline_config.device
    )
    classifier_config = ClassifierConfig(
        num_classes=10,  # 테스트용 작은 클래스 수
        input_size=224,  # 테스트용 작은 크기
        device=basic_pipeline_config.device,
        mixed_precision=False,
        torch_compile=False,
        channels_last=False
    )
    
    config = PipelineConfig(
        default_mode="single",
        detection_config=detector_config,
        classification_config=classifier_config,
        device=basic_pipeline_config.device,
        enable_optimization=False
    )
    
    return PillSnapPipeline(config=config)


class TestPipelineConfig:
    """PipelineConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = PipelineConfig()
        assert config.default_mode == "single"
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.enable_optimization == True
        assert config.detection_confidence_threshold == 0.3
        assert config.detection_iou_threshold == 0.45
    
    def test_config_validation(self):
        """설정 유효성 검증 테스트"""
        # 유효하지 않은 모드
        with pytest.raises(AssertionError):
            PipelineConfig(default_mode="invalid")
        
        # 유효하지 않은 임계값
        with pytest.raises(AssertionError):
            PipelineConfig(detection_confidence_threshold=1.5)
        
        with pytest.raises(AssertionError):
            PipelineConfig(detection_iou_threshold=-0.1)
        
        # 유효하지 않은 크롭 설정
        with pytest.raises(AssertionError):
            PipelineConfig(crop_padding=1.5)
        
        with pytest.raises(AssertionError):
            PipelineConfig(min_crop_size=0)
        
        with pytest.raises(AssertionError):
            PipelineConfig(max_crop_size=32, min_crop_size=64)
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = PipelineConfig(
            default_mode="combo",
            device="cpu",
            batch_size=8,
            detection_confidence_threshold=0.5,
            crop_padding=0.2
        )
        assert config.default_mode == "combo"
        assert config.device == "cpu"
        assert config.batch_size == 8
        assert config.detection_confidence_threshold == 0.5
        assert config.crop_padding == 0.2


class TestPipelineResult:
    """PipelineResult 클래스 테스트"""
    
    def test_empty_result(self):
        """빈 파이프라인 결과 테스트"""
        result = PipelineResult(
            mode_used="single",
            mode_requested="single"
        )
        assert len(result) == 0
        assert result.mode_used == "single"
        assert result.mode_requested == "single"
    
    def test_single_mode_result(self):
        """Single 모드 결과 테스트"""
        classification_result = ClassificationResult(
            logits=torch.randn(2, 10),
            probabilities=torch.softmax(torch.randn(2, 10), dim=1),
            predicted_classes=torch.tensor([3, 7]),
            confidence_scores=torch.tensor([0.85, 0.92])
        )
        
        result = PipelineResult(
            mode_used="single",
            mode_requested="single",
            classification_result=classification_result,
            timing={"classification": 15.2, "total": 15.2},
            input_shape=(2, 3, 224, 224)
        )
        
        assert len(result) == 2
        assert result.mode_used == "single"
        assert result.timing["total"] == 15.2
        assert result.input_shape == (2, 3, 224, 224)
    
    def test_combo_mode_result(self):
        """Combo 모드 결과 테스트"""
        detection_result = DetectionResult(
            boxes=torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]),
            scores=torch.tensor([0.9, 0.8]),
            class_ids=torch.tensor([0, 0]),
            image_shape=(640, 640)
        )
        
        cropped_classifications = [
            ClassificationResult(
                logits=torch.randn(1, 10),
                probabilities=torch.softmax(torch.randn(1, 10), dim=1),
                predicted_classes=torch.tensor([2]),
                confidence_scores=torch.tensor([0.88])
            ),
            ClassificationResult(
                logits=torch.randn(1, 10),
                probabilities=torch.softmax(torch.randn(1, 10), dim=1),
                predicted_classes=torch.tensor([5]),
                confidence_scores=torch.tensor([0.76])
            )
        ]
        
        result = PipelineResult(
            mode_used="combo",
            mode_requested="combo",
            detection_result=detection_result,
            cropped_classifications=cropped_classifications,
            timing={"detection": 25.1, "classification": 12.3, "total": 37.4}
        )
        
        assert len(result) == 2
        assert result.mode_used == "combo"
        assert result.timing["total"] == 37.4
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        classification_result = ClassificationResult(
            logits=torch.randn(1, 5),
            probabilities=torch.softmax(torch.randn(1, 5), dim=1),
            predicted_classes=torch.tensor([2]),
            confidence_scores=torch.tensor([0.85])
        )
        
        result = PipelineResult(
            mode_used="single",
            mode_requested="single",
            classification_result=classification_result,
            timing={"classification": 15.2, "total": 15.2},
            input_shape=(1, 3, 224, 224)
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['mode_used'] == "single"
        assert result_dict['mode_requested'] == "single"
        assert result_dict['count'] == 1
        assert 'classification' in result_dict
        assert 'timing' in result_dict
        assert 'input_shape' in result_dict
    
    def test_get_predictions_single(self):
        """Single 모드 예측 결과 테스트"""
        classification_result = ClassificationResult(
            logits=torch.randn(2, 5),
            probabilities=torch.softmax(torch.randn(2, 5), dim=1),
            predicted_classes=torch.tensor([1, 3]),
            confidence_scores=torch.tensor([0.82, 0.91])
        )
        
        result = PipelineResult(
            mode_used="single",
            mode_requested="single",
            classification_result=classification_result
        )
        
        predictions = result.get_predictions()
        assert len(predictions) == 2
        assert predictions[0]['class_id'] == 1
        assert predictions[0]['confidence'] == pytest.approx(0.82, abs=1e-3)
        assert predictions[0]['bbox'] is None
        assert predictions[0]['mode'] == 'single'
        assert predictions[1]['class_id'] == 3
        assert predictions[1]['confidence'] == pytest.approx(0.91, abs=1e-3)
    
    def test_get_predictions_combo(self):
        """Combo 모드 예측 결과 테스트"""
        detection_result = DetectionResult(
            boxes=torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
            scores=torch.tensor([0.9]),
            class_ids=torch.tensor([0]),
            image_shape=(640, 640)
        )
        
        cropped_classifications = [
            ClassificationResult(
                logits=torch.randn(1, 5),
                probabilities=torch.softmax(torch.randn(1, 5), dim=1),
                predicted_classes=torch.tensor([2]),
                confidence_scores=torch.tensor([0.88])
            )
        ]
        
        result = PipelineResult(
            mode_used="combo",
            mode_requested="combo",
            detection_result=detection_result,
            cropped_classifications=cropped_classifications
        )
        
        predictions = result.get_predictions()
        assert len(predictions) == 1
        assert predictions[0]['class_id'] == 2
        assert predictions[0]['confidence'] == pytest.approx(0.88, abs=1e-3)
        assert predictions[0]['bbox'] == pytest.approx([0.1, 0.1, 0.5, 0.5], abs=1e-3)
        assert predictions[0]['detection_confidence'] == pytest.approx(0.9, abs=1e-3)
        assert predictions[0]['mode'] == 'combo'


class TestPillSnapPipeline:
    """PillSnapPipeline 클래스 테스트"""
    
    def test_pipeline_initialization(self, basic_pipeline_config):
        """파이프라인 초기화 테스트"""
        pipeline = PillSnapPipeline(config=basic_pipeline_config)
        
        assert pipeline.config == basic_pipeline_config
        assert hasattr(pipeline, 'classifier')
        assert hasattr(pipeline, 'detector')
        assert hasattr(pipeline, 'logger')
    
    def test_get_model_info(self, pipeline):
        """모델 정보 조회 테스트"""
        info = pipeline.get_model_info()
        
        assert isinstance(info, dict)
        required_keys = ['pipeline_config', 'detector', 'classifier', 'total_parameters']
        for key in required_keys:
            assert key in info
        
        assert info['pipeline_config']['default_mode'] == "single"
        assert info['total_parameters'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
    def test_gpu_allocation(self):
        """GPU 할당 테스트"""
        config = PipelineConfig(device="cuda", enable_optimization=False)
        pipeline = PillSnapPipeline(config=config)
        
        # 분류기가 GPU에 있는지 확인
        classifier_device = next(pipeline.classifier.parameters()).device
        assert classifier_device.type == "cuda"
    
    def test_cpu_allocation(self):
        """CPU 할당 테스트"""
        config = PipelineConfig(device="cpu", enable_optimization=False)
        pipeline = PillSnapPipeline(config=config)
        
        # 분류기가 CPU에 있는지 확인
        classifier_device = next(pipeline.classifier.parameters()).device
        assert classifier_device.type == "cpu"
    
    def test_predict_single_mode(self, pipeline):
        """Single 모드 예측 테스트"""
        device = pipeline.config.device
        batch_size = 2
        
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        result = pipeline.predict(x, mode="single")
        
        assert isinstance(result, PipelineResult)
        assert result.mode_used == "single"
        assert result.mode_requested == "single"
        assert len(result) == batch_size
        assert result.classification_result is not None
        assert result.detection_result is None
        assert result.cropped_classifications is None
        assert 'classification' in result.timing
        assert 'total' in result.timing
    
    def test_predict_combo_mode(self, pipeline):
        """Combo 모드 예측 테스트"""
        device = pipeline.config.device
        batch_size = 1  # Combo 모드는 현재 단일 이미지만 지원
        
        x = torch.randn(batch_size, 3, 640, 640, device=device)
        
        result = pipeline.predict(x, mode="combo")
        
        assert isinstance(result, PipelineResult)
        assert result.mode_used == "combo"
        assert result.mode_requested == "combo"
        assert result.classification_result is None
        assert result.detection_result is not None
        assert result.cropped_classifications is not None
        assert 'detection' in result.timing
        assert 'classification' in result.timing
        assert 'total' in result.timing
    
    def test_predict_default_mode(self, pipeline):
        """기본 모드 예측 테스트"""
        device = pipeline.config.device
        x = torch.randn(1, 3, 224, 224, device=device)
        
        # 모드를 지정하지 않으면 기본 모드 사용
        result = pipeline.predict(x)
        
        assert result.mode_used == pipeline.config.default_mode
        assert result.mode_requested == pipeline.config.default_mode
    
    def test_predict_with_custom_thresholds(self, pipeline):
        """사용자 정의 임계값 예측 테스트"""
        device = pipeline.config.device
        x = torch.randn(1, 3, 640, 640, device=device)
        
        result = pipeline.predict(
            x, 
            mode="combo",
            confidence_threshold=0.7,
            temperature=2.0
        )
        
        assert isinstance(result, PipelineResult)
        assert result.mode_used == "combo"
    
    def test_predict_batch(self, pipeline):
        """배치 예측 테스트"""
        device = pipeline.config.device
        total_size = 3
        
        x = torch.randn(total_size, 3, 224, 224, device=device)
        
        results = pipeline.predict_batch(x, mode="single", batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == total_size
        for result in results:
            assert isinstance(result, PipelineResult)
            assert result.mode_used == "single"
    
    @pytest.mark.parametrize("mode", ["single", "combo"])
    def test_different_modes(self, pipeline, mode):
        """다양한 모드 테스트"""
        device = pipeline.config.device
        input_size = 640 if mode == "combo" else 224
        x = torch.randn(1, 3, input_size, input_size, device=device)
        
        result = pipeline.predict(x, mode=mode)
        
        assert result.mode_used == mode
        assert result.mode_requested == mode
        assert len(result.timing) >= 2  # 최소 total과 다른 하나
    
    def test_save_and_load_models(self, pipeline, tmp_path):
        """모델 저장/로드 테스트"""
        save_dir = tmp_path / "pipeline_models"
        
        # 모델 저장
        pipeline.save_models(save_dir)
        assert save_dir.exists()
        assert (save_dir / "detector.pth").exists()
        assert (save_dir / "classifier.pth").exists()
        assert (save_dir / "pipeline_config.pth").exists()
        
        # 분류기만 로드 테스트 (YOLO 모델 로드는 복잡하므로 별도 처리)
        classifier_path = save_dir / "classifier.pth"
        if classifier_path.exists():
            pipeline.classifier.load_model(classifier_path)


class TestPipelineHelperFunctions:
    """헬퍼 함수 테스트"""
    
    def test_create_pillsnap_pipeline(self):
        """파이프라인 생성 헬퍼 함수 테스트"""
        pipeline = create_pillsnap_pipeline(
            default_mode="combo",
            device="cpu",
            num_classes=100,
            detector_input_size=320,
            classifier_input_size=224
        )
        
        assert isinstance(pipeline, PillSnapPipeline)
        assert pipeline.config.default_mode == "combo"
        assert pipeline.config.device == "cpu"
        assert pipeline.classifier.config.num_classes == 100
        assert pipeline.detector.config.input_size == 320
        assert pipeline.classifier.config.input_size == 224
    
    def test_create_pipeline_with_defaults(self):
        """기본값으로 파이프라인 생성 테스트"""
        pipeline = create_pillsnap_pipeline(device="cpu")
        
        assert pipeline.config.default_mode == "single"
        assert pipeline.classifier.config.num_classes == 4523
        assert pipeline.detector.config.input_size == 640
        assert pipeline.classifier.config.input_size == 384


class TestPipelineErrorHandling:
    """오류 처리 테스트"""
    
    def test_invalid_input_dimensions(self, pipeline):
        """유효하지 않은 입력 차원 테스트"""
        device = pipeline.config.device
        
        # 잘못된 차원의 입력
        invalid_inputs = [
            torch.randn(3, 224, 224, device=device),  # 배치 차원 누락
            torch.randn(1, 4, 224, 224, device=device),  # 잘못된 채널 수
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(Exception):
                pipeline.predict(invalid_input)
    
    def test_empty_input(self, pipeline):
        """빈 입력 테스트"""
        device = pipeline.config.device
        
        # 배치 크기가 0인 입력
        empty_input = torch.empty(0, 3, 224, 224, device=device)
        
        result = pipeline.predict(empty_input, mode="single")
        assert len(result) == 0
    
    def test_nonexistent_model_load(self, pipeline):
        """존재하지 않는 모델 로드 테스트"""
        with pytest.raises(FileNotFoundError):
            pipeline.load_models("/path/that/does/not/exist")


@pytest.mark.integration
class TestPipelineIntegration:
    """통합 테스트"""
    
    def test_full_pipeline_workflow(self):
        """전체 파이프라인 워크플로우 테스트"""
        # 1. 파이프라인 생성
        pipeline = create_pillsnap_pipeline(
            default_mode="single",
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_classes=10,  # 테스트용
            detector_input_size=320,
            classifier_input_size=224
        )
        
        # 2. 모델 정보 확인
        info = pipeline.get_model_info()
        assert info['total_parameters'] > 0
        
        # 3. 다양한 입력 크기 테스트
        device = pipeline.config.device
        test_cases = [
            (1, 224, "single"),
            (2, 224, "single"),
            (1, 320, "combo")
        ]
        
        for batch_size, img_size, mode in test_cases:
            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            
            # 예측
            result = pipeline.predict(x, mode=mode)
            assert len(result) <= batch_size  # Combo 모드에서는 검출 결과에 따라 달라질 수 있음
            
            # 예측 결과 확인
            predictions = result.get_predictions()
            assert isinstance(predictions, list)
            
            # 딕셔너리 변환
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
        
        # 4. 배치 처리 테스트
        large_x = torch.randn(5, 3, 224, 224, device=device)
        batch_results = pipeline.predict_batch(large_x, mode="single", batch_size=2)
        assert len(batch_results) == 5
        
        # 5. 모델 저장/로드 테스트 (임시 디렉터리)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "models"
            pipeline.save_models(save_dir)
            
            # 분류기만 로드 테스트 (YOLO 모델 로드는 별도 처리)
            classifier_path = save_dir / "classifier.pth"
            if classifier_path.exists():
                pipeline.classifier.load_model(classifier_path)


if __name__ == "__main__":
    # 기본 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])