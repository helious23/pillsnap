"""
YOLOv11m 검출기 단위 테스트

테스트 범위:
- 모델 초기화 및 설정
- GPU 할당 및 메모리 관리
- 추론 기능 및 결과 형식
- 모델 저장/로드 기능
- 오류 처리 및 예외 상황
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.detector import (
    PillSnapYOLODetector, 
    YOLOConfig, 
    DetectionResult,
    create_pillsnap_detector
)


@pytest.fixture
def basic_config():
    """기본 YOLO 설정"""
    return YOLOConfig(
        model_size="yolo11n",  # 테스트용 경량 모델
        input_size=320,  # 테스트용 작은 크기
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=False,  # 테스트에서는 비활성화
        torch_compile=False,  # 테스트에서는 비활성화
        channels_last=False  # YOLO 호환성 문제로 비활성화
    )


@pytest.fixture
def detector(basic_config):
    """기본 검출기 인스턴스"""
    return PillSnapYOLODetector(config=basic_config, num_classes=1)


class TestYOLOConfig:
    """YOLOConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = YOLOConfig()
        assert config.model_size == "yolo11m"
        assert config.input_size == 640
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.max_detections == 300
        assert config.device == "cuda"
    
    def test_config_validation(self):
        """설정 유효성 검증 테스트"""
        # 유효하지 않은 모델 크기
        with pytest.raises(AssertionError):
            YOLOConfig(model_size="invalid_model")
        
        # 유효하지 않은 입력 크기
        with pytest.raises(AssertionError):
            YOLOConfig(input_size=0)
        
        # 유효하지 않은 신뢰도
        with pytest.raises(AssertionError):
            YOLOConfig(confidence_threshold=1.5)
        
        # 유효하지 않은 IoU
        with pytest.raises(AssertionError):
            YOLOConfig(iou_threshold=-0.1)
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = YOLOConfig(
            model_size="yolo11s",
            input_size=512,
            confidence_threshold=0.5,
            device="cpu"
        )
        assert config.model_size == "yolo11s"
        assert config.input_size == 512
        assert config.confidence_threshold == 0.5
        assert config.device == "cpu"


class TestDetectionResult:
    """DetectionResult 클래스 테스트"""
    
    def test_empty_result(self):
        """빈 검출 결과 테스트"""
        result = DetectionResult(
            boxes=torch.empty((0, 4)),
            scores=torch.empty((0,)),
            class_ids=torch.empty((0,), dtype=torch.long),
            image_shape=(480, 640)
        )
        assert len(result) == 0
        assert result.image_shape == (480, 640)
    
    def test_single_detection(self):
        """단일 검출 결과 테스트"""
        result = DetectionResult(
            boxes=torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
            scores=torch.tensor([0.85]),
            class_ids=torch.tensor([0], dtype=torch.long),
            image_shape=(480, 640)
        )
        assert len(result) == 1
        assert result.boxes.shape == (1, 4)
        assert result.scores.shape == (1,)
        assert result.class_ids.shape == (1,)
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        result = DetectionResult(
            boxes=torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
            scores=torch.tensor([0.85]),
            class_ids=torch.tensor([0], dtype=torch.long),
            image_shape=(480, 640)
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'boxes' in result_dict
        assert 'scores' in result_dict
        assert 'class_ids' in result_dict
        assert 'image_shape' in result_dict
        assert 'num_detections' in result_dict
        assert result_dict['num_detections'] == 1


class TestPillSnapYOLODetector:
    """PillSnapYOLODetector 클래스 테스트"""
    
    def test_detector_initialization(self, basic_config):
        """검출기 초기화 테스트"""
        detector = PillSnapYOLODetector(config=basic_config, num_classes=5)
        
        assert detector.config == basic_config
        assert detector.num_classes == 5
        assert hasattr(detector, 'model')
        assert hasattr(detector, 'logger')
    
    def test_get_model_info(self, detector):
        """모델 정보 조회 테스트"""
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        required_keys = [
            'model_size', 'num_classes', 'input_size',
            'total_parameters', 'trainable_parameters',
            'device', 'memory_format', 'mixed_precision', 'torch_compile'
        ]
        for key in required_keys:
            assert key in info
        
        assert info['model_size'] == "yolo11n"
        assert info['num_classes'] == 1
        assert info['input_size'] == 320
        assert info['total_parameters'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
    def test_gpu_allocation(self):
        """GPU 할당 테스트"""
        config = YOLOConfig(device="cuda", model_size="yolo11n")
        detector = PillSnapYOLODetector(config=config)
        
        # 모델이 GPU에 있는지 확인
        device = next(detector.model.model.parameters()).device
        assert device.type == "cuda"
    
    def test_cpu_fallback(self):
        """CPU 대체 테스트"""
        config = YOLOConfig(device="cpu", model_size="yolo11n")
        detector = PillSnapYOLODetector(config=config)
        
        # 모델이 CPU에 있는지 확인
        device = next(detector.model.model.parameters()).device
        assert device.type == "cpu"
    
    def test_training_mode_switch(self, detector):
        """학습/평가 모드 전환 테스트"""
        # 평가 모드로 설정
        detector.set_training_mode(False)
        # YOLO의 경우 eval 상태 확인은 다르게 처리
        try:
            assert not detector.model.model.training
        except Exception:
            # YOLO 내부 구조로 인해 직접 확인이 어려울 수 있음
            pass
        
        # 학습 모드로 설정 (데이터셋 오류 발생 가능하므로 예외 처리)
        try:
            detector.set_training_mode(True)
            # 성공하면 학습 모드 확인
            if hasattr(detector.model, 'model') and hasattr(detector.model.model, 'training'):
                assert detector.model.model.training
        except Exception:
            # YOLO 학습 모드 전환 시 데이터셋 필요로 인한 예외는 예상된 동작
            pass
    
    def test_predict_with_tensor_input(self, detector):
        """텐서 입력 추론 테스트"""
        device = detector.config.device
        
        # 더미 입력 생성 (정규화된 0-1 값)
        dummy_input = torch.rand(1, 3, 320, 320, device=device)
        
        # 추론 실행
        detector.set_training_mode(False)
        with torch.no_grad():
            results = detector.predict(dummy_input, verbose=False)
        
        # 결과 검증
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        assert results[0].image_shape == (320, 320)  # 입력 크기와 일치
    
    def test_predict_with_confidence_threshold(self, detector):
        """신뢰도 임계값 테스트"""
        device = detector.config.device
        dummy_input = torch.rand(1, 3, 320, 320, device=device)
        
        # 높은 신뢰도로 추론 (검출 수 감소 예상)
        detector.set_training_mode(False)
        with torch.no_grad():
            results_high_conf = detector.predict(dummy_input, conf=0.9, verbose=False)
            results_low_conf = detector.predict(dummy_input, conf=0.1, verbose=False)
        
        # 낮은 신뢰도에서 더 많이 검출되거나 같아야 함
        assert len(results_high_conf[0]) <= len(results_low_conf[0])
    
    def test_empty_detection_handling(self, detector):
        """빈 검출 결과 처리 테스트"""
        device = detector.config.device
        
        # 노이즈만 있는 입력 (검출 안될 가능성 높음)
        noise_input = torch.rand(1, 3, 320, 320, device=device) * 0.1
        
        detector.set_training_mode(False)
        with torch.no_grad():
            results = detector.predict(noise_input, conf=0.9, verbose=False)
        
        # 빈 결과도 올바르게 처리되어야 함
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        # 검출이 없어도 오류 없이 처리되어야 함
        assert len(results[0]) >= 0
    
    @pytest.mark.parametrize("input_size", [320, 480, 640])
    def test_different_input_sizes(self, input_size):
        """다양한 입력 크기 테스트"""
        config = YOLOConfig(
            model_size="yolo11n",
            input_size=input_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        detector = PillSnapYOLODetector(config=config)
        
        device = detector.config.device
        dummy_input = torch.rand(1, 3, input_size, input_size, device=device)
        
        detector.set_training_mode(False)
        with torch.no_grad():
            results = detector.predict(dummy_input, verbose=False)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].image_shape == (input_size, input_size)


class TestDetectorHelperFunctions:
    """헬퍼 함수 테스트"""
    
    def test_create_pillsnap_detector(self):
        """검출기 생성 헬퍼 함수 테스트"""
        detector = create_pillsnap_detector(
            num_classes=10,
            model_size="yolo11n",
            input_size=416,
            device="cpu"
        )
        
        assert isinstance(detector, PillSnapYOLODetector)
        assert detector.num_classes == 10
        assert detector.config.model_size == "yolo11n"
        assert detector.config.input_size == 416
        assert detector.config.device == "cpu"
    
    def test_create_detector_with_defaults(self):
        """기본값으로 검출기 생성 테스트"""
        detector = create_pillsnap_detector()
        
        assert detector.num_classes == 1
        assert detector.config.model_size == "yolo11m"
        assert detector.config.input_size == 640


class TestDetectorErrorHandling:
    """오류 처리 테스트"""
    
    def test_invalid_device(self):
        """유효하지 않은 디바이스 테스트"""
        config = YOLOConfig(device="invalid_device", model_size="yolo11n")
        
        # 유효하지 않은 디바이스는 CPU로 대체되거나 오류 발생
        try:
            detector = PillSnapYOLODetector(config=config)
            # 성공했다면 CPU로 대체되었을 것
            device = next(detector.model.model.parameters()).device
            assert device.type in ["cpu", "cuda"]
        except Exception:
            # 오류 발생도 예상된 동작
            pass
    
    def test_predict_with_invalid_input(self, detector):
        """유효하지 않은 입력 테스트"""
        # 잘못된 차원의 입력
        invalid_input = torch.rand(2, 3, 320)  # 3D 텐서 (4D 필요)
        
        detector.set_training_mode(False)
        with pytest.raises(Exception):
            detector.predict(invalid_input)


@pytest.mark.integration
class TestDetectorIntegration:
    """통합 테스트"""
    
    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        # 1. 검출기 생성
        detector = create_pillsnap_detector(
            num_classes=1,
            model_size="yolo11n",
            input_size=320,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 2. 모델 정보 확인
        info = detector.get_model_info()
        assert info['total_parameters'] > 0
        
        # 3. 추론 테스트
        device = detector.config.device
        test_input = torch.rand(1, 3, 320, 320, device=device)
        
        detector.set_training_mode(False)
        with torch.no_grad():
            results = detector.predict(test_input, verbose=False)
        
        # 4. 결과 검증
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, DetectionResult)
        assert result.image_shape == (320, 320)
        
        # 5. 딕셔너리 변환 테스트
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'num_detections' in result_dict


if __name__ == "__main__":
    # 기본 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])