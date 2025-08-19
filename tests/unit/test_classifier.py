"""
EfficientNetV2-S 분류기 단위 테스트

테스트 범위:
- 모델 초기화 및 설정
- GPU 할당 및 메모리 관리
- 분류 기능 및 결과 형식
- Top-K 예측 및 특징 추출
- 배치 처리 및 온도 스케일링
- 모델 저장/로드 기능
- 오류 처리 및 예외 상황
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.classifier import (
    PillSnapClassifier, 
    ClassifierConfig, 
    ClassificationResult,
    create_pillsnap_classifier
)


@pytest.fixture
def basic_config():
    """기본 분류기 설정"""
    return ClassifierConfig(
        model_name="tf_efficientnetv2_s",
        num_classes=100,  # 테스트용 작은 클래스 수
        input_size=224,  # 테스트용 작은 크기
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=False,  # 테스트에서는 비활성화
        torch_compile=False,  # 테스트에서는 비활성화
        channels_last=False,  # 테스트용 비활성화
        dropout_rate=0.0,  # 테스트용 비활성화
        drop_path_rate=0.0  # 테스트용 비활성화
    )


@pytest.fixture
def classifier(basic_config):
    """기본 분류기 인스턴스"""
    return PillSnapClassifier(config=basic_config)


class TestClassifierConfig:
    """ClassifierConfig 클래스 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = ClassifierConfig()
        assert config.model_name == "tf_efficientnetv2_s"
        assert config.num_classes == 4523
        assert config.input_size == 384
        assert config.pretrained == True
        assert config.device == "cuda"
        assert config.mixed_precision == True
    
    def test_config_validation(self):
        """설정 유효성 검증 테스트"""
        # 유효하지 않은 클래스 수
        with pytest.raises(AssertionError):
            ClassifierConfig(num_classes=0)
        
        # 유효하지 않은 입력 크기
        with pytest.raises(AssertionError):
            ClassifierConfig(input_size=0)
        
        # 유효하지 않은 드롭아웃
        with pytest.raises(AssertionError):
            ClassifierConfig(dropout_rate=1.5)
        
        # 유효하지 않은 드롭 패스
        with pytest.raises(AssertionError):
            ClassifierConfig(drop_path_rate=-0.1)
        
        # 유효하지 않은 라벨 스무딩
        with pytest.raises(AssertionError):
            ClassifierConfig(label_smoothing=1.5)
    
    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = ClassifierConfig(
            model_name="tf_efficientnetv2_s",
            num_classes=1000,
            input_size=224,
            dropout_rate=0.2,
            device="cpu"
        )
        assert config.model_name == "tf_efficientnetv2_s"
        assert config.num_classes == 1000
        assert config.input_size == 224
        assert config.dropout_rate == 0.2
        assert config.device == "cpu"


class TestClassificationResult:
    """ClassificationResult 클래스 테스트"""
    
    def test_empty_result(self):
        """빈 분류 결과 테스트"""
        result = ClassificationResult(
            logits=torch.empty((0, 100)),
            probabilities=torch.empty((0, 100)),
            predicted_classes=torch.empty((0,), dtype=torch.long),
            confidence_scores=torch.empty((0,))
        )
        assert len(result) == 0
    
    def test_single_classification(self):
        """단일 분류 결과 테스트"""
        result = ClassificationResult(
            logits=torch.randn(1, 100),
            probabilities=torch.softmax(torch.randn(1, 100), dim=1),
            predicted_classes=torch.tensor([42], dtype=torch.long),
            confidence_scores=torch.tensor([0.85])
        )
        assert len(result) == 1
        assert result.logits.shape == (1, 100)
        assert result.probabilities.shape == (1, 100)
        assert result.predicted_classes.shape == (1,)
        assert result.confidence_scores.shape == (1,)
    
    def test_batch_classification(self):
        """배치 분류 결과 테스트"""
        batch_size = 4
        num_classes = 100
        
        result = ClassificationResult(
            logits=torch.randn(batch_size, num_classes),
            probabilities=torch.softmax(torch.randn(batch_size, num_classes), dim=1),
            predicted_classes=torch.randint(0, num_classes, (batch_size,)),
            confidence_scores=torch.rand(batch_size)
        )
        assert len(result) == batch_size
        assert result.logits.shape == (batch_size, num_classes)
        assert result.probabilities.shape == (batch_size, num_classes)
        assert result.predicted_classes.shape == (batch_size,)
        assert result.confidence_scores.shape == (batch_size,)
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        result = ClassificationResult(
            logits=torch.randn(2, 10),
            probabilities=torch.softmax(torch.randn(2, 10), dim=1),
            predicted_classes=torch.tensor([3, 7]),
            confidence_scores=torch.tensor([0.85, 0.92])
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'logits' in result_dict
        assert 'probabilities' in result_dict
        assert 'predicted_classes' in result_dict
        assert 'confidence_scores' in result_dict
        assert 'batch_size' in result_dict
        assert result_dict['batch_size'] == 2
    
    def test_top_k_predictions(self):
        """Top-K 예측 테스트"""
        # 확실한 확률 분포 생성
        probs = torch.zeros(2, 10)
        probs[0, [0, 3, 7]] = torch.tensor([0.6, 0.3, 0.1])  # 첫 번째 샘플
        probs[1, [1, 5, 9]] = torch.tensor([0.5, 0.3, 0.2])  # 두 번째 샘플
        
        result = ClassificationResult(
            logits=torch.empty(2, 10),
            probabilities=probs,
            predicted_classes=torch.tensor([0, 1]),
            confidence_scores=torch.tensor([0.6, 0.5])
        )
        
        top_k = result.get_top_k_predictions(k=3)
        assert isinstance(top_k, dict)
        assert 'top_k_classes' in top_k
        assert 'top_k_probabilities' in top_k
        assert 'k' in top_k
        assert top_k['k'] == 3
        assert len(top_k['top_k_classes']) == 2  # 배치 크기
        assert len(top_k['top_k_classes'][0]) == 3  # k=3


class TestPillSnapClassifier:
    """PillSnapClassifier 클래스 테스트"""
    
    def test_classifier_initialization(self, basic_config):
        """분류기 초기화 테스트"""
        classifier = PillSnapClassifier(config=basic_config)
        
        assert classifier.config == basic_config
        assert hasattr(classifier, 'backbone')
        assert hasattr(classifier, 'logger')
    
    def test_get_model_info(self, classifier):
        """모델 정보 조회 테스트"""
        info = classifier.get_model_info()
        
        assert isinstance(info, dict)
        required_keys = [
            'model_name', 'num_classes', 'input_size',
            'total_parameters', 'trainable_parameters',
            'device', 'memory_format', 'mixed_precision', 'torch_compile',
            'dropout_rate', 'drop_path_rate'
        ]
        for key in required_keys:
            assert key in info
        
        assert info['model_name'] == "tf_efficientnetv2_s"
        assert info['num_classes'] == 100
        assert info['input_size'] == 224
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
    def test_gpu_allocation(self):
        """GPU 할당 테스트"""
        config = ClassifierConfig(
            device="cuda", 
            num_classes=10, 
            input_size=224,
            channels_last=False
        )
        classifier = PillSnapClassifier(config=config)
        
        # 모델이 GPU에 있는지 확인
        device = next(classifier.parameters()).device
        assert device.type == "cuda"
    
    def test_cpu_allocation(self):
        """CPU 할당 테스트"""
        config = ClassifierConfig(
            device="cpu", 
            num_classes=10, 
            input_size=224,
            channels_last=False
        )
        classifier = PillSnapClassifier(config=config)
        
        # 모델이 CPU에 있는지 확인
        device = next(classifier.parameters()).device
        assert device.type == "cpu"
    
    def test_forward_pass(self, classifier):
        """순전파 테스트"""
        device = classifier.config.device
        batch_size = 2
        input_size = classifier.config.input_size
        
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier.forward(x)
        
        assert logits.shape == (batch_size, classifier.config.num_classes)
        assert not torch.isnan(logits).any()
        assert torch.isfinite(logits).all()
    
    def test_predict_single_batch(self, classifier):
        """단일 배치 예측 테스트"""
        device = classifier.config.device
        batch_size = 3
        input_size = classifier.config.input_size
        
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)
        
        result = classifier.predict(x)
        
        assert isinstance(result, ClassificationResult)
        assert len(result) == batch_size
        assert result.probabilities.shape == (batch_size, classifier.config.num_classes)
        assert result.predicted_classes.shape == (batch_size,)
        assert result.confidence_scores.shape == (batch_size,)
        
        # 확률의 합이 1에 가까운지 확인
        prob_sums = result.probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
        
        # 신뢰도가 확률의 최댓값과 일치하는지 확인
        max_probs, _ = torch.max(result.probabilities, dim=1)
        assert torch.allclose(result.confidence_scores, max_probs, atol=1e-6)
    
    def test_predict_with_temperature(self, classifier):
        """온도 스케일링 예측 테스트"""
        device = classifier.config.device
        x = torch.randn(1, 3, classifier.config.input_size, classifier.config.input_size, device=device)
        
        # 온도 1.0 (기본)
        result_normal = classifier.predict(x, temperature=1.0)
        
        # 온도 2.0 (더 부드러운 분포)
        result_soft = classifier.predict(x, temperature=2.0)
        
        # 온도 0.5 (더 날카로운 분포)
        result_sharp = classifier.predict(x, temperature=0.5)
        
        # 온도가 높을수록 더 균등한 분포가 되어야 함
        entropy_normal = -torch.sum(result_normal.probabilities * torch.log(result_normal.probabilities + 1e-8))
        entropy_soft = -torch.sum(result_soft.probabilities * torch.log(result_soft.probabilities + 1e-8))
        entropy_sharp = -torch.sum(result_sharp.probabilities * torch.log(result_sharp.probabilities + 1e-8))
        
        assert entropy_soft >= entropy_normal >= entropy_sharp
    
    def test_predict_with_logits(self, classifier):
        """로짓 반환 예측 테스트"""
        device = classifier.config.device
        x = torch.randn(2, 3, classifier.config.input_size, classifier.config.input_size, device=device)
        
        result_no_logits = classifier.predict(x, return_logits=False)
        result_with_logits = classifier.predict(x, return_logits=True)
        
        # 로짓이 요청되지 않으면 빈 텐서
        assert result_no_logits.logits.numel() == 0
        
        # 로짓이 요청되면 적절한 크기
        assert result_with_logits.logits.shape == (2, classifier.config.num_classes)
    
    def test_predict_batch(self, classifier):
        """배치 단위 예측 테스트"""
        device = classifier.config.device
        total_size = 7  # 배치 크기로 나누어 떨어지지 않는 수
        batch_size = 3
        input_size = classifier.config.input_size
        
        x = torch.randn(total_size, 3, input_size, input_size, device=device)
        
        result = classifier.predict_batch(x, batch_size=batch_size)
        
        assert isinstance(result, ClassificationResult)
        assert len(result) == total_size
        assert result.probabilities.shape == (total_size, classifier.config.num_classes)
    
    def test_extract_features(self, classifier):
        """특징 추출 테스트"""
        device = classifier.config.device
        batch_size = 2
        input_size = classifier.config.input_size
        
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)
        
        features = classifier.extract_features(x)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == batch_size
        assert features.ndim == 2  # [batch_size, feature_dim]
        assert not torch.isnan(features).any()
    
    @pytest.mark.parametrize("input_size", [224, 256, 384])
    def test_different_input_sizes(self, input_size):
        """다양한 입력 크기 테스트"""
        config = ClassifierConfig(
            num_classes=10,
            input_size=input_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            channels_last=False
        )
        classifier = PillSnapClassifier(config=config)
        
        device = classifier.config.device
        x = torch.randn(1, 3, input_size, input_size, device=device)
        
        result = classifier.predict(x)
        
        assert isinstance(result, ClassificationResult)
        assert len(result) == 1
        assert result.probabilities.shape == (1, 10)
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, classifier, batch_size):
        """다양한 배치 크기 테스트"""
        device = classifier.config.device
        input_size = classifier.config.input_size
        
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)
        
        result = classifier.predict(x)
        
        assert len(result) == batch_size
        assert result.probabilities.shape == (batch_size, classifier.config.num_classes)


class TestClassifierHelperFunctions:
    """헬퍼 함수 테스트"""
    
    def test_create_pillsnap_classifier(self):
        """분류기 생성 헬퍼 함수 테스트"""
        classifier = create_pillsnap_classifier(
            num_classes=1000,
            model_name="tf_efficientnetv2_s",
            input_size=224,
            device="cpu",
            pretrained=False  # 테스트 속도 향상
        )
        
        assert isinstance(classifier, PillSnapClassifier)
        assert classifier.config.num_classes == 1000
        assert classifier.config.model_name == "tf_efficientnetv2_s"
        assert classifier.config.input_size == 224
        assert classifier.config.device == "cpu"
        assert classifier.config.pretrained == False
    
    def test_create_classifier_with_defaults(self):
        """기본값으로 분류기 생성 테스트"""
        classifier = create_pillsnap_classifier(pretrained=False)  # 테스트 속도 향상
        
        assert classifier.config.num_classes == 4523
        assert classifier.config.model_name == "tf_efficientnetv2_s"
        assert classifier.config.input_size == 384


class TestClassifierErrorHandling:
    """오류 처리 테스트"""
    
    def test_invalid_input_dimensions(self, classifier):
        """유효하지 않은 입력 차원 테스트"""
        device = classifier.config.device
        
        # 잘못된 차원의 입력
        invalid_inputs = [
            torch.randn(3, 224, 224, device=device),  # 채널 차원 누락
            torch.randn(1, 4, 224, 224, device=device),  # 잘못된 채널 수
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(Exception):
                classifier.predict(invalid_input)
        
        # 잘못된 이미지 크기는 EfficientNet에서 자동으로 처리되므로 테스트에서 제외
    
    def test_empty_input(self, classifier):
        """빈 입력 테스트"""
        device = classifier.config.device
        input_size = classifier.config.input_size
        
        # 배치 크기가 0인 입력
        empty_input = torch.empty(0, 3, input_size, input_size, device=device)
        
        result = classifier.predict(empty_input)
        assert len(result) == 0
    
    def test_model_save_load(self, classifier, tmp_path):
        """모델 저장/로드 테스트"""
        # 테스트 입력 생성
        device = classifier.config.device
        input_size = classifier.config.input_size
        test_input = torch.randn(1, 3, input_size, input_size, device=device)
        
        # 저장 전 예측 결과
        original_result = classifier.predict(test_input)
        
        # 모델 저장
        save_path = tmp_path / "test_classifier.pth"
        classifier.save_model(save_path)
        assert save_path.exists()
        
        # 같은 분류기에 로드
        classifier.load_model(save_path)
        
        # 동일한 입력에 대해 동일한 출력인지 확인
        loaded_result = classifier.predict(test_input)
        
        assert torch.allclose(
            original_result.probabilities, 
            loaded_result.probabilities, 
            atol=1e-5
        )
    
    def test_nonexistent_model_load(self, classifier):
        """존재하지 않는 모델 로드 테스트"""
        with pytest.raises(FileNotFoundError):
            classifier.load_model("/path/that/does/not/exist.pth")


@pytest.mark.integration
class TestClassifierIntegration:
    """통합 테스트"""
    
    def test_full_classification_pipeline(self):
        """전체 분류 파이프라인 테스트"""
        # 1. 분류기 생성
        classifier = create_pillsnap_classifier(
            num_classes=100,
            input_size=224,
            device="cuda" if torch.cuda.is_available() else "cpu",
            pretrained=False  # 테스트 속도 향상
        )
        
        # 2. 모델 정보 확인
        info = classifier.get_model_info()
        assert info['total_parameters'] > 0
        
        # 3. 다양한 크기의 배치 테스트
        device = classifier.config.device
        test_cases = [1, 2, 5, 8]
        
        for batch_size in test_cases:
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            
            # 예측
            result = classifier.predict(x)
            assert len(result) == batch_size
            
            # Top-K 예측
            top_k = result.get_top_k_predictions(k=5)
            assert len(top_k['top_k_classes']) == batch_size
            
            # 특징 추출
            features = classifier.extract_features(x)
            assert features.shape[0] == batch_size
        
        # 4. 온도 스케일링 테스트
        x = torch.randn(1, 3, 224, 224, device=device)
        for temperature in [0.5, 1.0, 2.0]:
            result = classifier.predict(x, temperature=temperature)
            assert len(result) == 1
        
        # 5. 배치 처리 테스트
        large_x = torch.randn(15, 3, 224, 224, device=device)
        batch_result = classifier.predict_batch(large_x, batch_size=4)
        assert len(batch_result) == 15


if __name__ == "__main__":
    # 기본 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])