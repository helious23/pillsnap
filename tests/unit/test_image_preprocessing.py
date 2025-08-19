"""
이미지 전처리 파이프라인 단위 테스트

Two-Stage Conditional Pipeline 이미지 전처리 검증:
- 검출용/분류용 리사이즈 정확성
- 이미지 품질 검증 기능
- 배치 처리 성능
- 메모리 최적화 확인
"""

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.image_preprocessing import (
    TwoStageImagePreprocessor,
    ImageProcessingConfig,
    ImageFormatValidator,
    ImageResizeProcessor,
    PipelineStage,
    image_preprocessing_factory
)


class TestImageFormatValidator:
    """이미지 포맷 검증기 테스트"""
    
    def test_supported_formats_validation(self):
        """지원 포맷 검증"""
        validator = ImageFormatValidator()
        
        # 지원 포맷
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for ext in supported_extensions:
            assert ext in validator.SUPPORTED_FORMATS
        
        # 미지원 포맷
        unsupported_extensions = ['.gif', '.svg', '.raw', '.cr2']
        for ext in unsupported_extensions:
            assert ext not in validator.SUPPORTED_FORMATS
    
    def test_validation_with_valid_image(self):
        """유효한 이미지 검증"""
        # 테스트용 이미지 생성
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_image = Image.new('RGB', (640, 480), color='red')
            test_image.save(f.name)
            temp_path = Path(f.name)
        
        try:
            validator = ImageFormatValidator()
            result = validator.validate_image_file(temp_path)
            
            assert result['is_valid'] == True
            assert len(result['errors']) == 0
            assert result['metadata']['width'] == 640
            assert result['metadata']['height'] == 480
            assert result['metadata']['mode'] == 'RGB'
            
        finally:
            temp_path.unlink()
    
    def test_validation_with_nonexistent_file(self):
        """존재하지 않는 파일 검증"""
        validator = ImageFormatValidator()
        fake_path = Path("/nonexistent/fake_image.jpg")
        
        result = validator.validate_image_file(fake_path)
        
        assert result['is_valid'] == False
        assert any("존재하지 않음" in error for error in result['errors'])
    
    def test_validation_with_unsupported_format(self):
        """지원하지 않는 포맷 검증"""
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            temp_path = Path(f.name)
            temp_path.write_text("fake content")
        
        try:
            validator = ImageFormatValidator()
            result = validator.validate_image_file(temp_path)
            
            assert result['is_valid'] == False
            assert any("지원하지 않는 포맷" in error for error in result['errors'])
            
        finally:
            temp_path.unlink()


class TestImageResizeProcessor:
    """이미지 리사이즈 처리기 테스트"""
    
    @pytest.fixture
    def config(self):
        return ImageProcessingConfig()
    
    @pytest.fixture
    def resizer(self, config):
        return ImageResizeProcessor(config)
    
    @pytest.fixture
    def test_image(self):
        """테스트용 이미지 생성 (800x600)"""
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    def test_resize_for_detection_aspect_ratio_preservation(self, resizer, test_image):
        """검출용 리사이즈 종횡비 유지 확인"""
        resized = resizer.resize_for_detection(test_image)
        
        # 출력 크기 확인
        assert resized.shape == (640, 640, 3)
        
        # 패딩이 올바르게 적용되었는지 확인 (검은색 패딩)
        # 원본이 800x600이므로 640x480으로 스케일링 후 상하 패딩
        # 상단과 하단에 검은색 패딩이 있어야 함
        top_padding = resized[0, :, :]  # 첫 번째 행
        bottom_padding = resized[-1, :, :]  # 마지막 행
        
        # 패딩 영역이 검은색인지 확인 (일부 픽셀만 확인)
        assert np.all(top_padding[:100] == 0) or np.all(bottom_padding[:100] == 0)
    
    def test_resize_for_classification_center_crop(self, resizer, test_image):
        """분류용 리사이즈 중앙 크롭 확인"""
        resized = resizer.resize_for_classification(test_image)
        
        # 출력 크기 확인
        assert resized.shape == (384, 384, 3)
        
        # 크롭된 이미지에 실제 내용이 있는지 확인 (모두 0이 아님)
        assert not np.all(resized == 0)
    
    def test_smart_resize_pipeline_stages(self, resizer, test_image):
        """지능형 리사이즈 파이프라인 단계별 확인"""
        # 검출 단계
        detection_result = resizer.smart_resize(test_image, PipelineStage.DETECTION)
        assert detection_result.shape == (640, 640, 3)
        
        # 분류 단계
        classification_result = resizer.smart_resize(test_image, PipelineStage.CLASSIFICATION)
        assert classification_result.shape == (384, 384, 3)
        
        # 지원하지 않는 단계
        with pytest.raises(ValueError):
            resizer.smart_resize(test_image, PipelineStage.AUGMENTATION)
    
    def test_different_input_sizes(self, resizer):
        """다양한 입력 크기 처리 확인"""
        test_cases = [
            (100, 100),    # 정사각형 작은 이미지
            (1920, 1080),  # 와이드스크린
            (480, 640),    # 세로 긴 이미지
            (2048, 1024),  # 매우 큰 이미지
        ]
        
        for width, height in test_cases:
            test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # 검출용 리사이즈
            detection_result = resizer.resize_for_detection(test_img)
            assert detection_result.shape == (640, 640, 3)
            
            # 분류용 리사이즈
            classification_result = resizer.resize_for_classification(test_img)
            assert classification_result.shape == (384, 384, 3)


class TestTwoStageImagePreprocessor:
    """Two-Stage 이미지 전처리기 테스트"""
    
    @pytest.fixture
    def preprocessor(self):
        config = ImageProcessingConfig(
            batch_size=4,  # 테스트용 작은 배치
            num_workers=2
        )
        return TwoStageImagePreprocessor(config)
    
    @pytest.fixture
    def test_image_file(self):
        """테스트용 이미지 파일 생성"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # 640x480 RGB 이미지 생성
            test_image = Image.new('RGB', (640, 480), color=(128, 64, 192))
            test_image.save(f.name, quality=95)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # 클린업
        temp_path.unlink()
    
    def test_load_and_validate_image_success(self, preprocessor, test_image_file):
        """이미지 로드 및 검증 성공 케이스"""
        is_valid, image_array, validation_info = preprocessor.load_and_validate_image(test_image_file)
        
        assert is_valid == True
        assert image_array is not None
        assert image_array.shape == (480, 640, 3)  # Height x Width x Channels
        assert image_array.dtype == np.uint8
        assert validation_info['is_valid'] == True
        assert validation_info['metadata']['width'] == 640
        assert validation_info['metadata']['height'] == 480
    
    def test_load_and_validate_image_failure(self, preprocessor):
        """이미지 로드 및 검증 실패 케이스"""
        fake_path = Path("/nonexistent/fake.jpg")
        is_valid, image_array, validation_info = preprocessor.load_and_validate_image(fake_path)
        
        assert is_valid == False
        assert image_array is None
        assert validation_info['is_valid'] == False
        assert len(validation_info['errors']) > 0
    
    def test_preprocess_for_detection_training(self, preprocessor, test_image_file):
        """검출용 전처리 (훈련 모드)"""
        success, tensor, info = preprocessor.preprocess_for_detection(test_image_file, is_training=True)
        
        if not success:
            print(f"전처리 실패: {info}")
        
        assert success == True, f"전처리 실패: {info}"
        assert tensor is not None
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 640, 640)  # C x H x W
        assert tensor.dtype == torch.float32
        assert info['stage'] == 'detection'
        assert info['is_training'] == True
        assert 'processing_time_ms' in info
    
    def test_preprocess_for_detection_validation(self, preprocessor, test_image_file):
        """검출용 전처리 (검증 모드)"""
        success, tensor, info = preprocessor.preprocess_for_detection(test_image_file, is_training=False)
        
        assert success == True
        assert tensor is not None
        assert tensor.shape == (3, 640, 640)
        assert info['stage'] == 'detection'
        assert info['is_training'] == False
    
    def test_preprocess_for_classification_training(self, preprocessor, test_image_file):
        """분류용 전처리 (훈련 모드)"""
        success, tensor, info = preprocessor.preprocess_for_classification(test_image_file, is_training=True)
        
        assert success == True
        assert tensor is not None
        assert tensor.shape == (3, 384, 384)
        assert info['stage'] == 'classification'
        assert info['is_training'] == True
    
    def test_preprocess_for_classification_validation(self, preprocessor, test_image_file):
        """분류용 전처리 (검증 모드)"""
        success, tensor, info = preprocessor.preprocess_for_classification(test_image_file, is_training=False)
        
        assert success == True
        assert tensor is not None
        assert tensor.shape == (3, 384, 384)
        assert info['stage'] == 'classification'
        assert info['is_training'] == False
    
    def test_batch_preprocess_detection(self, preprocessor):
        """배치 전처리 - 검출 단계"""
        # 여러 테스트 이미지 생성
        test_images = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=f'_test_{i}.jpg', delete=False) as f:
                test_image = Image.new('RGB', (400, 300), color=(i*50, 100, 150))
                test_image.save(f.name)
                test_images.append(Path(f.name))
        
        try:
            results = preprocessor.batch_preprocess(
                test_images, 
                PipelineStage.DETECTION, 
                is_training=True
            )
            
            assert results['batch_stats']['total_images'] == 3
            assert results['batch_stats']['successful'] == 3
            assert results['batch_stats']['failed'] == 0
            assert len(results['successful_tensors']) == 3
            assert len(results['failed_paths']) == 0
            
            # 모든 텐서가 올바른 형태인지 확인
            for tensor in results['successful_tensors']:
                assert tensor.shape == (3, 640, 640)
            
        finally:
            # 클린업
            for path in test_images:
                path.unlink()
    
    def test_batch_preprocess_classification(self, preprocessor):
        """배치 전처리 - 분류 단계"""
        # 테스트 이미지 생성 (일부는 유효하지 않은 경로)
        test_images = []
        
        # 유효한 이미지 2개
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=f'_valid_{i}.jpg', delete=False) as f:
                test_image = Image.new('RGB', (500, 400), color=(200, i*100, 50))
                test_image.save(f.name)
                test_images.append(Path(f.name))
        
        # 유효하지 않은 경로 1개
        test_images.append(Path("/fake/invalid_path.jpg"))
        
        try:
            results = preprocessor.batch_preprocess(
                test_images, 
                PipelineStage.CLASSIFICATION, 
                is_training=False
            )
            
            assert results['batch_stats']['total_images'] == 3
            assert results['batch_stats']['successful'] == 2
            assert results['batch_stats']['failed'] == 1
            assert len(results['successful_tensors']) == 2
            assert len(results['failed_paths']) == 1
            
            # 성공한 텐서들 확인
            for tensor in results['successful_tensors']:
                assert tensor.shape == (3, 384, 384)
            
        finally:
            # 유효한 파일들만 클린업
            for path in test_images[:2]:
                if path.exists():
                    path.unlink()
    
    def test_performance_stats_tracking(self, preprocessor, test_image_file):
        """성능 통계 추적 확인"""
        # 초기 상태
        initial_stats = preprocessor.get_performance_stats()
        assert initial_stats['processed_images'] == 0
        
        # 몇 개 이미지 처리
        for _ in range(3):
            preprocessor.preprocess_for_detection(test_image_file, is_training=True)
        
        # 통계 확인
        final_stats = preprocessor.get_performance_stats()
        assert final_stats['processed_images'] == 3
        assert len(final_stats['processing_time_ms']) == 3
        assert 'avg_processing_time_ms' in final_stats
        assert final_stats['avg_processing_time_ms'] > 0
    
    def test_memory_format_optimization(self, preprocessor, test_image_file):
        """메모리 포맷 최적화 확인"""
        success, tensor, info = preprocessor.preprocess_for_detection(test_image_file)
        
        assert success == True
        # 3D 텐서에서는 channels_last를 적용할 수 없으므로 기본 텐서 속성만 확인
        assert tensor.dtype == torch.float32
        assert tensor.shape == (3, 640, 640)
        # 텐서가 유효한 범위에 있는지 확인 (정규화 후)
        assert tensor.min() >= -3.0  # 정규화 후 대략적인 최소값
        assert tensor.max() <= 3.0   # 정규화 후 대략적인 최대값


class TestImagePreprocessingFactory:
    """이미지 전처리 팩토리 테스트"""
    
    def test_stage1_preprocessing_factory(self):
        """Stage 1용 전처리기 팩토리"""
        preprocessor = image_preprocessing_factory("stage1")
        
        assert isinstance(preprocessor, TwoStageImagePreprocessor)
        assert preprocessor.config.detection_size == (640, 640)
        assert preprocessor.config.classification_size == (384, 384)
        assert preprocessor.config.batch_size == 16  # Stage 1용 설정
        assert preprocessor.config.memory_format == "channels_last"
    
    def test_default_preprocessing_factory(self):
        """기본 전처리기 팩토리"""
        preprocessor = image_preprocessing_factory("default")
        
        assert isinstance(preprocessor, TwoStageImagePreprocessor)
        assert preprocessor.config.detection_size == (640, 640)
        assert preprocessor.config.classification_size == (384, 384)


class TestImagePreprocessingIntegration:
    """이미지 전처리 통합 테스트"""
    
    def test_end_to_end_preprocessing_workflow(self):
        """End-to-End 전처리 워크플로우"""
        preprocessor = image_preprocessing_factory("stage1")
        
        # 다양한 크기의 테스트 이미지 생성
        test_cases = [
            (800, 600, 'landscape'),
            (600, 800, 'portrait'),
            (512, 512, 'square'),
            (1920, 1080, 'hd')
        ]
        
        for width, height, name in test_cases:
            with tempfile.NamedTemporaryFile(suffix=f'_{name}.jpg', delete=False) as f:
                test_image = Image.new('RGB', (width, height), color=(100, 150, 200))
                test_image.save(f.name, quality=95)
                temp_path = Path(f.name)
            
            try:
                # 검출용 전처리
                det_success, det_tensor, det_info = preprocessor.preprocess_for_detection(
                    temp_path, is_training=True
                )
                assert det_success == True
                assert det_tensor.shape == (3, 640, 640)
                
                # 분류용 전처리
                cls_success, cls_tensor, cls_info = preprocessor.preprocess_for_classification(
                    temp_path, is_training=False
                )
                assert cls_success == True
                assert cls_tensor.shape == (3, 384, 384)
                
                # 처리 시간이 합리적인지 확인 (5초 미만)
                assert det_info['processing_time_ms'] < 5000
                assert cls_info['processing_time_ms'] < 5000
                
            finally:
                temp_path.unlink()
    
    def test_error_handling_robustness(self):
        """에러 처리 견고성 테스트"""
        preprocessor = image_preprocessing_factory("stage1")
        
        # 다양한 에러 케이스
        error_cases = [
            Path("/nonexistent/path.jpg"),  # 존재하지 않는 파일
            Path("fake_image.gif"),         # 지원하지 않는 포맷
        ]
        
        for error_path in error_cases:
            # 검출용 전처리 에러 처리
            det_success, det_tensor, det_info = preprocessor.preprocess_for_detection(error_path)
            assert det_success == False
            assert det_tensor is None
            assert 'errors' in det_info
            
            # 분류용 전처리 에러 처리
            cls_success, cls_tensor, cls_info = preprocessor.preprocess_for_classification(error_path)
            assert cls_success == False
            assert cls_tensor is None
            assert 'errors' in cls_info


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])