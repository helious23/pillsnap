"""
Progressive Validation 샘플링 시스템 단위 테스트

테스트 범위:
- Stage1SamplingStrategy 설정 검증
- ProgressiveValidationSampler 핵심 기능
- 이미지 품질 검증
- 샘플 분포 검증
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.sampling import (
    Stage1SamplingStrategy,
    ProgressiveValidationSampler,
    SamplingStats,
    validate_sample_distribution
)


class TestStage1SamplingStrategy:
    """Stage1SamplingStrategy 테스트"""
    
    def test_default_strategy_valid(self):
        """기본 전략 설정 유효성 테스트"""
        strategy = Stage1SamplingStrategy()
        
        assert strategy.target_images == 5000
        assert strategy.target_classes == 50
        assert strategy.images_per_class == 100
        assert strategy.min_images_per_class == 80
        assert strategy.max_images_per_class == 120
        assert 0 < strategy.quality_threshold <= 1
        assert 0 < strategy.single_combo_ratio < 1
        assert strategy.seed == 42
    
    def test_custom_strategy_validation(self):
        """커스텀 전략 유효성 검증 테스트"""
        # 유효한 커스텀 설정
        strategy = Stage1SamplingStrategy(
            target_images=1000,
            target_classes=10,
            images_per_class=100,
            quality_threshold=0.9,
            single_combo_ratio=0.8
        )
        assert strategy.target_images == 1000
        assert strategy.target_classes == 10
        
        # 잘못된 설정들 테스트
        with pytest.raises(AssertionError):
            Stage1SamplingStrategy(target_images=-1)
        
        with pytest.raises(AssertionError):
            Stage1SamplingStrategy(quality_threshold=1.5)
        
        with pytest.raises(AssertionError):
            Stage1SamplingStrategy(single_combo_ratio=0)
    
    def test_images_per_class_auto_adjustment(self):
        """클래스당 이미지 수 자동 조정 테스트"""
        # 나누어 떨어지지 않는 경우
        strategy = Stage1SamplingStrategy(
            target_images=1001,  # 10으로 나누어떨어지지 않음
            target_classes=10
        )
        # 자동으로 100으로 조정되어야 함
        assert strategy.images_per_class == 100


class TestProgressiveValidationSampler:
    """ProgressiveValidationSampler 테스트"""
    
    @pytest.fixture
    def mock_data_structure(self):
        """테스트용 데이터 구조 Mock"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 가짜 데이터 구조 생성
            single_dir = temp_path / "data/train/images/single"
            combo_dir = temp_path / "data/train/images/combination"
            
            # TS 디렉토리와 K-코드 디렉토리 생성
            for i in range(3):
                ts_single = single_dir / f"TS_{i}"
                ts_combo = combo_dir / f"TS_{i}"
                
                for j in range(20):  # 20개 K-코드
                    k_code = f"K{i:02d}{j:03d}"
                    
                    # Single 이미지
                    k_single_dir = ts_single / k_code
                    k_single_dir.mkdir(parents=True, exist_ok=True)
                    for img_idx in range(100):  # 클래스당 100개 이미지
                        img_file = k_single_dir / f"{k_code}_{img_idx:03d}.jpg"
                        img_file.write_text("fake_image_data")
                    
                    # Combo 이미지 (절반만)
                    if j < 10:
                        k_combo_dir = ts_combo / k_code
                        k_combo_dir.mkdir(parents=True, exist_ok=True)
                        for img_idx in range(50):  # 클래스당 50개 이미지
                            img_file = k_combo_dir / f"{k_code}_{img_idx:03d}.jpg"
                            img_file.write_text("fake_image_data")
            
            yield temp_path
    
    @pytest.fixture
    def sampler(self, mock_data_structure):
        """테스트용 샘플러 인스턴스"""
        strategy = Stage1SamplingStrategy(
            target_images=500,  # 테스트용으로 작은 값
            target_classes=5,
            seed=42
        )
        return ProgressiveValidationSampler(str(mock_data_structure), strategy)
    
    def test_sampler_initialization(self, sampler):
        """샘플러 초기화 테스트"""
        assert sampler.strategy.target_images == 500
        assert sampler.strategy.target_classes == 5
        assert sampler.artifacts_dir.name == "sampling"
    
    @patch('src.data.sampling.Image.open')
    def test_validate_image_quality(self, mock_image_open, sampler):
        """이미지 품질 검증 테스트"""
        # 유효한 이미지 Mock
        mock_img = Mock()
        mock_img.size = (256, 256)
        mock_img.mode = 'RGB'
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        test_path = Path("/fake/image.jpg")
        assert sampler.validate_image_quality(test_path) == True
        
        # 너무 작은 이미지
        mock_img.size = (16, 16)
        assert sampler.validate_image_quality(test_path) == False
        
        # 잘못된 모드
        mock_img.size = (256, 256)
        mock_img.mode = 'UNKNOWN'
        mock_img.convert.side_effect = Exception("Cannot convert")
        assert sampler.validate_image_quality(test_path) == False
        
        # 파일 열기 실패
        mock_image_open.side_effect = Exception("Cannot open")
        assert sampler.validate_image_quality(test_path) == False
    
    def test_scan_available_data(self, sampler):
        """데이터 스캔 기능 테스트"""
        scan_result = sampler.scan_available_data()
        
        assert 'single' in scan_result
        assert 'combo' in scan_result
        assert 'all_k_codes' in scan_result
        assert 'k_code_counts' in scan_result
        
        # 생성한 K-코드들이 모두 스캔되었는지 확인
        assert len(scan_result['all_k_codes']) > 0
        assert len(scan_result['k_code_counts']) > 0
    
    def test_select_target_classes(self, sampler):
        """목표 클래스 선택 테스트"""
        # Mock 데이터로 K-코드 카운트 생성
        k_code_counts = {f"K{i:05d}": 100 + i for i in range(100)}
        
        selected = sampler.select_target_classes(k_code_counts)
        
        assert len(selected) == sampler.strategy.target_classes
        
        # 이미지 수가 많은 순으로 선택되었는지 확인
        for i in range(len(selected) - 1):
            assert k_code_counts[selected[i]] >= k_code_counts[selected[i + 1]]
    
    def test_select_target_classes_insufficient_data(self, sampler):
        """데이터 부족시 예외 처리 테스트"""
        # 충분한 이미지가 없는 K-코드들만 있는 경우
        k_code_counts = {f"K{i:05d}": 10 for i in range(10)}  # 모두 min_images_per_class 미만
        
        with pytest.raises(ValueError, match="충분한 이미지가 있는 K-코드가 부족"):
            sampler.select_target_classes(k_code_counts)
    
    @patch.object(ProgressiveValidationSampler, 'validate_image_quality')
    def test_sample_images_for_class(self, mock_validate, sampler):
        """클래스별 이미지 샘플링 테스트"""
        mock_validate.return_value = True  # 모든 이미지가 유효하다고 가정
        
        k_code = "K00001"
        single_images = [Path(f"single_{i}.jpg") for i in range(100)]
        combo_images = [Path(f"combo_{i}.jpg") for i in range(50)]
        
        sampled_single, sampled_combo = sampler.sample_images_for_class(
            k_code, single_images, combo_images
        )
        
        # 목표 개수대로 샘플링되었는지 확인
        total_sampled = len(sampled_single) + len(sampled_combo)
        assert total_sampled == sampler.strategy.images_per_class
        
        # 비율이 대략적으로 맞는지 확인
        expected_single = int(sampler.strategy.images_per_class * sampler.strategy.single_combo_ratio)
        assert abs(len(sampled_single) - expected_single) <= 10  # 10개 오차 허용


class TestSamplingValidation:
    """샘플링 검증 기능 테스트"""
    
    def test_validate_sample_distribution_valid(self):
        """유효한 샘플 분포 검증 테스트"""
        valid_sample = {
            'stats': {
                'sampled_classes': 50,
                'sampled_images': 5000,
                'quality_pass_rate': 0.98
            },
            'samples': {
                f'K{i:05d}': {
                    'total_images': 100,
                    'single_count': 70,
                    'combo_count': 30
                } for i in range(50)
            }
        }
        
        assert validate_sample_distribution(valid_sample) == True
    
    def test_validate_sample_distribution_invalid_counts(self):
        """잘못된 개수로 검증 실패 테스트"""
        invalid_sample = {
            'stats': {
                'sampled_classes': 45,  # 50개가 아님
                'sampled_images': 5000,
                'quality_pass_rate': 0.98
            },
            'samples': {}
        }
        
        assert validate_sample_distribution(invalid_sample) == False
    
    def test_validate_sample_distribution_poor_quality(self):
        """품질 부족으로 검증 실패 테스트"""
        invalid_sample = {
            'stats': {
                'sampled_classes': 50,
                'sampled_images': 5000,
                'quality_pass_rate': 0.8  # 95% 미만
            },
            'samples': {
                f'K{i:05d}': {
                    'total_images': 100,
                    'single_count': 70,
                    'combo_count': 30
                } for i in range(50)
            }
        }
        
        assert validate_sample_distribution(invalid_sample) == False
    
    def test_validate_sample_distribution_uneven_distribution(self):
        """불균등 분포로 검증 실패 테스트"""
        # 클래스별 이미지 수가 크게 다른 경우
        samples = {}
        for i in range(50):
            # 첫 10개는 200개, 나머지는 50개로 편차 크게 만들기
            img_count = 200 if i < 10 else 50
            samples[f'K{i:05d}'] = {
                'total_images': img_count,
                'single_count': int(img_count * 0.7),
                'combo_count': int(img_count * 0.3)
            }
        
        invalid_sample = {
            'stats': {
                'sampled_classes': 50,
                'sampled_images': sum(s['total_images'] for s in samples.values()),
                'quality_pass_rate': 0.98
            },
            'samples': samples
        }
        
        assert validate_sample_distribution(invalid_sample) == False


class TestSamplingStats:
    """SamplingStats 테스트"""
    
    def test_sampling_stats_creation(self):
        """SamplingStats 생성 테스트"""
        stats = SamplingStats(
            total_images=526000,
            total_classes=5000,
            sampled_images=5000,
            sampled_classes=50,
            images_per_class={'K00001': 100, 'K00002': 100},
            single_pill_ratio=0.7,
            combo_pill_ratio=0.3,
            quality_pass_rate=0.98
        )
        
        assert stats.total_images == 526000
        assert stats.sampled_images == 5000
        assert stats.quality_pass_rate == 0.98
        assert abs(stats.single_pill_ratio + stats.combo_pill_ratio - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])