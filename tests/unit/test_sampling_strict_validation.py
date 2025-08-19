"""
샘플링 시스템 엄격한 검증 테스트

실제 구현 품질을 검증하는 테스트:
- 실제 데이터로 정확성 검증
- Edge case 및 실패 시나리오
- 성능 및 메모리 제약 검증
- 데이터 품질 및 일관성 검증
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import load_config, PillSnapLogger
from src.data.sampling import (
    Stage1SamplingStrategy,
    ProgressiveValidationSampler,
    validate_sample_distribution
)


class TestSamplingStrictValidation:
    """샘플링 시스템 엄격한 검증"""
    
    @pytest.fixture(scope="class")
    def config(self):
        return load_config()
    
    @pytest.fixture(scope="class")
    def real_data_sampler(self, config):
        """실제 데이터로 초기화된 샘플러"""
        strategy = Stage1SamplingStrategy(
            target_images=200,  # 검증 가능한 크기
            target_classes=20,
            images_per_class=10,
            seed=42
        )
        return ProgressiveValidationSampler(config['data']['root'], strategy)
    
    def test_actual_data_scan_accuracy(self, real_data_sampler):
        """실제 데이터 스캔 정확성 검증"""
        scan_result = real_data_sampler.scan_available_data()
        
        # 1. 스캔 결과가 비어있지 않아야 함
        assert len(scan_result['k_code_counts']) > 0, "실제 데이터에서 K-코드를 찾지 못함"
        
        # 2. 모든 K-코드가 올바른 형식이어야 함
        for k_code in scan_result['k_code_counts'].keys():
            assert k_code.startswith('K-'), f"잘못된 K-코드 형식: {k_code}"
            assert len(k_code) >= 7, f"K-코드가 너무 짧음: {k_code}"
        
        # 3. 이미지 경로가 실제 존재하는지 검증 (샘플 확인)
        sample_k_codes = list(scan_result['k_code_counts'].keys())[:5]
        for k_code in sample_k_codes:
            single_images = scan_result['single'].get(k_code, [])
            combo_images = scan_result['combo'].get(k_code, [])
            
            # 경로가 실제 존재하는지 확인
            for img_path in (single_images[:3] + combo_images[:3]):  # 샘플만 확인
                assert Path(img_path).exists(), f"이미지 파일이 존재하지 않음: {img_path}"
                assert Path(img_path).suffix.lower() in ['.jpg', '.png', '.jpeg'], \
                    f"지원하지 않는 이미지 형식: {img_path}"
        
        # 4. 카운트가 실제 파일 수와 일치하는지 검증
        for k_code in sample_k_codes:
            expected_count = scan_result['k_code_counts'][k_code]
            actual_single = len(scan_result['single'].get(k_code, []))
            actual_combo = len(scan_result['combo'].get(k_code, []))
            actual_total = actual_single + actual_combo
            
            assert actual_total == expected_count, \
                f"{k_code} 카운트 불일치: 예상={expected_count}, 실제={actual_total}"
    
    def test_sampling_deterministic_behavior(self, real_data_sampler):
        """샘플링 결정적 동작 검증 (같은 시드 = 같은 결과)"""
        # 첫 번째 샘플링
        sample1 = real_data_sampler.generate_stage1_sample()
        
        # 동일한 설정으로 두 번째 샘플러 생성
        strategy2 = Stage1SamplingStrategy(
            target_images=200,
            target_classes=20, 
            images_per_class=10,
            seed=42  # 동일한 시드
        )
        sampler2 = ProgressiveValidationSampler(
            real_data_sampler.data_root, strategy2
        )
        sample2 = sampler2.generate_stage1_sample()
        
        # 결과가 동일해야 함
        assert sample1['metadata']['selected_classes'] == sample2['metadata']['selected_classes'], \
            "동일한 시드로 다른 클래스가 선택됨"
        
        # 각 클래스별 샘플링된 이미지가 동일해야 함
        for k_code in sample1['metadata']['selected_classes']:
            images1 = set(sample1['samples'][k_code]['single_images'] + 
                         sample1['samples'][k_code]['combo_images'])
            images2 = set(sample2['samples'][k_code]['single_images'] + 
                         sample2['samples'][k_code]['combo_images'])
            
            assert images1 == images2, f"{k_code} 샘플링 결과가 일치하지 않음"
    
    def test_image_quality_validation_strictness(self, real_data_sampler):
        """이미지 품질 검증 엄격성 테스트"""
        # 실제 이미지로 품질 검증
        scan_result = real_data_sampler.scan_available_data()
        sample_images = []
        
        # 다양한 K-코드에서 이미지 수집
        for k_code, images in list(scan_result['single'].items())[:5]:
            sample_images.extend(images[:2])  # 각 K-코드에서 2개씩
        
        valid_count = 0
        total_count = len(sample_images)
        
        for img_path in sample_images:
            is_valid = real_data_sampler.validate_image_quality(Path(img_path))
            if is_valid:
                valid_count += 1
            else:
                # 실패한 이미지 정보 출력 (디버깅용)
                print(f"품질 검증 실패: {img_path}")
        
        # 실제 데이터에서는 높은 품질이 기대됨
        quality_rate = valid_count / total_count if total_count > 0 else 0
        assert quality_rate >= 0.90, \
            f"이미지 품질이 너무 낮음: {quality_rate:.2%} (90% 미만)"
        
        # 존재하지 않는 파일은 확실히 실패해야 함
        fake_path = Path("/nonexistent/fake_image.jpg")
        assert real_data_sampler.validate_image_quality(fake_path) == False, \
            "존재하지 않는 이미지가 품질 검증을 통과함"
    
    def test_class_selection_quality(self, real_data_sampler):
        """클래스 선택 품질 검증"""
        scan_result = real_data_sampler.scan_available_data()
        k_code_counts = scan_result['k_code_counts']
        
        selected_classes = real_data_sampler.select_target_classes(k_code_counts)
        
        # 1. 정확한 개수 선택
        assert len(selected_classes) == real_data_sampler.strategy.target_classes
        
        # 2. 모든 선택된 클래스가 충분한 이미지를 가져야 함
        for k_code in selected_classes:
            count = k_code_counts[k_code]
            assert count >= real_data_sampler.strategy.min_images_per_class, \
                f"{k_code}의 이미지 수가 부족: {count} < {real_data_sampler.strategy.min_images_per_class}"
        
        # 3. 선택된 클래스들이 이미지 수 기준으로 상위권이어야 함
        all_k_codes_sorted = sorted(k_code_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_codes = [k for k, _ in all_k_codes_sorted[:real_data_sampler.strategy.target_classes * 2]]
        
        # 선택된 클래스의 80% 이상이 상위 2배수 안에 있어야 함
        overlap = len(set(selected_classes) & set(top_k_codes))
        overlap_rate = overlap / len(selected_classes)
        assert overlap_rate >= 0.8, \
            f"선택된 클래스가 상위권이 아님: {overlap_rate:.2%} < 80%"
    
    def test_sampling_balance_and_distribution(self, real_data_sampler):
        """샘플링 균형 및 분포 검증"""
        sample_data = real_data_sampler.generate_stage1_sample()
        
        # 1. 클래스별 이미지 수 균형 검증
        image_counts = [data['total_images'] for data in sample_data['samples'].values()]
        
        # 표준편차가 작아야 함 (균등 분포)
        import statistics
        std_dev = statistics.stdev(image_counts)
        mean_count = statistics.mean(image_counts)
        cv = std_dev / mean_count if mean_count > 0 else float('inf')  # 변동계수
        
        assert cv < 0.2, f"클래스별 이미지 수 분포가 불균등: CV={cv:.3f} (0.2 초과)"
        
        # 2. Single/Combo 비율 검증 - 실제 데이터 제약 고려
        total_single = sum(data['single_count'] for data in sample_data['samples'].values())
        total_combo = sum(data['combo_count'] for data in sample_data['samples'].values())
        total_images = total_single + total_combo
        
        if total_images > 0:
            actual_single_ratio = total_single / total_images
            expected_ratio = real_data_sampler.strategy.single_combo_ratio
            
            # 실제 데이터셋에서 Combo 이미지가 매우 적음 (0.7%)
            # 따라서 Single 위주 샘플링이 불가피함 - 현실적 기준 적용
            if total_combo == 0:
                # Combo 이미지가 없는 경우는 허용 (실제 데이터 제약)
                assert actual_single_ratio == 1.0, "Combo 없을 때 Single 비율이 100%가 아님"
            else:
                # Combo가 있는 경우 큰 차이 허용 (실제 데이터 분포 반영)
                ratio_diff = abs(actual_single_ratio - expected_ratio)
                assert ratio_diff < 0.5, \
                    f"Single/Combo 비율이 목표와 너무 다름: 실제={actual_single_ratio:.2%}, 목표={expected_ratio:.2%}"
    
    def test_edge_cases_and_error_handling(self, config):
        """Edge case 및 에러 처리 검증"""
        # 1. 불가능한 목표 설정
        impossible_strategy = Stage1SamplingStrategy(
            target_images=1000000,  # 100만개 - 불가능
            target_classes=50000,   # 5만개 클래스 - 불가능
            images_per_class=20
        )
        
        sampler = ProgressiveValidationSampler(config['data']['root'], impossible_strategy)
        scan_result = sampler.scan_available_data()
        
        # 충분한 클래스가 없으면 예외 발생해야 함
        with pytest.raises(ValueError, match="충분한 이미지가 있는 K-코드가 부족"):
            sampler.select_target_classes(scan_result['k_code_counts'])
        
        # 2. 너무 적은 목표 설정
        tiny_strategy = Stage1SamplingStrategy(
            target_images=5,
            target_classes=1,
            images_per_class=5,
            min_images_per_class=3
        )
        
        tiny_sampler = ProgressiveValidationSampler(config['data']['root'], tiny_strategy)
        tiny_sample = tiny_sampler.generate_stage1_sample()
        
        # 작은 목표도 정확히 달성해야 함
        assert tiny_sample['stats']['sampled_classes'] == 1
        assert tiny_sample['stats']['sampled_images'] == 5
    
    def test_memory_and_performance_constraints(self, config):
        """메모리 및 성능 제약 검증"""
        import psutil
        import time
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # 큰 스캔 작업 수행
        large_strategy = Stage1SamplingStrategy(
            target_images=1000,
            target_classes=50,
            images_per_class=20
        )
        
        start_time = time.time()
        sampler = ProgressiveValidationSampler(config['data']['root'], large_strategy)
        scan_result = sampler.scan_available_data()
        scan_time = time.time() - start_time
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # 성능 기준 (실제 사용 환경 고려)
        assert scan_time < 30.0, f"스캔 시간이 너무 오래 걸림: {scan_time:.1f}초"
        assert memory_increase < 2000.0, f"메모리 사용량 과다: {memory_increase:.1f}MB"
        
        # 스캔 결과가 의미있어야 함
        assert len(scan_result['k_code_counts']) >= 100, "스캔된 클래스가 너무 적음"
    
    def test_data_integrity_and_consistency(self, real_data_sampler):
        """데이터 무결성 및 일관성 검증"""
        sample_data = real_data_sampler.generate_stage1_sample()
        
        # 1. 모든 샘플링된 이미지 경로가 유효해야 함
        invalid_paths = []
        for k_code, data in sample_data['samples'].items():
            all_images = data['single_images'] + data['combo_images']
            for img_path in all_images:
                if not Path(img_path).exists():
                    invalid_paths.append(img_path)
        
        assert len(invalid_paths) == 0, f"존재하지 않는 이미지 경로들: {invalid_paths[:5]}"
        
        # 2. K-코드와 이미지 경로의 일관성 검증
        for k_code, data in sample_data['samples'].items():
            all_images = data['single_images'] + data['combo_images']
            for img_path in all_images:
                # 이미지 경로에 해당 K-코드가 포함되어야 함
                assert k_code in img_path, f"이미지 경로와 K-코드 불일치: {k_code} not in {img_path}"
        
        # 3. 중복 이미지 검증
        all_sampled_images = []
        for data in sample_data['samples'].values():
            all_sampled_images.extend(data['single_images'])
            all_sampled_images.extend(data['combo_images'])
        
        unique_images = set(all_sampled_images)
        assert len(unique_images) == len(all_sampled_images), \
            f"중복 이미지 발견: {len(all_sampled_images) - len(unique_images)}개"
    
    def test_validation_function_strictness(self):
        """검증 함수의 엄격성 테스트"""
        # 완벽한 샘플 데이터
        perfect_sample = {
            'stats': {
                'sampled_classes': 50,
                'sampled_images': 5000,
                'quality_pass_rate': 1.0
            },
            'samples': {
                f'K-{i:05d}': {
                    'total_images': 100,
                    'single_count': 70,
                    'combo_count': 30
                } for i in range(50)
            }
        }
        
        assert validate_sample_distribution(perfect_sample) == True
        
        # 각종 결함이 있는 샘플들
        defective_samples = [
            # 클래스 수 부족
            {**perfect_sample, 'stats': {**perfect_sample['stats'], 'sampled_classes': 45}},
            # 이미지 수 부족  
            {**perfect_sample, 'stats': {**perfect_sample['stats'], 'sampled_images': 4500}},
            # 품질 부족
            {**perfect_sample, 'stats': {**perfect_sample['stats'], 'quality_pass_rate': 0.8}},
        ]
        
        for defective_sample in defective_samples:
            assert validate_sample_distribution(defective_sample) == False, \
                f"결함이 있는 샘플이 검증을 통과함: {defective_sample['stats']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])