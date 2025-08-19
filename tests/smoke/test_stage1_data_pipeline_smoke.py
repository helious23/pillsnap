"""
Stage 1 데이터 파이프라인 스모크 테스트

핵심 기능들이 기본적으로 작동하는지 빠르게 검증:
- 샘플링 시스템 기본 동작
- 레지스트리 구축 기본 동작  
- 전체 워크플로우 End-to-End
- 실제 데이터 소량(10개 K-코드, 50개 이미지)으로 빠른 테스트
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import load_config, PillSnapLogger
from src.data.sampling import (
    Stage1SamplingStrategy,
    ProgressiveValidationSampler,
    validate_sample_distribution
)
from src.data.pharmaceutical_code_registry import (
    PharmaceuticalCodeRegistry,
    drug_metadata_validator
)


class TestStage1DataPipelineSmoke:
    """Stage 1 데이터 파이프라인 스모크 테스트"""
    
    @pytest.fixture(scope="class")
    def config(self):
        """설정 로드"""
        return load_config()
    
    @pytest.fixture(scope="class")
    def logger(self):
        """로거 인스턴스"""
        return PillSnapLogger(__name__)
    
    def test_smoke_sampling_basic_functionality(self, config, logger):
        """샘플링 시스템 기본 기능 스모크 테스트"""
        logger.info("🚬 샘플링 시스템 스모크 테스트 시작...")
        
        # 매우 작은 크기로 빠른 테스트
        smoke_strategy = Stage1SamplingStrategy(
            target_images=50,      # 50개 이미지만
            target_classes=5,      # 5개 클래스만
            images_per_class=10,   # 클래스당 10개
            min_images_per_class=8,
            seed=42
        )
        
        data_root = config['data']['root']
        sampler = ProgressiveValidationSampler(data_root, smoke_strategy)
        
        # 데이터 스캔 테스트
        logger.info("  📊 데이터 스캔 테스트...")
        scan_result = sampler.scan_available_data()
        
        assert 'single' in scan_result
        assert 'combo' in scan_result
        assert 'k_code_counts' in scan_result
        assert len(scan_result['k_code_counts']) > 0
        
        logger.info(f"    스캔된 K-코드: {len(scan_result['k_code_counts'])}개")
        
        # 클래스 선택 테스트
        logger.info("  🎯 클래스 선택 테스트...")
        selected_classes = sampler.select_target_classes(scan_result['k_code_counts'])
        
        assert len(selected_classes) == 5
        assert all(k_code.startswith('K-') for k_code in selected_classes)
        
        logger.info(f"    선택된 클래스: {selected_classes[:3]}...")
        
        # 샘플 생성 테스트 (전체가 아닌 간단한 검증만)
        logger.info("  🎲 샘플 생성 기본 테스트...")
        
        # 첫 번째 클래스만 테스트
        test_k_code = selected_classes[0]
        single_images = scan_result['single'].get(test_k_code, [])
        combo_images = scan_result['combo'].get(test_k_code, [])
        
        if single_images or combo_images:
            sampled_single, sampled_combo = sampler.sample_images_for_class(
                test_k_code, single_images[:20], combo_images[:20]  # 최대 20개씩만
            )
            
            total_sampled = len(sampled_single) + len(sampled_combo)
            assert total_sampled <= 10  # 클래스당 최대 10개
            assert total_sampled > 0    # 최소 1개는 샘플링
            
            logger.info(f"    {test_k_code}: {total_sampled}개 샘플링 성공")
        
        logger.info("  ✅ 샘플링 시스템 스모크 테스트 통과")
    
    def test_smoke_registry_basic_functionality(self, config, logger):
        """레지스트리 시스템 기본 기능 스모크 테스트"""
        logger.info("🚬 레지스트리 시스템 스모크 테스트 시작...")
        
        data_root = config['data']['root']
        registry = PharmaceuticalCodeRegistry(data_root)
        
        # 메타데이터 소스 스캔 테스트
        logger.info("  🔍 메타데이터 소스 스캔 테스트...")
        sources = registry.scan_drug_metadata_sources()
        
        assert isinstance(sources, dict)
        assert len(sources) > 0
        
        if 'directory_k_codes' in sources:
            k_codes = sources['directory_k_codes']
            assert len(k_codes) > 0
            assert all(k_code.startswith('K-') for k_code in k_codes[:10])
        
        logger.info(f"    발견된 소스: {list(sources.keys())}")
        
        # 기본 레코드 생성 테스트
        logger.info("  💊 기본 레코드 생성 테스트...")
        
        # 간단한 모의 Stage 1 샘플 생성
        mock_sample_data = {
            'metadata': {
                'stage': 1,
                'selected_classes': ['K-000001', 'K-000002', 'K-000003']
            },
            'samples': {
                'K-000001': {'total_images': 10, 'single_count': 7, 'combo_count': 3},
                'K-000002': {'total_images': 10, 'single_count': 8, 'combo_count': 2},
                'K-000003': {'total_images': 10, 'single_count': 6, 'combo_count': 4}
            }
        }
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_sample_data, f)
            temp_file = Path(f.name)
        
        try:
            # 레지스트리 구축 테스트
            success = registry.build_drug_registry_from_stage1_sample(temp_file)
            
            assert success == True
            assert len(registry.drug_records) == 3
            assert registry.stage1_mapping is not None
            
            # 기본 조회 기능 테스트
            record = registry.lookup_drug_by_k_code('K-000001')
            assert record is not None
            assert record.k_code == 'K-000001'
            assert record.stage1_class_id == 0
            
            # 분류 매핑 테스트
            mapping = registry.get_classification_mapping_for_stage1()
            assert mapping.total_classes == 3
            assert len(mapping.k_code_to_class_id) == 3
            
            logger.info(f"    레코드 생성: {len(registry.drug_records)}개")
            logger.info(f"    매핑 클래스: {mapping.total_classes}개")
            
        finally:
            temp_file.unlink()
        
        logger.info("  ✅ 레지스트리 시스템 스모크 테스트 통과")
    
    def test_smoke_end_to_end_minimal_workflow(self, config, logger):
        """End-to-End 최소 워크플로우 스모크 테스트"""
        logger.info("🚬 End-to-End 워크플로우 스모크 테스트 시작...")
        
        data_root = config['data']['root']
        
        # 1. 매우 작은 샘플링
        logger.info("  1️⃣ 미니 샘플링...")
        mini_strategy = Stage1SamplingStrategy(
            target_images=20,     # 20개만
            target_classes=2,     # 2개 클래스만
            images_per_class=10,
            seed=42
        )
        
        sampler = ProgressiveValidationSampler(data_root, mini_strategy)
        
        # 데이터 스캔 후 상위 2개 클래스만 선택
        scan_result = sampler.scan_available_data()
        k_code_counts = scan_result['k_code_counts']
        
        # 이미지가 많은 상위 2개 클래스 선택
        top_k_codes = sorted(k_code_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        selected_k_codes = [k_code for k_code, _ in top_k_codes]
        
        logger.info(f"    선택된 K-코드: {selected_k_codes}")
        
        # 간단한 샘플 데이터 생성
        mini_sample_data = {
            'metadata': {
                'stage': 1,
                'selected_classes': selected_k_codes
            },
            'samples': {
                k_code: {
                    'total_images': 10,
                    'single_count': 7,
                    'combo_count': 3
                } for k_code in selected_k_codes
            }
        }
        
        # 2. 레지스트리 구축
        logger.info("  2️⃣ 레지스트리 구축...")
        registry = PharmaceuticalCodeRegistry(data_root)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mini_sample_data, f)
            temp_sample_file = Path(f.name)
        
        try:
            success = registry.build_drug_registry_from_stage1_sample(temp_sample_file)
            assert success == True
            
            # 3. 기본 검증
            logger.info("  3️⃣ 기본 검증...")
            
            # 레코드 수 확인
            assert len(registry.drug_records) == 2
            
            # 각 레코드 확인
            for k_code in selected_k_codes:
                record = registry.lookup_drug_by_k_code(k_code)
                assert record is not None
                assert record.k_code == k_code
                assert record.edi_code is not None
                assert record.stage1_class_id is not None
            
            # 분류 매핑 확인
            mapping = registry.get_classification_mapping_for_stage1()
            assert mapping.total_classes == 2
            
            # 예측 기능 확인
            test_class_id = 0
            predicted_edi = registry.predict_edi_code_from_class_id(test_class_id)
            assert predicted_edi is not None
            
            # 4. 저장/로드 테스트
            logger.info("  4️⃣ 저장/로드 테스트...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 임시 레지스트리 디렉토리 설정
                registry.registry_dir = Path(temp_dir)
                
                # 저장
                saved_path = registry.save_registry_to_artifacts("smoke_test_registry.json")
                assert saved_path.exists()
                
                # 로드
                from src.data.pharmaceutical_code_registry import load_pharmaceutical_registry_from_artifacts
                loaded_registry = load_pharmaceutical_registry_from_artifacts(saved_path)
                
                # 로드 검증
                assert len(loaded_registry.drug_records) == len(registry.drug_records)
                assert loaded_registry.stage1_mapping.total_classes == mapping.total_classes
                
                # 동일한 K-코드로 조회 테스트
                for k_code in selected_k_codes:
                    original_record = registry.lookup_drug_by_k_code(k_code)
                    loaded_record = loaded_registry.lookup_drug_by_k_code(k_code)
                    
                    assert loaded_record is not None
                    assert loaded_record.k_code == original_record.k_code
                    assert loaded_record.edi_code == original_record.edi_code
            
            logger.info("  ✅ End-to-End 워크플로우 스모크 테스트 통과")
            
        finally:
            temp_sample_file.unlink()
    
    def test_smoke_performance_basic_check(self, config, logger):
        """기본 성능 체크 스모크 테스트"""
        logger.info("🚬 기본 성능 체크 스모크 테스트 시작...")
        
        import time
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"  📊 초기 메모리: {initial_memory:.1f} MB")
        
        # 데이터 스캔 성능 테스트
        start_time = time.time()
        
        data_root = config['data']['root']
        sampler = ProgressiveValidationSampler(data_root, Stage1SamplingStrategy(
            target_images=100, target_classes=5, seed=42
        ))
        
        scan_result = sampler.scan_available_data()
        scan_time = time.time() - start_time
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        logger.info(f"  ⏱️  스캔 시간: {scan_time:.2f}초")
        logger.info(f"  💾 메모리 증가: {memory_increase:.1f} MB")
        logger.info(f"  📈 스캔된 K-코드: {len(scan_result['k_code_counts'])}개")
        
        # 기본 성능 기준 (246만개 이미지 스캔을 고려한 현실적 기준)
        assert scan_time < 60.0, f"스캔 시간이 너무 오래 걸림: {scan_time:.2f}초"
        assert memory_increase < 1000.0, f"메모리 사용량이 너무 많음: {memory_increase:.1f} MB"  # 246만개 경로 저장 고려
        assert len(scan_result['k_code_counts']) > 100, "스캔된 K-코드가 너무 적음"
        
        logger.info("  ✅ 기본 성능 체크 통과")
    
    def test_smoke_error_handling(self, config, logger):
        """기본 에러 처리 스모크 테스트"""
        logger.info("🚬 에러 처리 스모크 테스트 시작...")
        
        # 1. 잘못된 데이터 루트 (예외가 발생하지 않고 빈 결과를 반환함)
        logger.info("  🚫 잘못된 데이터 루트 테스트...")
        sampler = ProgressiveValidationSampler("/nonexistent/path", Stage1SamplingStrategy())
        scan_result = sampler.scan_available_data()
        
        # 빈 결과가 반환되어야 함
        assert len(scan_result['k_code_counts']) == 0
        
        # 2. 잘못된 샘플 파일
        logger.info("  🚫 잘못된 샘플 파일 테스트...")
        registry = PharmaceuticalCodeRegistry(config['data']['root'])
        
        result = registry.build_drug_registry_from_stage1_sample(Path("/nonexistent/file.json"))
        assert result == False  # 실패해야 함
        
        # 3. 빈 레지스트리 검증
        logger.info("  🚫 빈 레지스트리 검증 테스트...")
        empty_registry = PharmaceuticalCodeRegistry("/tmp")
        validation_results = drug_metadata_validator(empty_registry)
        
        # 모든 검증이 실패해야 함
        assert not all(validation_results.values())
        
        logger.info("  ✅ 에러 처리 스모크 테스트 통과")


if __name__ == "__main__":
    # 스모크 테스트만 실행
    pytest.main([__file__, "-v", "-s"])