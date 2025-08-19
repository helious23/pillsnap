"""
Stage 1 샘플링-레지스트리 통합 테스트

샘플링 시스템과 레지스트리 시스템 간 데이터 일관성 및 상호 작용 검증:
- 샘플링 결과와 레지스트리 구축 결과의 데이터 일관성
- 클래스 ID와 K-코드 매핑의 정확성
- 저장/로드 후 데이터 무결성
- 실제 Stage 1 샘플 데이터와의 통합
"""

import pytest
import json
import tempfile
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
    drug_metadata_validator,
    load_pharmaceutical_registry_from_artifacts
)


class TestStage1SamplingRegistryIntegration:
    """Stage 1 샘플링-레지스트리 통합 테스트"""
    
    @pytest.fixture(scope="class")
    def config(self):
        """설정 로드"""
        return load_config()
    
    @pytest.fixture(scope="class")
    def logger(self):
        """로거 인스턴스"""
        return PillSnapLogger(__name__)
    
    @pytest.fixture(scope="class")
    def integration_strategy(self):
        """통합 테스트용 샘플링 전략"""
        return Stage1SamplingStrategy(
            target_images=100,     # 통합 테스트용 중간 크기
            target_classes=10,     # 10개 클래스
            images_per_class=10,
            min_images_per_class=8,
            seed=42
        )
    
    def test_sampling_to_registry_data_consistency(self, config, integration_strategy, logger):
        """샘플링 결과와 레지스트리 데이터 일관성 테스트"""
        logger.info("🔗 샘플링-레지스트리 데이터 일관성 테스트...")
        
        data_root = config['data']['root']
        
        # 1. 샘플링 실행
        sampler = ProgressiveValidationSampler(data_root, integration_strategy)
        sample_data = sampler.generate_stage1_sample()
        
        # 샘플링 기본 검증 (validate_sample_distribution은 50개 클래스 하드코딩되어 있음)
        assert sample_data['stats']['sampled_classes'] == integration_strategy.target_classes
        assert sample_data['stats']['sampled_images'] == integration_strategy.target_images
        
        selected_k_codes = sample_data['metadata']['selected_classes']
        sample_stats = sample_data['stats']
        
        logger.info(f"  샘플링 결과: {len(selected_k_codes)}개 클래스, {sample_stats['sampled_images']}개 이미지")
        
        # 2. 임시 파일에 샘플 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_sample_path = Path(f.name)
        
        try:
            # 3. 레지스트리 구축
            registry = PharmaceuticalCodeRegistry(data_root)
            success = registry.build_drug_registry_from_stage1_sample(temp_sample_path)
            
            assert success == True
            
            # 4. 데이터 일관성 검증
            
            # 4.1 K-코드 일치 검증
            registry_k_codes = set(registry.drug_records.keys())
            sample_k_codes = set(selected_k_codes)
            
            assert registry_k_codes == sample_k_codes, \
                f"K-코드 불일치: 샘플링={len(sample_k_codes)}, 레지스트리={len(registry_k_codes)}"
            
            # 4.2 클래스 ID 연속성 검증
            class_ids = [record.stage1_class_id for record in registry.drug_records.values()]
            expected_class_ids = list(range(len(selected_k_codes)))
            
            assert sorted(class_ids) == expected_class_ids, \
                f"클래스 ID 불일치: 예상={expected_class_ids}, 실제={sorted(class_ids)}"
            
            # 4.3 이미지 수 일치 검증
            total_sample_images = sum(data['total_images'] for data in sample_data['samples'].values())
            total_registry_images = sum(record.image_count or 0 for record in registry.drug_records.values())
            
            assert total_sample_images == total_registry_images, \
                f"이미지 수 불일치: 샘플={total_sample_images}, 레지스트리={total_registry_images}"
            
            # 4.4 개별 K-코드별 이미지 수 검증
            for k_code in selected_k_codes:
                sample_image_count = sample_data['samples'][k_code]['total_images']
                registry_record = registry.lookup_drug_by_k_code(k_code)
                
                assert registry_record is not None, f"레지스트리에서 {k_code} 찾을 수 없음"
                assert registry_record.image_count == sample_image_count, \
                    f"{k_code} 이미지 수 불일치: 샘플={sample_image_count}, 레지스트리={registry_record.image_count}"
            
            logger.info("  ✅ 샘플링-레지스트리 데이터 일관성 검증 통과")
            
        finally:
            temp_sample_path.unlink()
    
    def test_class_id_mapping_accuracy(self, config, integration_strategy, logger):
        """클래스 ID 매핑 정확성 테스트"""
        logger.info("🎯 클래스 ID 매핑 정확성 테스트...")
        
        data_root = config['data']['root']
        
        # 샘플링 및 레지스트리 구축
        sampler = ProgressiveValidationSampler(data_root, integration_strategy)
        sample_data = sampler.generate_stage1_sample()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_path = Path(f.name)
        
        try:
            registry = PharmaceuticalCodeRegistry(data_root)
            registry.build_drug_registry_from_stage1_sample(temp_path)
            
            mapping = registry.get_classification_mapping_for_stage1()
            selected_k_codes = sample_data['metadata']['selected_classes']
            
            # 1. 양방향 매핑 일관성 검증
            for i, k_code in enumerate(selected_k_codes):
                # K-코드 → 클래스 ID
                class_id_from_mapping = mapping.k_code_to_class_id.get(k_code)
                assert class_id_from_mapping == i, \
                    f"{k_code}의 클래스 ID 불일치: 예상={i}, 매핑={class_id_from_mapping}"
                
                # 클래스 ID → EDI 코드
                edi_code_from_mapping = mapping.class_id_to_edi_code.get(i)
                assert edi_code_from_mapping is not None, f"클래스 ID {i}에 대한 EDI 코드 없음"
                
                # 레코드와 매핑 일치 검증
                record = registry.lookup_drug_by_k_code(k_code)
                assert record.stage1_class_id == i, \
                    f"{k_code} 레코드의 클래스 ID 불일치: 예상={i}, 레코드={record.stage1_class_id}"
                assert record.edi_code == edi_code_from_mapping, \
                    f"{k_code} EDI 코드 불일치: 매핑={edi_code_from_mapping}, 레코드={record.edi_code}"
            
            # 2. 예측 기능 정확성 검증
            for i in range(len(selected_k_codes)):
                predicted_edi = registry.predict_edi_code_from_class_id(i)
                expected_edi = mapping.class_id_to_edi_code[i]
                
                assert predicted_edi == expected_edi, \
                    f"클래스 ID {i} EDI 예측 실패: 예상={expected_edi}, 예측={predicted_edi}"
            
            # 3. 분류 타겟 변환 정확성 검증
            for k_code in selected_k_codes:
                record = registry.lookup_drug_by_k_code(k_code)
                edi_code, class_id = record.to_classification_target()
                
                expected_class_id = mapping.k_code_to_class_id[k_code]
                expected_edi = mapping.class_id_to_edi_code[expected_class_id]
                
                assert class_id == expected_class_id, \
                    f"{k_code} 분류 타겟 클래스 ID 불일치: 예상={expected_class_id}, 실제={class_id}"
                assert edi_code == expected_edi, \
                    f"{k_code} 분류 타겟 EDI 불일치: 예상={expected_edi}, 실제={edi_code}"
            
            logger.info("  ✅ 클래스 ID 매핑 정확성 검증 통과")
            
        finally:
            temp_path.unlink()
    
    def test_save_load_roundtrip_integrity(self, config, integration_strategy, logger):
        """저장/로드 라운드트립 무결성 테스트"""
        logger.info("💾 저장/로드 라운드트립 무결성 테스트...")
        
        data_root = config['data']['root']
        
        # 원본 레지스트리 구축
        sampler = ProgressiveValidationSampler(data_root, integration_strategy)
        sample_data = sampler.generate_stage1_sample()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_sample_path = Path(f.name)
        
        try:
            original_registry = PharmaceuticalCodeRegistry(data_root)
            original_registry.build_drug_registry_from_stage1_sample(temp_sample_path)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 저장 경로 설정
                original_registry.registry_dir = Path(temp_dir)
                
                # 저장
                saved_path = original_registry.save_registry_to_artifacts("integration_test_registry.json")
                assert saved_path.exists()
                
                # 로드
                loaded_registry = load_pharmaceutical_registry_from_artifacts(saved_path)
                
                # 무결성 검증
                
                # 1. 레코드 수 일치
                assert len(loaded_registry.drug_records) == len(original_registry.drug_records)
                
                # 2. 모든 K-코드와 레코드 일치
                for k_code, original_record in original_registry.drug_records.items():
                    loaded_record = loaded_registry.lookup_drug_by_k_code(k_code)
                    
                    assert loaded_record is not None, f"{k_code} 로드 후 찾을 수 없음"
                    
                    # 주요 필드 일치 검증
                    assert loaded_record.k_code == original_record.k_code
                    assert loaded_record.edi_code == original_record.edi_code
                    assert loaded_record.drug_name_kor == original_record.drug_name_kor
                    assert loaded_record.manufacturer == original_record.manufacturer
                    assert loaded_record.stage1_class_id == original_record.stage1_class_id
                    assert loaded_record.image_count == original_record.image_count
                    assert loaded_record.data_quality_score == original_record.data_quality_score
                
                # 3. 매핑 정보 일치
                original_mapping = original_registry.get_classification_mapping_for_stage1()
                loaded_mapping = loaded_registry.get_classification_mapping_for_stage1()
                
                assert loaded_mapping.total_classes == original_mapping.total_classes
                assert loaded_mapping.k_code_to_class_id == original_mapping.k_code_to_class_id
                assert loaded_mapping.class_id_to_edi_code == original_mapping.class_id_to_edi_code
                
                # 4. 기능 동작 일치
                for k_code in original_registry.drug_records.keys():
                    # 조회 기능
                    original_lookup = original_registry.lookup_drug_by_k_code(k_code)
                    loaded_lookup = loaded_registry.lookup_drug_by_k_code(k_code)
                    
                    assert loaded_lookup.edi_code == original_lookup.edi_code
                    
                    # 예측 기능
                    class_id = original_lookup.stage1_class_id
                    original_prediction = original_registry.predict_edi_code_from_class_id(class_id)
                    loaded_prediction = loaded_registry.predict_edi_code_from_class_id(class_id)
                    
                    assert loaded_prediction == original_prediction
                
                # 5. 검증 결과 일치
                original_validation = drug_metadata_validator(original_registry)
                loaded_validation = drug_metadata_validator(loaded_registry)
                
                assert loaded_validation == original_validation
                
                logger.info("  ✅ 저장/로드 라운드트립 무결성 검증 통과")
            
        finally:
            temp_sample_path.unlink()
    
    def test_real_stage1_sample_integration(self, config, logger):
        """실제 Stage 1 샘플과의 통합 테스트"""
        logger.info("📄 실제 Stage 1 샘플 통합 테스트...")
        
        # 실제 Stage 1 샘플 파일 확인
        stage1_sample_path = Path("artifacts/stage1/sampling/stage1_sample_test.json")
        
        if not stage1_sample_path.exists():
            logger.warning(f"  ⚠️  실제 Stage 1 샘플 파일이 없음: {stage1_sample_path}")
            pytest.skip("실제 Stage 1 샘플 파일이 없어 테스트 스킵")
            return
        
        # 실제 샘플 데이터 로드
        with open(stage1_sample_path, 'r', encoding='utf-8') as f:
            real_sample_data = json.load(f)
        
        logger.info(f"  실제 샘플: {len(real_sample_data['metadata']['selected_classes'])}개 클래스")
        
        # 레지스트리 구축
        data_root = config['data']['root']
        registry = PharmaceuticalCodeRegistry(data_root)
        
        success = registry.build_drug_registry_from_stage1_sample(stage1_sample_path)
        assert success == True
        
        # 실제 데이터와 레지스트리 일치 검증
        selected_k_codes = real_sample_data['metadata']['selected_classes']
        
        # 1. 모든 K-코드가 레지스트리에 존재하는지 확인
        for k_code in selected_k_codes:
            record = registry.lookup_drug_by_k_code(k_code)
            assert record is not None, f"실제 샘플의 {k_code}가 레지스트리에 없음"
            
            # 이미지 수 일치 확인
            expected_image_count = real_sample_data['samples'][k_code]['total_images']
            assert record.image_count == expected_image_count, \
                f"{k_code} 이미지 수 불일치: 샘플={expected_image_count}, 레지스트리={record.image_count}"
        
        # 2. 전체 통계 일치 확인
        real_stats = real_sample_data['stats']
        
        assert len(registry.drug_records) == real_stats['sampled_classes']
        
        total_registry_images = sum(record.image_count or 0 for record in registry.drug_records.values())
        assert total_registry_images == real_stats['sampled_images']
        
        # 3. 매핑 시스템 정상 작동 확인
        mapping = registry.get_classification_mapping_for_stage1()
        assert mapping.total_classes == len(selected_k_codes)
        
        # 4. 검증 시스템 정상 작동 확인
        validation_results = drug_metadata_validator(registry)
        assert all(validation_results.values()), f"실제 샘플 검증 실패: {validation_results}"
        
        logger.info("  ✅ 실제 Stage 1 샘플 통합 테스트 통과")
    
    def test_cross_system_data_flow(self, config, integration_strategy, logger):
        """시스템 간 데이터 흐름 테스트"""
        logger.info("🔄 시스템 간 데이터 흐름 테스트...")
        
        data_root = config['data']['root']
        
        # 시나리오: 샘플링 → 레지스트리 → 저장 → 로드 → 새 레지스트리 → 비교
        
        # 1. 첫 번째 샘플링
        sampler1 = ProgressiveValidationSampler(data_root, integration_strategy)
        sample_data1 = sampler1.generate_stage1_sample()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data1, f)
            temp_sample1 = Path(f.name)
        
        try:
            # 2. 첫 번째 레지스트리 구축
            registry1 = PharmaceuticalCodeRegistry(data_root)
            registry1.build_drug_registry_from_stage1_sample(temp_sample1)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                registry1.registry_dir = Path(temp_dir)
                
                # 3. 저장
                saved_path = registry1.save_registry_to_artifacts("flow_test_registry.json")
                
                # 4. 새로운 환경에서 로드
                registry2 = load_pharmaceutical_registry_from_artifacts(saved_path)
                
                # 5. 두 번째 샘플링 (동일한 시드)
                sampler2 = ProgressiveValidationSampler(data_root, integration_strategy)
                sample_data2 = sampler2.generate_stage1_sample()
                
                # 6. 데이터 흐름 일관성 검증
                
                # 샘플링 결과 일치 (동일한 시드)
                assert sample_data1['metadata']['selected_classes'] == sample_data2['metadata']['selected_classes']
                
                # 레지스트리 결과 일치
                for k_code in sample_data1['metadata']['selected_classes']:
                    record1 = registry1.lookup_drug_by_k_code(k_code)
                    record2 = registry2.lookup_drug_by_k_code(k_code)
                    
                    assert record1 is not None and record2 is not None
                    assert record1.k_code == record2.k_code
                    assert record1.edi_code == record2.edi_code
                    assert record1.stage1_class_id == record2.stage1_class_id
                
                # 예측 결과 일치
                for class_id in range(integration_strategy.target_classes):
                    pred1 = registry1.predict_edi_code_from_class_id(class_id)
                    pred2 = registry2.predict_edi_code_from_class_id(class_id)
                    assert pred1 == pred2
                
                logger.info("  ✅ 시스템 간 데이터 흐름 검증 통과")
        
        finally:
            temp_sample1.unlink()


if __name__ == "__main__":
    # 통합 테스트만 실행
    pytest.main([__file__, "-v", "-s"])