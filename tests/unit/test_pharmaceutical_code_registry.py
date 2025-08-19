"""
의약품 코드 레지스트리 시스템 단위 테스트

테스트 범위:
- DrugIdentificationRecord 데이터 모델 검증
- PharmaceuticalCodeRegistry 핵심 기능
- Stage1ClassificationMapping 매핑 로직
- drug_metadata_validator 검증 기능
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd

from src.data.pharmaceutical_code_registry import (
    DrugIdentificationRecord,
    PharmaceuticalCodeRegistry,
    Stage1ClassificationMapping,
    drug_metadata_validator,
    load_pharmaceutical_registry_from_artifacts
)


class TestDrugIdentificationRecord:
    """DrugIdentificationRecord 테스트"""
    
    def test_valid_drug_record_creation(self):
        """유효한 의약품 레코드 생성 테스트"""
        record = DrugIdentificationRecord(
            k_code="K-029066",
            edi_code="EDI12345",
            drug_name_kor="아스피린정100mg",
            drug_name_eng="Aspirin Tab 100mg",
            manufacturer="제약회사A",
            dosage_form="정제",
            strength="100mg",
            shape="원형",
            color="흰색",
            marking="A100",
            approval_date="2020-01-01",
            stage1_class_id=0,
            image_count=100,
            data_quality_score=0.95
        )
        
        assert record.k_code == "K-029066"
        assert record.edi_code == "EDI12345"
        assert record.drug_name_kor == "아스피린정100mg"
        assert record.stage1_class_id == 0
        assert record.image_count == 100
        assert record.data_quality_score == 0.95
    
    def test_invalid_k_code_format(self):
        """잘못된 K-코드 형식 검증 테스트"""
        with pytest.raises(ValueError, match="잘못된 K-코드 형식"):
            DrugIdentificationRecord(
                k_code="INVALID",  # K-로 시작하지 않음
                edi_code="EDI12345",
                drug_name_kor="테스트약",
                drug_name_eng=None,
                manufacturer="테스트회사",
                dosage_form="정제",
                strength=None,
                shape=None,
                color=None,
                marking=None,
                approval_date=None
            )
    
    def test_missing_edi_code(self):
        """EDI 코드 누락 검증 테스트"""
        with pytest.raises(ValueError, match="EDI 코드가 필요합니다"):
            DrugIdentificationRecord(
                k_code="K-123456",
                edi_code="",  # 빈 EDI 코드
                drug_name_kor="테스트약",
                drug_name_eng=None,
                manufacturer="테스트회사",
                dosage_form="정제",
                strength=None,
                shape=None,
                color=None,
                marking=None,
                approval_date=None
            )
    
    def test_missing_korean_drug_name(self):
        """한국어 의약품명 누락 검증 테스트"""
        with pytest.raises(ValueError, match="한국어 의약품명이 필요합니다"):
            DrugIdentificationRecord(
                k_code="K-123456",
                edi_code="EDI12345",
                drug_name_kor="",  # 빈 한국어 이름
                drug_name_eng=None,
                manufacturer="테스트회사",
                dosage_form="정제",
                strength=None,
                shape=None,
                color=None,
                marking=None,
                approval_date=None
            )
    
    def test_to_classification_target(self):
        """분류 타겟 변환 테스트"""
        record = DrugIdentificationRecord(
            k_code="K-123456",
            edi_code="EDI12345",
            drug_name_kor="테스트약",
            drug_name_eng=None,
            manufacturer="테스트회사",
            dosage_form="정제",
            strength=None,
            shape=None,
            color=None,
            marking=None,
            approval_date=None,
            stage1_class_id=5
        )
        
        edi_code, class_id = record.to_classification_target()
        assert edi_code == "EDI12345"
        assert class_id == 5
    
    def test_to_classification_target_no_stage1_id(self):
        """Stage 1 클래스 ID 없는 경우 테스트"""
        record = DrugIdentificationRecord(
            k_code="K-123456",
            edi_code="EDI12345",
            drug_name_kor="테스트약",
            drug_name_eng=None,
            manufacturer="테스트회사",
            dosage_form="정제",
            strength=None,
            shape=None,
            color=None,
            marking=None,
            approval_date=None
        )
        
        edi_code, class_id = record.to_classification_target()
        assert edi_code == "EDI12345"
        assert class_id == -1


class TestStage1ClassificationMapping:
    """Stage1ClassificationMapping 테스트"""
    
    def test_default_mapping_creation(self):
        """기본 매핑 생성 테스트"""
        mapping = Stage1ClassificationMapping()
        
        assert mapping.total_classes == 50
        assert mapping.k_code_to_class_id == {}
        assert mapping.class_id_to_edi_code == {}
        assert mapping.class_weights == {}
    
    def test_custom_mapping_creation(self):
        """커스텀 매핑 생성 테스트"""
        k_code_mapping = {"K-123456": 0, "K-789012": 1}
        class_mapping = {0: "EDI12345", 1: "EDI67890"}
        weights = {0: 1.0, 1: 1.2}
        
        mapping = Stage1ClassificationMapping(
            total_classes=2,
            k_code_to_class_id=k_code_mapping,
            class_id_to_edi_code=class_mapping,
            class_weights=weights
        )
        
        assert mapping.total_classes == 2
        assert mapping.k_code_to_class_id == k_code_mapping
        assert mapping.class_id_to_edi_code == class_mapping
        assert mapping.class_weights == weights


class TestPharmaceuticalCodeRegistry:
    """PharmaceuticalCodeRegistry 테스트"""
    
    @pytest.fixture
    def mock_data_structure(self):
        """테스트용 데이터 구조 Mock"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 가짜 디렉토리 구조 생성
            single_dir = temp_path / "data/train/images/single/TS_1"
            combo_dir = temp_path / "data/train/images/combination/TS_1"
            doc_dir = temp_path / "document"
            
            # K-코드 디렉토리 생성
            for i in range(5):
                k_code = f"K-{i:06d}"
                (single_dir / k_code).mkdir(parents=True, exist_ok=True)
                (combo_dir / k_code).mkdir(parents=True, exist_ok=True)
            
            # 문서 디렉토리 생성
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # 가짜 Excel 파일 생성
            df = pd.DataFrame({
                'K코드': [f"{i:06d}" for i in range(5)],
                '의약품명': [f"테스트약품_{i}" for i in range(5)],
                '제조회사': [f"제약회사_{i}" for i in range(5)]
            })
            df.to_excel(doc_dir / "single_list.xlsx", index=False)
            
            yield temp_path
    
    @pytest.fixture
    def registry(self, mock_data_structure):
        """테스트용 레지스트리 인스턴스"""
        return PharmaceuticalCodeRegistry(str(mock_data_structure))
    
    @pytest.fixture
    def mock_stage1_sample(self):
        """테스트용 Stage 1 샘플 데이터"""
        return {
            'metadata': {
                'stage': 1,
                'selected_classes': [f"K-{i:06d}" for i in range(5)]
            },
            'samples': {
                f"K-{i:06d}": {
                    'total_images': 100,
                    'single_count': 70,
                    'combo_count': 30
                } for i in range(5)
            }
        }
    
    def test_registry_initialization(self, registry):
        """레지스트리 초기화 테스트"""
        assert registry.data_root.exists()
        assert registry.registry_dir.name == "registry"
        assert len(registry.drug_records) == 0
        assert registry.stage1_mapping is None
    
    def test_scan_drug_metadata_sources(self, registry):
        """메타데이터 소스 스캔 테스트"""
        sources = registry.scan_drug_metadata_sources()
        
        assert 'directory_k_codes' in sources
        assert isinstance(sources['directory_k_codes'], list)
        assert len(sources['directory_k_codes']) > 0
    
    def test_extract_k_codes_from_directory_structure(self, registry):
        """디렉토리에서 K-코드 추출 테스트"""
        k_codes = registry._extract_k_codes_from_directory_structure()
        
        assert len(k_codes) == 5
        assert all(k_code.startswith('K-') for k_code in k_codes)
        assert k_codes == sorted(k_codes)  # 정렬되어 있는지 확인
    
    @patch('pandas.read_excel')
    def test_parse_excel_metadata(self, mock_read_excel, registry):
        """Excel 메타데이터 파싱 테스트"""
        # Mock DataFrame 생성
        mock_df = pd.DataFrame({
            'K코드': ['123456', '789012'],
            '의약품명': ['테스트약품1', '테스트약품2'],
            '제조회사': ['제약회사1', '제약회사2']
        })
        mock_read_excel.return_value = mock_df
        
        records = registry.parse_excel_metadata(Path("fake_excel.xlsx"))
        
        assert len(records) == 2
        assert records[0].k_code == "K-123456"
        assert records[0].drug_name_kor == "테스트약품1"
        assert records[0].manufacturer == "제약회사1"
        assert records[1].k_code == "K-789012"
    
    def test_safe_get_column_value(self, registry):
        """안전한 컬럼값 추출 테스트"""
        row = pd.Series({'K코드': 'K-123456', '의약품명': '테스트약', '기타': None})
        
        # 존재하는 컬럼
        value = registry._safe_get_column_value(row, ['K코드', 'K_CODE'])
        assert value == 'K-123456'
        
        # 존재하지 않는 컬럼
        value = registry._safe_get_column_value(row, ['없는컬럼1', '없는컬럼2'])
        assert value is None
        
        # NaN 값
        value = registry._safe_get_column_value(row, ['기타'])
        assert value is None
    
    def test_build_drug_registry_from_stage1_sample(self, registry, mock_stage1_sample):
        """Stage 1 샘플에서 레지스트리 구축 테스트"""
        # 임시 파일에 샘플 데이터 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_stage1_sample, f)
            temp_file = Path(f.name)
        
        try:
            success = registry.build_drug_registry_from_stage1_sample(temp_file)
            
            assert success == True
            assert len(registry.drug_records) == 5
            assert registry.stage1_mapping is not None
            assert registry.stage1_mapping.total_classes == 5
            
            # 첫 번째 레코드 확인
            first_k_code = "K-000000"
            assert first_k_code in registry.drug_records
            
            record = registry.drug_records[first_k_code]
            assert record.stage1_class_id == 0
            assert record.image_count == 100
            assert record.data_quality_score == 1.0
            
        finally:
            temp_file.unlink()  # 임시 파일 삭제
    
    def test_get_classification_mapping_for_stage1(self, registry, mock_stage1_sample):
        """Stage 1 분류 매핑 반환 테스트"""
        # 먼저 레지스트리 구축
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_stage1_sample, f)
            temp_file = Path(f.name)
        
        try:
            registry.build_drug_registry_from_stage1_sample(temp_file)
            mapping = registry.get_classification_mapping_for_stage1()
            
            assert mapping.total_classes == 5
            assert len(mapping.k_code_to_class_id) == 5
            assert len(mapping.class_id_to_edi_code) == 5
            
        finally:
            temp_file.unlink()
    
    def test_get_classification_mapping_before_build(self, registry):
        """레지스트리 구축 전 매핑 요청 테스트"""
        with pytest.raises(ValueError, match="Stage 1 매핑이 구축되지 않았습니다"):
            registry.get_classification_mapping_for_stage1()
    
    def test_lookup_drug_by_k_code(self, registry):
        """K-코드로 의약품 조회 테스트"""
        # 테스트 레코드 추가
        test_record = DrugIdentificationRecord(
            k_code="K-123456",
            edi_code="EDI12345",
            drug_name_kor="테스트약",
            drug_name_eng=None,
            manufacturer="테스트회사",
            dosage_form="정제",
            strength=None,
            shape=None,
            color=None,
            marking=None,
            approval_date=None
        )
        registry.drug_records["K-123456"] = test_record
        
        # 존재하는 K-코드 조회
        found_record = registry.lookup_drug_by_k_code("K-123456")
        assert found_record is not None
        assert found_record.drug_name_kor == "테스트약"
        
        # 존재하지 않는 K-코드 조회
        not_found = registry.lookup_drug_by_k_code("K-999999")
        assert not_found is None
    
    def test_predict_edi_code_from_class_id(self, registry, mock_stage1_sample):
        """클래스 ID에서 EDI 코드 예측 테스트"""
        # 레지스트리 구축
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_stage1_sample, f)
            temp_file = Path(f.name)
        
        try:
            registry.build_drug_registry_from_stage1_sample(temp_file)
            
            # 유효한 클래스 ID
            edi_code = registry.predict_edi_code_from_class_id(0)
            assert edi_code is not None
            assert edi_code.startswith("EDI")
            
            # 무효한 클래스 ID
            invalid_edi = registry.predict_edi_code_from_class_id(999)
            assert invalid_edi is None
            
        finally:
            temp_file.unlink()


class TestDrugMetadataValidator:
    """drug_metadata_validator 테스트"""
    
    def test_valid_registry_validation(self):
        """유효한 레지스트리 검증 테스트"""
        # 유효한 레지스트리 생성
        registry = PharmaceuticalCodeRegistry("/tmp")
        
        # 50개 레코드 생성
        for i in range(50):
            k_code = f"K-{i:06d}"
            record = DrugIdentificationRecord(
                k_code=k_code,
                edi_code=f"EDI{i:05d}",
                drug_name_kor=f"테스트약품_{i}",
                drug_name_eng=None,
                manufacturer=f"제약회사_{i}",
                dosage_form="정제",
                strength=None,
                shape=None,
                color=None,
                marking=None,
                approval_date=None,
                stage1_class_id=i
            )
            registry.drug_records[k_code] = record
        
        # Stage 1 매핑 생성
        registry.stage1_mapping = Stage1ClassificationMapping(
            total_classes=50,
            k_code_to_class_id={f"K-{i:06d}": i for i in range(50)},
            class_id_to_edi_code={i: f"EDI{i:05d}" for i in range(50)},
            class_weights={i: 1.0 for i in range(50)}
        )
        
        validation_results = drug_metadata_validator(registry)
        
        assert validation_results['record_count_valid'] == True
        assert validation_results['k_code_format_valid'] == True
        assert validation_results['class_id_continuity_valid'] == True
        assert validation_results['edi_code_unique'] == True
        assert validation_results['required_fields_complete'] == True
    
    def test_invalid_registry_validation(self):
        """무효한 레지스트리 검증 테스트"""
        registry = PharmaceuticalCodeRegistry("/tmp")
        
        # 잘못된 레코드들 생성
        # 1. 개수 부족 (10개만)
        for i in range(10):
            k_code = f"INVALID{i}"  # 잘못된 K-코드 형식
            record = DrugIdentificationRecord(
                k_code=f"K-{i:06d}",  # 실제로는 유효한 형식 사용
                edi_code=f"EDI00000",  # 중복 EDI 코드
                drug_name_kor="",  # 빈 이름
                drug_name_eng=None,
                manufacturer="",  # 빈 제조회사
                dosage_form="정제",
                strength=None,
                shape=None,
                color=None,
                marking=None,
                approval_date=None
            )
            registry.drug_records[k_code] = record  # key는 잘못된 형식 사용
        
        validation_results = drug_metadata_validator(registry)
        
        assert validation_results['record_count_valid'] == False
        assert validation_results['k_code_format_valid'] == False
        assert validation_results['edi_code_unique'] == False
        assert validation_results['required_fields_complete'] == False


class TestLoadPharmaceuticalRegistry:
    """load_pharmaceutical_registry_from_artifacts 테스트"""
    
    def test_load_registry_from_valid_artifacts(self):
        """유효한 아티팩트에서 레지스트리 로드 테스트"""
        # 테스트용 레지스트리 데이터 생성
        registry_data = {
            'metadata': {
                'total_records': 2,
                'stage1_classes': 2,
                'created_at': '2023-01-01T00:00:00',
                'version': '1.0.0'
            },
            'drug_records': {
                'K-123456': {
                    'k_code': 'K-123456',
                    'edi_code': 'EDI12345',
                    'drug_name_kor': '테스트약1',
                    'drug_name_eng': None,
                    'manufacturer': '제약회사1',
                    'dosage_form': '정제',
                    'strength': None,
                    'shape': None,
                    'color': None,
                    'marking': None,
                    'approval_date': None,
                    'stage1_class_id': 0,
                    'image_count': 100,
                    'data_quality_score': 1.0
                }
            },
            'stage1_mapping': {
                'total_classes': 2,
                'k_code_to_class_id': {'K-123456': 0},
                'class_id_to_edi_code': {0: 'EDI12345'},
                'class_weights': {0: 1.0}
            }
        }
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(registry_data, f)
            temp_file = Path(f.name)
        
        try:
            with patch('src.data.pharmaceutical_code_registry.load_config') as mock_config:
                mock_config.return_value = {'data': {'root': '/tmp'}}
                
                registry = load_pharmaceutical_registry_from_artifacts(temp_file)
                
                assert len(registry.drug_records) == 1
                assert 'K-123456' in registry.drug_records
                assert registry.stage1_mapping is not None
                assert registry.stage1_mapping.total_classes == 2
                
        finally:
            temp_file.unlink()
    
    def test_load_registry_from_invalid_file(self):
        """잘못된 파일에서 레지스트리 로드 테스트"""
        with pytest.raises(Exception):
            load_pharmaceutical_registry_from_artifacts(Path("nonexistent.json"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])