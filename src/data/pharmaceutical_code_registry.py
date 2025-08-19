"""
의약품 코드 레지스트리 시스템

K-코드(Korean Drug Code) → EDI 코드 매핑 및 의약품 메타데이터 관리:
- PharmaceuticalCodeRegistry: 전체 코드 매핑 시스템
- DrugIdentificationRecord: 개별 의약품 식별 레코드
- Stage1ClassificationMapper: Stage 1용 50개 클래스 매핑
- drug_metadata_validator: 의약품 정보 무결성 검증
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

from src.utils.core import PillSnapLogger, load_config


@dataclass
class DrugIdentificationRecord:
    """개별 의약품 식별 레코드"""
    k_code: str                    # 한국 의약품 코드 (예: K-029066)
    edi_code: str                  # EDI 표준 코드 (분류 타겟)
    drug_name_kor: str             # 한국어 의약품명
    drug_name_eng: Optional[str]   # 영어 의약품명
    manufacturer: str              # 제조회사
    dosage_form: str              # 제형 (정제, 캡슐 등)
    strength: Optional[str]        # 함량 정보
    shape: Optional[str]           # 모양 정보
    color: Optional[str]           # 색상 정보
    marking: Optional[str]         # 각인 정보
    approval_date: Optional[str]   # 허가일자
    
    # Stage 1 전용 필드
    stage1_class_id: Optional[int] = None  # Stage 1에서의 클래스 ID (0-49)
    image_count: Optional[int] = None      # 사용 가능한 이미지 수
    data_quality_score: Optional[float] = None  # 데이터 품질 점수 (0-1)
    
    def __post_init__(self):
        """레코드 유효성 검증"""
        if not self.k_code.startswith('K-'):
            raise ValueError(f"잘못된 K-코드 형식: {self.k_code}")
        
        if not self.edi_code:
            raise ValueError(f"EDI 코드가 필요합니다: {self.k_code}")
        
        if not self.drug_name_kor:
            raise ValueError(f"한국어 의약품명이 필요합니다: {self.k_code}")
    
    def to_classification_target(self) -> Tuple[str, int]:
        """분류 모델용 타겟 정보 반환"""
        return (self.edi_code, self.stage1_class_id or -1)


@dataclass
class Stage1ClassificationMapping:
    """Stage 1 분류 매핑 정보"""
    total_classes: int = 50
    k_code_to_class_id: Dict[str, int] = None
    class_id_to_edi_code: Dict[int, str] = None
    class_weights: Dict[int, float] = None  # 클래스별 가중치 (불균형 대응)
    
    def __post_init__(self):
        if self.k_code_to_class_id is None:
            self.k_code_to_class_id = {}
        if self.class_id_to_edi_code is None:
            self.class_id_to_edi_code = {}
        if self.class_weights is None:
            self.class_weights = {}


class PharmaceuticalCodeRegistry:
    """의약품 코드 레지스트리 메인 클래스"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.logger = PillSnapLogger(__name__)
        
        # 레지스트리 데이터 저장 경로
        self.registry_dir = Path("artifacts/stage1/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # 의약품 레코드 저장소
        self.drug_records: Dict[str, DrugIdentificationRecord] = {}
        self.stage1_mapping: Optional[Stage1ClassificationMapping] = None
        
        self.logger.info(f"PharmaceuticalCodeRegistry 초기화")
        self.logger.info(f"데이터 루트: {self.data_root}")
        self.logger.info(f"레지스트리 디렉토리: {self.registry_dir}")
    
    def scan_drug_metadata_sources(self) -> Dict[str, Path]:
        """의약품 메타데이터 소스 스캔"""
        self.logger.info("의약품 메타데이터 소스 스캔...")
        
        sources = {}
        
        # 1. 조합 리스트 (combination_list.xlsx)
        combo_list_path = self.data_root / "document" / "combination_list.xlsx"
        if combo_list_path.exists():
            sources['combination_list'] = combo_list_path
            self.logger.info(f"발견: {combo_list_path}")
        
        # 2. 단일 리스트 (single_list.xlsx)
        single_list_path = self.data_root / "document" / "single_list.xlsx"
        if single_list_path.exists():
            sources['single_list'] = single_list_path
            self.logger.info(f"발견: {single_list_path}")
        
        # 3. 데이터 문서 (data_document.pdf) - 참고용
        data_doc_path = self.data_root / "document" / "data_document.pdf"
        if data_doc_path.exists():
            sources['data_document'] = data_doc_path
            self.logger.info(f"발견: {data_doc_path}")
        
        # 4. 디렉토리 기반 K-코드 스캔
        k_codes_from_dirs = self._extract_k_codes_from_directory_structure()
        if k_codes_from_dirs:
            sources['directory_k_codes'] = k_codes_from_dirs
            self.logger.info(f"디렉토리에서 추출된 K-코드: {len(k_codes_from_dirs)}개")
        
        self.logger.info(f"메타데이터 소스 스캔 완료: {len(sources)}개 소스")
        return sources
    
    def _extract_k_codes_from_directory_structure(self) -> List[str]:
        """디렉토리 구조에서 K-코드 추출"""
        k_codes = set()
        
        # Single 약품 디렉토리 스캔
        single_dir = self.data_root / "data/train/images/single"
        if single_dir.exists():
            for ts_dir in single_dir.glob("TS_*"):
                for k_code_dir in ts_dir.iterdir():
                    if k_code_dir.is_dir() and k_code_dir.name.startswith('K-'):
                        k_codes.add(k_code_dir.name)
        
        # Combination 약품 디렉토리 스캔
        combo_dir = self.data_root / "data/train/images/combination"
        if combo_dir.exists():
            for ts_dir in combo_dir.glob("TS_*"):
                for k_code_dir in ts_dir.iterdir():
                    if k_code_dir.is_dir() and k_code_dir.name.startswith('K-'):
                        k_codes.add(k_code_dir.name)
        
        return sorted(list(k_codes))
    
    def parse_excel_metadata(self, excel_path: Path) -> List[DrugIdentificationRecord]:
        """Excel 파일에서 의약품 메타데이터 파싱"""
        self.logger.info(f"Excel 메타데이터 파싱: {excel_path}")
        
        try:
            # pandas로 Excel 읽기
            df = pd.read_excel(excel_path)
            self.logger.info(f"Excel 로드 완료: {len(df)}행 × {len(df.columns)}열")
            
            # 컬럼명 출력 (디버깅용)
            self.logger.info(f"컬럼: {list(df.columns)}")
            
            records = []
            
            # Excel 구조에 따라 파싱 로직 조정 필요
            # 현재는 기본적인 구조 가정
            for idx, row in df.iterrows():
                try:
                    # 컬럼명이 확실하지 않으므로 안전한 방식으로 접근
                    k_code = self._safe_get_column_value(row, ['K코드', 'K_CODE', 'k_code', 'KCODE'])
                    drug_name = self._safe_get_column_value(row, ['의약품명', '제품명', 'DRUG_NAME', 'drug_name'])
                    manufacturer = self._safe_get_column_value(row, ['제조회사', '업체명', 'MANUFACTURER', 'manufacturer'])
                    
                    if not k_code or not drug_name:
                        continue
                    
                    # K-코드 정규화
                    if not k_code.startswith('K-'):
                        k_code = f"K-{k_code}"
                    
                    # 임시 EDI 코드 생성 (실제로는 별도 매핑 테이블 필요)
                    edi_code = f"EDI{hash(k_code) % 100000:05d}"
                    
                    record = DrugIdentificationRecord(
                        k_code=k_code,
                        edi_code=edi_code,
                        drug_name_kor=str(drug_name).strip(),
                        drug_name_eng=None,
                        manufacturer=str(manufacturer).strip() if manufacturer else "Unknown",
                        dosage_form="정제",  # 기본값
                        strength=None,
                        shape=None,
                        color=None,
                        marking=None,
                        approval_date=None
                    )
                    
                    records.append(record)
                    
                except Exception as e:
                    self.logger.warning(f"행 {idx} 파싱 실패: {e}")
                    continue
            
            self.logger.info(f"Excel 파싱 완료: {len(records)}개 레코드")
            return records
            
        except Exception as e:
            self.logger.error(f"Excel 파싱 실패: {e}")
            return []
    
    def _safe_get_column_value(self, row: pd.Series, possible_columns: List[str]) -> Optional[str]:
        """안전한 컬럼값 추출"""
        for col in possible_columns:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]).strip()
        return None
    
    def build_drug_registry_from_stage1_sample(self, stage1_sample_path: Path) -> bool:
        """Stage 1 샘플에서 의약품 레지스트리 구축"""
        self.logger.info(f"Stage 1 샘플에서 레지스트리 구축: {stage1_sample_path}")
        
        try:
            # Stage 1 샘플 데이터 로드
            with open(stage1_sample_path, 'r', encoding='utf-8') as f:
                stage1_data = json.load(f)
            
            selected_k_codes = stage1_data['metadata']['selected_classes']
            self.logger.info(f"선택된 K-코드: {len(selected_k_codes)}개")
            
            # 메타데이터 소스 스캔
            metadata_sources = self.scan_drug_metadata_sources()
            
            # Excel 메타데이터 파싱
            all_records = []
            for source_type, source_path in metadata_sources.items():
                if source_type in ['combination_list', 'single_list'] and isinstance(source_path, Path):
                    records = self.parse_excel_metadata(source_path)
                    all_records.extend(records)
            
            # K-코드별 레코드 매핑
            k_code_to_record = {record.k_code: record for record in all_records}
            
            # Stage 1 K-코드에 대한 레코드 구축
            for class_id, k_code in enumerate(selected_k_codes):
                if k_code in k_code_to_record:
                    # 기존 레코드 사용
                    record = k_code_to_record[k_code]
                    record.stage1_class_id = class_id
                else:
                    # 기본 레코드 생성
                    record = DrugIdentificationRecord(
                        k_code=k_code,
                        edi_code=f"EDI{class_id:05d}",  # 임시 EDI 코드
                        drug_name_kor=f"약품_{k_code}",  # 임시 이름
                        drug_name_eng=None,
                        manufacturer="Unknown",
                        dosage_form="정제",
                        strength=None,
                        shape=None,
                        color=None,
                        marking=None,
                        approval_date=None,
                        stage1_class_id=class_id
                    )
                
                # 이미지 수 정보 추가
                if k_code in stage1_data['samples']:
                    record.image_count = stage1_data['samples'][k_code]['total_images']
                    record.data_quality_score = 1.0  # 기본값
                
                self.drug_records[k_code] = record
            
            # Stage 1 매핑 정보 구축
            self.stage1_mapping = Stage1ClassificationMapping(
                total_classes=len(selected_k_codes),
                k_code_to_class_id={k_code: record.stage1_class_id 
                                   for k_code, record in self.drug_records.items()},
                class_id_to_edi_code={record.stage1_class_id: record.edi_code 
                                     for record in self.drug_records.values()},
                class_weights={i: 1.0 for i in range(len(selected_k_codes))}  # 균등 가중치
            )
            
            self.logger.info(f"레지스트리 구축 완료: {len(self.drug_records)}개 레코드")
            return True
            
        except Exception as e:
            self.logger.error(f"레지스트리 구축 실패: {e}")
            return False
    
    def get_classification_mapping_for_stage1(self) -> Stage1ClassificationMapping:
        """Stage 1용 분류 매핑 반환"""
        if self.stage1_mapping is None:
            raise ValueError("Stage 1 매핑이 구축되지 않았습니다")
        return self.stage1_mapping
    
    def lookup_drug_by_k_code(self, k_code: str) -> Optional[DrugIdentificationRecord]:
        """K-코드로 의약품 정보 조회"""
        return self.drug_records.get(k_code)
    
    def predict_edi_code_from_class_id(self, class_id: int) -> Optional[str]:
        """클래스 ID에서 EDI 코드 예측"""
        if self.stage1_mapping is None:
            return None
        return self.stage1_mapping.class_id_to_edi_code.get(class_id)
    
    def save_registry_to_artifacts(self, filename: str = "drug_registry_stage1.json") -> Path:
        """레지스트리를 아티팩트로 저장"""
        output_path = self.registry_dir / filename
        
        registry_data = {
            'metadata': {
                'total_records': len(self.drug_records),
                'stage1_classes': self.stage1_mapping.total_classes if self.stage1_mapping else 0,
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'drug_records': {k: asdict(v) for k, v in self.drug_records.items()},
            'stage1_mapping': asdict(self.stage1_mapping) if self.stage1_mapping else None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"레지스트리 저장 완료: {output_path}")
        return output_path
    
    def export_stage1_class_mapping_csv(self) -> Path:
        """Stage 1 클래스 매핑을 CSV로 내보내기"""
        output_path = self.registry_dir / "stage1_class_mapping.csv"
        
        if not self.stage1_mapping:
            raise ValueError("Stage 1 매핑이 없습니다")
        
        rows = []
        for k_code, record in self.drug_records.items():
            rows.append({
                'class_id': record.stage1_class_id,
                'k_code': k_code,
                'edi_code': record.edi_code,
                'drug_name_kor': record.drug_name_kor,
                'manufacturer': record.manufacturer,
                'image_count': record.image_count or 0
            })
        
        df = pd.DataFrame(rows)
        df.sort_values('class_id', inplace=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"클래스 매핑 CSV 저장: {output_path}")
        return output_path


def drug_metadata_validator(registry: PharmaceuticalCodeRegistry) -> Dict[str, bool]:
    """의약품 메타데이터 무결성 검증"""
    logger = PillSnapLogger(__name__)
    
    validation_results = {}
    
    try:
        # 1. 레코드 개수 검증
        record_count = len(registry.drug_records)
        validation_results['record_count_valid'] = record_count == 50
        logger.info(f"레코드 개수: {record_count}/50 {'✅' if validation_results['record_count_valid'] else '❌'}")
        
        # 2. K-코드 형식 검증
        invalid_k_codes = [k for k in registry.drug_records.keys() if not k.startswith('K-')]
        validation_results['k_code_format_valid'] = len(invalid_k_codes) == 0
        logger.info(f"K-코드 형식: {'✅' if validation_results['k_code_format_valid'] else '❌'} "
                   f"(잘못된 형식: {len(invalid_k_codes)}개)")
        
        # 3. 클래스 ID 연속성 검증
        if registry.stage1_mapping:
            class_ids = set(registry.stage1_mapping.k_code_to_class_id.values())
            expected_class_ids = set(range(50))
            validation_results['class_id_continuity_valid'] = class_ids == expected_class_ids
            logger.info(f"클래스 ID 연속성: {'✅' if validation_results['class_id_continuity_valid'] else '❌'}")
        
        # 4. EDI 코드 중복 검증
        edi_codes = [record.edi_code for record in registry.drug_records.values()]
        validation_results['edi_code_unique'] = len(edi_codes) == len(set(edi_codes))
        logger.info(f"EDI 코드 유일성: {'✅' if validation_results['edi_code_unique'] else '❌'}")
        
        # 5. 필수 필드 검증
        missing_required_fields = 0
        for record in registry.drug_records.values():
            if not all([record.k_code, record.edi_code, record.drug_name_kor, record.manufacturer]):
                missing_required_fields += 1
        
        validation_results['required_fields_complete'] = missing_required_fields == 0
        logger.info(f"필수 필드 완성도: {'✅' if validation_results['required_fields_complete'] else '❌'} "
                   f"(누락: {missing_required_fields}개)")
        
        # 전체 검증 결과
        all_valid = all(validation_results.values())
        logger.info(f"전체 검증 결과: {'✅ 통과' if all_valid else '❌ 실패'}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"검증 중 오류: {e}")
        return {'validation_error': False}


def load_pharmaceutical_registry_from_artifacts(registry_path: Path) -> PharmaceuticalCodeRegistry:
    """아티팩트에서 의약품 레지스트리 로드"""
    logger = PillSnapLogger(__name__)
    
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
        
        # 임시 레지스트리 생성
        config = load_config()
        registry = PharmaceuticalCodeRegistry(config['data']['root'])
        
        # 의약품 레코드 복원
        for k_code, record_data in registry_data['drug_records'].items():
            record = DrugIdentificationRecord(**record_data)
            registry.drug_records[k_code] = record
        
        # Stage 1 매핑 복원
        if registry_data['stage1_mapping']:
            mapping_data = registry_data['stage1_mapping']
            registry.stage1_mapping = Stage1ClassificationMapping(**mapping_data)
        
        logger.info(f"레지스트리 로드 완료: {len(registry.drug_records)}개 레코드")
        return registry
        
    except Exception as e:
        logger.error(f"레지스트리 로드 실패: {e}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    from src.utils.core import load_config
    
    config = load_config()
    registry = PharmaceuticalCodeRegistry(config['data']['root'])
    
    # Stage 1 샘플에서 레지스트리 구축
    stage1_sample_path = Path("artifacts/stage1/sampling/stage1_sample_test.json")
    
    if stage1_sample_path.exists():
        success = registry.build_drug_registry_from_stage1_sample(stage1_sample_path)
        
        if success:
            # 레지스트리 저장
            registry.save_registry_to_artifacts()
            registry.export_stage1_class_mapping_csv()
            
            # 검증
            validation_results = drug_metadata_validator(registry)
            print(f"검증 결과: {validation_results}")
        else:
            print("레지스트리 구축 실패")
    else:
        print(f"Stage 1 샘플 파일을 찾을 수 없습니다: {stage1_sample_path}")