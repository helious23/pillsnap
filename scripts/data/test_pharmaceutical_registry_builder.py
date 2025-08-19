#!/usr/bin/env python3
"""
의약품 코드 레지스트리 구축 및 검증 테스트

Stage 1 샘플에서 K-코드 → EDI 코드 매핑 시스템 구축 및 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import load_config, PillSnapLogger
from src.data.pharmaceutical_code_registry import (
    PharmaceuticalCodeRegistry,
    drug_metadata_validator,
    load_pharmaceutical_registry_from_artifacts
)


def test_pharmaceutical_registry_construction():
    """의약품 레지스트리 구축 전체 테스트"""
    logger = PillSnapLogger(__name__)
    
    try:
        # 1. 설정 로드
        logger.info("🔧 설정 로드 중...")
        config = load_config()
        data_root = config['data']['root']
        
        logger.info(f"📁 데이터 루트: {data_root}")
        
        # 2. 레지스트리 초기화
        logger.info("🏥 PharmaceuticalCodeRegistry 초기화...")
        registry = PharmaceuticalCodeRegistry(data_root)
        
        # 3. Stage 1 샘플 파일 확인
        stage1_sample_path = Path("artifacts/stage1/sampling/stage1_sample_test.json")
        if not stage1_sample_path.exists():
            raise FileNotFoundError(f"Stage 1 샘플 파일이 없습니다: {stage1_sample_path}")
        
        logger.info(f"📄 Stage 1 샘플 파일: {stage1_sample_path}")
        
        # 4. 메타데이터 소스 스캔
        logger.info("🔍 의약품 메타데이터 소스 스캔...")
        metadata_sources = registry.scan_drug_metadata_sources()
        
        logger.info(f"📊 발견된 메타데이터 소스:")
        for source_type, source_info in metadata_sources.items():
            if isinstance(source_info, Path):
                logger.info(f"  {source_type}: {source_info}")
            elif isinstance(source_info, list):
                logger.info(f"  {source_type}: {len(source_info)}개 항목")
        
        # 5. 레지스트리 구축
        logger.info("🏗️  레지스트리 구축 시작...")
        success = registry.build_drug_registry_from_stage1_sample(stage1_sample_path)
        
        if not success:
            logger.error("❌ 레지스트리 구축 실패")
            return False
        
        logger.info(f"✅ 레지스트리 구축 성공: {len(registry.drug_records)}개 의약품 레코드")
        
        # 6. 매핑 정보 확인
        logger.info("🗺️  Stage 1 분류 매핑 확인...")
        stage1_mapping = registry.get_classification_mapping_for_stage1()
        
        logger.info(f"📈 매핑 정보:")
        logger.info(f"  총 클래스: {stage1_mapping.total_classes}개")
        logger.info(f"  K-코드 매핑: {len(stage1_mapping.k_code_to_class_id)}개")
        logger.info(f"  EDI 코드 매핑: {len(stage1_mapping.class_id_to_edi_code)}개")
        
        # 7. 샘플 의약품 정보 출력
        logger.info("💊 샘플 의약품 정보 (처음 5개):")
        for i, (k_code, record) in enumerate(list(registry.drug_records.items())[:5]):
            logger.info(f"  {i+1}. {k_code}:")
            logger.info(f"     EDI: {record.edi_code}")
            logger.info(f"     이름: {record.drug_name_kor}")
            logger.info(f"     제조회사: {record.manufacturer}")
            logger.info(f"     클래스 ID: {record.stage1_class_id}")
            logger.info(f"     이미지 수: {record.image_count}")
        
        # 8. 분류 기능 테스트
        logger.info("🎯 분류 기능 테스트...")
        test_k_code = list(registry.drug_records.keys())[0]
        test_record = registry.lookup_drug_by_k_code(test_k_code)
        
        if test_record:
            logger.info(f"  K-코드 조회 테스트: {test_k_code} → {test_record.drug_name_kor}")
            
            # EDI 코드 예측 테스트
            predicted_edi = registry.predict_edi_code_from_class_id(test_record.stage1_class_id)
            logger.info(f"  EDI 코드 예측 테스트: 클래스 {test_record.stage1_class_id} → {predicted_edi}")
            
            # 분류 타겟 변환 테스트
            edi_code, class_id = test_record.to_classification_target()
            logger.info(f"  분류 타겟 변환: ({edi_code}, {class_id})")
        
        # 9. 레지스트리 저장
        logger.info("💾 레지스트리 아티팩트 저장...")
        registry_path = registry.save_registry_to_artifacts("drug_registry_stage1_test.json")
        csv_path = registry.export_stage1_class_mapping_csv()
        
        logger.info(f"  JSON 레지스트리: {registry_path}")
        logger.info(f"  CSV 매핑: {csv_path}")
        
        # 10. 검증 수행
        logger.info("✅ 레지스트리 무결성 검증...")
        validation_results = drug_metadata_validator(registry)
        
        logger.info(f"📋 검증 결과:")
        for check_name, result in validation_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            logger.info(f"  {check_name}: {status}")
        
        all_valid = all(validation_results.values())
        
        # 11. 저장된 레지스트리 로드 테스트
        logger.info("🔄 저장된 레지스트리 로드 테스트...")
        loaded_registry = load_pharmaceutical_registry_from_artifacts(registry_path)
        
        logger.info(f"  로드된 레코드 수: {len(loaded_registry.drug_records)}개")
        logger.info(f"  로드된 매핑 클래스 수: {loaded_registry.stage1_mapping.total_classes}개")
        
        # 로드 검증
        original_k_codes = set(registry.drug_records.keys())
        loaded_k_codes = set(loaded_registry.drug_records.keys())
        load_success = original_k_codes == loaded_k_codes
        
        logger.info(f"  로드 검증: {'✅ 성공' if load_success else '❌ 실패'}")
        
        # 12. 최종 결과
        logger.info("🎉 레지스트리 구축 및 검증 완료!")
        
        final_success = all_valid and load_success
        logger.info(f"📊 최종 상태:")
        logger.info(f"  의약품 레코드: {len(registry.drug_records)}개")
        logger.info(f"  분류 클래스: {stage1_mapping.total_classes}개")
        logger.info(f"  무결성 검증: {'✅ 통과' if all_valid else '❌ 실패'}")
        logger.info(f"  저장/로드: {'✅ 통과' if load_success else '❌ 실패'}")
        
        return final_success
        
    except Exception as e:
        logger.error(f"❌ 레지스트리 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def demonstrate_classification_workflow():
    """분류 워크플로우 시연"""
    logger = PillSnapLogger(__name__)
    
    try:
        logger.info("🎭 분류 워크플로우 시연...")
        
        # 저장된 레지스트리 로드
        registry_path = Path("artifacts/stage1/registry/drug_registry_stage1_test.json")
        if not registry_path.exists():
            logger.error(f"레지스트리 파일이 없습니다: {registry_path}")
            return False
        
        registry = load_pharmaceutical_registry_from_artifacts(registry_path)
        mapping = registry.get_classification_mapping_for_stage1()
        
        logger.info("🔄 모의 분류 시나리오:")
        
        # 시나리오 1: K-코드로 조회
        sample_k_code = list(registry.drug_records.keys())[0]
        logger.info(f"  시나리오 1 - K-코드 조회: {sample_k_code}")
        
        record = registry.lookup_drug_by_k_code(sample_k_code)
        if record:
            logger.info(f"    결과: {record.drug_name_kor} (제조: {record.manufacturer})")
            logger.info(f"    EDI 코드: {record.edi_code}")
            logger.info(f"    클래스 ID: {record.stage1_class_id}")
        
        # 시나리오 2: 분류 모델 출력 (클래스 ID) → EDI 코드
        test_class_id = 0
        logger.info(f"  시나리오 2 - 클래스 ID → EDI 변환: {test_class_id}")
        
        predicted_edi = registry.predict_edi_code_from_class_id(test_class_id)
        logger.info(f"    예측된 EDI: {predicted_edi}")
        
        # 시나리오 3: 전체 매핑 통계
        logger.info(f"  시나리오 3 - 매핑 통계:")
        logger.info(f"    K-코드 → 클래스 ID 매핑: {len(mapping.k_code_to_class_id)}개")
        logger.info(f"    클래스 ID → EDI 매핑: {len(mapping.class_id_to_edi_code)}개")
        
        # 클래스별 의약품 수 분포
        class_distribution = {}
        for record in registry.drug_records.values():
            class_id = record.stage1_class_id
            if class_id is not None:
                class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
        
        logger.info(f"    클래스별 분포: {len(class_distribution)}개 클래스")
        logger.info(f"    평균 클래스당 약품 수: {len(registry.drug_records) / len(class_distribution):.1f}개")
        
        logger.info("✅ 분류 워크플로우 시연 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 워크플로우 시연 실패: {e}")
        return False


def main():
    """메인 함수"""
    print("🏥 PillSnap 의약품 코드 레지스트리 테스트 시작")
    print("=" * 70)
    
    # 1. 레지스트리 구축 테스트
    construction_success = test_pharmaceutical_registry_construction()
    
    print("=" * 70)
    
    if construction_success:
        # 2. 분류 워크플로우 시연
        workflow_success = demonstrate_classification_workflow()
        
        print("=" * 70)
        
        if workflow_success:
            print("✅ 모든 테스트 완료 - 의약품 코드 레지스트리 준비됨")
            sys.exit(0)
        else:
            print("❌ 워크플로우 시연 실패")
            sys.exit(1)
    else:
        print("❌ 레지스트리 구축 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()