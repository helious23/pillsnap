#!/usr/bin/env python3
"""
Stage 1 샘플링 시스템 실제 데이터 테스트

실제 526만개 이미지 데이터에서 5,000개 이미지, 50개 클래스 샘플링 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import load_config, PillSnapLogger
from src.data.sampling import (
    Stage1SamplingStrategy,
    ProgressiveValidationSampler,
    validate_sample_distribution
)


def test_stage1_sampling():
    """Stage 1 샘플링 전체 테스트"""
    logger = PillSnapLogger(__name__)
    
    try:
        # 1. 설정 로드
        logger.info("🔧 설정 로드 중...")
        config = load_config()
        data_root = config['data']['root']
        
        logger.info(f"📁 데이터 루트: {data_root}")
        
        # 데이터 루트 존재 확인
        if not Path(data_root).exists():
            raise FileNotFoundError(f"데이터 루트가 존재하지 않습니다: {data_root}")
        
        # 2. 샘플링 전략 설정
        logger.info("🎯 Stage 1 샘플링 전략 설정...")
        strategy = Stage1SamplingStrategy(
            target_images=5000,
            target_classes=50,
            images_per_class=100,
            quality_threshold=0.95,
            single_combo_ratio=0.7,
            seed=42
        )
        
        logger.info(f"  목표 이미지: {strategy.target_images}개")
        logger.info(f"  목표 클래스: {strategy.target_classes}개")
        logger.info(f"  클래스당 이미지: {strategy.images_per_class}개")
        logger.info(f"  Single/Combo 비율: {strategy.single_combo_ratio:.1%}/{1-strategy.single_combo_ratio:.1%}")
        
        # 3. 샘플러 초기화
        logger.info("🔄 샘플러 초기화...")
        sampler = ProgressiveValidationSampler(data_root, strategy)
        
        # 4. 데이터 스캔
        logger.info("🔍 데이터 스캔 시작...")
        scan_result = sampler.scan_available_data()
        
        total_k_codes = len(scan_result['all_k_codes'])
        total_images = sum(scan_result['k_code_counts'].values())
        single_images = sum(len(imgs) for imgs in scan_result['single'].values())
        combo_images = sum(len(imgs) for imgs in scan_result['combo'].values())
        
        logger.info(f"📊 스캔 결과:")
        logger.info(f"  총 K-코드: {total_k_codes:,}개")
        logger.info(f"  총 이미지: {total_images:,}개")
        logger.info(f"  Single 이미지: {single_images:,}개 ({single_images/total_images:.1%})")
        logger.info(f"  Combo 이미지: {combo_images:,}개 ({combo_images/total_images:.1%})")
        
        # 5. 목표 클래스 선택 가능성 확인
        sufficient_k_codes = [
            k_code for k_code, count in scan_result['k_code_counts'].items()
            if count >= strategy.min_images_per_class
        ]
        
        logger.info(f"📋 충분한 이미지가 있는 K-코드: {len(sufficient_k_codes)}개")
        
        if len(sufficient_k_codes) < strategy.target_classes:
            logger.error(f"❌ Stage 1 샘플링 불가능: "
                        f"필요 {strategy.target_classes}개, 사용 가능 {len(sufficient_k_codes)}개")
            return False
        
        # 6. Stage 1 샘플 생성
        logger.info("🎲 Stage 1 샘플링 실행...")
        sample_data = sampler.generate_stage1_sample()
        
        # 7. 샘플 저장
        logger.info("💾 샘플 데이터 저장...")
        output_path = sampler.save_sample(sample_data, "stage1_sample_test.json")
        
        # 8. 샘플 검증
        logger.info("✅ 샘플 분포 검증...")
        is_valid = validate_sample_distribution(sample_data)
        
        if is_valid:
            logger.info("🎉 Stage 1 샘플링 테스트 성공!")
            logger.info(f"📄 샘플 파일: {output_path}")
            
            # 통계 출력
            stats = sample_data['stats']
            logger.info(f"📈 최종 통계:")
            logger.info(f"  샘플링된 이미지: {stats['sampled_images']:,}개")
            logger.info(f"  샘플링된 클래스: {stats['sampled_classes']}개")
            logger.info(f"  품질 통과율: {stats['quality_pass_rate']:.2%}")
            logger.info(f"  Single 비율: {stats['single_pill_ratio']:.1%}")
            logger.info(f"  Combo 비율: {stats['combo_pill_ratio']:.1%}")
            
            return True
        else:
            logger.error("❌ 샘플 분포 검증 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ Stage 1 샘플링 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """메인 함수"""
    print("🏥 PillSnap Stage 1 샘플링 테스트 시작")
    print("=" * 60)
    
    success = test_stage1_sampling()
    
    print("=" * 60)
    if success:
        print("✅ Stage 1 샘플링 테스트 완료")
        sys.exit(0)
    else:
        print("❌ Stage 1 샘플링 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()