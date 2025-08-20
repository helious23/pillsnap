#!/usr/bin/env python3
"""
Stage 2 Progressive Validation 샘플링 실행 스크립트
25,000개 이미지, 250개 클래스 샘플링
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.progressive_validation_sampler import (
    ProgressiveValidationSampler,
    Stage2SamplingStrategy,
    validate_sample_distribution
)
from src.utils.core import load_config, PillSnapLogger

def main():
    logger = PillSnapLogger(__name__)
    logger.info("=== Stage 2 Progressive Validation 샘플링 시작 ===")
    
    # 환경 설정
    config = load_config()
    data_root = "/mnt/data/pillsnap_dataset"  # 원본 데이터 경로
    
    logger.info(f"원본 데이터 루트: {data_root}")
    logger.info("Stage 2 목표: 25,000개 이미지, 250개 클래스")
    
    # Stage 2 샘플링 전략 생성
    strategy = Stage2SamplingStrategy()
    logger.info(f"샘플링 전략: {strategy.target_images}개 이미지, {strategy.target_classes}개 클래스")
    logger.info(f"Single:Combo 비율: {strategy.single_combo_ratio:.1%}:{1-strategy.single_combo_ratio:.1%}")
    
    # 샘플러 초기화
    sampler = ProgressiveValidationSampler(data_root, strategy)
    
    try:
        # Stage 2 샘플 생성
        logger.info("Stage 2 샘플링 실행...")
        sample_data = sampler.generate_stage2_sample()
        
        # 샘플 저장
        output_path = sampler.save_sample(sample_data, "stage2_sample.json")
        logger.info(f"Stage 2 샘플 저장 완료: {output_path}")
        
        # 검증
        logger.info("Stage 2 샘플 검증 중...")
        is_valid = validate_stage2_distribution(sample_data)
        
        if is_valid:
            logger.info("✅ Stage 2 샘플링 성공!")
            logger.info("다음 단계: SSD로 데이터 이전")
        else:
            logger.error("❌ Stage 2 샘플링 검증 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ Stage 2 샘플링 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def validate_stage2_distribution(sample_data: dict) -> bool:
    """Stage 2 샘플 분포 검증"""
    logger = PillSnapLogger(__name__)
    
    try:
        stats = sample_data['stats']
        samples = sample_data['samples']
        
        # 기본 검증
        assert stats['sampled_classes'] == 250, f"클래스 수 불일치: {stats['sampled_classes']} != 250"
        assert stats['sampled_images'] == 25000, f"이미지 수 불일치: {stats['sampled_images']} != 25000"
        
        # 클래스별 이미지 수 검증
        for k_code, data in samples.items():
            assert 80 <= data['total_images'] <= 120, \
                f"{k_code}: 이미지 수 범위 벗어남 ({data['total_images']})"
        
        # 품질 통과율 검증
        assert stats['quality_pass_rate'] >= 0.95, \
            f"품질 통과율 부족: {stats['quality_pass_rate']:.2%} < 95%"
        
        logger.info("✅ Stage 2 샘플 분포 검증 통과")
        return True
        
    except AssertionError as e:
        logger.error(f"❌ Stage 2 샘플 분포 검증 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 검증 중 오류: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)