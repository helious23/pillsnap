"""
PillSnap 데이터 파이프라인 모듈

Two-Stage Conditional Pipeline을 위한 데이터 처리 시스템:
- Progressive Validation Stage 1: 526만 → 5,000개 이미지, 50개 클래스
- Single 약품: EfficientNetV2-S 직접 분류 (384px)
- Combination 약품: YOLOv11m 검출 → 크롭 → 분류 (640px → 384px)
"""

from .progressive_validation_sampler import (
    ProgressiveValidationSampler,
    Stage1SamplingStrategy,
    validate_sample_distribution
)

# 현재 구현된 모듈만 import
# from .mapping import (
#     KCodeEDIMapper,
#     DrugMetadata,
#     load_drug_metadata
# )

# from .preprocessing import (
#     PillImagePreprocessor,
#     SinglePillPreprocessor, 
#     ComboPillPreprocessor
# )

# from .loaders import (
#     PillSnapDataLoader,
#     SinglePillDataLoader,
#     ComboPillDataLoader,
#     TwoStageDataLoader
# )

# from .caching import (
#     LMDBCache,
#     PillSnapCacheManager
# )

__all__ = [
    # Sampling (현재 구현됨)
    'ProgressiveValidationSampler',
    'Stage1SamplingStrategy', 
    'validate_sample_distribution',
    
    # TODO: 아래 모듈들은 순차적으로 구현 예정
    # # Mapping
    # 'KCodeEDIMapper',
    # 'DrugMetadata',
    # 'load_drug_metadata',
    
    # # Preprocessing
    # 'PillImagePreprocessor',
    # 'SinglePillPreprocessor',
    # 'ComboPillPreprocessor',
    
    # # Data Loaders
    # 'PillSnapDataLoader',
    # 'SinglePillDataLoader', 
    # 'ComboPillDataLoader',
    # 'TwoStageDataLoader',
    
    # # Caching
    # 'LMDBCache',
    # 'PillSnapCacheManager'
]