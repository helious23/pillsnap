"""
Pipeline Mode Resolver - Single Source of Truth
파이프라인 모드 결정을 위한 단일 진실 소스
"""

from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

PipelineMode = Literal["single", "combo"]


@dataclass
class PipelineModeConfig:
    """파이프라인 모드 설정"""
    default_mode: PipelineMode = "single"
    auto_fallback: bool = False  # 항상 False (자동 판단 제거)
    confidence_threshold: float = 0.3
    
    def __post_init__(self):
        if self.auto_fallback:
            logger.warning("auto_fallback is deprecated and will be ignored. User selection only.")
            self.auto_fallback = False


class PipelineModeResolver:
    """
    파이프라인 모드 결정을 위한 단일 진실 소스
    모든 모드 결정은 이 클래스를 통해서만 수행
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: config.yaml의 data 섹션
        """
        self.config = PipelineModeConfig(
            default_mode=config.get("default_mode", "single"),
            auto_fallback=False,  # 강제로 False
            confidence_threshold=config.get("single_confidence_threshold", 0.3)
        )
        logger.info(f"PipelineModeResolver initialized with default_mode={self.config.default_mode}")
    
    def resolve_mode(
        self, 
        user_mode: Optional[PipelineMode] = None,
        hint: Optional[str] = None
    ) -> tuple[PipelineMode, str]:
        """
        파이프라인 모드 결정 (사용자 제어 기반)
        
        Args:
            user_mode: 사용자가 명시적으로 선택한 모드
            hint: 모드 선택 힌트 (로깅용)
            
        Returns:
            (선택된 모드, 선택 이유)
        """
        # 1. 사용자 명시적 선택이 최우선
        if user_mode is not None:
            if user_mode not in ["single", "combo"]:
                logger.warning(f"Invalid mode '{user_mode}', falling back to default")
                return self.config.default_mode, f"invalid_mode_fallback_to_{self.config.default_mode}"
            return user_mode, f"user_explicit_selection_{user_mode}"
        
        # 2. 기본 모드 사용
        reason = f"default_mode_{self.config.default_mode}"
        if hint:
            reason += f"_hint_{hint}"
        
        return self.config.default_mode, reason
    
    def validate_mode_consistency(self, mode: PipelineMode) -> bool:
        """
        모드 일관성 검증
        
        Args:
            mode: 검증할 모드
            
        Returns:
            유효한 모드인지 여부
        """
        return mode in ["single", "combo"]
    
    def get_mode_requirements(self, mode: PipelineMode) -> Dict[str, Any]:
        """
        모드별 요구사항 반환
        
        Args:
            mode: 파이프라인 모드
            
        Returns:
            모드별 요구사항 딕셔너리
        """
        if mode == "single":
            return {
                "models_required": ["classification"],
                "expected_input": "single_pill_image",
                "output_format": "direct_classification",
                "confidence_threshold": self.config.confidence_threshold,
                "preprocessing": {
                    "resize": 384,
                    "normalize": "imagenet"
                }
            }
        elif mode == "combo":
            return {
                "models_required": ["detection", "classification"],
                "expected_input": "combination_pill_image",
                "output_format": "detection_then_classification",
                "confidence_threshold": self.config.confidence_threshold,
                "preprocessing": {
                    "detection_resize": 640,
                    "classification_resize": 384,
                    "normalize": "imagenet"
                }
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def log_mode_decision(
        self,
        mode: PipelineMode,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        모드 결정 로깅 (감사 추적용)
        
        Args:
            mode: 선택된 모드
            reason: 선택 이유
            metadata: 추가 메타데이터
        """
        log_entry = {
            "mode": mode,
            "reason": reason,
            "default_mode": self.config.default_mode,
            "timestamp": None,  # Will be added by logger
        }
        
        if metadata:
            log_entry.update(metadata)
        
        logger.info(f"Pipeline mode resolved: {log_entry}")
        
        # 감사 로그 (선택적)
        if metadata and metadata.get("audit_log", False):
            logger.info(f"[AUDIT] Pipeline mode decision: {log_entry}")


# 전역 싱글톤 인스턴스
_resolver_instance: Optional[PipelineModeResolver] = None


def get_pipeline_resolver(config: Optional[Dict[str, Any]] = None) -> PipelineModeResolver:
    """
    파이프라인 리졸버 싱글톤 반환
    
    Args:
        config: 초기화 시 사용할 설정 (첫 호출 시만 필요)
        
    Returns:
        PipelineModeResolver 인스턴스
    """
    global _resolver_instance
    
    if _resolver_instance is None:
        if config is None:
            raise ValueError("Config must be provided for first initialization")
        _resolver_instance = PipelineModeResolver(config)
    
    return _resolver_instance


def reset_resolver():
    """리졸버 인스턴스 리셋 (테스트용)"""
    global _resolver_instance
    _resolver_instance = None