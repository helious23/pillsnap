"""
설정 로더 모듈

목적: config.yaml을 안전하게 읽고 기본값을 채우는 설정 관리
입력: 
    - 프로젝트 루트의 config.yaml (선택적)
    - 환경변수 및 기본값
출력:
    - 검증된 AppCfg 설정 객체
검증 포인트:
    - config.yaml 파싱 오류 처리
    - 필수 필드 기본값 적용
    - default_mode 지원 (프론트엔드 기본값 vs 실행 모드 분리)
    - 경로 정책 준수 확인 (path_policy.PathPolicyValidator)
    - 데이터 루트 경로 자동 보완
    - 알 수 없는 키/잘못된 값에 대한 안전한 폴백
"""

import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Literal
import logging

try:
    from pydantic import BaseModel, Field, ConfigDict
    V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, Field
        V2 = False
    except ImportError:
        # TODO: Pydantic 미존재 시 dataclass fallback 구현
        from dataclasses import dataclass, field
        V2 = None

import paths
from src.core.path_policy import PathPolicyValidator

logger = logging.getLogger(__name__)


def _sanitize_data_config(data_dict: dict) -> dict:
    """
    data 섹션 설정 검증 및 정리
    
    Args:
        data_dict: 원본 data 설정 딕셔너리
        
    Returns:
        검증된 data 설정 딕셔너리
    """
    cleaned = data_dict.copy()
    
    # pipeline_mode 검증
    if 'pipeline_mode' in cleaned:
        if cleaned['pipeline_mode'] not in ["single", "combo"]:
            print(f"[config] warning: invalid value for data.pipeline_mode='{cleaned['pipeline_mode']}', fallback='single'")
            cleaned['pipeline_mode'] = "single"
    
    # default_mode 검증
    if 'default_mode' in cleaned:
        if cleaned['default_mode'] not in ["single", "combo"]:
            print(f"[config] warning: invalid value for data.default_mode='{cleaned['default_mode']}', fallback='single'")
            cleaned['default_mode'] = "single"
    
    # 알 수 없는 키 경고 및 필터링
    known_keys = {'root', 'pipeline_mode', 'default_mode', 'image_exts', 'label_ext'}
    
    # 알 수 없는 키들 로깅
    unknown_keys = set(cleaned.keys()) - known_keys
    if unknown_keys:
        print(f"[config] warning: unknown key ignored in data: {', '.join(unknown_keys)}")
    
    # DataCfg에서 사용할 수 있는 키만 필터링
    filtered = {k: v for k, v in cleaned.items() if k in known_keys}
    
    return filtered


if V2 is not None:
    class DataCfg(BaseModel):
        """데이터 설정"""
        # 경로 루트 (없으면 paths.get_data_root() 채움)
        root: Optional[str] = None
        # 실행 모드: 실제 파이프라인에서 사용할 모드
        pipeline_mode: Literal["single", "combo"] = "single"
        # UI 기본 선택: 프론트엔드 초기값(실행 모드 강제 X)
        default_mode: Literal["single", "combo"] = "single"
        image_exts: List[str] = Field(default=[".jpg", ".jpeg", ".png"])
        label_ext: str = ".json"
        
        # Pydantic extra 필드 무시
        if V2:
            model_config = ConfigDict(extra="ignore")
        else:
            class Config:
                extra = "ignore"
    
    class PreprocessCfg(BaseModel):
        """전처리 설정"""
        manifest_filename: str = "manifest_stage1.csv"
        quarantine_dirname: str = "_quarantine"
        
        # Pydantic extra 필드 무시
        if V2:
            model_config = ConfigDict(extra="ignore")
        else:
            class Config:
                extra = "ignore"
    
    class ValidationCfg(BaseModel):
        """검증 설정"""
        enable_angle_rules: bool = False
        label_size_range: Optional[Tuple[int, int]] = None
        
        # Pydantic extra 필드 무시
        if V2:
            model_config = ConfigDict(extra="ignore")
        else:
            class Config:
                extra = "ignore"
    
    class AppCfg(BaseModel):
        """애플리케이션 전체 설정"""
        data: DataCfg = Field(default_factory=DataCfg)
        preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
        validation: ValidationCfg = Field(default_factory=ValidationCfg)
        
        # Pydantic extra 필드 무시
        if V2:
            model_config = ConfigDict(extra="ignore")
        else:
            class Config:
                extra = "ignore"

else:
    # TODO: Pydantic 미존재 시 dataclass fallback
    @dataclass
    class DataCfg:
        """데이터 설정"""
        root: Optional[str] = None
        pipeline_mode: str = "single"
        default_mode: str = "single"
        image_exts: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
        label_ext: str = ".json"
    
    @dataclass
    class PreprocessCfg:
        """전처리 설정"""
        manifest_filename: str = "manifest_stage1.csv"
        quarantine_dirname: str = "_quarantine"
    
    @dataclass
    class ValidationCfg:
        """검증 설정"""
        enable_angle_rules: bool = False
        label_size_range: Optional[Tuple[int, int]] = None
    
    @dataclass
    class AppCfg:
        """애플리케이션 전체 설정"""
        data: DataCfg = field(default_factory=DataCfg)
        preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
        validation: ValidationCfg = field(default_factory=ValidationCfg)


def load_config() -> AppCfg:
    """
    설정 파일 로드 및 AppCfg 객체 생성
    
    Returns:
        AppCfg: 검증된 설정 객체
        
    Raises:
        None: 모든 오류는 로그로만 기록하고 기본값 사용
    """
    # 1) config.yaml 읽기
    config_path = Path("config.yaml")
    config_dict = {}
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}. Using defaults.")
            config_dict = {}
    else:
        logger.info("config.yaml not found, using default configuration")
    
    # 2) dict → AppCfg 파싱 (누락은 기본값, 값 검증)
    try:
        # data 섹션 전처리 및 검증
        data_dict = config_dict.get('data', {})
        data_dict = _sanitize_data_config(data_dict)
        
        if V2 is not None:
            # Pydantic 모델 파싱
            data_cfg = DataCfg(**data_dict)
            preprocess_cfg = PreprocessCfg(**config_dict.get('preprocess', {}))
            validation_cfg = ValidationCfg(**config_dict.get('validation', {}))
            app_cfg = AppCfg(
                data=data_cfg,
                preprocess=preprocess_cfg,
                validation=validation_cfg
            )
        else:
            # dataclass 파싱
            data_cfg = DataCfg(**data_dict)
            preprocess_cfg = PreprocessCfg(**config_dict.get('preprocess', {}))
            validation_cfg = ValidationCfg(**config_dict.get('validation', {}))
            app_cfg = AppCfg(
                data=data_cfg,
                preprocess=preprocess_cfg,
                validation=validation_cfg
            )
        
        logger.info("Configuration parsed successfully")
        
    except Exception as e:
        print(f"Failed to parse configuration: {e}. Using all defaults.")
        # 완전 기본값으로 폴백
        app_cfg = AppCfg()
    
    # 3) data.root가 None이면 pillsnap.paths.get_data_root()로 대체
    if app_cfg.data.root is None:
        try:
            default_root = paths.get_data_root()
            app_cfg.data.root = str(default_root)
            logger.info(f"Set data.root to default: {app_cfg.data.root}")
        except Exception as e:
            logger.error(f"Failed to get default data root: {e}")
            app_cfg.data.root = "./data"  # 최후 폴백
    
    # 4) 경로 정책 검증 (경고만 출력, 중단 없음)
    try:
        validator = PathPolicyValidator()
        valid, message = validator.validate_path(app_cfg.data.root, purpose="data")
        
        if valid:
            logger.info(f"Path policy validation passed: {message}")
        else:
            logger.warning(f"Path policy validation failed: {message}")
        
    except Exception as e:
        logger.warning(f"Path policy validation error: {e}")
    
    # 5) AppCfg 반환
    logger.info(f"Configuration loaded successfully. Data root: {app_cfg.data.root}")
    return app_cfg


def get_default_config() -> AppCfg:
    """
    기본 설정 객체 반환 (config.yaml 무시)
    
    Returns:
        AppCfg: 기본값으로 채워진 설정 객체
    """
    return AppCfg()