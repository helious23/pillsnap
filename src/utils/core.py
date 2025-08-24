"""
PillSnap ML 유틸리티 함수들
- 설정 파일 로딩 (config.yaml)
- 로깅 시스템
- 경로 검증
- Git SHA 추출
- 시드 설정
"""

import os
import sys
import yaml
import logging
import subprocess
import random
import numpy as np
import torch
import json
import re
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, Set, List
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# 한국 시간대 정의
KST = timezone(timedelta(hours=9))


class ConfigLoader:
    """
    설정 파일 로더 클래스 (1단계 강화)
    - config.yaml 안전 로딩
    - 환경변수 오버라이드 지원
    - 경로 검증
    - YAML 중복 키 탐지 및 머지 규칙
    - 정책 충돌 자동 해소
    - 최종 설정 스냅샷 저장
    """
    
    def __init__(self, config_path: str = "config.yaml", cli_overrides: Optional[Dict[str, Any]] = None):
        """
        Args:
            config_path: config.yaml 파일 경로 (프로젝트 루트 기준)
            cli_overrides: CLI 인자 오버라이드 (최우선 적용)
        """
        self.config_path = config_path
        self.project_root = Path("/home/max16/pillsnap")
        self.cli_overrides = cli_overrides or {}
        self.merge_log = []  # 머지 과정 로그
        
    def _load_config_instance(self) -> Dict[str, Any]:
        """
        config.yaml 파일을 로딩하고 검증합니다. (1단계 강화)
        
        Returns:
            Dict: 설정 딕셔너리
            
        Raises:
            FileNotFoundError: config.yaml 파일이 없는 경우
            yaml.YAMLError: YAML 파싱 오류
            ValueError: 필수 설정이 누락된 경우
        """
        config_file_path = self.project_root / self.config_path
        
        # 1) config.yaml 파일 존재 확인
        if not config_file_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file_path}")
            
        # 2) YAML 중복 키 탐지 전처리
        self._check_duplicate_yaml_keys(config_file_path)
            
        # 3) YAML 파일 로딩
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML 파싱 오류: {e}")
            
        # 4) 머지 순서 적용: base < stage_overrides < CLI
        config = self._apply_merge_hierarchy(base_config)
        
        # 5) 정책 충돌 자동 해소
        config = self._resolve_policy_conflicts(config)
        
        # 6) 환경변수 오버라이드 적용
        config = self._apply_env_overrides(config)
        
        # 7) 경로 검증 및 정규화
        config = self._validate_and_normalize_paths(config)
        
        # 8) 필수 설정 검증
        self._validate_required_settings(config)
        
        # 9) 최종 설정 스냅샷 저장
        self._save_config_snapshot(config)
        
        return config
    
    @classmethod 
    def load_config(cls, config_path: str = "config.yaml", cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        정적 메서드로 config.yaml 파일을 로딩합니다. (1단계 강화)
        
        Args:
            config_path: config.yaml 파일 경로
            cli_overrides: CLI 인자 오버라이드 (최우선 적용)
            
        Returns:
            Dict: 설정 딕셔너리
        """
        loader = cls(config_path, cli_overrides)
        return loader._load_config_instance()
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        환경변수로 설정 오버라이드 적용
        
        지원하는 환경변수:
        - PILLSNAP_DATA_ROOT: 데이터 루트 경로
        - PILLSNAP_EXP_DIR: 실험 디렉토리 경로
        """
        # 데이터 루트 경로 오버라이드
        if data_root := os.getenv("PILLSNAP_DATA_ROOT"):
            if "data" not in config:
                config["data"] = {}
            config["data"]["root"] = data_root
            print(f"📁 환경변수 적용: PILLSNAP_DATA_ROOT = {data_root}")
            
        # 실험 디렉토리 오버라이드  
        if exp_dir := os.getenv("PILLSNAP_EXP_DIR"):
            if "paths" not in config:
                config["paths"] = {}
            config["paths"]["exp_dir"] = exp_dir
            print(f"📁 환경변수 적용: PILLSNAP_EXP_DIR = {exp_dir}")
            
        return config
    
    def _validate_and_normalize_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        경로 설정 검증 및 정규화
        - WSL 절대 경로만 허용 (/mnt/...)
        - Windows 경로 (C:\) 금지
        """
        paths_to_check = [
            ("data", "root"),
            ("paths", "exp_dir"),
        ]
        
        for section, key in paths_to_check:
            if section in config and key in config[section]:
                path = config[section][key]
                
                # Windows 경로 금지
                if isinstance(path, str) and (path.startswith("C:") or "\\\\" in path):
                    raise ValueError(f"Windows 경로는 사용할 수 없습니다: {section}.{key} = {path}")
                
                # WSL 절대 경로 강제
                if isinstance(path, str) and not path.startswith("/"):
                    raise ValueError(f"절대 경로를 사용해야 합니다: {section}.{key} = {path}")
                    
                # 경로 정규화
                if isinstance(path, str):
                    config[section][key] = str(Path(path).resolve())
                    
        return config
    
    def _validate_required_settings(self, config: Dict[str, Any]) -> None:
        """
        필수 설정 항목 검증
        """
        required_sections = [
            "progressive_validation",
            "pipeline", 
            "data",
            "paths"
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"필수 설정 섹션이 누락되었습니다: {section}")
                
        # Progressive Validation 설정 검증
        pv = config.get("progressive_validation", {})
        if not pv.get("enabled", False):
            raise ValueError("Progressive Validation이 비활성화되어 있습니다")
            
        current_stage = pv.get("current_stage")
        if current_stage not in [1, 2, 3, 4]:
            raise ValueError(f"잘못된 current_stage 값: {current_stage} (1-4 범위)")
    
    def _check_duplicate_yaml_keys(self, config_file_path: Path) -> None:
        """
        YAML 파일의 중복 키 탐지 (1단계 강화)
        발견 시 실패 또는 강경 경고 후 중단
        """
        with open(config_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 루트 레벨 키들을 찾기 위한 정규식
        # 주석과 들여쓰기가 없는 키들만 탐지
        root_key_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*):.*$', re.MULTILINE)
        root_keys = root_key_pattern.findall(content)
        
        # 중복 키 탐지
        key_counts = defaultdict(int)
        for key in root_keys:
            key_counts[key] += 1
        
        duplicate_keys = [key for key, count in key_counts.items() if count > 1]
        
        if duplicate_keys:
            raise ValueError(
                f"🚨 YAML 중복 루트 키 발견: {duplicate_keys}\n"
                f"파일: {config_file_path}\n"
                f"중복 키를 제거하거나 병합한 후 다시 시도하세요."
            )
        
        self.merge_log.append(f"✅ YAML 중복 키 검사 통과: {len(root_keys)}개 루트 키")
    
    def _apply_merge_hierarchy(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        머지 순서 적용: base < stage_overrides[current_stage] < CLI
        """
        config = base_config.copy()
        
        # 1) Stage Override 적용
        pv_config = config.get("progressive_validation", {})
        current_stage = pv_config.get("current_stage")
        
        if current_stage and "stage_overrides" in config:
            stage_key = f"stage_{current_stage}"
            stage_overrides = config["stage_overrides"].get(stage_key, {})
            
            if stage_overrides:
                config = self._deep_merge(config, stage_overrides)
                self.merge_log.append(f"🔄 Stage {current_stage} overrides 적용: {len(stage_overrides)} 키")
        
        # 2) CLI Override 적용 (최우선)
        if self.cli_overrides:
            config = self._deep_merge(config, self.cli_overrides)
            self.merge_log.append(f"🔧 CLI overrides 적용: {len(self.cli_overrides)} 키")
        
        return config
    
    def _resolve_policy_conflicts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        정책 충돌 자동 해소
        - copy-paste: 램프 정책이 있으면 고정 확률 무시
        - confidence: auto 튜닝 값이 있으면 하드코딩 무시
        """
        # Copy-Paste 정책 충돌 해소
        if self._has_copy_paste_ramp_policy(config):
            if self._remove_copy_paste_fixed_values(config):
                self.merge_log.append("🔧 Copy-Paste: 램프 정책 우선, 고정 확률 제거")
        
        # Confidence 정책 충돌 해소
        if self._has_confidence_auto_tuning(config):
            if self._remove_confidence_hardcoded_values(config):
                self.merge_log.append("🔧 Confidence: 자동 튜닝 우선, 하드코딩 값 제거")
        
        return config
    
    def _save_config_snapshot(self, config: Dict[str, Any]) -> None:
        """
        최종 머지 결과 스냅샷 저장 (재현성/디버깅)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # artifacts/config_snapshots 디렉토리 생성
        snapshot_dir = self.project_root / "artifacts" / "config_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # 스냅샷 파일 경로
        snapshot_file = snapshot_dir / f"config_merged_{timestamp}.json"
        
        # 메타데이터 추가
        snapshot_data = {
            "timestamp": timestamp,
            "config_path": str(self.config_path),
            "cli_overrides": self.cli_overrides,
            "merge_log": self.merge_log,
            "final_config": config
        }
        
        # JSON으로 저장
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📸 설정 스냅샷 저장: {snapshot_file}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """딥 머지 유틸리티"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _has_copy_paste_ramp_policy(self, config: Dict[str, Any]) -> bool:
        """Copy-Paste 램프 정책 존재 여부 확인"""
        try:
            augmentation = config.get("data", {}).get("augmentation", {})
            copy_paste = augmentation.get("copy_paste", {})
            return "ramp_schedule" in copy_paste
        except (KeyError, TypeError):
            return False
    
    def _remove_copy_paste_fixed_values(self, config: Dict[str, Any]) -> bool:
        """Copy-Paste 고정 확률값 제거"""
        try:
            copy_paste = config["data"]["augmentation"]["copy_paste"]
            removed = False
            
            # 고정 확률 키들 제거
            fixed_keys = ["probability", "fixed_prob", "static_prob"]
            for key in fixed_keys:
                if key in copy_paste:
                    del copy_paste[key]
                    removed = True
            
            return removed
        except (KeyError, TypeError):
            return False
    
    def _has_confidence_auto_tuning(self, config: Dict[str, Any]) -> bool:
        """Confidence 자동 튜닝 존재 여부 확인"""
        try:
            logging_config = config.get("logging", {})
            confidence_tuning = logging_config.get("confidence_tuning", {})
            return confidence_tuning.get("enabled", False)
        except (KeyError, TypeError):
            return False
    
    def _remove_confidence_hardcoded_values(self, config: Dict[str, Any]) -> bool:
        """하드코딩된 Confidence 값들 제거"""
        removed = False
        
        try:
            # Detection 설정에서 하드코딩 confidence 제거
            det_config = config.get("models", {}).get("detector", {})
            if "confidence_threshold" in det_config:
                del det_config["confidence_threshold"]
                removed = True
            
            # Inference 설정에서 하드코딩 confidence 제거 (auto tuning 결과 사용)
            inf_config = config.get("inference", {})
            if "confidence" in inf_config:
                del inf_config["confidence"]
                removed = True
            
            return removed
        except (KeyError, TypeError):
            return False


class ConfigProvider:
    """
    Singleton 설정 제공자
    - 전역 단일 인스턴스로 설정 관리
    - Thread-safe 구현
    - 런타임 오버라이드 지원
    - 설정 변경 추적
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._config = None
        self._config_path = "config.yaml"
        self._cli_overrides = {}
        self._runtime_overrides = {}
        self._loader = None
        self._change_history = []
        
    def load(self, config_path: str = "config.yaml", cli_overrides: Optional[Dict[str, Any]] = None) -> None:
        """
        설정 초기 로드
        
        Args:
            config_path: config.yaml 파일 경로
            cli_overrides: CLI 인자 오버라이드
        """
        self._config_path = config_path
        self._cli_overrides = cli_overrides or {}
        
        # ConfigLoader 사용하여 로드
        self._loader = ConfigLoader(config_path, cli_overrides)
        self._config = self._loader._load_config_instance()
        
        # 변경 이력 기록
        self._change_history.append({
            "timestamp": datetime.now(KST).isoformat(),
            "action": "initial_load",
            "config_path": config_path,
            "cli_overrides": cli_overrides
        })
        
        print(f"✅ ConfigProvider 초기화 완료: {config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        중첩된 키 경로로 설정값 가져오기
        
        Args:
            key_path: 점(.)으로 구분된 키 경로 (예: "models.classifier.lr")
            default: 기본값
            
        Returns:
            설정값 또는 기본값
        """
        if self._config is None:
            self.load()  # 자동 로드
        
        # 런타임 오버라이드 우선 확인
        if key_path in self._runtime_overrides:
            return self._runtime_overrides[key_path]
        
        # 중첩 키 탐색
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any, persist: bool = False) -> None:
        """
        런타임 설정값 오버라이드
        
        Args:
            key_path: 점(.)으로 구분된 키 경로
            value: 설정할 값
            persist: config.yaml에 영구 저장 여부
        """
        # 런타임 오버라이드 저장
        self._runtime_overrides[key_path] = value
        
        # 변경 이력 기록
        self._change_history.append({
            "timestamp": datetime.now(KST).isoformat(),
            "action": "runtime_override",
            "key_path": key_path,
            "value": value,
            "persist": persist
        })
        
        # 영구 저장 옵션
        if persist:
            self._persist_to_yaml(key_path, value)
        
        print(f"🔧 설정 오버라이드: {key_path} = {value}")
    
    def _persist_to_yaml(self, key_path: str, value: Any) -> None:
        """config.yaml에 변경사항 저장"""
        # 구현 예정: YAML 파일 업데이트
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """전체 설정 딕셔너리 반환"""
        if self._config is None:
            self.load()
        
        # 런타임 오버라이드 적용된 설정 반환
        config_copy = self._config.copy()
        
        # 런타임 오버라이드 적용
        for key_path, value in self._runtime_overrides.items():
            keys = key_path.split('.')
            target = config_copy
            
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            target[keys[-1]] = value
        
        return config_copy
    
    def reload(self) -> None:
        """설정 재로드 (런타임 오버라이드 유지)"""
        runtime_overrides_backup = self._runtime_overrides.copy()
        
        self.load(self._config_path, self._cli_overrides)
        
        # 런타임 오버라이드 복원
        self._runtime_overrides = runtime_overrides_backup
        
        print("🔄 설정 재로드 완료 (런타임 오버라이드 유지)")
    
    def get_change_history(self) -> List[Dict[str, Any]]:
        """설정 변경 이력 반환"""
        return self._change_history.copy()
    
    def clear_runtime_overrides(self) -> None:
        """런타임 오버라이드 초기화"""
        self._runtime_overrides.clear()
        
        self._change_history.append({
            "timestamp": datetime.now(KST).isoformat(),
            "action": "clear_runtime_overrides"
        })
        
        print("🧹 런타임 오버라이드 초기화")


# 전역 ConfigProvider 인스턴스
config_provider = ConfigProvider()


def load_config(config_path: str = "config.yaml", cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    설정 파일 로딩 함수 (편의 함수, ConfigProvider 사용)
    
    Args:
        config_path: config.yaml 파일 경로
        cli_overrides: CLI 인자 오버라이드
        
    Returns:
        Dict: 설정 딕셔너리
    """
    config_provider.load(config_path, cli_overrides)
    return config_provider.get_config()


def get_git_sha() -> str:
    """
    현재 Git 커밋 SHA 추출
    
    Returns:
        str: Git SHA (7자리) 또는 "nogit"
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return "nogit"


def utc_timestamp() -> str:
    """
    UTC 타임스탬프 생성
    
    Returns:
        str: YYYYMMDD-HHMMSS 형식
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    재현 가능한 시드 설정
    
    Args:
        seed: 시드 값
        deterministic: 결정적 알고리즘 사용 여부 (성능 저하)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        # RTX 5080 추가 최적화
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    디렉토리 존재 보장 (없으면 생성)
    
    Args:
        path: 디렉토리 경로
        
    Returns:
        Path: 생성된 디렉토리 경로
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class PillSnapLogger:
    """
    PillSnap ML 전용 로깅 시스템
    - 콘솔 + 파일 동시 출력
    - 단계별 진행 상황 추적
    - 에러 디버깅용 상세 로그
    - 성능 메트릭 로깅
    """
    
    def __init__(self, name: str = "pillsnap", log_dir: Optional[str] = None, level: str = "info"):
        """
        Args:
            name: 로거 이름
            log_dir: 로그 파일 저장 디렉토리 (None이면 exp_dir/logs 사용)
            level: 로그 레벨 (debug/info/warning/error/critical)
        """
        self.name = name
        self.level = level.upper()
        
        # 로그 디렉토리 설정
        if log_dir is None:
            config = load_config()
            log_dir = Path(config["paths"]["exp_dir"]) / "logs"
        
        self.log_dir = ensure_dir(log_dir)
        
        # 로거 생성
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """로깅 시스템 설정"""
        
        # 기존 핸들러 제거 (중복 방지)
        logger = logging.getLogger(self.name)
        logger.handlers.clear()
        
        # 로그 레벨 설정
        log_level = getattr(logging, self.level, logging.INFO)
        logger.setLevel(log_level)
        
        # KST 시간대를 사용하는 커스텀 포맷터
        class KSTFormatter(logging.Formatter):
            """한국 시간대(KST)를 사용하는 로그 포맷터"""
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, tz=KST)
                if datefmt:
                    return dt.strftime(datefmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 포맷터 정의
        detailed_formatter = KSTFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = KSTFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 1) 콘솔 핸들러 (간단한 포맷)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # 2) 파일 핸들러 (상세한 포맷)
        # 로그 디렉토리가 없으면 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{self.name}_{datetime.now(tz=KST).strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 레벨 저장
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # 2-1) /tmp/stage3_training_output.log 파일 핸들러 추가 (모니터링용)
        if self.name in ["pillsnap", "__main__", "src.training.train_stage3_two_stage"]:
            monitor_log_file = Path("/tmp/stage3_training_output.log")
            monitor_handler = logging.FileHandler(monitor_log_file, encoding='utf-8', mode='a')
            monitor_handler.setLevel(logging.INFO)
            monitor_handler.setFormatter(simple_formatter)
            logger.addHandler(monitor_handler)
        
        # 3) 에러 전용 파일 핸들러
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now(tz=KST).strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # 4) Ultralytics YOLO 로거도 같은 핸들러로 리다이렉트
        if self.name in ["pillsnap", "__main__", "src.training.train_stage3_two_stage"]:
            # YOLO 로거 설정
            yolo_logger = logging.getLogger('ultralytics')
            yolo_logger.setLevel(logging.INFO)
            yolo_logger.handlers.clear()  # 기존 핸들러 제거
            
            # 모니터링 로그 파일로 리다이렉트
            monitor_log_file = Path("/tmp/stage3_training_output.log")
            yolo_monitor_handler = logging.FileHandler(monitor_log_file, encoding='utf-8', mode='a')
            yolo_monitor_handler.setLevel(logging.INFO)
            yolo_monitor_handler.setFormatter(simple_formatter)
            yolo_logger.addHandler(yolo_monitor_handler)
            
            # 콘솔 출력도 추가
            yolo_console_handler = logging.StreamHandler(sys.stdout)
            yolo_console_handler.setLevel(logging.INFO)
            yolo_console_handler.setFormatter(simple_formatter)
            yolo_logger.addHandler(yolo_console_handler)
        
        return logger
    
    def info(self, message: str, **kwargs) -> None:
        """정보 메시지 로깅"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """디버그 메시지 로깅"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """경고 메시지 로깅"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """에러 메시지 로깅"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """치명적 에러 메시지 로깅"""
        self.logger.critical(message, **kwargs)
    
    def step(self, step_name: str, message: str = "") -> None:
        """단계별 진행 상황 로깅 (눈에 띄는 포맷)"""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🚀 STEP: {step_name}")
        if message:
            self.logger.info(f"📝 {message}")
        self.logger.info(f"{separator}")
    
    def metric(self, name: str, value: float, unit: str = "", step: Optional[int] = None) -> None:
        """성능 메트릭 로깅"""
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"📊 METRIC: {name} = {value:.4f}{unit}{step_info}")
    
    def success(self, message: str) -> None:
        """성공 메시지 로깅"""
        self.logger.info(f"✅ SUCCESS: {message}")
    
    def failure(self, message: str) -> None:
        """실패 메시지 로깅"""
        self.logger.error(f"❌ FAILURE: {message}")
    
    def timer_start(self, operation: str) -> datetime:
        """타이머 시작"""
        start_time = datetime.now()
        self.logger.info(f"⏱️  START: {operation}")
        return start_time
    
    def timer_end(self, operation: str, start_time: datetime) -> float:
        """타이머 종료 및 경과 시간 로깅"""
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"⏱️  END: {operation} (took {elapsed:.2f}s)")
        return elapsed
    
    @staticmethod
    def get_logger(name: str = "pillsnap", log_dir: Optional[str] = None, level: str = "info") -> 'PillSnapLogger':
        """
        PillSnap 로거 생성 정적 메서드
        
        Args:
            name: 로거 이름
            log_dir: 로그 디렉토리 (None이면 기본값 사용)
            level: 로그 레벨
            
        Returns:
            PillSnapLogger: 로거 인스턴스
        """
        return PillSnapLogger(name=name, log_dir=log_dir, level=level)


# ConfigLoader 정적 메서드 추가
ConfigLoader.load_config_static = staticmethod(lambda config_path="config.yaml": ConfigLoader(config_path).load_config())


def build_logger(name: str = "pillsnap", log_dir: Optional[str] = None, level: str = "info") -> PillSnapLogger:
    """
    PillSnap 로거 생성 함수 (편의 함수)
    
    Args:
        name: 로거 이름
        log_dir: 로그 디렉토리 
        level: 로그 레벨
        
    Returns:
        PillSnapLogger: 설정된 로거 인스턴스
    """
    return PillSnapLogger(name=name, log_dir=log_dir, level=level)


if __name__ == "__main__":
    print("🧪 PillSnap ML 유틸리티 시스템 테스트")
    print("=" * 60)
    
    # 1) 설정 로딩 테스트
    print("\n1️⃣ 설정 파일 로딩 테스트")
    try:
        config = load_config()
        print("✅ config.yaml 로딩 성공")
        print(f"📊 Progressive Validation Stage: {config['progressive_validation']['current_stage']}")
        print(f"📁 데이터 루트: {config['data']['root']}")
        print(f"📁 실험 디렉토리: {config['paths']['exp_dir']}")
        print(f"🔀 파이프라인 모드: {config['pipeline']['mode']}")
        print(f"🏷️  Git SHA: {get_git_sha()}")
    except Exception as e:
        print(f"❌ 설정 로딩 실패: {e}")
        sys.exit(1)
    
    # 2) 로깅 시스템 테스트
    print("\n2️⃣ 로깅 시스템 테스트")
    try:
        logger = build_logger("test", level="info")
        
        logger.step("로깅 시스템 테스트", "모든 로그 레벨과 기능을 테스트합니다")
        
        # 기본 로그 레벨 테스트
        logger.info("일반 정보 메시지입니다")
        logger.warning("경고 메시지입니다")
        logger.error("에러 메시지입니다")
        
        # 특수 기능 테스트
        logger.success("성공적으로 완료되었습니다")
        logger.metric("accuracy", 0.9234, "%")
        logger.metric("loss", 0.1456, "", step=100)
        
        # 타이머 테스트
        import time
        start_time = logger.timer_start("샘플 작업")
        time.sleep(0.1)  # 0.1초 대기
        logger.timer_end("샘플 작업", start_time)
        
        logger.success("로깅 시스템 테스트 완료")
        print(f"✅ 로그 파일 저장 위치: {logger.log_dir}")
        
    except Exception as e:
        print(f"❌ 로깅 시스템 실패: {e}")
        sys.exit(1)
    
    # 3) 유틸리티 함수 테스트
    print("\n3️⃣ 유틸리티 함수 테스트")
    try:
        # 시드 설정 테스트
        set_seed(42, deterministic=False)
        print("✅ 시드 설정 완료")
        
        # 타임스탬프 테스트
        timestamp = utc_timestamp()
        print(f"✅ UTC 타임스탬프: {timestamp}")
        
        # 디렉토리 생성 테스트
        test_dir = ensure_dir("/tmp/pillsnap_test")
        print(f"✅ 테스트 디렉토리 생성: {test_dir}")
        
        print("✅ 모든 유틸리티 함수 테스트 통과")
        
    except Exception as e:
        print(f"❌ 유틸리티 함수 실패: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 PillSnap ML 유틸리티 시스템 테스트 완료!")
    print("   모든 기능이 정상적으로 작동합니다.")