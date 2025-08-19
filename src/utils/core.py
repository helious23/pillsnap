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
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone


class ConfigLoader:
    """
    설정 파일 로더 클래스
    - config.yaml 안전 로딩
    - 환경변수 오버라이드 지원
    - 경로 검증
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: config.yaml 파일 경로 (프로젝트 루트 기준)
        """
        self.config_path = config_path
        self.project_root = Path("/home/max16/pillsnap")
        
    def load_config(self) -> Dict[str, Any]:
        """
        config.yaml 파일을 로딩하고 검증합니다.
        
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
            
        # 2) YAML 파일 로딩
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML 파싱 오류: {e}")
            
        # 3) 환경변수 오버라이드 적용
        config = self._apply_env_overrides(config)
        
        # 4) 경로 검증 및 정규화
        config = self._validate_and_normalize_paths(config)
        
        # 5) 필수 설정 검증
        self._validate_required_settings(config)
        
        return config
    
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


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    설정 파일 로딩 함수 (편의 함수)
    
    Args:
        config_path: config.yaml 파일 경로
        
    Returns:
        Dict: 설정 딕셔너리
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()


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
        
        # 포맷터 정의
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
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
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 파일에는 모든 레벨 저장
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # 3) 에러 전용 파일 핸들러
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
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