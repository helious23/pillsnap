"""
경로 유틸리티 모듈

목적: path_policy.py 규칙을 준수하는 안전한 경로 관리 유틸리티 제공
입력: 
    - 환경 변수 PILLSNAP_DATA_ROOT (선택적)
    - 문자열 또는 Path 객체 경로
출력:
    - WSL 환경 여부 판단 (bool)
    - 검증된 데이터 루트 경로 (Path)
    - 정규화된 절대 경로 (Path)
검증 포인트:
    - PathPolicyValidator를 통한 경로 정책 준수 확인
    - WSL vs 일반 Linux 환경 구분
    - 환경 변수 우선순위 적용
"""

import os
from pathlib import Path
from src.core.path_policy import PathPolicyValidator


def is_wsl() -> bool:
    """
    WSL 환경 여부 확인
    
    Returns:
        bool: WSL 환경이면 True, 그렇지 않으면 False
    """
    try:
        with open('/proc/sys/kernel/osrelease', 'r') as f:
            content = f.read().lower()
            return 'microsoft' in content
    except (FileNotFoundError, PermissionError):
        return False


def get_data_root() -> Path:
    """
    데이터 루트 경로 반환 (우선순위 적용)
    
    우선순위:
    1. 환경변수 PILLSNAP_DATA_ROOT
    2. WSL 환경: /mnt/data/AIHub
    3. 기타 환경: ./data
    
    Returns:
        Path: 검증된 데이터 루트 경로
    """
    # 1. 환경변수 우선
    env_root = os.getenv('PILLSNAP_DATA_ROOT')
    if env_root:
        data_root = Path(env_root)
        print(f"Using PILLSNAP_DATA_ROOT: {data_root}")
    else:
        # 2. WSL vs 기타 환경
        if is_wsl():
            data_root = Path('/mnt/data/AIHub')
            print(f"WSL detected, using default: {data_root}")
        else:
            data_root = Path('./data')
            print(f"Non-WSL environment, using default: {data_root}")
    
    # 3. 경로 정책 검증 (로그만, 실패해도 경로 반환)
    validator = PathPolicyValidator()
    valid, message = validator.validate_path(str(data_root), purpose="data")
    
    if valid:
        print(f"Path policy validation passed: {message}")
    else:
        print(f"Path policy validation warning: {message}")
    
    return data_root


def norm(p: str | Path) -> Path:
    """
    경로 정규화 및 절대경로 변환
    
    Args:
        p: 정규화할 경로 (문자열 또는 Path 객체)
        
    Returns:
        Path: 정규화된 절대경로
    """
    path = Path(p)
    return path.resolve()