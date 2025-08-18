"""
WSL/Windows Path Policy Validation
WSL/Windows 경로 정책 검증 및 변환
"""

import os
import re
import platform
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path, PureWindowsPath, PurePosixPath
from enum import Enum

logger = logging.getLogger(__name__)


class PathContext(Enum):
    """경로 컨텍스트"""
    WSL = "wsl"              # WSL 내부 실행
    WINDOWS = "windows"      # Windows 네이티브 실행
    DOCKER = "docker"        # Docker 컨테이너
    UNKNOWN = "unknown"


class PathType(Enum):
    """경로 타입"""
    WSL_NATIVE = "wsl_native"       # /mnt/...
    WINDOWS_NATIVE = "windows"      # C:\...
    WSL_HOME = "wsl_home"          # /home/...
    RELATIVE = "relative"           # ./...
    INVALID = "invalid"


class PathPolicyValidator:
    """
    경로 정책 검증 및 변환 관리자
    WSL과 Windows 경로를 안전하게 관리
    """
    
    # 금지된 Windows 경로 패턴
    FORBIDDEN_WINDOWS_PATTERNS = [
        r'^[A-Za-z]:\\',           # C:\ 형식
        r'^\\\\',                   # UNC 경로
        r'^[A-Za-z]:/',            # C:/ 형식 (혼용)
    ]
    
    # 허용된 Windows 도구 경로 (예외)
    ALLOWED_WINDOWS_TOOLS = [
        r'C:\\ProgramData\\Cloudflare',
        r'C:\\Program Files\\Cloudflare',
        r'C:\\Scripts',
    ]
    
    # WSL 경로 패턴
    WSL_PATTERNS = {
        'mnt': r'^/mnt/[a-z]/.*',      # /mnt/c/...
        'home': r'^/home/.*',          # /home/user/...
        'root': r'^/.*',               # 기타 Linux 경로
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 경로 정책 설정
        """
        self.config = config or {}
        self.context = self._detect_context()
        self.violations = []
        
        logger.info(f"PathPolicyValidator initialized: context={self.context.value}")
    
    def _detect_context(self) -> PathContext:
        """실행 컨텍스트 감지"""
        system = platform.system().lower()
        
        # WSL 감지
        if 'microsoft' in platform.release().lower():
            return PathContext.WSL
        
        # Windows 감지
        if system == 'windows':
            return PathContext.WINDOWS
        
        # Docker 감지
        if os.path.exists('/.dockerenv'):
            return PathContext.DOCKER
        
        # Linux/Mac
        if system in ['linux', 'darwin']:
            return PathContext.WSL  # WSL과 동일하게 처리
        
        return PathContext.UNKNOWN
    
    def validate_path(self, path: str, purpose: str = "general") -> Tuple[bool, str]:
        """
        경로 검증
        
        Args:
            path: 검증할 경로
            purpose: 경로 용도 (general, cloudflare, data, code)
            
        Returns:
            (유효 여부, 메시지)
        """
        path_type = self._identify_path_type(path)
        
        # WSL 컨텍스트에서의 검증
        if self.context == PathContext.WSL:
            # Windows 경로 사용 금지 (예외 제외)
            if path_type == PathType.WINDOWS_NATIVE:
                if purpose == "cloudflare" and self._is_allowed_windows_tool(path):
                    return True, "Allowed Windows tool path"
                
                msg = f"Windows path forbidden in WSL context: {path}"
                self._log_violation(path, msg)
                return False, msg
            
            # WSL 경로 권장
            if path_type in [PathType.WSL_NATIVE, PathType.WSL_HOME]:
                return True, "Valid WSL path"
            
            # 상대 경로 허용
            if path_type == PathType.RELATIVE:
                return True, "Relative path allowed"
        
        # Windows 컨텍스트에서의 검증
        elif self.context == PathContext.WINDOWS:
            # Cloudflare 도구는 Windows 경로 필수
            if purpose == "cloudflare":
                if path_type == PathType.WINDOWS_NATIVE:
                    return True, "Valid Windows path for Cloudflare"
                else:
                    return False, "Cloudflare requires Windows path"
            
            # 데이터/코드는 WSL 경로 권장
            if purpose in ["data", "code"]:
                if path_type == PathType.WSL_NATIVE:
                    logger.warning(f"WSL path in Windows context may need conversion: {path}")
        
        return True, "Path validation passed"
    
    def _identify_path_type(self, path: str) -> PathType:
        """경로 타입 식별"""
        # Windows 경로 체크
        for pattern in self.FORBIDDEN_WINDOWS_PATTERNS:
            if re.match(pattern, path):
                return PathType.WINDOWS_NATIVE
        
        # WSL 경로 체크
        if re.match(self.WSL_PATTERNS['mnt'], path):
            return PathType.WSL_NATIVE
        if re.match(self.WSL_PATTERNS['home'], path):
            return PathType.WSL_HOME
        
        # 상대 경로 체크
        if path.startswith('./') or path.startswith('../'):
            return PathType.RELATIVE
        
        # 절대 Linux 경로
        if path.startswith('/'):
            return PathType.WSL_HOME
        
        return PathType.INVALID
    
    def _is_allowed_windows_tool(self, path: str) -> bool:
        """허용된 Windows 도구 경로인지 확인"""
        for allowed_pattern in self.ALLOWED_WINDOWS_TOOLS:
            if path.startswith(allowed_pattern):
                return True
        return False
    
    def convert_path(self, path: str, target: PathContext) -> str:
        """
        경로 변환
        
        Args:
            path: 변환할 경로
            target: 대상 컨텍스트
            
        Returns:
            변환된 경로
        """
        path_type = self._identify_path_type(path)
        
        # WSL → Windows 변환
        if target == PathContext.WINDOWS:
            if path_type == PathType.WSL_NATIVE:
                # /mnt/c/... → C:\...
                match = re.match(r'^/mnt/([a-z])/(.*)', path)
                if match:
                    drive = match.group(1).upper()
                    rest = match.group(2).replace('/', '\\')
                    return f"{drive}:\\{rest}"
            
            elif path_type == PathType.WSL_HOME:
                # WSL 홈 디렉토리는 Windows에서 접근 가능
                # /home/user/... → \\wsl$\Ubuntu\home\user\...
                win_path = path.replace('/', '\\')
                return f"\\\\wsl$\\Ubuntu{win_path}"
        
        # Windows → WSL 변환
        elif target == PathContext.WSL:
            if path_type == PathType.WINDOWS_NATIVE:
                # C:\... → /mnt/c/...
                match = re.match(r'^([A-Za-z]):\\(.*)$', path)
                if match:
                    drive = match.group(1).lower()
                    rest = match.group(2).replace('\\', '/')
                    return f"/mnt/{drive}/{rest}"
        
        # 변환 불필요 또는 불가능
        return path
    
    def validate_config_paths(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        설정 파일의 모든 경로 검증
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            위반 사항 리스트
        """
        violations = []
        
        # 검증할 경로 키들
        path_keys = [
            ("paths.exp_dir", "data"),
            ("paths.data_root", "data"),
            ("data.root", "data"),
            ("data.train.single_images", "data"),
            ("data.val.single_images", "data"),
            ("export.out_dir", "data"),
            ("logging.windows_integration.cloudflared_config", "cloudflare"),
        ]
        
        for key_path, purpose in path_keys:
            value = self._get_nested_value(config, key_path)
            if value and isinstance(value, str):
                valid, msg = self.validate_path(value, purpose)
                if not valid:
                    violations.append({
                        "key": key_path,
                        "value": value,
                        "purpose": purpose,
                        "message": msg
                    })
        
        return violations
    
    def _get_nested_value(self, config: Dict, key_path: str) -> Any:
        """중첩된 딕셔너리 값 획득"""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value
    
    def _log_violation(self, path: str, message: str):
        """경로 정책 위반 로깅"""
        violation = {
            "path": path,
            "message": message,
            "context": self.context.value,
            "timestamp": None  # Logger가 추가
        }
        
        self.violations.append(violation)
        logger.error(f"[PATH_VIOLATION] {message}")
    
    def get_safe_paths(self) -> Dict[str, str]:
        """안전한 기본 경로 반환"""
        if self.context == PathContext.WSL:
            return {
                "code_root": "/home/max16/pillsnap",
                "data_root": "/mnt/data/pillsnap_dataset",
                "exp_dir": "/mnt/data/exp/exp01",
                "venv": "$HOME/pillsnap/.venv"
            }
        elif self.context == PathContext.WINDOWS:
            return {
                "cloudflare_config": "C:\\ProgramData\\Cloudflare\\cloudflared",
                "cloudflare_logs": "C:\\ProgramData\\Cloudflare\\cloudflared\\logs",
                "scripts": "C:\\Scripts"
            }
        else:
            return {}
    
    def ensure_wsl_path(self, path: str) -> str:
        """WSL 경로 보장 (필요시 변환)"""
        path_type = self._identify_path_type(path)
        
        if path_type == PathType.WINDOWS_NATIVE:
            converted = self.convert_path(path, PathContext.WSL)
            logger.warning(f"Converted Windows path to WSL: {path} → {converted}")
            return converted
        
        return path
    
    def ensure_windows_path(self, path: str) -> str:
        """Windows 경로 보장 (필요시 변환)"""
        path_type = self._identify_path_type(path)
        
        if path_type in [PathType.WSL_NATIVE, PathType.WSL_HOME]:
            converted = self.convert_path(path, PathContext.WINDOWS)
            logger.warning(f"Converted WSL path to Windows: {path} → {converted}")
            return converted
        
        return path
    
    def get_violations_summary(self) -> str:
        """위반 사항 요약"""
        if not self.violations:
            return "No path policy violations detected"
        
        summary = f"Found {len(self.violations)} path policy violations:\n"
        for v in self.violations:
            summary += f"  - {v['path']}: {v['message']}\n"
        
        return summary


def validate_project_paths(config: Dict[str, Any]) -> bool:
    """
    프로젝트 경로 정책 검증 헬퍼
    
    Args:
        config: 프로젝트 설정
        
    Returns:
        모든 경로가 유효한지 여부
    """
    validator = PathPolicyValidator(config)
    violations = validator.validate_config_paths(config)
    
    if violations:
        logger.error(f"Path policy violations found: {len(violations)}")
        for v in violations:
            logger.error(f"  {v['key']}: {v['message']}")
        return False
    
    logger.info("All paths validated successfully")
    return True


def get_wsl_safe_path(path: str) -> str:
    """
    WSL 안전 경로 반환 헬퍼
    
    Args:
        path: 원본 경로
        
    Returns:
        WSL 호환 경로
    """
    validator = PathPolicyValidator()
    return validator.ensure_wsl_path(path)