"""
경로 유틸리티 테스트

테스트 목적:
- is_wsl() WSL 환경 감지 정확성 확인
- get_data_root() 우선순위 로직 검증
- norm() 경로 정규화 기능 확인
- 환경변수 설정/해제에 따른 동작 검증
"""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open

# 프로젝트 루트를 sys.path에 추가 (import를 위해)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import paths


class TestWSLDetection:
    """WSL 환경 감지 테스트"""
    
    def test_is_wsl_with_microsoft_in_osrelease(self):
        """microsoft가 포함된 osrelease 파일에서 WSL로 감지"""
        mock_content = "4.19.104-microsoft-standard\n"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            assert paths.is_wsl() is True
    
    def test_is_wsl_without_microsoft_in_osrelease(self):
        """microsoft가 없는 osrelease 파일에서 일반 Linux로 감지"""
        mock_content = "5.15.0-generic\n"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            assert paths.is_wsl() is False
    
    def test_is_wsl_file_not_found(self):
        """osrelease 파일이 없는 경우 False 반환"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert paths.is_wsl() is False


class TestDataRoot:
    """데이터 루트 경로 테스트"""
    
    def test_get_data_root_with_env_var(self):
        """환경변수가 설정된 경우 해당 값 사용"""
        test_path = "/tmp/test_data"
        with patch.dict(os.environ, {'PILLSNAP_DATA_ROOT': test_path}):
            result = paths.get_data_root()
            assert str(result) == test_path
    
    def test_get_data_root_wsl_default(self):
        """WSL 환경에서 환경변수 미설정 시 기본값 사용"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('paths.is_wsl', return_value=True):
                result = paths.get_data_root()
                assert str(result) == '/mnt/data/AIHub'
    
    def test_get_data_root_non_wsl_default(self):
        """비WSL 환경에서 환경변수 미설정 시 기본값 사용"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('paths.is_wsl', return_value=False):
                result = paths.get_data_root()
                assert result.name == 'data'  # ./data의 마지막 부분
    
    def test_get_data_root_env_priority(self):
        """환경변수가 WSL 감지보다 우선"""
        custom_path = "/custom/data/path"
        with patch.dict(os.environ, {'PILLSNAP_DATA_ROOT': custom_path}):
            with patch('paths.is_wsl', return_value=True):
                result = paths.get_data_root()
                assert str(result) == custom_path
    
    def test_get_data_root_env_unset_cases(self):
        """PILLSNAP_DATA_ROOT 미설정 케이스 검증"""
        # 환경변수가 완전히 제거된 상태에서 테스트
        env_backup = os.environ.get('PILLSNAP_DATA_ROOT')
        try:
            # 환경변수 완전 제거
            if 'PILLSNAP_DATA_ROOT' in os.environ:
                del os.environ['PILLSNAP_DATA_ROOT']
            
            # WSL 환경에서 기본값
            with patch('paths.is_wsl', return_value=True):
                result = paths.get_data_root()
                assert str(result) == '/mnt/data/AIHub'
            
            # 비WSL 환경에서 기본값
            with patch('paths.is_wsl', return_value=False):
                result = paths.get_data_root()
                assert result.name == 'data'
        finally:
            # 환경변수 복원
            if env_backup is not None:
                os.environ['PILLSNAP_DATA_ROOT'] = env_backup
    
    def test_get_data_root_priority_order(self):
        """우선순위 확인: 환경변수 > WSL 기본값 > 로컬 ./data"""
        # 1. 환경변수 최우선
        env_path = "/priority/env/path"
        with patch.dict(os.environ, {'PILLSNAP_DATA_ROOT': env_path}):
            with patch('paths.is_wsl', return_value=True):
                result = paths.get_data_root()
                assert str(result) == env_path
        
        # 2. WSL 기본값 (환경변수 없음)
        with patch.dict(os.environ, {}, clear=True):
            with patch('paths.is_wsl', return_value=True):
                result = paths.get_data_root()
                assert str(result) == '/mnt/data/AIHub'
        
        # 3. 로컬 ./data (환경변수 없음, 비WSL)
        with patch.dict(os.environ, {}, clear=True):
            with patch('paths.is_wsl', return_value=False):
                result = paths.get_data_root()
                assert result.name == 'data'


class TestPathNormalization:
    """경로 정규화 테스트"""
    
    def test_norm_relative_path(self):
        """상대경로를 절대경로로 변환"""
        result = paths.norm("./test/path")
        assert result.is_absolute()
        assert "test" in str(result)
        assert "path" in str(result)
    
    def test_norm_absolute_path(self):
        """절대경로 입력 시 정규화만 수행"""
        abs_path = "/tmp/absolute/path"
        result = paths.norm(abs_path)
        assert result.is_absolute()
        assert str(result).endswith("absolute/path")
    
    def test_norm_path_object(self):
        """Path 객체 입력도 처리"""
        path_obj = Path("./some/path")
        result = paths.norm(path_obj)
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_norm_current_directory(self):
        """현재 디렉토리 정규화"""
        result = paths.norm(".")
        assert result.is_absolute()
        assert result.exists()  # 현재 디렉토리는 존재해야 함
    
    def test_norm_unix_slash_handling(self):
        """유닉스 스타일 슬래시 처리 검증"""
        unix_path = "/mnt/data/dataset/images"
        result = paths.norm(unix_path)
        assert result.is_absolute()
        assert "/" in str(result)  # 유닉스 스타일 슬래시 유지
        assert str(result) == unix_path  # 변화 없이 유지
    
    def test_norm_windows_backslash_simulation(self):
        """Windows 백슬래시 경로 시뮬레이션 (WSL에서 처리)"""
        # Windows 스타일 경로를 입력했을 때의 동작
        windows_style = r"C:\Users\data\dataset"
        
        # WSL 환경에서는 이런 경로가 그대로 처리될 수 있음
        # 하지만 norm 함수는 현재 시스템에 맞게 정규화
        result = paths.norm(windows_style)
        assert result.is_absolute()
        # WSL에서는 / 스타일로 변환됨
        assert "/" in str(result)
    
    def test_norm_mixed_separators(self):
        """혼합된 경로 구분자 처리"""
        mixed_path = "./data\\subfolder/images"
        result = paths.norm(mixed_path)
        assert result.is_absolute()
        # 정규화 후 일관된 구분자 사용
        normalized_str = str(result)
        # WSL/Linux에서는 / 구분자 사용
        assert "/" in normalized_str
    
    def test_norm_drive_letter_simulation(self):
        """드라이브 문자 경로 시뮬레이션"""
        # Windows 드라이브 문자 스타일
        drive_paths = [
            "C:/data/dataset",
            "D:\\images\\train",
            "/mnt/c/data/dataset"  # WSL 마운트 스타일
        ]
        
        for test_path in drive_paths:
            result = paths.norm(test_path)
            assert result.is_absolute()
            assert isinstance(result, Path)


class TestIntegration:
    """통합 테스트"""
    
    def test_all_functions_return_path_objects(self):
        """모든 함수가 적절한 타입 반환"""
        # get_data_root는 Path 반환
        data_root = paths.get_data_root()
        assert isinstance(data_root, Path)
        
        # norm은 Path 반환
        normalized = paths.norm("./test")
        assert isinstance(normalized, Path)
        
        # is_wsl은 bool 반환
        wsl_status = paths.is_wsl()
        assert isinstance(wsl_status, bool)
    
    def test_functions_work_without_errors(self):
        """모든 함수가 예외 없이 실행"""
        try:
            paths.is_wsl()
            paths.get_data_root()
            paths.norm("./test")
        except Exception as e:
            pytest.fail(f"Functions should not raise exceptions: {e}")