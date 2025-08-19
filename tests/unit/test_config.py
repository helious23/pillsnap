"""
config.py 모듈 유닛 테스트

테스트 목적:
- config.yaml 파싱 및 기본값 적용 검증
- data.root 자동 보완 로직 검증  
- 환경변수 우선순위 확인
- Pydantic/dataclass 양쪽 모드 지원 확인
"""

from pathlib import Path
from unittest.mock import patch, mock_open

# 프로젝트 루트를 sys.path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfigLoading:
    """config.yaml 로딩 테스트"""
    
    def test_load_config_with_empty_file(self):
        """빈 config.yaml에서 기본값 사용"""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = config.load_config()
                
                # 기본값 확인
                assert ".jpg" in cfg.data.image_exts
                assert cfg.data.label_ext == ".json"
                assert cfg.preprocess.manifest_filename == "manifest_stage1.csv"
                assert cfg.preprocess.quarantine_dirname == "_quarantine"
                assert cfg.validation.enable_angle_rules is False
                assert cfg.validation.label_size_range is None
    
    def test_load_config_no_file(self):
        """config.yaml 없을 때 기본값 사용"""
        with patch("pathlib.Path.exists", return_value=False):
            cfg = config.load_config()
            
            # 기본값 확인
            assert ".jpg" in cfg.data.image_exts
            assert cfg.preprocess.manifest_filename == "manifest_stage1.csv"
            assert cfg.validation.enable_angle_rules is False
    
    def test_load_config_with_partial_data(self):
        """일부 설정만 있는 config.yaml 처리"""
        partial_config = """
        data:
          image_exts: [".png", ".tiff"]
        preprocess:
          manifest_filename: "custom_manifest.csv"
        """
        
        with patch("builtins.open", mock_open(read_data=partial_config)):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = config.load_config()
                
                # 설정된 값 확인
                assert cfg.data.image_exts == [".png", ".tiff"]
                assert cfg.preprocess.manifest_filename == "custom_manifest.csv"
                
                # 기본값 유지 확인
                assert cfg.data.label_ext == ".json"
                assert cfg.preprocess.quarantine_dirname == "_quarantine"
                assert cfg.validation.enable_angle_rules is False
    
    def test_load_config_invalid_yaml(self):
        """잘못된 YAML 파일에서 기본값으로 폴백"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = config.load_config()
                
                # 모든 값이 기본값이어야 함
                assert ".jpg" in cfg.data.image_exts
                assert cfg.preprocess.manifest_filename == "manifest_stage1.csv"


class TestDataRootHandling:
    """data.root 처리 로직 테스트"""
    
    def test_data_root_from_env_var(self, monkeypatch):
        """환경변수 PILLSNAP_DATA_ROOT가 설정된 경우"""
        custom_path = "/mnt/data/custom"
        monkeypatch.setenv("PILLSNAP_DATA_ROOT", custom_path)
        
        with patch("pathlib.Path.exists", return_value=False):
            cfg = config.load_config()
            assert cfg.data.root == custom_path
    
    def test_data_root_from_get_data_root(self, monkeypatch):
        """환경변수 미설정 시 paths.get_data_root() 사용"""
        # 환경변수 제거
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        # paths.get_data_root() mock
        mock_data_root = Path("/mnt/data/AIHub")
        with patch("paths.get_data_root", return_value=mock_data_root):
            with patch("pathlib.Path.exists", return_value=False):
                cfg = config.load_config()
                assert cfg.data.root == str(mock_data_root)
    
    def test_data_root_config_yaml_null(self, monkeypatch):
        """config.yaml에서 data.root가 null인 경우"""
        config_with_null_root = """
        data:
          root: null
          image_exts: [".jpg"]
        """
        
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        mock_data_root = Path("/mnt/data/fallback")
        
        with patch("builtins.open", mock_open(read_data=config_with_null_root)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("paths.get_data_root", return_value=mock_data_root):
                    cfg = config.load_config()
                    assert cfg.data.root == str(mock_data_root)
    
    def test_data_root_config_yaml_override(self, monkeypatch):
        """config.yaml에 data.root가 설정된 경우 환경변수보다 우선"""
        config_with_root = """
        data:
          root: "/config/specified/path"
          image_exts: [".jpg"]
        """
        
        monkeypatch.setenv("PILLSNAP_DATA_ROOT", "/env/path")
        
        with patch("builtins.open", mock_open(read_data=config_with_root)):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = config.load_config()
                # config.yaml의 값이 우선해야 함
                assert cfg.data.root == "/config/specified/path"


class TestValidationSettings:
    """검증 설정 테스트"""
    
    def test_validation_defaults(self):
        """검증 설정 기본값 확인"""
        with patch("pathlib.Path.exists", return_value=False):
            cfg = config.load_config()
            
            assert cfg.validation.enable_angle_rules is False
            assert cfg.validation.label_size_range is None
    
    def test_validation_custom_settings(self):
        """검증 설정 커스텀 값"""
        validation_config = """
        validation:
          enable_angle_rules: true
          label_size_range: [10, 1000]
        """
        
        with patch("builtins.open", mock_open(read_data=validation_config)):
            with patch("pathlib.Path.exists", return_value=True):
                cfg = config.load_config()
                
                assert cfg.validation.enable_angle_rules is True
                assert cfg.validation.label_size_range == (10, 1000)


class TestErrorHandling:
    """오류 처리 테스트"""
    
    def test_get_data_root_failure_fallback(self, monkeypatch):
        """paths.get_data_root() 실패 시 폴백"""
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        with patch("paths.get_data_root", side_effect=Exception("Mock error")):
            with patch("pathlib.Path.exists", return_value=False):
                cfg = config.load_config()
                # 폴백 경로 확인
                assert cfg.data.root == "./data"
    
    def test_path_policy_validation_error(self):
        """경로 정책 검증 실패 시에도 설정 반환"""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("src.core.path_policy.PathPolicyValidator") as mock_validator:
                # 검증 실패 시뮬레이션
                mock_instance = mock_validator.return_value
                mock_instance.validate_path.side_effect = Exception("Validation error")
                
                cfg = config.load_config()
                # 설정은 여전히 반환되어야 함
                assert cfg.data.image_exts == [".jpg", ".jpeg", ".png"]


class TestDefaultConfig:
    """기본 설정 함수 테스트"""
    
    def test_get_default_config(self):
        """get_default_config() 함수 동작 확인"""
        cfg = config.get_default_config()
        
        assert cfg.data.root is None
        assert ".jpg" in cfg.data.image_exts
        assert cfg.data.label_ext == ".json"
        assert cfg.preprocess.manifest_filename == "manifest_stage1.csv"
        assert cfg.validation.enable_angle_rules is False


class TestDefaultModeAndPipelineMode:
    """default_mode와 pipeline_mode 테스트"""
    
    def test_default_mode_and_pipeline_mode_ok(self, monkeypatch):
        """루트 config.yaml에서 default_mode와 pipeline_mode 정상 로딩"""
        # 환경 정리
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        # 루트 config.yaml 사용
        cfg = config.load_config()
        
        # pipeline_mode, default_mode가 유효한 값인지 확인
        assert cfg.data.pipeline_mode in {"single", "combo"}
        assert cfg.data.default_mode in {"single", "combo"}
    
    def test_default_mode_only_fallback(self, tmp_path, monkeypatch):
        """default_mode만 제공되고 pipeline_mode 누락 시 기본값 사용"""
        # 임시 config.yaml 생성
        config_content = """
        data:
          default_mode: "combo"
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        # 임시 디렉토리로 변경
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        cfg = config.load_config()
        
        # 기대값 확인
        assert cfg.data.default_mode == "combo"
        assert cfg.data.pipeline_mode == "single"  # 명시 없으면 기본값
    
    def test_invalid_modes_are_sanitized(self, tmp_path, monkeypatch, capsys):
        """잘못된 모드 값들이 정정되는지 확인"""
        # 임시 config.yaml 생성
        config_content = """
        data:
          pipeline_mode: "foobar"
          default_mode: "zzz"
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        # 임시 디렉토리로 변경
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        cfg = config.load_config()
        
        # 두 값 모두 "single"로 정정되어야 함
        assert cfg.data.pipeline_mode == "single"
        assert cfg.data.default_mode == "single"
        
        # 경고 메시지 확인
        captured = capsys.readouterr()
        assert "invalid value for data.pipeline_mode" in captured.out
        assert "invalid value for data.default_mode" in captured.out
    
    def test_extra_keys_ignored(self, tmp_path, monkeypatch):
        """알 수 없는 키가 무시되는지 확인"""
        # 임시 config.yaml 생성
        config_content = """
        data:
          pipeline_mode: "single"
          default_mode: "single"
          extra_foo: 1
          unknown_bar: "test"
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        # 임시 디렉토리로 변경
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PILLSNAP_DATA_ROOT", raising=False)
        
        # 예외 없이 통과해야 함
        cfg = config.load_config()
        
        # 정상 설정은 유지
        assert cfg.data.pipeline_mode == "single"
        assert cfg.data.default_mode == "single"
        
        # extra 속성은 없어야 함
        assert not hasattr(cfg.data, 'extra_foo')
        assert not hasattr(cfg.data, 'unknown_bar')


class TestIntegration:
    """통합 테스트"""
    
    def test_config_types_consistency(self):
        """설정 객체 타입 일관성 확인"""
        cfg = config.load_config()
        
        # 타입 확인
        assert isinstance(cfg.data.image_exts, list)
        assert isinstance(cfg.data.label_ext, str)
        assert isinstance(cfg.preprocess.manifest_filename, str)
        assert isinstance(cfg.validation.enable_angle_rules, bool)
        
        # data.root는 None이거나 문자열
        assert cfg.data.root is None or isinstance(cfg.data.root, str)
        
        # 새로운 필드들 타입 확인
        assert isinstance(cfg.data.pipeline_mode, str)
        assert isinstance(cfg.data.default_mode, str)
    
    def test_pydantic_fallback_behavior(self):
        """Pydantic 사용 가능 여부와 관계없이 동작"""
        # 현재 V2 상태 확인
        original_v2 = config.V2
        
        try:
            # V2 있는 경우와 없는 경우 모두 테스트
            for v2_state in [True, False, None]:
                with patch.object(config, 'V2', v2_state):
                    cfg = config.load_config()
                    
                    # 기본 기능 동작 확인
                    assert hasattr(cfg, 'data')
                    assert hasattr(cfg, 'preprocess')
                    assert hasattr(cfg, 'validation')
                    
                    # 새로운 필드들도 확인
                    assert hasattr(cfg.data, 'pipeline_mode')
                    assert hasattr(cfg.data, 'default_mode')
                    
        finally:
            # 원래 상태 복원
            config.V2 = original_v2