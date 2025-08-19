"""
엔트리포인트 테스트 (pillsnap.stage1.verify, pillsnap.stage1.run)

목적: Stage 1 CLI 엔트리포인트의 임포트/실행/산출물 검증
테스트 범위:
- 모듈 임포트 확인
- 환경변수 기반 실행
- 빠른 스모크 테스트 (limit=50)
- 산출물 스키마 검증
"""

import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch


class TestEntrypoints:
    """엔트리포인트 테스트"""
    
    def test_import_entrypoints(self):
        """모듈 임포트 확인"""
        try:
            from pillsnap.stage1 import verify, run
            from pillsnap.stage1.verify import main as verify_main  
            from pillsnap.stage1.run import main as run_main
            
            # 함수 존재 확인
            assert callable(verify_main)
            assert callable(run_main)
            
        except ImportError as e:
            pytest.fail(f"엔트리포인트 임포트 실패: {e}")
    
    @patch.dict(os.environ, {"PILLSNAP_DATA_ROOT": "/mnt/data/pillsnap_dataset/data"})
    def test_verify_smoke_test(self, tmp_path):
        """verify 스모크 테스트"""
        # 실제 데이터 루트가 없으면 스킵
        data_root = os.environ.get("PILLSNAP_DATA_ROOT")
        if not Path(data_root).exists():
            pytest.skip(f"데이터 루트 없음: {data_root}")
        
        from pillsnap.stage1.verify import main as verify_main
        
        # 아주 빠른 스모크 (30초 제한, 50개 샘플)
        try:
            exit_code = verify_main(max_seconds=30, sample_limit=50)
            # 데이터가 있으면 성공(0), 없어도 실행은 되어야 함
            assert exit_code in [0, 1]
            
        except Exception as e:
            # 예상치 못한 예외는 실패
            pytest.fail(f"verify 실행 중 예외: {e}")
    
    @patch.dict(os.environ, {"PILLSNAP_DATA_ROOT": "/mnt/data/pillsnap_dataset/data"})  
    def test_run_smoke_test(self, tmp_path):
        """run 스모크 테스트"""
        # 실제 데이터 루트가 없으면 스킵
        data_root = os.environ.get("PILLSNAP_DATA_ROOT")
        if not Path(data_root).exists():
            pytest.skip(f"데이터 루트 없음: {data_root}")
        
        from pillsnap.stage1.run import main as run_main
        
        # 임시 매니페스트 경로
        test_manifest = tmp_path / "manifest_test.csv"
        
        try:
            # 아주 빠른 스모크 (50개 샘플만)
            exit_code = run_main(limit=50, manifest=str(test_manifest))
            
            # 실행은 되어야 함
            assert exit_code in [0, 1]
            
            # 매니페스트 생성 확인 (실행이 성공했다면)
            if exit_code == 0 and test_manifest.exists():
                self._verify_manifest_schema(test_manifest)
            
        except Exception as e:
            pytest.fail(f"run 실행 중 예외: {e}")
    
    def _verify_manifest_schema(self, manifest_path: Path):
        """매니페스트 CSV 스키마 검증"""
        try:
            df = pd.read_csv(manifest_path)
            
            # 필수 컬럼 확인
            required_columns = ['image_path', 'label_path', 'code', 'is_pair']
            missing_columns = set(required_columns) - set(df.columns)
            
            assert len(missing_columns) == 0, f"누락된 컬럼: {missing_columns}"
            
            # 데이터 타입 검증 (행이 있는 경우만)
            if len(df) > 0:
                assert df['image_path'].dtype == 'object'
                assert df['label_path'].dtype == 'object' 
                assert df['code'].dtype == 'object'
                assert df['is_pair'].dtype in ['bool', 'object']  # bool 또는 문자열
            
        except Exception as e:
            pytest.fail(f"매니페스트 스키마 검증 실패: {e}")
    
    def test_schema_validation_empty_manifest(self, tmp_path):
        """빈 매니페스트 스키마 검증"""
        # 빈 매니페스트 생성
        empty_manifest = tmp_path / "empty_manifest.csv"
        
        # 헤더만 있는 CSV
        df_empty = pd.DataFrame(columns=['image_path', 'label_path', 'code', 'is_pair'])
        df_empty.to_csv(empty_manifest, index=False)
        
        # 스키마 검증 (행이 0개여도 통과해야 함)
        self._verify_manifest_schema(empty_manifest)
    
    @patch.dict(os.environ, {"PILLSNAP_DATA_ROOT": "/nonexistent/path"})
    def test_invalid_data_root_handling(self):
        """존재하지 않는 데이터 루트 처리"""
        from pillsnap.stage1.verify import main as verify_main
        
        # 존재하지 않는 경로로 실행 → 실패 코드 1 반환해야 함
        exit_code = verify_main(max_seconds=5, sample_limit=10)
        assert exit_code == 1
    
    def test_environment_variable_priority(self):
        """환경변수 우선순위 확인"""
        test_path = "/test/custom/path"
        
        with patch.dict(os.environ, {"PILLSNAP_DATA_ROOT": test_path}):
            import config
            cfg = config.load_config()
            
            # 환경변수가 우선 적용되어야 함
            assert cfg.data.root == test_path


if __name__ == "__main__":
    # 스탠드얼론 실행
    print("🧪 엔트리포인트 테스트 실행...")
    
    # 기본 임포트 테스트
    try:
        from pillsnap.stage1 import verify, run
        print("✅ 임포트 성공")
    except ImportError as e:
        print(f"❌ 임포트 실패: {e}")
        exit(1)
    
    # 환경변수 확인
    data_root = os.environ.get("PILLSNAP_DATA_ROOT")
    if data_root and Path(data_root).exists():
        print(f"✅ 데이터 루트 확인: {data_root}")
    else:
        print(f"⚠️  데이터 루트 없음: {data_root}")
        print("💡 설정 방법: export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset/data")
    
    print("✅ 기본 검증 완료")
    print("\n전체 테스트 실행:")
    print("  pytest tests/test_entrypoints.py -v")