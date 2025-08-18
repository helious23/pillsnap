"""
dataset.preprocess 모듈 유닛 테스트

테스트 목적:
- preprocess 함수의 기본 동작 검증
- 경로 정규화 및 파일 존재성 확인
- 중복 코드 제거 로직 검증
- CSV 저장 및 DataFrame 일치성 확인
- 예외 상황 처리 검증
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.preprocess import preprocess, build_summary_dict, validate_preprocessed_data


class SimpleConfig:
    """테스트용 간단한 설정 객체"""
    def __init__(self, manifest_filename="manifest_stage1.csv", quarantine_dirname="_quarantine"):
        self.manifest_filename = manifest_filename
        self.quarantine_dirname = quarantine_dirname


class TestPreprocessBasicFlow:
    """preprocess 함수 기본 동작 테스트"""
    
    def test_preprocess_basic_flow(self, tmp_path):
        """기본 전처리 플로우 테스트 - 매칭/미매칭 파일 정확한 처리"""
        # 가짜 데이터셋 구성
        (tmp_path / "a.jpg").touch()
        (tmp_path / "a.json").touch()
        (tmp_path / "b.jpg").touch()  # 라벨 없음
        # c.json은 생성하지 않음 (이미지 없음)
        
        # DataFrame 구성 (절대경로 변환 전 상대경로로)
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "b.jpg"), "label_path": None, "code": "b", "is_pair": False},
            {"image_path": None, "label_path": str(tmp_path / "c.json"), "code": "c", "is_pair": False},
        ])
        
        # 설정 객체
        cfg = SimpleConfig()
        
        # artifacts 디렉토리 설정
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 반환 DataFrame 행 수 (a.jpg+a.json 매칭만 남아야 함)
        assert len(result_df) == 1, f"Expected 1 row, got {len(result_df)}"
        
        # 검증: code 값
        assert result_df["code"].tolist() == ["a"], f"Expected code 'a', got {result_df['code'].tolist()}"
        
        # 검증: 모든 경로가 절대경로인지 확인
        for idx, row in result_df.iterrows():
            if row["image_path"] is not None:
                assert Path(row["image_path"]).is_absolute(), f"image_path should be absolute: {row['image_path']}"
                assert isinstance(row["image_path"], str), "image_path should be string"
            
            if row["label_path"] is not None:
                assert Path(row["label_path"]).is_absolute(), f"label_path should be absolute: {row['label_path']}"
                assert isinstance(row["label_path"], str), "label_path should be string"
        
        # 검증: CSV 파일 저장 확인
        csv_path = artifacts_dir / "manifest_stage1.csv"
        assert csv_path.exists(), f"CSV file should be saved at {csv_path}"
        
        # 검증: 저장된 CSV 내용 확인
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(result_df), "Saved CSV should have same row count as returned DataFrame"
        assert saved_df["code"].tolist() == ["a"], "Saved CSV should contain correct code"
    
    def test_preprocess_duplicate_code(self, tmp_path):
        """중복 코드 제거 테스트"""
        # 동일한 코드를 가진 두 파일 세트 생성
        (tmp_path / "a1.jpg").touch()
        (tmp_path / "a1.json").touch()
        (tmp_path / "a2.jpg").touch()
        (tmp_path / "a2.json").touch()
        
        # DataFrame 구성 (같은 code="a")
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a1.jpg"), "label_path": str(tmp_path / "a1.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "a2.jpg"), "label_path": str(tmp_path / "a2.json"), "code": "a", "is_pair": True},
        ])
        
        cfg = SimpleConfig()
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 중복 제거 후 행 수
        assert len(result_df) == 1, f"Expected 1 row after deduplication, got {len(result_df)}"
        
        # 검증: 마지막 항목이 유지되었는지 확인 (keep='last')
        assert "a2.jpg" in result_df.iloc[0]["image_path"], "Should keep the last duplicate item"
        assert "a2.json" in result_df.iloc[0]["label_path"], "Should keep the last duplicate item"
    
    def test_preprocess_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError 발생 확인"""
        # 필수 컬럼 일부가 누락된 DataFrame
        df = pd.DataFrame([
            {"image_path": "/path/to/a.jpg", "code": "a", "is_pair": True},
            # label_path 컬럼 누락
        ])
        
        cfg = SimpleConfig()
        
        # ValueError 발생 확인
        with pytest.raises(ValueError) as exc_info:
            preprocess(df, cfg, artifacts_dir="test_artifacts")
        
        assert "Missing required columns" in str(exc_info.value)
        assert "label_path" in str(exc_info.value)
    
    def test_preprocess_keeps_only_existing_files(self, tmp_path):
        """실제 파일만 존재하는 경우만 유지하는지 확인"""
        # 이미지 파일만 생성 (라벨 파일은 생성하지 않음)
        (tmp_path / "exists.jpg").touch()
        # nonexistent.json은 생성하지 않음
        
        # DataFrame 구성 (존재하지 않는 라벨 파일 경로 포함)
        df = pd.DataFrame([
            {
                "image_path": str(tmp_path / "exists.jpg"), 
                "label_path": str(tmp_path / "nonexistent.json"),  # 실제로는 없는 파일
                "code": "test", 
                "is_pair": True
            },
        ])
        
        cfg = SimpleConfig()
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 존재하지 않는 라벨 파일로 인해 행이 제거되어야 함
        assert len(result_df) == 0, f"Expected 0 rows due to missing label file, got {len(result_df)}"


class TestPreprocessValidation:
    """전처리 검증 함수 테스트"""
    
    def test_validate_preprocessed_data_valid(self, tmp_path):
        """유효한 전처리 데이터 검증"""
        # 테스트 파일 생성
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.json").touch()
        
        # 유효한 DataFrame
        df = pd.DataFrame([
            {
                "image_path": str(tmp_path / "test.jpg"),
                "label_path": str(tmp_path / "test.json"),
                "code": "test",
                "is_pair": True
            }
        ])
        
        # 검증 실행
        result = validate_preprocessed_data(df)
        
        # 검증: 결과가 유효해야 함
        assert result["valid"] is True
        assert len(result["issues"]) == 0
    
    def test_validate_preprocessed_data_invalid(self):
        """유효하지 않은 전처리 데이터 검증"""
        # 필수 컬럼이 누락된 DataFrame
        df = pd.DataFrame([
            {"image_path": "/path/to/test.jpg", "code": "test"}
            # label_path, is_pair 컬럼 누락
        ])
        
        # 검증 실행
        result = validate_preprocessed_data(df)
        
        # 검증: 결과가 무효해야 함
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert any("Missing columns" in issue for issue in result["issues"])


class TestSummaryAndHelpers:
    """요약 및 헬퍼 함수 테스트"""
    
    def test_build_summary_dict(self):
        """요약 딕셔너리 생성 테스트"""
        summary = build_summary_dict(
            initial_count=100,
            final_count=85,
            missing_image=5,
            missing_label=3,
            duplicate_code=7
        )
        
        # 검증: 기본 통계
        assert summary["initial_count"] == 100
        assert summary["final_count"] == 85
        assert summary["total_removed"] == 15
        assert summary["removal_rate_percent"] == 15.0
        
        # 검증: 세부 분석
        assert summary["breakdown"]["missing_image"] == 5
        assert summary["breakdown"]["missing_label"] == 3
        assert summary["breakdown"]["duplicate_code"] == 7
    
    def test_build_summary_dict_zero_initial(self):
        """초기 카운트가 0인 경우 처리"""
        summary = build_summary_dict(
            initial_count=0,
            final_count=0
        )
        
        # 검증: 0으로 나누기 오류 없이 처리
        assert summary["removal_rate_percent"] == 0
        assert summary["total_removed"] == 0


class TestPreprocessEdgeCases:
    """전처리 엣지 케이스 테스트"""
    
    def test_preprocess_empty_dataframe(self, tmp_path):
        """빈 DataFrame 처리"""
        # 빈 DataFrame (컬럼은 존재)
        df = pd.DataFrame(columns=["image_path", "label_path", "code", "is_pair"])
        
        cfg = SimpleConfig()
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 빈 결과
        assert len(result_df) == 0
        
        # 검증: CSV 파일은 생성되어야 함
        csv_path = artifacts_dir / "manifest_stage1.csv"
        assert csv_path.exists()
        
        # 검증: 저장된 CSV도 빈 내용 (헤더 있거나 완전히 빈 파일)
        try:
            saved_df = pd.read_csv(csv_path)
            assert len(saved_df) == 0
        except pd.errors.EmptyDataError:
            # 빈 DataFrame의 경우 완전히 빈 CSV가 생성될 수 있음
            with open(csv_path, 'r') as f:
                content = f.read().strip()
                # 빈 파일이거나 헤더만 있는 경우 모두 허용
                assert content == "" or content == "image_path,label_path,code,is_pair", \
                    f"CSV should be empty or contain only header, got: '{content}'"
    
    def test_preprocess_custom_manifest_filename(self, tmp_path):
        """커스텀 매니페스트 파일명 테스트"""
        # 테스트 데이터 생성
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.json").touch()
        
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "test.jpg"), "label_path": str(tmp_path / "test.json"), "code": "test", "is_pair": True}
        ])
        
        # 커스텀 설정
        cfg = SimpleConfig(manifest_filename="custom_manifest.csv")
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 커스텀 파일명으로 저장
        custom_csv_path = artifacts_dir / "custom_manifest.csv"
        assert custom_csv_path.exists(), f"Custom manifest should be saved at {custom_csv_path}"
        
        # 검증: 기본 파일명은 생성되지 않음
        default_csv_path = artifacts_dir / "manifest_stage1.csv"
        assert not default_csv_path.exists(), "Default manifest should not be created"
    
    def test_preprocess_all_none_paths(self, tmp_path):
        """모든 경로가 None인 경우 처리"""
        df = pd.DataFrame([
            {"image_path": None, "label_path": None, "code": "none_test", "is_pair": False}
        ])
        
        cfg = SimpleConfig()
        artifacts_dir = tmp_path / "artifacts"
        
        # preprocess 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: None 경로로 인해 모든 행 제거
        assert len(result_df) == 0, "All rows should be removed due to None paths"


class TestIntegration:
    """통합 테스트"""
    
    def test_full_preprocessing_workflow(self, tmp_path):
        """전체 전처리 워크플로우 통합 테스트"""
        # 복잡한 시나리오 구성
        test_files = [
            ("valid1.jpg", "valid1.json", "valid1", True),    # 유효한 쌍
            ("valid2.jpg", "valid2.json", "valid2", True),    # 유효한 쌍
            ("orphan.jpg", None, "orphan", False),            # 고아 이미지
            (None, "orphan.json", "orphan_label", False),     # 고아 라벨
            ("dup1.jpg", "dup1.json", "duplicate", True),     # 중복 코드 1
            ("dup2.jpg", "dup2.json", "duplicate", True),     # 중복 코드 2 (유지될 것)
        ]
        
        # 실제 존재하는 파일만 생성
        for img_file, lbl_file, code, is_pair in test_files:
            if img_file and img_file != "orphan.jpg":  # orphan.jpg는 의도적으로 생성하지 않음
                (tmp_path / img_file).touch()
            if lbl_file and lbl_file != "orphan.json":  # orphan.json도 의도적으로 생성하지 않음
                (tmp_path / lbl_file).touch()
        
        # DataFrame 구성
        df_data = []
        for img_file, lbl_file, code, is_pair in test_files:
            img_path = str(tmp_path / img_file) if img_file else None
            lbl_path = str(tmp_path / lbl_file) if lbl_file else None
            df_data.append({
                "image_path": img_path,
                "label_path": lbl_path,
                "code": code,
                "is_pair": is_pair
            })
        
        df = pd.DataFrame(df_data)
        
        cfg = SimpleConfig()
        artifacts_dir = tmp_path / "artifacts"
        
        # 전처리 실행
        result_df = preprocess(df, cfg, artifacts_dir=artifacts_dir)
        
        # 검증: 예상 결과
        # - valid1, valid2: 유지
        # - orphan: 파일 없어서 제거
        # - orphan_label: 파일 없어서 제거
        # - duplicate: 마지막 것만 유지 (dup2)
        expected_codes = {"valid1", "valid2", "duplicate"}
        actual_codes = set(result_df["code"].tolist())
        
        assert actual_codes == expected_codes, f"Expected codes {expected_codes}, got {actual_codes}"
        assert len(result_df) == 3, f"Expected 3 final rows, got {len(result_df)}"
        
        # 검증: 중복 제거에서 마지막 항목 유지
        duplicate_row = result_df[result_df["code"] == "duplicate"].iloc[0]
        assert "dup2.jpg" in duplicate_row["image_path"], "Should keep the last duplicate (dup2)"
        
        # 검증: 모든 경로가 절대경로
        for idx, row in result_df.iterrows():
            if row["image_path"]:
                assert Path(row["image_path"]).is_absolute()
            if row["label_path"]:
                assert Path(row["label_path"]).is_absolute()
        
        # 검증: CSV 파일 생성 및 내용 일치
        csv_path = artifacts_dir / "manifest_stage1.csv"
        assert csv_path.exists()
        
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(result_df)
        assert set(saved_df["code"]) == set(result_df["code"])