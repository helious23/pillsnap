"""
dataset.validate 모듈 유닛 테스트

테스트 목적:
- 검증기가 빈/정상/이상/경계 케이스를 모두 안전하게 판정하는지 확인
- ValidationReport의 모든 필드 검증
- 각 검증 규칙(R0~R5)의 동작 확인
- 파일 존재성 정책 및 임계값 처리 검증
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.validate import validate_manifest, ValidationReport, validate_manifest_from_csv, generate_validation_summary, create_validation_config


# 공통 준비: 더미 설정 객체
def _create_vcfg(enable_angle_rules=False, label_size_range=None):
    """더미 검증 설정 객체 생성"""
    return type("VCfg", (object,), {
        "enable_angle_rules": enable_angle_rules,
        "label_size_range": label_size_range
    })()


def _mk_file(path, size=0):
    """지정 바이트 수의 파일 생성 (라벨 사이즈 범위 검증용)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'x' * size)


class TestValidateBasicCases:
    """기본 검증 시나리오 테스트"""
    
    def test_validate_basic_pass(self, tmp_path):
        """기본 통과 케이스 - 모든 파일 존재, 높은 pair_rate"""
        # 테스트 파일 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 50)
        _mk_file(tmp_path / "b.jpg", 100)
        _mk_file(tmp_path / "b.json", 50)
        
        # DataFrame 구성
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "b.jpg"), "label_path": str(tmp_path / "b.json"), "code": "b", "is_pair": True},
        ])
        
        # 검증 설정
        vcfg = _create_vcfg(enable_angle_rules=False, label_size_range=None)
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True, min_pair_rate=0.8)
        
        # 검증: 통과해야 함
        assert report.passed is True
        assert not report.errors
        assert report.stats["pairs"] == 2
        assert report.stats["pair_rate"] == 1.0
        assert report.stats["rows"] == 2
        assert report.stats["missing_image"] == 0
        assert report.stats["missing_label"] == 0
        assert report.stats["duplicate_codes"] == 0
    
    def test_pair_rate_threshold_fail(self, tmp_path):
        """pair_rate 임계값 미달로 실패"""
        # 테스트 파일 생성 (일부만)
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 50)
        _mk_file(tmp_path / "c.jpg", 100)
        # c.json은 생성하지 않음 (라벨 없음)
        
        # DataFrame 구성
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "c.jpg"), "label_path": None, "code": "c", "is_pair": False},
        ])
        
        vcfg = _create_vcfg()
        
        # pair_rate = 1/2 = 0.5, min_pair_rate = 0.8
        report = validate_manifest(df, vcfg, require_files_exist=True, min_pair_rate=0.8)
        
        # 검증: 실패해야 함
        assert report.passed is False
        assert any("pair_rate_below" in error for error in report.errors)
        assert report.stats["pair_rate"] == 0.5
        assert report.stats["pairs"] == 1
    
    def test_duplicate_codes_fail(self, tmp_path):
        """중복 코드로 인한 실패"""
        # 테스트 파일 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 50)
        _mk_file(tmp_path / "a2.jpg", 100)
        _mk_file(tmp_path / "a2.json", 50)
        
        # DataFrame 구성 (같은 코드 "a" 사용)
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "a2.jpg"), "label_path": str(tmp_path / "a2.json"), "code": "a", "is_pair": True},
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 실패해야 함
        assert report.passed is False
        assert any("duplicate_codes" in error for error in report.errors)
        assert report.stats["duplicate_codes"] == 1  # 하나의 중복
    
    def test_missing_required_columns_error(self):
        """필수 컬럼 누락으로 인한 에러"""
        # 필수 컬럼 중 일부가 누락된 DataFrame
        df = pd.DataFrame([
            {"image_path": "/path/to/a.jpg", "code": "a", "is_pair": True},
            # label_path 컬럼 누락
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=False)
        
        # 검증: 실패해야 함
        assert report.passed is False
        assert any("missing_columns" in error for error in report.errors)
        assert "label_path" in report.errors[0]


class TestFileExistencePolicy:
    """파일 존재성 정책 테스트"""
    
    def test_file_existence_policy(self, tmp_path):
        """파일 존재성 정책 - is_pair 여부에 따른 에러/경고 구분"""
        # 일부 파일만 생성
        _mk_file(tmp_path / "a.jpg", 100)
        # a.json은 생성하지 않음 (is_pair=True인데 파일 없음 → 에러)
        # c.jpg는 생성하지 않음 (is_pair=False인데 파일 없음 → 경고)
        _mk_file(tmp_path / "c.json", 50)
        
        # DataFrame 구성
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "c.jpg"), "label_path": str(tmp_path / "c.json"), "code": "c", "is_pair": False},
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 에러와 경고 구분
        assert report.passed is False  # 에러가 있으므로 실패
        
        # is_pair=True & 파일 없음 → 에러
        assert any("missing_label_file" in error and "code: a" in error for error in report.errors)
        
        # is_pair=False & 파일 없음 → 경고
        assert any("missing_image_file" in warning and "code: c" in warning for warning in report.warnings)
        
        # 통계 확인
        assert report.stats["missing_image"] == 1
        assert report.stats["missing_label"] == 1
    
    def test_missing_path_for_pair(self, tmp_path):
        """is_pair=True인데 경로가 None인 경우"""
        # DataFrame 구성 (is_pair=True인데 경로가 None)
        df = pd.DataFrame([
            {"image_path": None, "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "b.jpg"), "label_path": None, "code": "b", "is_pair": True},
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 에러 발생
        assert report.passed is False
        assert any("missing_image_path" in error for error in report.errors)
        assert any("missing_label_path" in error for error in report.errors)


class TestLabelSizeRange:
    """라벨 파일 크기 범위 검증 테스트"""
    
    def test_label_size_range(self, tmp_path):
        """라벨 파일 크기 범위 검증"""
        # 라벨 파일 크기를 다양하게 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 100)  # 범위 밖 (< 180)
        
        _mk_file(tmp_path / "b.jpg", 100)
        _mk_file(tmp_path / "b.json", 200)  # 범위 안 (180-220)
        
        _mk_file(tmp_path / "c.jpg", 100)
        _mk_file(tmp_path / "c.json", 260)  # 범위 밖 (> 220)
        
        # DataFrame 구성
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "b.jpg"), "label_path": str(tmp_path / "b.json"), "code": "b", "is_pair": True},
            {"image_path": str(tmp_path / "c.jpg"), "label_path": str(tmp_path / "c.json"), "code": "c", "is_pair": True},
        ])
        
        # 검증 설정 (크기 범위 180-220 bytes)
        vcfg = _create_vcfg(label_size_range=(180, 220))
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 실패해야 함 (2개 파일이 범위 밖)
        assert report.passed is False
        assert any("label_size_out_of_range: 2" in error for error in report.errors)
    
    def test_label_size_range_no_violations(self, tmp_path):
        """라벨 파일 크기 범위 - 위반 없음"""
        # 모든 라벨 파일을 범위 안으로 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 200)  # 범위 안
        
        _mk_file(tmp_path / "b.jpg", 100)
        _mk_file(tmp_path / "b.json", 180)  # 범위 안 (경계값)
        
        # DataFrame 구성
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
            {"image_path": str(tmp_path / "b.jpg"), "label_path": str(tmp_path / "b.json"), "code": "b", "is_pair": True},
        ])
        
        vcfg = _create_vcfg(label_size_range=(180, 220))
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 통과해야 함
        assert report.passed is True
        assert not any("label_size_out_of_range" in error for error in report.errors)


class TestSpecialCases:
    """특수 케이스 테스트"""
    
    def test_empty_df_is_ok(self):
        """빈 DataFrame - 정상 처리"""
        # 빈 DataFrame (컬럼은 정확히 존재)
        df = pd.DataFrame(columns=["image_path", "label_path", "code", "is_pair"])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 통과해야 함
        assert report.passed is True
        assert report.stats["rows"] == 0
        assert report.stats["pair_rate"] == 0.0
        assert report.stats["pairs"] == 0
    
    def test_empty_df_with_correct_columns(self):
        """빈 DataFrame (0행, 4컬럼) → passed=True"""
        # 정확한 스키마를 가진 빈 DataFrame
        df = pd.DataFrame(columns=["image_path", "label_path", "code", "is_pair"])
        assert len(df) == 0
        assert len(df.columns) == 4
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 빈 상태에서도 통과
        assert report.passed is True
        assert report.stats["rows"] == 0
        assert report.stats["pairs"] == 0
        assert report.stats["pair_rate"] == 0.0
        assert report.stats["missing_image"] == 0
        assert report.stats["missing_label"] == 0
        assert report.stats["duplicate_codes"] == 0
        assert len(report.errors) == 0
    
    def test_missing_columns_failure(self):
        """잘못된 컬럼 누락 → passed=False, 'missing_columns' 포함"""
        # 필수 컬럼 중 일부 누락된 DataFrame
        df = pd.DataFrame([
            {"image_path": "/test/a.jpg", "code": "a", "is_pair": True},
            # label_path 컬럼이 누락됨
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=False)
        
        # 검증: 실패해야 함
        assert report.passed is False
        assert any("missing_columns" in error for error in report.errors)
        # 누락된 컬럼명이 에러 메시지에 포함되어야 함
        missing_cols_error = next(error for error in report.errors if "missing_columns" in error)
        assert "label_path" in missing_cols_error
    
    def test_multiple_missing_columns(self):
        """여러 컬럼 누락"""
        # 여러 필수 컬럼이 누락된 DataFrame
        df = pd.DataFrame([
            {"image_path": "/test/a.jpg"},
            # label_path, code, is_pair 모두 누락
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=False)
        
        # 검증: 실패 및 누락된 모든 컬럼 언급
        assert report.passed is False
        missing_cols_error = next(error for error in report.errors if "missing_columns" in error)
        assert "label_path" in missing_cols_error
        assert "code" in missing_cols_error
        assert "is_pair" in missing_cols_error
    
    def test_invalid_file_paths_failure(self, tmp_path):
        """파일 경로 일부 무효 → passed=False, invalid path count 반영"""
        # 일부 파일만 생성 (유효)
        _mk_file(tmp_path / "valid.jpg", 100)
        _mk_file(tmp_path / "valid.json", 50)
        
        # DataFrame 구성 (일부 경로 무효)
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "valid.jpg"), "label_path": str(tmp_path / "valid.json"), "code": "valid", "is_pair": True},
            {"image_path": "/nonexistent/invalid.jpg", "label_path": "/nonexistent/invalid.json", "code": "invalid", "is_pair": True},
            {"image_path": str(tmp_path / "missing.jpg"), "label_path": str(tmp_path / "missing.json"), "code": "missing", "is_pair": True},
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행 (파일 존재성 검사 활성화)
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 실패해야 함
        assert report.passed is False
        
        # 무효한 파일 경로 수가 통계에 반영되어야 함
        assert report.stats["missing_image"] == 2  # invalid.jpg, missing.jpg
        assert report.stats["missing_label"] == 2  # invalid.json, missing.json
        
        # 에러 메시지에 누락된 파일들이 언급되어야 함
        missing_file_errors = [error for error in report.errors if "missing_" in error and "_file" in error]
        assert len(missing_file_errors) >= 2  # 최소 2개 파일에 대한 에러
    
    def test_duplicate_codes_with_count(self, tmp_path):
        """중복 code → passed=False, duplicate_codes 반영"""
        # 테스트 파일 생성
        _mk_file(tmp_path / "a1.jpg", 100)
        _mk_file(tmp_path / "a1.json", 50)
        _mk_file(tmp_path / "a2.jpg", 100)
        _mk_file(tmp_path / "a2.json", 50)
        _mk_file(tmp_path / "b1.jpg", 100)
        _mk_file(tmp_path / "b1.json", 50)
        _mk_file(tmp_path / "b2.jpg", 100)
        _mk_file(tmp_path / "b2.json", 50)
        
        # DataFrame 구성 (중복 코드 포함)
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a1.jpg"), "label_path": str(tmp_path / "a1.json"), "code": "duplicate_a", "is_pair": True},
            {"image_path": str(tmp_path / "a2.jpg"), "label_path": str(tmp_path / "a2.json"), "code": "duplicate_a", "is_pair": True},  # 중복
            {"image_path": str(tmp_path / "b1.jpg"), "label_path": str(tmp_path / "b1.json"), "code": "duplicate_b", "is_pair": True},
            {"image_path": str(tmp_path / "b2.jpg"), "label_path": str(tmp_path / "b2.json"), "code": "duplicate_b", "is_pair": True},  # 중복
        ])
        
        vcfg = _create_vcfg()
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 실패해야 함
        assert report.passed is False
        
        # 중복 코드 수가 정확히 계산되어야 함
        assert report.stats["duplicate_codes"] == 2  # duplicate_a, duplicate_b 2개
        
        # 중복 코드 에러 메시지 확인
        duplicate_error = next(error for error in report.errors if "duplicate_codes" in error)
        assert "2" in duplicate_error  # 중복 수가 메시지에 포함
    
    def test_angle_rules_placeholder(self, tmp_path):
        """각도 규칙 placeholder 테스트"""
        # 기본 유효한 데이터 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 50)
        
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
        ])
        
        # 각도 규칙 활성화
        vcfg = _create_vcfg(enable_angle_rules=True)
        
        # 검증 실행
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 검증: 통과하되 경고 있어야 함
        assert report.passed is True  # 다른 에러가 없으므로 통과
        assert any("angle_rules" in warning and "skipped" in warning for warning in report.warnings)
    
    def test_require_files_exist_false(self, tmp_path):
        """파일 존재성 검증 비활성화"""
        # 파일을 생성하지 않음
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "nonexistent.jpg"), "label_path": str(tmp_path / "nonexistent.json"), "code": "a", "is_pair": True},
        ])
        
        vcfg = _create_vcfg()
        
        # 파일 존재성 검증 비활성화
        report = validate_manifest(df, vcfg, require_files_exist=False)
        
        # 검증: 파일이 없어도 통과해야 함
        assert report.passed is True
        assert report.stats["missing_image"] == 0  # 검증하지 않았으므로 0
        assert report.stats["missing_label"] == 0


class TestUtilityFunctions:
    """유틸리티 함수 테스트"""
    
    def test_validate_manifest_from_csv(self, tmp_path):
        """CSV에서 매니페스트 검증"""
        # 테스트 파일 생성
        _mk_file(tmp_path / "test.jpg", 100)
        _mk_file(tmp_path / "test.json", 50)
        
        # CSV 파일 생성
        csv_content = f"""image_path,label_path,code,is_pair
{tmp_path / "test.jpg"},{tmp_path / "test.json"},test,True"""
        
        csv_path = tmp_path / "manifest.csv"
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        vcfg = _create_vcfg()
        
        # CSV에서 검증
        report = validate_manifest_from_csv(csv_path, vcfg, require_files_exist=True)
        
        # 검증: 정상 처리
        assert report.passed is True
        assert report.stats["rows"] == 1
        assert report.stats["pairs"] == 1
    
    def test_validate_manifest_from_empty_csv(self, tmp_path):
        """빈 CSV 파일 처리"""
        # 빈 CSV 파일 생성
        csv_path = tmp_path / "empty_manifest.csv"
        csv_path.touch()
        
        vcfg = _create_vcfg()
        
        # 빈 CSV에서 검증
        report = validate_manifest_from_csv(csv_path, vcfg, require_files_exist=True)
        
        # 검증: 빈 DataFrame으로 처리
        assert report.passed is True
        assert report.stats["rows"] == 0
    
    def test_generate_validation_summary(self):
        """검증 요약 생성"""
        # 샘플 리포트 생성
        report = ValidationReport(
            passed=False,
            errors=["duplicate_codes: 2", "pair_rate_below: 0.500 < 0.800"],
            warnings=["angle_rules: skipped (not implemented)"],
            stats={"rows": 10, "pairs": 5, "pair_rate": 0.5, "missing_image": 1, "missing_label": 0, "duplicate_codes": 2}
        )
        
        # 요약 생성
        summary = generate_validation_summary(report)
        
        # 검증: 요약 내용 확인
        assert "FAILED" in summary
        assert "Total rows: 10" in summary
        assert "Pairs: 5" in summary
        assert "Pair rate: 0.500" in summary
        assert "duplicate_codes: 2" in summary
        assert "angle_rules: skipped" in summary
    
    def test_create_validation_config(self):
        """검증 설정 생성 헬퍼"""
        # 기본 설정
        vcfg1 = create_validation_config()
        assert vcfg1.enable_angle_rules is False
        assert vcfg1.label_size_range is None
        
        # 커스텀 설정
        vcfg2 = create_validation_config(enable_angle_rules=True, label_size_range=(100, 500))
        assert vcfg2.enable_angle_rules is True
        assert vcfg2.label_size_range == (100, 500)


class TestErrorHandling:
    """에러 처리 테스트"""
    
    def test_csv_read_error(self, tmp_path):
        """CSV 읽기 에러 처리"""
        # 잘못된 CSV 파일 생성
        csv_path = tmp_path / "invalid.csv"
        with open(csv_path, 'w') as f:
            f.write("invalid,csv,content\nwith,mismatched,columns,too,many")
        
        vcfg = _create_vcfg()
        
        # 에러가 있는 CSV에서 검증
        report = validate_manifest_from_csv(csv_path, vcfg)
        
        # 검증: 에러 보고
        assert report.passed is False
        # CSV 읽기 에러나 다른 파싱 에러가 발생할 수 있음
        assert len(report.errors) > 0
    
    def test_file_stat_error(self, tmp_path):
        """파일 stat 에러 처리 (권한 등)"""
        # 정상 파일 생성
        _mk_file(tmp_path / "a.jpg", 100)
        _mk_file(tmp_path / "a.json", 50)
        
        df = pd.DataFrame([
            {"image_path": str(tmp_path / "a.jpg"), "label_path": str(tmp_path / "a.json"), "code": "a", "is_pair": True},
        ])
        
        # 존재하지 않는 범위로 설정 (실제 파일 크기 체크에서 에러 발생하지 않음)
        vcfg = _create_vcfg(label_size_range=(1000, 2000))
        
        # 검증 실행 (파일이 작아서 범위 밖으로 감지됨)
        report = validate_manifest(df, vcfg, require_files_exist=True)
        
        # 크기 범위 밖이므로 에러 발생
        assert report.passed is False
        assert any("label_size_out_of_range" in error for error in report.errors)