"""
데이터 검증 모듈

목적: 매니페스트(또는 scan→preprocess 결과 DataFrame)에 대해 품질 게이트를 수행하고 리포트를 생성
입력:
    - df: 검증할 DataFrame (필수 컬럼: image_path, label_path, code, is_pair)
    - vcfg: 검증 설정 객체 (enable_angle_rules, label_size_range 속성 포함)
    - 검증 옵션: require_files_exist, min_pair_rate
출력:
    - ValidationReport: 검증 결과 리포트 (passed, errors, warnings, stats)
검증 포인트:
    - R0: 필수 컬럼 존재 확인
    - R1: 중복 code 검출
    - R2: 파일 존재성 검증 (옵션)
    - R3: pair_rate 임계값 검증
    - R4: 라벨 파일 크기 범위 검증
    - R5: 각도 규칙 검증 (placeholder)
    - 통계 집계 및 최종 pass/fail 판정
"""

import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Any


@dataclass
class ValidationReport:
    """검증 결과 리포트"""
    passed: bool
    errors: List[str]
    warnings: List[str] 
    stats: Dict[str, Union[int, float]]


def validate_manifest(
    df: pd.DataFrame,
    vcfg: Any,
    *,
    require_files_exist: bool = True,
    min_pair_rate: Optional[float] = None
) -> ValidationReport:
    """
    매니페스트 DataFrame 검증
    
    Args:
        df: 검증할 DataFrame
        vcfg: 검증 설정 객체 (enable_angle_rules, label_size_range 속성 필요)
        require_files_exist: 파일 존재성 검증 여부
        min_pair_rate: 최소 페어 비율 임계값
        
    Returns:
        ValidationReport: 검증 결과
    """
    errors = []
    warnings = []
    
    # R0) 필수 컬럼 존재 확인
    required_columns = ["image_path", "label_path", "code", "is_pair"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"missing_columns: {','.join(missing_columns)}")
        # 필수 컬럼이 없으면 더 이상 검증 불가
        return ValidationReport(
            passed=False,
            errors=errors,
            warnings=warnings,
            stats={"rows": len(df), "pairs": 0, "pair_rate": 0.0, 
                   "missing_image": 0, "missing_label": 0, "duplicate_codes": 0}
        )
    
    # 기본 통계 계산
    rows = len(df)
    pairs = int(df["is_pair"].sum()) if rows > 0 else 0
    pair_rate = float(pairs / rows) if rows > 0 else 0.0
    
    # R1) 중복 code 검출
    duplicate_codes = df["code"].duplicated().sum()
    if duplicate_codes > 0:
        errors.append(f"duplicate_codes: {duplicate_codes}")
    
    # R2) 파일 존재성 검증
    missing_image_count = 0
    missing_label_count = 0
    
    if require_files_exist and rows > 0:
        # 이미지 파일 존재성 검증
        for idx, row in df.iterrows():
            # 이미지 파일 검증
            if row["image_path"] is not None and not pd.isna(row["image_path"]):
                if not Path(row["image_path"]).exists():
                    missing_image_count += 1
                    if row["is_pair"]:
                        errors.append(f"missing_image_file: {row['image_path']} (code: {row['code']})")
                    else:
                        warnings.append(f"missing_image_file: {row['image_path']} (code: {row['code']})")
            elif row["is_pair"]:
                # is_pair=True인데 image_path가 None이면 에러
                missing_image_count += 1
                errors.append(f"missing_image_path: code {row['code']} marked as pair but no image_path")
            
            # 라벨 파일 검증
            if row["label_path"] is not None and not pd.isna(row["label_path"]):
                if not Path(row["label_path"]).exists():
                    missing_label_count += 1
                    if row["is_pair"]:
                        errors.append(f"missing_label_file: {row['label_path']} (code: {row['code']})")
                    else:
                        warnings.append(f"missing_label_file: {row['label_path']} (code: {row['code']})")
            elif row["is_pair"]:
                # is_pair=True인데 label_path가 None이면 에러
                missing_label_count += 1
                errors.append(f"missing_label_path: code {row['code']} marked as pair but no label_path")
    
    # R3) pair_rate 임계값 검증
    if min_pair_rate is not None and pair_rate < min_pair_rate:
        errors.append(f"pair_rate_below: {pair_rate:.3f} < {min_pair_rate:.3f}")
    
    # R4) 라벨 파일 크기 범위 검증
    label_size_violations = 0
    if hasattr(vcfg, 'label_size_range') and vcfg.label_size_range is not None:
        min_size, max_size = vcfg.label_size_range
        
        for idx, row in df.iterrows():
            if row["label_path"] is not None and not pd.isna(row["label_path"]):
                label_path = Path(row["label_path"])
                if label_path.exists():
                    try:
                        file_size = label_path.stat().st_size
                        if not (min_size <= file_size <= max_size):
                            label_size_violations += 1
                    except OSError:
                        warnings.append(f"label_size_check_failed: {row['label_path']}")
        
        if label_size_violations > 0:
            errors.append(f"label_size_out_of_range: {label_size_violations}")
    
    # R5) 각도 규칙 검증 (placeholder)
    if hasattr(vcfg, 'enable_angle_rules') and vcfg.enable_angle_rules:
        warnings.append("angle_rules: skipped (not implemented)")
    
    # 통계 집계
    stats = {
        "rows": rows,
        "pairs": pairs,
        "pair_rate": pair_rate,
        "missing_image": missing_image_count,
        "missing_label": missing_label_count,
        "duplicate_codes": duplicate_codes
    }
    
    # 최종 판정
    passed = len(errors) == 0
    
    return ValidationReport(
        passed=passed,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


def validate_manifest_from_csv(
    csv_path: Union[str, Path],
    vcfg: Any,
    *,
    require_files_exist: bool = True,
    min_pair_rate: Optional[float] = None
) -> ValidationReport:
    """
    CSV 파일에서 매니페스트를 읽어 검증
    
    Args:
        csv_path: 매니페스트 CSV 파일 경로
        vcfg: 검증 설정 객체
        require_files_exist: 파일 존재성 검증 여부
        min_pair_rate: 최소 페어 비율 임계값
        
    Returns:
        ValidationReport: 검증 결과
    """
    try:
        df = pd.read_csv(csv_path)
        return validate_manifest(df, vcfg, 
                               require_files_exist=require_files_exist,
                               min_pair_rate=min_pair_rate)
    except pd.errors.EmptyDataError:
        # 빈 CSV 파일인 경우
        empty_df = pd.DataFrame(columns=["image_path", "label_path", "code", "is_pair"])
        return validate_manifest(empty_df, vcfg,
                               require_files_exist=require_files_exist,
                               min_pair_rate=min_pair_rate)
    except Exception as e:
        return ValidationReport(
            passed=False,
            errors=[f"csv_read_error: {str(e)}"],
            warnings=[],
            stats={"rows": 0, "pairs": 0, "pair_rate": 0.0,
                   "missing_image": 0, "missing_label": 0, "duplicate_codes": 0}
        )


def generate_validation_summary(report: ValidationReport) -> str:
    """
    검증 결과 요약 문자열 생성
    
    Args:
        report: 검증 결과 리포트
        
    Returns:
        str: 요약 문자열
    """
    status = "PASSED" if report.passed else "FAILED"
    
    summary = f"""Validation Report: {status}
==========================================
Statistics:
  - Total rows: {report.stats['rows']}
  - Pairs: {report.stats['pairs']}
  - Pair rate: {report.stats['pair_rate']:.3f}
  - Missing images: {report.stats['missing_image']}
  - Missing labels: {report.stats['missing_label']}
  - Duplicate codes: {report.stats['duplicate_codes']}

Issues:
  - Errors: {len(report.errors)}
  - Warnings: {len(report.warnings)}
"""
    
    if report.errors:
        summary += "\nErrors:\n"
        for error in report.errors:
            summary += f"  - {error}\n"
    
    if report.warnings:
        summary += "\nWarnings:\n"
        for warning in report.warnings:
            summary += f"  - {warning}\n"
    
    return summary


def create_validation_config(
    enable_angle_rules: bool = False,
    label_size_range: Optional[Tuple[int, int]] = None
) -> object:
    """
    검증 설정 객체 생성 헬퍼 함수
    
    Args:
        enable_angle_rules: 각도 규칙 활성화 여부
        label_size_range: 라벨 파일 크기 범위 (min, max) bytes
        
    Returns:
        검증 설정 객체
    """
    class ValidationConfig:
        def __init__(self, enable_angle_rules: bool, label_size_range: Optional[Tuple[int, int]]):
            self.enable_angle_rules = enable_angle_rules
            self.label_size_range = label_size_range
    
    return ValidationConfig(enable_angle_rules, label_size_range)