"""
데이터 전처리 모듈

목적: scan 결과를 정규화하여 CSV 매니페스트를 생성
입력:
    - df: 스캔 결과 DataFrame (scan.py의 출력)
    - cfg: 설정 객체 (manifest_filename, quarantine_dirname 속성 포함)
    - artifacts_dir: 출력 디렉토리 경로
출력:
    - 정규화된 DataFrame
    - artifacts/{manifest_filename} CSV 파일 생성
검증 포인트:
    - 기본 컬럼 존재 보장 및 TypeError 처리
    - 경로 정규화 (paths.norm() 사용하여 절대경로 변환)
    - 파일 존재성 재검증 및 누락 파일 제거
    - code 중복 제거 (마지막 항목 유지)
    - CSV 저장과 DataFrame 행 수 일치 확인
"""

import os
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
import logging

import paths

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    cfg: Any,
    *,
    artifacts_dir: Union[str, Path] = "artifacts"
) -> pd.DataFrame:
    """
    스캔 결과 DataFrame을 정규화하여 CSV 매니페스트 생성
    
    Args:
        df: 스캔 결과 DataFrame
        cfg: 설정 객체 (manifest_filename, quarantine_dirname 속성 필요)
        artifacts_dir: 출력 디렉토리 경로
        
    Returns:
        pd.DataFrame: 정규화된 DataFrame
        
    Raises:
        ValueError: 필수 컬럼이 누락된 경우
    """
    print(f"🔄 Starting preprocessing with {len(df)} rows...")
    
    # 1) 기본 컬럼 존재 보장
    required_columns = ["image_path", "label_path", "code", "is_pair"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"✅ Required columns verified: {required_columns}")
    
    # 작업용 DataFrame 복사
    working_df = df.copy()
    summary = {}
    
    # 2) 경로 정규화
    print("📁 Normalizing file paths...")
    
    def normalize_path(path_str):
        """경로 정규화 헬퍼 함수"""
        if path_str is None or pd.isna(path_str):
            return None
        try:
            return str(paths.norm(path_str))
        except Exception as e:
            logger.warning(f"Failed to normalize path '{path_str}': {e}")
            return None
    
    working_df["image_path"] = working_df["image_path"].apply(normalize_path)
    working_df["label_path"] = working_df["label_path"].apply(normalize_path)
    
    print(f"📁 Path normalization completed")
    
    # 3) 파일 존재 재검증
    print("🔍 Re-validating file existence...")
    
    initial_count = len(working_df)
    
    # 이미지 파일 존재 확인
    def file_exists(path_str):
        """파일 존재 확인 헬퍼 함수"""
        if path_str is None or pd.isna(path_str):
            return False
        return Path(path_str).exists()
    
    # 이미지 파일 누락 제거 (None이거나 파일이 없는 경우)
    def should_keep_image_row(row):
        if row["image_path"] is None or pd.isna(row["image_path"]):
            return False
        return Path(row["image_path"]).exists()
    
    def should_keep_label_row(row):
        if row["label_path"] is None or pd.isna(row["label_path"]):
            return False
        return Path(row["label_path"]).exists()
    
    # 이미지 파일 필터링
    before_image_filter = len(working_df)
    working_df = working_df[working_df.apply(should_keep_image_row, axis=1)]
    missing_image_count = before_image_filter - len(working_df)
    
    # 라벨 파일 필터링
    before_label_filter = len(working_df)
    working_df = working_df[working_df.apply(should_keep_label_row, axis=1)]
    missing_label_count = before_label_filter - len(working_df)
    
    summary["missing_image"] = missing_image_count
    summary["missing_label"] = missing_label_count
    
    print(f"🗑️  Removed {missing_image_count} rows with missing images")
    print(f"🗑️  Removed {missing_label_count} rows with missing labels")
    
    # 4) code 중복 제거
    print("🔄 Removing duplicate codes...")
    
    before_dedup_count = len(working_df)
    
    # 마지막 항목 유지 (keep='last')
    working_df = working_df.drop_duplicates(subset=["code"], keep="last")
    
    after_dedup_count = len(working_df)
    duplicate_count = before_dedup_count - after_dedup_count
    
    summary["duplicate_code"] = duplicate_count
    
    print(f"🗑️  Removed {duplicate_count} duplicate codes")
    
    # 스키마 보존 - 빈 결과라도 필수 컬럼 구조 유지
    required_cols = ["image_path", "label_path", "code", "is_pair"]
    
    # 필수 컬럼이 존재하는지 확인하고 없으면 추가
    for col in required_cols:
        if col not in working_df.columns:
            working_df[col] = pd.Series(dtype="object" if col != "is_pair" else "bool")
    
    # 컬럼 순서 고정
    working_df = working_df[required_cols]
    
    # 빈 DataFrame이면 스키마만 유지 (0,4)
    if len(working_df) == 0:
        working_df = pd.DataFrame(columns=required_cols)
        print("📋 Empty result - preserving schema (0,4)")
    else:
        # 5) 재현성을 위한 정렬 (CSV 저장 전 필수)
        print("🔄 Sorting DataFrame for reproducibility...")
        working_df = working_df.sort_values(
            ["code", "image_path", "label_path"]
        ).reset_index(drop=True)
        print(f"✅ Sorted by code, image_path, label_path")
    
    # 6) artifacts_dir 생성
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Artifacts directory ensured: {artifacts_path}")
    
    # 7) CSV 저장 (index=False 고정)
    manifest_path = artifacts_path / cfg.manifest_filename
    working_df.to_csv(manifest_path, index=False)
    
    print(f"💾 Manifest saved: {manifest_path}")
    print(f"📊 Final DataFrame: {len(working_df)} rows")
    
    # 8) 요약 출력
    total_removed = initial_count - len(working_df)
    print(f"📈 Preprocessing summary:")
    print(f"   Initial rows: {initial_count}")
    print(f"   Final rows: {len(working_df)}")
    print(f"   Total removed: {total_removed}")
    print(f"   Removal breakdown: {summary}")
    
    # 저장된 CSV와 DataFrame 행 수 일치 확인
    try:
        saved_df = pd.read_csv(manifest_path)
        if len(saved_df) != len(working_df):
            logger.warning(f"Row count mismatch: DataFrame={len(working_df)}, CSV={len(saved_df)}")
        else:
            print(f"✅ CSV verification passed: {len(saved_df)} rows")
    except pd.errors.EmptyDataError:
        # 빈 DataFrame인 경우 CSV에 헤더만 있을 수 있음
        if len(working_df) == 0:
            print(f"✅ CSV verification passed: empty DataFrame saved correctly")
        else:
            logger.warning(f"CSV is empty but DataFrame has {len(working_df)} rows")
    
    return working_df


def build_summary_dict(
    initial_count: int,
    final_count: int,
    missing_image: int = 0,
    missing_label: int = 0,
    duplicate_code: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    전처리 요약 정보 생성
    
    Args:
        initial_count: 초기 행 수
        final_count: 최종 행 수
        missing_image: 누락된 이미지 파일 수
        missing_label: 누락된 라벨 파일 수
        duplicate_code: 중복 코드 수
        **kwargs: 추가 통계 정보
        
    Returns:
        Dict[str, Any]: 요약 정보 딕셔너리
    """
    total_removed = initial_count - final_count
    removal_rate = (total_removed / initial_count * 100) if initial_count > 0 else 0
    
    summary = {
        "initial_count": initial_count,
        "final_count": final_count,
        "total_removed": total_removed,
        "removal_rate_percent": round(removal_rate, 2),
        "breakdown": {
            "missing_image": missing_image,
            "missing_label": missing_label,
            "duplicate_code": duplicate_code
        }
    }
    
    # 추가 정보 병합
    summary.update(kwargs)
    
    return summary


def validate_preprocessed_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    전처리된 데이터 검증
    
    Args:
        df: 전처리된 DataFrame
        
    Returns:
        Dict[str, Any]: 검증 결과
    """
    validation_results = {
        "valid": True,
        "issues": []
    }
    
    # 1. 필수 컬럼 존재 확인
    required_columns = ["image_path", "label_path", "code", "is_pair"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results["valid"] = False
        validation_results["issues"].append(f"Missing columns: {missing_columns}")
        return validation_results  # 컬럼이 없으면 더 이상 검증 불가
    
    # 2. 파일 경로 유효성 확인
    if len(df) > 0:
        # None이 아닌 경로들이 실제 파일을 가리키는지 확인
        invalid_images = 0
        invalid_labels = 0
        
        for idx, row in df.iterrows():
            if row["image_path"] is not None and not Path(row["image_path"]).exists():
                invalid_images += 1
            if row["label_path"] is not None and not Path(row["label_path"]).exists():
                invalid_labels += 1
        
        if invalid_images > 0:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Invalid image paths: {invalid_images}")
        
        if invalid_labels > 0:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Invalid label paths: {invalid_labels}")
        
        # 3. 코드 중복 확인
        duplicate_codes = df["code"].duplicated().sum()
        if duplicate_codes > 0:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Duplicate codes found: {duplicate_codes}")
    
    return validation_results