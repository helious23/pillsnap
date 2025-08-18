"""
데이터셋 스캔 모듈

목적: 루트 디렉토리를 스트리밍으로 스캔하여 이미지-라벨 쌍 테이블과 통계를 생성
입력: 
    - root: 데이터셋 루트 디렉토리 경로
    - image_exts: 이미지 파일 확장자 리스트 (예: ['.jpg', '.png'])
    - label_ext: 라벨 파일 확장자 (예: '.json')
출력:
    - DataFrame: 컬럼 ["image_path", "label_path", "code", "is_pair"]
    - 통계 딕셔너리: {"images": N, "labels": M, "pairs": K, "mismatches": X}
검증 포인트:
    - 대용량 디렉토리에서 메모리 효율적 스트리밍 처리
    - 예외 파일 스킵으로 견고성 확보
    - 이미지-라벨 basename 매칭 정확도
    - 통계 정보의 일관성 (images + labels = pairs + mismatches)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Tuple
import logging
from collections import defaultdict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


def scan_dataset(
    root: Union[str, Path], 
    image_exts: List[str], 
    label_ext: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    데이터셋 디렉토리를 스트리밍으로 스캔하여 이미지-라벨 쌍 분석
    
    Args:
        root: 스캔할 루트 디렉토리
        image_exts: 이미지 파일 확장자 리스트 (예: ['.jpg', '.jpeg', '.png'])
        label_ext: 라벨 파일 확장자 (예: '.json')
        
    Returns:
        tuple[DataFrame, dict]: (이미지-라벨 쌍 테이블, 통계 정보)
            - DataFrame 컬럼: ["image_path", "label_path", "code", "is_pair"]
            - 통계: {"images": int, "labels": int, "pairs": int, "mismatches": int}
    """
    root_path = Path(root)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    
    logger.info(f"Starting dataset scan: {root_path}")
    
    # 파일 수집용 딕셔너리 (basename -> 경로)
    image_files = {}  # basename -> full_path
    label_files = {}  # basename -> full_path
    
    # 중복 basename 카운터
    duplicate_image_basenames = defaultdict(int)
    duplicate_label_basenames = defaultdict(int)
    
    # 확장자 정규화 (소문자로 통일)
    normalized_image_exts = [ext.lower() for ext in image_exts]
    normalized_label_ext = label_ext.lower()
    
    # 스트리밍 스캔 시작
    total_files = 0
    skipped_files = 0
    
    logger.info("Streaming file collection started...")
    
    # os.walk로 스트리밍 스캔
    for dirpath, _, filenames in os.walk(root_path):
        # tqdm 진행률 표시 (선택적)
        if TQDM_AVAILABLE and total_files % 1000 == 0:
            print(f"Processed {total_files} files...")
        
        for filename in filenames:
            total_files += 1
            
            try:
                file_path = Path(dirpath) / filename
                
                # 파일 확장자 추출 및 정규화
                file_ext = file_path.suffix.lower()
                basename = file_path.stem  # 확장자 제외한 파일명
                
                # 이미지 파일 분류
                if file_ext in normalized_image_exts:
                    if basename in image_files:
                        duplicate_image_basenames[basename] += 1
                        # 첫 번째 발견된 파일 유지
                        continue
                    image_files[basename] = str(file_path)
                
                # 라벨 파일 분류
                elif file_ext == normalized_label_ext:
                    if basename in label_files:
                        duplicate_label_basenames[basename] += 1
                        # 첫 번째 발견된 파일 유지
                        continue
                    label_files[basename] = str(file_path)
                
                # 기타 파일은 무시
                
            except Exception as e:
                logger.warning(f"Skipping file {filename}: {e}")
                skipped_files += 1
                continue
    
    logger.info(f"File collection completed: {total_files} files processed, {skipped_files} skipped")
    logger.info(f"Found {len(image_files)} images, {len(label_files)} labels")
    
    # 중복 basename 경고 요약 출력
    if duplicate_image_basenames or duplicate_label_basenames:
        print("\n=== Duplicate Basename Summary ===")
        
        if duplicate_image_basenames:
            # 상위 20개 이미지 중복
            sorted_img_dups = sorted(duplicate_image_basenames.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Duplicate image basenames (top {min(20, len(sorted_img_dups))}):")
            for basename, count in sorted_img_dups:
                print(f"  {basename}: {count} duplicates")
        
        if duplicate_label_basenames:
            # 상위 20개 라벨 중복
            sorted_lbl_dups = sorted(duplicate_label_basenames.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Duplicate label basenames (top {min(20, len(sorted_lbl_dups))}):")
            for basename, count in sorted_lbl_dups:
                print(f"  {basename}: {count} duplicates")
        
        print(f"Total unique duplicate image basenames: {len(duplicate_image_basenames)}")
        print(f"Total unique duplicate label basenames: {len(duplicate_label_basenames)}")
        print("=================================\n")
    
    # 이미지-라벨 매칭 및 DataFrame 생성
    df_rows = []
    
    # 모든 고유한 basename 수집
    all_basenames = set(image_files.keys()) | set(label_files.keys())
    
    logger.info(f"Matching {len(all_basenames)} unique basenames...")
    
    for code in all_basenames:
        image_path = image_files.get(code)
        label_path = label_files.get(code)
        
        # 쌍 여부 결정
        is_pair = (image_path is not None) and (label_path is not None)
        
        df_rows.append({
            "image_path": image_path,
            "label_path": label_path, 
            "code": code,
            "is_pair": is_pair
        })
    
    # DataFrame 생성
    df = pd.DataFrame(df_rows)
    
    # 통계 계산 (빈 DataFrame 처리)
    if len(df) > 0:
        pairs_count = df["is_pair"].sum()
        mismatches_count = len(df) - pairs_count
    else:
        pairs_count = 0
        mismatches_count = 0
    
    stats = {
        "images": len(image_files),
        "labels": len(label_files),
        "pairs": pairs_count,
        "mismatches": mismatches_count,
        "total_files_scanned": total_files,
        "skipped_files": skipped_files
    }
    
    # 통계 검증 및 추가 계산
    orphaned_images = stats["images"] - stats["pairs"]  # 라벨 없는 이미지
    orphaned_labels = stats["labels"] - stats["pairs"]  # 이미지 없는 라벨
    total_orphans = orphaned_images + orphaned_labels
    
    if stats["mismatches"] != total_orphans:
        logger.warning(f"Statistics inconsistency detected: mismatches={stats['mismatches']}, expected_orphans={total_orphans}")
    
    # 추가 통계 정보
    stats.update({
        "orphaned_images": orphaned_images,
        "orphaned_labels": orphaned_labels,
        "pair_rate": stats["pairs"] / len(df) if len(df) > 0 else 0.0
    })
    
    logger.info(f"Scan completed successfully: {stats}")
    
    return df, stats


def scan_dataset_summary(
    root: Union[str, Path],
    image_exts: List[str],
    label_ext: str,
    max_examples: int = 5
) -> str:
    """
    데이터셋 스캔 결과를 요약 문자열로 반환
    
    Args:
        root: 스캔할 루트 디렉토리
        image_exts: 이미지 파일 확장자 리스트
        label_ext: 라벨 파일 확장자
        max_examples: 표시할 예시 파일 개수
        
    Returns:
        str: 스캔 결과 요약 문자열
    """
    try:
        df, stats = scan_dataset(root, image_exts, label_ext)
        
        summary = f"""
Dataset Scan Summary
====================
Root Directory: {root}
Total Files Scanned: {stats['total_files_scanned']}
Skipped Files: {stats['skipped_files']}

File Counts:
- Images: {stats['images']}
- Labels: {stats['labels']}
- Matched Pairs: {stats['pairs']}
- Orphaned Files: {stats['mismatches']}
  - Orphaned Images: {stats['orphaned_images']}
  - Orphaned Labels: {stats['orphaned_labels']}

Pair Rate: {stats['pair_rate']:.2%}

Sample Files:
"""
        
        # 쌍이 있는 예시 (빈 DataFrame 처리)
        if len(df) > 0:
            paired_samples = df[df["is_pair"]].head(max_examples)
            if len(paired_samples) > 0:
                summary += "\nMatched Pairs:\n"
                for idx in range(len(paired_samples)):
                    row = paired_samples.iloc[idx]
                    summary += f"  {row['code']} -> {Path(row['image_path']).name} + {Path(row['label_path']).name}\n"
            
            # 고아 파일 예시
            orphaned_samples = df[~df["is_pair"]].head(max_examples)
            if len(orphaned_samples) > 0:
                summary += "\nOrphaned Files:\n"
                for idx in range(len(orphaned_samples)):
                    row = orphaned_samples.iloc[idx]
                    if row['image_path']:
                        summary += f"  {row['code']} -> {Path(row['image_path']).name} (no label)\n"
                    elif row['label_path']:
                        summary += f"  {row['code']} -> {Path(row['label_path']).name} (no image)\n"
        
        return summary
        
    except Exception as e:
        return f"Dataset scan failed: {e}"


def validate_scan_results(df: pd.DataFrame, stats: Dict[str, int]) -> List[str]:
    """
    스캔 결과의 일관성을 검증
    
    Args:
        df: 스캔 결과 DataFrame
        stats: 통계 딕셔너리
        
    Returns:
        List[str]: 발견된 문제점 리스트 (빈 리스트면 모든 검증 통과)
    """
    issues = []
    
    # 1. DataFrame 기본 구조 검증
    expected_columns = {"image_path", "label_path", "code", "is_pair"}
    actual_columns = set(df.columns)
    
    if expected_columns != actual_columns:
        issues.append(f"Invalid DataFrame columns: expected {expected_columns}, got {actual_columns}")
    
    # 2. 필수 통계 키 존재 확인
    required_stats = {"images", "labels", "pairs", "mismatches"}
    missing_stats = required_stats - set(stats.keys())
    
    if missing_stats:
        issues.append(f"Missing statistics: {missing_stats}")
    
    # 3. 통계 값 일관성 검증
    if len(df) > 0:
        actual_pairs = df["is_pair"].sum()
        if actual_pairs != stats["pairs"]:
            issues.append(f"Pair count mismatch: DataFrame={actual_pairs}, stats={stats['pairs']}")
        
        actual_mismatches = len(df) - actual_pairs
        if actual_mismatches != stats["mismatches"]:
            issues.append(f"Mismatch count inconsistency: DataFrame={actual_mismatches}, stats={stats['mismatches']}")
    
    # 4. 음수 값 검증
    for key, value in stats.items():
        if isinstance(value, (int, float)) and value < 0:
            issues.append(f"Negative statistic value: {key}={value}")
    
    # 5. 논리적 일관성 검증
    if "pairs" in stats and "images" in stats and "labels" in stats:
        if stats["pairs"] > min(stats["images"], stats["labels"]):
            issues.append(f"Impossible pair count: pairs={stats['pairs']} > min(images={stats['images']}, labels={stats['labels']})")
    
    return issues