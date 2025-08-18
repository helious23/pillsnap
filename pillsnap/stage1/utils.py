"""
Stage 1 유틸리티 함수들

목적: Stage 1 파이프라인에서 사용되는 도우미 함수들
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


def build_edi_classes(
    manifest_csv: Union[str, Path],
    outfile: Union[str, Path] = "artifacts/classes_step11.json",
) -> Dict[str, int]:
    """
    매니페스트에서 EDI 코드를 추출하여 클래스 맵 생성

    목적: EDI 코드를 class_id로 매핑하는 사전 생성
    입력: manifest CSV 파일 (edi_code 컬럼 필요)
    출력: {edi_code: class_id} 사전을 JSON으로 저장
    검증: EDI 코드 정렬 후 일관된 class_id 부여

    Args:
        manifest_csv: 매니페스트 CSV 파일 경로
        outfile: 출력 JSON 파일 경로

    Returns:
        Dict[str, int]: {edi_code: class_id} 매핑 사전
    """
    print(f"📊 Loading manifest from: {manifest_csv}")

    # CSV 로드
    try:
        df = pd.read_csv(manifest_csv)
        print(f"✅ Loaded {len(df)} rows from manifest")
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise

    # edi_code 컬럼 확인
    if "edi_code" not in df.columns:
        raise ValueError("edi_code column not found in manifest")

    # EDI 코드 추출 및 정규화
    edi_codes = df["edi_code"].dropna()

    # 문자열로 변환 및 정규화
    edi_codes = edi_codes.astype(str).str.strip()

    # 빈 문자열 제거
    edi_codes = edi_codes[edi_codes != ""]

    # 고유값 추출 및 정렬
    unique_edi = sorted(edi_codes.unique())

    excluded_count = len(df) - len(edi_codes)
    if excluded_count > 0:
        print(f"⚠️  Excluded {excluded_count} rows with missing/empty EDI codes")

    # class_id 매핑 생성
    class_map = {edi: idx for idx, edi in enumerate(unique_edi)}

    print(f"✅ Created class map: {len(class_map)} unique EDI codes")

    # 샘플 출력
    if len(class_map) > 0:
        sample_keys = list(class_map.keys())
        first_5 = sample_keys[:5]
        last_5 = sample_keys[-5:] if len(sample_keys) > 5 else []

        print(f"   First 5 EDI codes: {first_5}")
        if last_5:
            print(f"   Last 5 EDI codes: {last_5}")

    # JSON 저장
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved class map to: {outfile}")
    print(f"📈 Total {len(class_map)} EDI codes → class_id mapping")

    return class_map


def validate_class_map(
    class_map_file: Union[str, Path], manifest_csv: Optional[Union[str, Path]] = None
) -> Dict[str, any]:
    """
    클래스 맵 검증

    Args:
        class_map_file: 클래스 맵 JSON 파일 경로
        manifest_csv: 검증에 사용할 매니페스트 CSV (선택)

    Returns:
        검증 결과 딕셔너리
    """
    results = {"valid": True, "num_classes": 0, "issues": []}

    # JSON 로드
    try:
        with open(class_map_file, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        results["num_classes"] = len(class_map)
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Failed to load class map: {e}")
        return results

    # 기본 검증
    if len(class_map) == 0:
        results["valid"] = False
        results["issues"].append("Class map is empty")
        return results

    # class_id 연속성 검증
    class_ids = sorted(class_map.values())
    expected_ids = list(range(len(class_map)))

    if class_ids != expected_ids:
        results["valid"] = False
        results["issues"].append("Class IDs are not continuous from 0")

    # 매니페스트와 교차 검증 (선택)
    if manifest_csv:
        try:
            df = pd.read_csv(manifest_csv)
            manifest_edi = set(df["edi_code"].dropna().astype(str).str.strip())
            manifest_edi.discard("")

            class_edi = set(class_map.keys())

            # 매니페스트에만 있는 EDI
            only_in_manifest = manifest_edi - class_edi
            if only_in_manifest:
                results["issues"].append(
                    f"{len(only_in_manifest)} EDI codes in manifest but not in class map"
                )

            # 클래스 맵에만 있는 EDI
            only_in_class = class_edi - manifest_edi
            if only_in_class:
                results["issues"].append(
                    f"{len(only_in_class)} EDI codes in class map but not in manifest"
                )
        except Exception as e:
            results["issues"].append(f"Failed to validate against manifest: {e}")

    return results
