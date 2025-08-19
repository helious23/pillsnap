#!/usr/bin/env python3
"""
Step 3-4 스모크 테스트 스크립트

목적: 실제 config.yaml의 data.root에서 소규모 스캔 테스트
주의: 대용량 데이터셋 고려하여 최대 500개 파일만 샘플링
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from dataset.scan import scan_dataset
import os

def limited_scan_dataset(root, image_exts, label_ext, max_files=500):
    """
    스캔 작업을 최대 파일 수로 제한하는 래퍼 함수
    """
    print(f"🔍 Starting limited scan (max {max_files} files)...")
    
    # 파일 수집 제한을 위한 카운터
    file_count = 0
    
    # 원본 os.walk를 대체하는 제한된 버전
    original_walk = os.walk
    
    def limited_walk(path):
        nonlocal file_count
        for dirpath, dirnames, filenames in original_walk(path):
            if file_count >= max_files:
                print(f"⚠️  File limit ({max_files}) reached, stopping scan...")
                break
            
            # 파일 수 제한
            limited_filenames = filenames[:max(1, max_files - file_count)]
            file_count += len(limited_filenames)
            
            yield dirpath, dirnames, limited_filenames
    
    # 임시로 os.walk 교체
    os.walk = limited_walk
    
    try:
        result = scan_dataset(root, image_exts, label_ext)
        return result
    finally:
        # 원본 os.walk 복원
        os.walk = original_walk


def main():
    print("=== PillSnap Dataset Scan Smoke Test ===")
    print()
    
    # 1. 설정 로드
    print("📋 Loading configuration...")
    try:
        cfg = load_config()
        print(f"✅ Config loaded successfully")
        print(f"   data.root: {cfg.data.root}")
        print(f"   pipeline_mode: {cfg.data.pipeline_mode}")
        print(f"   default_mode: {cfg.data.default_mode}")
        print(f"   image_exts: {cfg.data.image_exts}")
        print(f"   label_ext: {cfg.data.label_ext}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    print()
    
    # 2. 경로 존재 확인
    data_root = Path(cfg.data.root)
    if not data_root.exists():
        print(f"❌ Data root does not exist: {data_root}")
        return 1
    
    print(f"📁 Data root exists: {data_root}")
    
    # 3. 제한된 스캔 실행
    print()
    print("🚀 Starting smoke test scan...")
    try:
        df, stats = limited_scan_dataset(
            root=cfg.data.root,
            image_exts=cfg.data.image_exts,
            label_ext=cfg.data.label_ext,
            max_files=500
        )
        
        print("✅ Scan completed successfully!")
        
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        return 1
    
    # 4. 결과 요약 출력
    print()
    print("📊 Scan Results Summary:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # 5. DataFrame 샘플 출력 (최대 5개)
    print()
    print("📋 Sample Data (max 5 rows):")
    print("-" * 40)
    if len(df) > 0:
        sample_df = df.head(5)
        for idx, row in sample_df.iterrows():
            img_name = Path(row['image_path']).name if row['image_path'] else "None"
            lbl_name = Path(row['label_path']).name if row['label_path'] else "None"
            pair_status = "✅" if row['is_pair'] else "❌"
            print(f"   {pair_status} {row['code']}: {img_name} + {lbl_name}")
    else:
        print("   (No data found)")
    
    print()
    print("✅ Smoke test completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())