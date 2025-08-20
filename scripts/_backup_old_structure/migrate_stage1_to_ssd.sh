#!/bin/bash

# Stage 1 데이터를 SSD로 이전하는 스크립트
set -euo pipefail

echo "🚀 Stage 1 데이터 SSD 이전 시작..."

# 경로 설정
SRC_DATA_ROOT="/mnt/data/pillsnap_dataset"
DST_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"
STAGE1_JSON="/home/max16/pillsnap/artifacts/stage1/sampling/stage1_sample.json"

# 기존 SSD 데이터 백업
if [ -d "$DST_DATA_ROOT" ]; then
    echo "📦 기존 SSD 데이터 백업..."
    mv "$DST_DATA_ROOT" "$DST_DATA_ROOT.backup.$(date +%Y%m%d_%H%M%S)"
fi

# 디렉토리 생성
echo "📁 SSD 디렉토리 구조 생성..."
mkdir -p "$DST_DATA_ROOT/data/train/images/single"
mkdir -p "$DST_DATA_ROOT/data/train/labels/single"

# Python을 통해 JSON 파싱하여 파일 복사
echo "📋 Stage 1 매니페스트에서 파일 목록 추출..."
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"

$VENV_PYTHON -c "
import json
import shutil
import os
from pathlib import Path

# Stage 1 JSON 파일 읽기
with open('$STAGE1_JSON', 'r', encoding='utf-8') as f:
    stage1_data = json.load(f)

src_root = Path('$SRC_DATA_ROOT')
dst_root = Path('$DST_DATA_ROOT')

total_files = 0
copied_files = 0
failed_files = 0

# K-code별로 반복
k_codes = list(stage1_data['samples'].keys())
total_single = sum(len(stage1_data['samples'][k]['single_images']) for k in k_codes)
total_combo = sum(len(stage1_data['samples'][k]['combo_images']) for k in k_codes)

print(f'📊 복사할 K-code 수: {len(k_codes)}')
print(f'📊 총 Single 이미지 수: {total_single}')
print(f'📊 총 Combo 이미지 수: {total_combo}')
print(f'📊 총 이미지 수: {total_single + total_combo}')

sample_count = 0
for i, k_code in enumerate(k_codes):
    if i % 10 == 0:
        print(f'진행률: {i}/{len(k_codes)} K-codes ({i*100//len(k_codes):.1f}%) - {sample_count} 파일 완료')
    
    k_data = stage1_data['samples'][k_code]
    
    # Single 이미지 처리
    for img_path in k_data['single_images']:
        sample_count += 1
        
        # 이미지 파일 복사
        src_img = Path(img_path)
        dst_img = dst_root / src_img.relative_to(src_root)
        
        # 레이블 파일 경로 생성 (TS_* -> TL_*)
        ts_name = src_img.parent.parent.name  # TS_13_single
        tl_name = ts_name.replace('TS_', 'TL_')  # TL_13_single
        src_label = src_img.parent.parent.parent.parent / 'labels' / 'single' / tl_name / f'{src_img.parent.name}_json' / f'{src_img.stem}.json'
        dst_label = dst_root / src_label.relative_to(src_root)
        
        try:
            # 디렉토리 생성
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                total_files += 1
                copied_files += 1
            else:
                print(f'⚠️ 이미지 파일 없음: {src_img}')
                failed_files += 1
                
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                total_files += 1
                copied_files += 1
            else:
                print(f'⚠️ 레이블 파일 없음: {src_label}')
                failed_files += 1
                
        except Exception as e:
            print(f'❌ 복사 실패 {k_code}: {e}')
            failed_files += 1
    
    # Combo 이미지 처리 (있는 경우)
    for img_path in k_data['combo_images']:
        sample_count += 1
        
        # 이미지 파일 복사
        src_img = Path(img_path)
        dst_img = dst_root / src_img.relative_to(src_root)
        
        # 레이블 파일 경로 생성 (TC_* -> combination)
        tc_name = src_img.parent.parent.name  # TC_*_combo
        src_label = src_img.parent.parent.parent.parent / 'labels' / 'combination' / tc_name / f'{src_img.parent.name}_json' / f'{src_img.stem}.json'
        dst_label = dst_root / src_label.relative_to(src_root)
        
        try:
            # 디렉토리 생성
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                total_files += 1
                copied_files += 1
            else:
                print(f'⚠️ Combo 이미지 파일 없음: {src_img}')
                failed_files += 1
                
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                total_files += 1
                copied_files += 1
            else:
                print(f'⚠️ Combo 레이블 파일 없음: {src_label}')
                failed_files += 1
                
        except Exception as e:
            print(f'❌ Combo 복사 실패 {k_code}: {e}')
            failed_files += 1

print(f'✅ 복사 완료!')
print(f'📊 총 파일: {total_files}개')
print(f'✅ 성공: {copied_files}개')
print(f'❌ 실패: {failed_files}개')
"

echo "🔍 SSD 데이터 검증..."
echo "Images: $(find "$DST_DATA_ROOT" -name '*.png' | wc -l)"
echo "Labels: $(find "$DST_DATA_ROOT" -name '*.json' | wc -l)"
echo "Total files: $(find "$DST_DATA_ROOT" -type f | wc -l)"

echo "✅ Stage 1 데이터 SSD 이전 완료!"