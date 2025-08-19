#!/bin/bash

# Stage 1 이미지만 SSD로 이전하는 스크립트 (간단 버전)
set -euo pipefail

echo "🚀 Stage 1 이미지만 SSD 이전 시작..."

# 경로 설정
SRC_DATA_ROOT="/mnt/data/pillsnap_dataset"
DST_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"
STAGE1_JSON="/home/max16/pillsnap/artifacts/stage1/sampling/stage1_sample.json"

# 기존 SSD 데이터 정리
if [ -d "$DST_DATA_ROOT" ]; then
    echo "📦 기존 SSD 데이터 정리..."
    rm -rf "$DST_DATA_ROOT"
fi

# 디렉토리 생성
echo "📁 SSD 디렉토리 구조 생성..."
mkdir -p "$DST_DATA_ROOT"

# Python을 통해 이미지만 복사
echo "📋 Stage 1 이미지 복사 시작..."
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

print(f'📊 복사할 K-code 수: {len(k_codes)}')
print(f'📊 총 이미지 수: {total_single}')

sample_count = 0
for i, k_code in enumerate(k_codes):
    if i % 10 == 0:
        print(f'진행률: {i}/{len(k_codes)} K-codes ({i*100//len(k_codes):.1f}%) - {sample_count} 이미지 완료')
    
    k_data = stage1_data['samples'][k_code]
    
    # Single 이미지만 처리
    for img_path in k_data['single_images']:
        sample_count += 1
        
        # 이미지 파일 복사
        src_img = Path(img_path)
        dst_img = dst_root / src_img.relative_to(src_root)
        
        try:
            # 디렉토리 생성
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_files += 1
            else:
                print(f'⚠️ 이미지 파일 없음: {src_img}')
                failed_files += 1
                
        except Exception as e:
            print(f'❌ 복사 실패 {k_code}: {e}')
            failed_files += 1

print(f'✅ 이미지 복사 완료!')
print(f'✅ 성공: {copied_files}개')
print(f'❌ 실패: {failed_files}개')
"

echo "🔍 SSD 데이터 검증..."
echo "Images: $(find "$DST_DATA_ROOT" -name '*.png' | wc -l)"
echo "Total files: $(find "$DST_DATA_ROOT" -type f | wc -l)"
echo "Disk usage: $(du -sh "$DST_DATA_ROOT")"

echo "✅ Stage 1 이미지 SSD 이전 완료!"