#!/usr/bin/env python3
"""
Stage 1과 Stage 2 클래스 중복 확인 스크립트
"""

import json
from pathlib import Path

def check_stage_overlap():
    # Stage 1 클래스 로드
    stage1_file = Path("artifacts/stage1/sampling/stage1_sample.json")
    stage2_file = Path("artifacts/stage2/sampling/stage2_sample.json")
    
    with open(stage1_file, 'r') as f:
        stage1_data = json.load(f)
    
    with open(stage2_file, 'r') as f:
        stage2_data = json.load(f)
    
    stage1_classes = set(stage1_data['metadata']['selected_classes'])
    stage2_classes = set(stage2_data['metadata']['selected_classes'])
    
    print(f"Stage 1 클래스 수: {len(stage1_classes)}")
    print(f"Stage 2 클래스 수: {len(stage2_classes)}")
    
    # 중복 확인
    overlap = stage1_classes & stage2_classes
    print(f"중복 클래스 수: {len(overlap)}")
    
    if overlap:
        print("중복 클래스들:")
        for cls in sorted(overlap):
            print(f"  {cls}")
    else:
        print("✅ 중복 없음 - 완전히 다른 클래스들")
    
    # Stage 2 전용 클래스
    stage2_only = stage2_classes - stage1_classes
    print(f"Stage 2 전용 클래스 수: {len(stage2_only)}")
    
    return len(overlap), len(stage1_classes), len(stage2_classes)

if __name__ == "__main__":
    check_stage_overlap()