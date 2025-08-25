#!/usr/bin/env python3
"""
CSV 파서 견고성 테스트
목적: results.csv의 여러 형식과 상황 처리 테스트
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
import pandas as pd
from datetime import datetime

def test_csv_parser():
    """CSV 파싱 견고성 테스트"""
    print("=== CSV 파서 견고성 테스트 ===")
    
    csv_path = Path("/home/max16/pillsnap/runs/detect/train/results.csv")
    
    # 1. 파일 존재 체크
    if not csv_path.exists():
        print("❌ results.csv 파일이 없습니다")
        return False
    
    print(f"✅ CSV 파일 존재: {csv_path}")
    
    # 2. 다양한 읽기 방법 시도
    methods_tested = []
    
    # 방법 1: 기본 읽기
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 기본 읽기 성공: {len(df)} 행")
        methods_tested.append("basic")
    except Exception as e:
        print(f"❌ 기본 읽기 실패: {e}")
    
    # 방법 2: 공백 처리
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        print(f"✅ 공백 처리 읽기 성공: {len(df)} 행")
        methods_tested.append("skipinitialspace")
    except Exception as e:
        print(f"❌ 공백 처리 읽기 실패: {e}")
    
    # 방법 3: 에러 무시
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"✅ 에러 무시 읽기 성공: {len(df)} 행")
        methods_tested.append("skip_bad_lines")
    except Exception as e:
        print(f"❌ 에러 무시 읽기 실패: {e}")
    
    # 3. 컬럼명 변형 처리
    if 'basic' in methods_tested:
        df = pd.read_csv(csv_path)
        
        # YOLO 버전별 컬럼명 매핑
        column_mappings = {
            # YOLOv8 형식
            'metrics/mAP50(B)': 'map50',
            'metrics/precision(B)': 'precision', 
            'metrics/recall(B)': 'recall',
            # YOLOv11 형식  
            'metrics/mAP50': 'map50',
            'metrics/precision': 'precision',
            'metrics/recall': 'recall',
            # 학습 손실
            'train/box_loss': 'box_loss',
            'train/cls_loss': 'cls_loss',
            'train/dfl_loss': 'dfl_loss'
        }
        
        print("\n컬럼명 매핑 테스트:")
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns:
                print(f"  ✅ {old_col} → {new_col}")
            else:
                # 컬럼명에 공백이 있을 수 있음
                stripped_cols = [col.strip() for col in df.columns]
                if old_col.strip() in stripped_cols:
                    print(f"  ✅ {old_col} → {new_col} (공백 제거)")
    
    # 4. 중복 행 처리
    if 'basic' in methods_tested:
        df = pd.read_csv(csv_path)
        duplicates = df.duplicated().sum()
        unique_epochs = df['epoch'].nunique() if 'epoch' in df.columns else 0
        
        print(f"\n중복 처리:")
        print(f"  전체 행: {len(df)}")
        print(f"  중복 행: {duplicates}")
        print(f"  고유 에폭: {unique_epochs}")
        
        if duplicates > 0:
            df_clean = df.drop_duplicates()
            print(f"  ✅ 중복 제거 후: {len(df_clean)} 행")
    
    # 5. 숫자 변환 테스트
    if 'basic' in methods_tested:
        df = pd.read_csv(csv_path)
        
        print("\n숫자 변환 테스트:")
        numeric_cols = ['epoch', 'metrics/mAP50(B)', 'train/box_loss']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col], errors='coerce')
                    invalid = values.isna().sum()
                    if invalid > 0:
                        print(f"  ⚠️ {col}: {invalid}개 변환 실패")
                    else:
                        print(f"  ✅ {col}: 모든 값 숫자 변환 성공")
                except Exception as e:
                    print(f"  ❌ {col}: {e}")
    
    # 6. 마지막 유효 메트릭 추출
    if 'basic' in methods_tested:
        df = pd.read_csv(csv_path)
        
        print("\n마지막 유효 메트릭:")
        if len(df) > 0:
            last_row = df.iloc[-1]
            
            # mAP 찾기
            map_col = None
            for col in ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50']:
                if col in df.columns:
                    map_col = col
                    break
            
            if map_col:
                last_map = last_row[map_col]
                print(f"  mAP@0.5: {last_map}")
            
            # 에폭 찾기
            if 'epoch' in df.columns:
                last_epoch = last_row['epoch']
                print(f"  Epoch: {last_epoch}")
    
    print("\n✅ CSV 파서 견고성 테스트 완료")
    return True

if __name__ == "__main__":
    success = test_csv_parser()
    sys.exit(0 if success else 1)