#!/usr/bin/env python3
"""
Manifest 빠른 점검 스크립트
1. Train/Val 클래스 일치성 확인
2. Single/Combination 비율 확인
"""

import pandas as pd
from pathlib import Path

def check_manifests():
    # Manifest 파일 경로
    train_path = Path("/home/max16/pillsnap/artifacts/stage3/manifest_train.csv")
    val_path = Path("/home/max16/pillsnap/artifacts/stage3/manifest_val.csv")
    
    # CSV 로드
    print("📂 Manifest 파일 로딩...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"  - Train: {len(train_df):,} samples")
    print(f"  - Val: {len(val_df):,} samples")
    print()
    
    # ========== 1. 클래스 일치성 확인 ==========
    print("=" * 60)
    print("1️⃣ TRAIN/VAL 클래스 일치성 확인")
    print("=" * 60)
    
    # 유니크 클래스 추출
    train_classes = set(train_df['mapping_code'].unique())
    val_classes = set(val_df['mapping_code'].unique())
    
    print(f"📊 유니크 클래스 수:")
    print(f"  - Train: {len(train_classes)} classes")
    print(f"  - Val: {len(val_classes)} classes")
    print()
    
    # 차집합 계산
    train_only = train_classes - val_classes
    val_only = val_classes - train_classes
    common = train_classes & val_classes
    
    print(f"✅ 공통 클래스: {len(common)} classes")
    print()
    
    if train_only:
        print(f"⚠️  Train에만 있는 클래스: {len(train_only)}개")
        print(f"   {sorted(list(train_only))[:10]}")
        if len(train_only) > 10:
            print(f"   ... and {len(train_only)-10} more")
    else:
        print("✅ Train에만 있는 클래스: 없음")
    print()
    
    if val_only:
        print(f"⚠️  Val에만 있는 클래스: {len(val_only)}개")
        print(f"   {sorted(list(val_only))[:10]}")
        if len(val_only) > 10:
            print(f"   ... and {len(val_only)-10} more")
    else:
        print("✅ Val에만 있는 클래스: 없음")
    print()
    
    # 클래스 분포 통계
    print("📈 클래스별 샘플 수 통계:")
    train_class_counts = train_df['mapping_code'].value_counts()
    val_class_counts = val_df['mapping_code'].value_counts()
    
    print(f"  Train - 평균: {train_class_counts.mean():.1f}, 최소: {train_class_counts.min()}, 최대: {train_class_counts.max()}")
    print(f"  Val   - 평균: {val_class_counts.mean():.1f}, 최소: {val_class_counts.min()}, 최대: {val_class_counts.max()}")
    print()
    
    # ========== 2. Single/Combination 비율 확인 ==========
    print("=" * 60)
    print("2️⃣ SINGLE vs COMBINATION 비율")
    print("=" * 60)
    
    # Train 비율
    train_single = len(train_df[train_df['image_type'] == 'single'])
    train_combo = len(train_df[train_df['image_type'] == 'combination'])
    train_total = len(train_df)
    
    print(f"📊 Train 데이터 비율:")
    print(f"  - Single:      {train_single:6,} ({train_single/train_total*100:5.1f}%)")
    print(f"  - Combination: {train_combo:6,} ({train_combo/train_total*100:5.1f}%)")
    print(f"  - Total:       {train_total:6,}")
    print()
    
    # Val 비율
    val_single = len(val_df[val_df['image_type'] == 'single'])
    val_combo = len(val_df[val_df['image_type'] == 'combination'])
    val_total = len(val_df)
    
    print(f"📊 Val 데이터 비율:")
    print(f"  - Single:      {val_single:6,} ({val_single/val_total*100:5.1f}%)")
    print(f"  - Combination: {val_combo:6,} ({val_combo/val_total*100:5.1f}%)")
    print(f"  - Total:       {val_total:6,}")
    print()
    
    # 경고 체크
    if train_combo / train_total < 0.1:
        print("⚠️  WARNING: Train Combination 비율이 10% 미만입니다!")
    if val_combo / val_total < 0.1:
        print("⚠️  WARNING: Val Combination 비율이 10% 미만입니다!")
    
    # ========== 3. 추가 분석 ==========
    print("=" * 60)
    print("3️⃣ 추가 분석")
    print("=" * 60)
    
    # 클래스별 Single/Combo 분포
    print("📊 클래스별 이미지 타입 분포 (상위 5개 클래스):")
    for i, code in enumerate(train_class_counts.head(5).index):
        train_class_df = train_df[train_df['mapping_code'] == code]
        single_cnt = len(train_class_df[train_class_df['image_type'] == 'single'])
        combo_cnt = len(train_class_df[train_class_df['image_type'] == 'combination'])
        print(f"  {code}: Single={single_cnt}, Combo={combo_cnt}")
    
    print()
    print("✅ 점검 완료!")
    
    # 결과 요약
    print()
    print("=" * 60)
    print("📋 요약")
    print("=" * 60)
    if train_only or val_only:
        print("❌ Train/Val 클래스 불일치 발견!")
        print(f"   - 수정 필요: {len(train_only)} + {len(val_only)} = {len(train_only) + len(val_only)} 클래스")
    else:
        print("✅ Train/Val 클래스 완전 일치!")
    
    if abs((train_combo/train_total) - (val_combo/val_total)) > 0.05:
        print("⚠️  Train/Val Single/Combo 비율 차이 5% 이상")
    else:
        print("✅ Train/Val Single/Combo 비율 유사")

if __name__ == "__main__":
    check_manifests()