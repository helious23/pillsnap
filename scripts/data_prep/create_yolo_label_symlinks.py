#!/usr/bin/env python3
"""
YOLO 라벨 심볼릭 링크 생성 스크립트
combination_yolo/*.txt 파일들을 올바른 이미지 경로 구조에 맞게 링크
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm

def create_yolo_label_symlinks():
    """YOLO 라벨 파일들을 이미지 구조에 맞게 심볼릭 링크 생성"""
    
    # 경로 설정
    data_root = Path('/home/max16/pillsnap_data')
    
    # 이미지 디렉토리
    img_train_dir = data_root / 'train/images/combination'
    img_val_dir = data_root / 'val/images/combination'
    
    # 원본 라벨 디렉토리 (flat structure)
    label_source_dir = data_root / 'train/labels/combination_yolo'
    
    # 타겟 라벨 디렉토리 (이미지와 같은 구조)
    label_train_dir = data_root / 'train/labels/combination'
    label_val_dir = data_root / 'val/labels/combination'
    
    print("🔧 YOLO 라벨 심볼릭 링크 생성 시작")
    print("-" * 50)
    
    # 1. 원본 라벨 파일 확인
    if not label_source_dir.exists():
        print(f"❌ 원본 라벨 디렉토리가 없습니다: {label_source_dir}")
        return False
        
    txt_files = list(label_source_dir.glob('*.txt'))
    print(f"✅ 원본 라벨 파일: {len(txt_files)}개")
    
    if not txt_files:
        print("❌ 라벨 파일이 없습니다")
        return False
    
    # 2. 기존 labels/combination 백업
    if label_train_dir.exists() and not label_train_dir.is_symlink():
        # TL_*_combo 폴더들이 있는 기존 디렉토리 백업
        backup_dir = label_train_dir.parent / 'combination_backup_json'
        if not backup_dir.exists():
            print(f"📦 기존 라벨 백업: {backup_dir}")
            shutil.move(str(label_train_dir), str(backup_dir))
        else:
            print(f"⚠️ 백업 이미 존재: {backup_dir}")
            # 기존 combination 삭제
            shutil.rmtree(label_train_dir)
    
    # 3. 새로운 라벨 디렉토리 생성
    label_train_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. train 이미지 구조 분석 및 심볼릭 링크 생성
    print("\n📂 이미지 디렉토리 구조 분석...")
    
    # TS_*_combo 폴더들
    ts_folders = sorted([d for d in img_train_dir.iterdir() if d.is_dir() and d.name.startswith('TS_')])
    print(f"   발견된 TS 폴더: {len(ts_folders)}개")
    
    created_links = 0
    missing_labels = 0
    
    for ts_folder in tqdm(ts_folders, desc="TS 폴더 처리"):
        # 해당 TS 폴더의 라벨 디렉토리 생성
        ts_label_dir = label_train_dir / ts_folder.name
        ts_label_dir.mkdir(exist_ok=True)
        
        # K-code 폴더들
        k_folders = [d for d in ts_folder.iterdir() if d.is_dir() and d.name.startswith('K-')]
        
        for k_folder in k_folders:
            # K-code 폴더의 라벨 디렉토리 생성
            k_label_dir = ts_label_dir / k_folder.name
            k_label_dir.mkdir(exist_ok=True)
            
            # 이미지 파일들
            img_files = list(k_folder.glob('*.png')) + list(k_folder.glob('*.jpg'))
            
            for img_file in img_files:
                # 대응하는 라벨 파일명
                label_name = img_file.stem + '.txt'
                source_label = label_source_dir / label_name
                
                if source_label.exists():
                    # 심볼릭 링크 생성
                    target_label = k_label_dir / label_name
                    if not target_label.exists():
                        target_label.symlink_to(source_label)
                        created_links += 1
                else:
                    missing_labels += 1
    
    print(f"\n📊 결과:")
    print(f"   ✅ 생성된 심볼릭 링크: {created_links}개")
    print(f"   ⚠️ 라벨 없는 이미지: {missing_labels}개")
    
    # 5. validation 디렉토리 처리 (필요시)
    if img_val_dir.exists():
        print("\n📂 Validation 디렉토리 처리...")
        # val도 train과 같은 라벨 사용 (스모크 테스트용)
        if not label_val_dir.exists():
            label_val_dir.symlink_to(label_train_dir)
            print(f"   ✅ Val 라벨 링크: {label_val_dir} -> {label_train_dir}")
    
    # 6. 검증
    print("\n🔍 검증 중...")
    
    # 샘플 확인
    sample_ts = ts_folders[0] if ts_folders else None
    if sample_ts:
        sample_k = next(sample_ts.iterdir(), None)
        if sample_k and sample_k.is_dir():
            sample_img = next(sample_k.glob('*.png'), next(sample_k.glob('*.jpg'), None))
            if sample_img:
                expected_label = label_train_dir / sample_ts.name / sample_k.name / (sample_img.stem + '.txt')
                if expected_label.exists():
                    print(f"   ✅ 샘플 확인 성공:")
                    print(f"      이미지: {sample_img.relative_to(data_root)}")
                    print(f"      라벨: {expected_label.relative_to(data_root)}")
                    
                    # 라벨 내용 확인
                    with open(expected_label) as f:
                        content = f.read().strip()
                        print(f"      내용: {content}")
                else:
                    print(f"   ❌ 샘플 라벨 없음: {expected_label}")
    
    return created_links > 0


def verify_yolo_structure():
    """YOLO가 기대하는 구조 확인"""
    
    data_root = Path('/home/max16/pillsnap_data')
    
    # 이미지와 라벨 짝 확인
    img_dir = data_root / 'train/images/combination/TS_1_combo'
    label_dir = data_root / 'train/labels/combination/TS_1_combo'
    
    if img_dir.exists() and label_dir.exists():
        # 첫 번째 K-folder 확인
        k_folder = next(img_dir.iterdir(), None)
        if k_folder and k_folder.is_dir():
            img_files = list(k_folder.glob('*.png'))[:3]
            
            print("\n✅ YOLO 구조 확인:")
            for img in img_files:
                label_path = label_dir / k_folder.name / (img.stem + '.txt')
                if label_path.exists():
                    print(f"   ✓ {img.name} -> {label_path.name}")
                else:
                    print(f"   ✗ {img.name} -> 라벨 없음")
    else:
        print("\n❌ 디렉토리 구조 확인 실패")


if __name__ == "__main__":
    # 심볼릭 링크 생성
    success = create_yolo_label_symlinks()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ YOLO 라벨 심볼릭 링크 생성 완료!")
        
        # 구조 검증
        verify_yolo_structure()
        
        print("\n💡 다음 단계:")
        print("   1. python scripts/test_detection_smoke.py")
        print("   2. Detection 학습 시작")
    else:
        print("\n❌ 심볼릭 링크 생성 실패")