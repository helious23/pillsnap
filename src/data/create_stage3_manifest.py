#!/usr/bin/env python3
"""
Stage 3 훈련용 Manifest 생성기 (Classification 중심)

Stage 3 Progressive Validation을 위한 manifest 생성:
- 100,000개 이미지 선택
- 1,000개 클래스 균등 분포
- Single/Combination 비율 95:5 (Classification 중심)
- 물리적 복사 없이 원본 경로 참조
- Detection 최소화, Classification 성능 극대화 전략
"""

import os
import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import hashlib
from datetime import datetime

from src.utils.core import PillSnapLogger


class Stage3ManifestCreator:
    """Stage 3 Manifest 생성기"""
    
    def __init__(self, data_root: Optional[str] = None):
        self.logger = PillSnapLogger(__name__)
        
        # 환경변수 우선, 기본값 대체
        self.data_root = Path(data_root or os.getenv('PILLSNAP_DATA_ROOT', '/home/max16/pillsnap_data'))
        
        # Stage 3 설정 (Classification 중심)
        self.target_samples = 100000
        self.target_classes = 1000
        self.samples_per_class = 100  # 100000 ÷ 1000 = 100
        self.single_ratio = 0.95  # Single 95%, Combination 5% (현실적 비율)
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.seed = 42
        
        # 출력 경로
        self.output_dir = Path("artifacts/stage3")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플링 상태
        self.available_classes = {}
        self.selected_classes = []
        self.manifest_data = []
        
        random.seed(self.seed)
        self.logger.info(f"Stage 3 Manifest Creator 초기화 (Classification 중심)")
        self.logger.info(f"데이터 루트: {self.data_root}")
        self.logger.info(f"목표: {self.target_samples:,}개 샘플, {self.target_classes:,}개 클래스")
        self.logger.info(f"데이터 비율: Single {self.single_ratio:.1%}, Combination {1-self.single_ratio:.1%}")
    
    def scan_available_data(self) -> Dict[str, Dict]:
        """사용 가능한 데이터 스캔"""
        self.logger.info("데이터 스캔 시작...")
        
        available_data = defaultdict(lambda: {'single': [], 'combination': []})
        
        # Single 이미지 스캔
        single_dir = self.data_root / "train/images/single"
        if single_dir.exists():
            for ts_dir in sorted(single_dir.glob("TS_*")):
                if not ts_dir.is_dir():
                    continue
                    
                for k_code_dir in ts_dir.iterdir():
                    if not k_code_dir.is_dir():
                        continue
                        
                    k_code = k_code_dir.name
                    images = list(k_code_dir.glob("*.jpg")) + list(k_code_dir.glob("*.png"))
                    
                    if images:
                        available_data[k_code]['single'].extend(images)
                        
        # Combination 이미지 스캔
        combo_dir = self.data_root / "train/images/combination"
        if combo_dir.exists():
            for ts_dir in sorted(combo_dir.glob("TS_*")):
                if not ts_dir.is_dir():
                    continue
                    
                for k_code_dir in ts_dir.iterdir():
                    if not k_code_dir.is_dir():
                        continue
                        
                    k_code = k_code_dir.name
                    images = list(k_code_dir.glob("*.jpg")) + list(k_code_dir.glob("*.png"))
                    
                    if images:
                        available_data[k_code]['combination'].extend(images)
        
        # 통계 출력
        total_classes = len(available_data)
        total_single = sum(len(v['single']) for v in available_data.values())
        total_combo = sum(len(v['combination']) for v in available_data.values())
        
        self.logger.info(f"스캔 완료: {total_classes:,}개 클래스")
        self.logger.info(f"  Single: {total_single:,}개 이미지")
        self.logger.info(f"  Combination: {total_combo:,}개 이미지")
        
        self.available_classes = dict(available_data)
        return self.available_classes
    
    def select_target_classes(self) -> List[str]:
        """Stage 3용 1000개 클래스 선택"""
        self.logger.info(f"목표 {self.target_classes:,}개 클래스 선택...")
        
        # 충분한 이미지가 있는 클래스만 필터링 (최소 80개)
        valid_classes = []
        for k_code, data in self.available_classes.items():
            total_images = len(data['single']) + len(data['combination'])
            if total_images >= 80:  # 최소 요구 이미지 수
                valid_classes.append((k_code, total_images))
        
        if len(valid_classes) < self.target_classes:
            self.logger.warning(
                f"충분한 클래스 부족: 필요 {self.target_classes}, 사용가능 {len(valid_classes)}"
            )
            # 사용 가능한 만큼만 사용
            self.target_classes = len(valid_classes)
        
        # 이미지 수가 많은 순으로 정렬 후 상위 N개 선택
        valid_classes.sort(key=lambda x: x[1], reverse=True)
        self.selected_classes = [k_code for k_code, _ in valid_classes[:self.target_classes]]
        
        self.logger.info(f"선택된 {len(self.selected_classes)}개 클래스")
        return self.selected_classes
    
    def sample_images_for_class(self, k_code: str) -> List[Dict]:
        """특정 클래스에서 이미지 샘플링"""
        class_data = self.available_classes[k_code]
        single_images = class_data['single']
        combo_images = class_data['combination']
        
        # 목표 샘플 수 계산
        target_single = int(self.samples_per_class * self.single_ratio)
        target_combo = self.samples_per_class - target_single
        
        # Single 샘플링
        sampled_single = []
        if single_images:
            n_sample = min(target_single, len(single_images))
            sampled_single = random.sample(single_images, n_sample)
        
        # Combination 샘플링 (현실적 제한 고려)
        sampled_combo = []
        if combo_images:
            # 가용한 Combination 데이터만큼만 사용 (최대 5개)
            n_sample = min(target_combo, len(combo_images), 5)  # 현실적 제한
            if n_sample > 0:
                sampled_combo = random.sample(combo_images, n_sample)
        
        # 부족한 경우 Single에서 보충 (Classification 중심 전략)
        total_sampled = len(sampled_single) + len(sampled_combo)
        if total_sampled < self.samples_per_class:
            shortage = self.samples_per_class - total_sampled
            
            # Single에서 추가 샘플링 (우선)
            if single_images and len(sampled_single) < len(single_images):
                additional = min(shortage, len(single_images) - len(sampled_single))
                remaining = [img for img in single_images if img not in sampled_single]
                if len(remaining) >= additional:
                    sampled_single.extend(random.sample(remaining, additional))
                else:
                    sampled_single.extend(remaining)  # 모든 남은 이미지 사용
                shortage = self.samples_per_class - len(sampled_single) - len(sampled_combo)
            
            # 여전히 부족하면 Single 이미지 중복 사용 허용
            if shortage > 0 and single_images:
                additional_needed = shortage
                for _ in range(additional_needed):
                    sampled_single.append(random.choice(single_images))
        
        # Manifest 레코드 생성
        records = []
        
        for img_path in sampled_single:
            records.append({
                'image_path': str(img_path),
                'mapping_code': k_code,
                'image_type': 'single',
                'source': 'train'
            })
        
        for img_path in sampled_combo:
            records.append({
                'image_path': str(img_path),
                'mapping_code': k_code,
                'image_type': 'combination',
                'source': 'train'
            })
        
        return records
    
    def create_manifest(self) -> pd.DataFrame:
        """Stage 3 Manifest 생성"""
        self.logger.info("Manifest 생성 시작...")
        
        # 1. 데이터 스캔
        if not self.available_classes:
            self.scan_available_data()
        
        # 2. 클래스 선택
        if not self.selected_classes:
            self.select_target_classes()
        
        # 3. 각 클래스별 샘플링
        all_records = []
        for idx, k_code in enumerate(self.selected_classes):
            if (idx + 1) % 100 == 0:
                self.logger.info(f"진행: {idx + 1}/{len(self.selected_classes)} 클래스")
            
            class_records = self.sample_images_for_class(k_code)
            all_records.extend(class_records)
        
        # 4. DataFrame 생성
        df = pd.DataFrame(all_records)
        
        # 5. 셔플링
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # 6. 통계 출력
        self.logger.info(f"Manifest 생성 완료:")
        self.logger.info(f"  총 샘플: {len(df):,}개")
        self.logger.info(f"  클래스 수: {df['mapping_code'].nunique():,}개")
        self.logger.info(f"  Single: {(df['image_type'] == 'single').sum():,}개")
        self.logger.info(f"  Combination: {(df['image_type'] == 'combination').sum():,}개")
        
        self.manifest_data = df
        return df
    
    def split_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Validation 분할 (클래스별 균등)"""
        train_records = []
        val_records = []
        
        # 클래스별로 분할
        for k_code in df['mapping_code'].unique():
            class_df = df[df['mapping_code'] == k_code]
            n_val = max(1, int(len(class_df) * self.val_ratio))
            
            # 셔플 후 분할
            class_df = class_df.sample(frac=1, random_state=self.seed)
            val_records.append(class_df.iloc[:n_val])
            train_records.append(class_df.iloc[n_val:])
        
        train_df = pd.concat(train_records, ignore_index=True)
        val_df = pd.concat(val_records, ignore_index=True)
        
        # 다시 셔플
        train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        self.logger.info(f"Train/Val 분할:")
        self.logger.info(f"  Train: {len(train_df):,}개 ({len(train_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Val: {len(val_df):,}개 ({len(val_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df
    
    def save_manifests(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Manifest 파일 저장"""
        # Train manifest
        train_path = self.output_dir / "manifest_train.csv"
        train_df.to_csv(train_path, index=False)
        self.logger.info(f"Train manifest 저장: {train_path}")
        
        # Validation manifest
        val_path = self.output_dir / "manifest_val.csv"
        val_df.to_csv(val_path, index=False)
        self.logger.info(f"Val manifest 저장: {val_path}")
        
        # 클래스 매핑 저장
        class_mapping = {
            k_code: idx for idx, k_code in enumerate(sorted(train_df['mapping_code'].unique()))
        }
        
        mapping_path = self.output_dir / "class_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        self.logger.info(f"클래스 매핑 저장: {mapping_path}")
        
        # 통계 리포트 저장
        stats = {
            'timestamp': datetime.now().isoformat(),
            'stage': 3,
            'total_samples': len(train_df) + len(val_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'num_classes': len(class_mapping),
            'single_ratio': (train_df['image_type'] == 'single').mean(),
            'combination_ratio': (train_df['image_type'] == 'combination').mean(),
            'samples_per_class': self.samples_per_class,
            'data_root': str(self.data_root),
            'manifest_checksum': {
                'train': hashlib.md5(open(train_path, 'rb').read()).hexdigest(),
                'val': hashlib.md5(open(val_path, 'rb').read()).hexdigest()
            }
        }
        
        stats_path = self.output_dir / "sampling_report.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"샘플링 리포트 저장: {stats_path}")
        
        return train_path, val_path, mapping_path, stats_path
    
    def run(self) -> Dict[str, Path]:
        """전체 Manifest 생성 프로세스 실행"""
        self.logger.info("=" * 60)
        self.logger.info("Stage 3 Manifest 생성 시작")
        self.logger.info("=" * 60)
        
        # 1. Manifest 생성
        df = self.create_manifest()
        
        # 2. Train/Val 분할
        train_df, val_df = self.split_train_val(df)
        
        # 3. 파일 저장
        train_path, val_path, mapping_path, stats_path = self.save_manifests(train_df, val_df)
        
        self.logger.success("Stage 3 Manifest 생성 완료!")
        
        return {
            'train_manifest': train_path,
            'val_manifest': val_path,
            'class_mapping': mapping_path,
            'sampling_report': stats_path
        }


def main():
    """CLI 엔트리포인트"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3 Manifest 생성기")
    parser.add_argument(
        '--data-root', 
        type=str, 
        default=None,
        help='데이터 루트 경로 (기본: 환경변수 PILLSNAP_DATA_ROOT)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 파일 생성 없이 검증만 수행'
    )
    
    args = parser.parse_args()
    
    # Manifest 생성기 실행
    creator = Stage3ManifestCreator(data_root=args.data_root)
    
    if args.dry_run:
        print("🔍 Dry Run 모드 - 데이터 스캔만 수행")
        available_classes = creator.scan_available_data()
        selected_classes = creator.select_target_classes()
        print(f"✅ 사용 가능 클래스: {len(available_classes):,}개")
        print(f"✅ 선택될 클래스: {len(selected_classes)}개")
        print(f"✅ 예상 샘플 수: {creator.target_samples:,}개")
    else:
        results = creator.run()
        print("\n📋 생성된 파일:")
        for name, path in results.items():
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()