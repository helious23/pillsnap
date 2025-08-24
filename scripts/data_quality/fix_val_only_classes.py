#!/usr/bin/env python3
"""
Val-only 클래스 처리 유틸리티
- Train에 없고 Val에만 있는 클래스 검출
- Val에서 제거 또는 Train에 추가 옵션
"""

import argparse
import random
from typing import Dict, List, Set, Tuple

import pandas as pd

from base import DataQualityBase


class ValOnlyClassHandler(DataQualityBase):
    """Val-only 클래스 처리"""
    
    def __init__(self, args):
        super().__init__(args)
        self.val_only_classes: Set[str] = set()
        self.train_only_classes: Set[str] = set()
        
    def find_class_mismatches(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
        """Train/Val 클래스 불일치 찾기"""
        train_classes = set(train_df['mapping_code'].unique())
        val_classes = set(val_df['mapping_code'].unique())
        
        train_only = train_classes - val_classes
        val_only = val_classes - train_classes
        common = train_classes & val_classes
        
        self.logger.info(f"Class distribution:")
        self.logger.info(f"  - Train classes: {len(train_classes)}")
        self.logger.info(f"  - Val classes: {len(val_classes)}")
        self.logger.info(f"  - Common classes: {len(common)}")
        self.logger.info(f"  - Train-only classes: {len(train_only)}")
        self.logger.info(f"  - Val-only classes: {len(val_only)}")
        
        return train_only, val_only
    
    def remove_val_only_classes(self, val_df: pd.DataFrame, val_only_classes: Set[str]) -> pd.DataFrame:
        """Val-only 클래스 제거"""
        initial_count = len(val_df)
        
        # Val-only 클래스 필터링
        val_df_clean = val_df[~val_df['mapping_code'].isin(val_only_classes)].copy()
        
        removed_count = initial_count - len(val_df_clean)
        self.logger.info(f"Removed {removed_count} samples from {len(val_only_classes)} val-only classes")
        
        return val_df_clean
    
    def add_to_train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                    val_only_classes: Set[str], max_per_class: int) -> pd.DataFrame:
        """Val-only 클래스 샘플을 Train에 추가"""
        samples_to_add = []
        
        for class_code in val_only_classes:
            class_samples = val_df[val_df['mapping_code'] == class_code]
            
            # 샘플 수 제한
            if len(class_samples) > max_per_class:
                # 랜덤 샘플링
                class_samples = class_samples.sample(n=max_per_class, random_state=42)
            
            samples_to_add.append(class_samples)
            self.logger.info(f"  Adding {len(class_samples)} samples of class {class_code} to train")
        
        if samples_to_add:
            # Train에 추가
            added_df = pd.concat(samples_to_add, ignore_index=True)
            train_df_extended = pd.concat([train_df, added_df], ignore_index=True)
            
            self.logger.info(f"Added total {len(added_df)} samples to train set")
            return train_df_extended
        
        return train_df
    
    def analyze_class_distribution(self, df: pd.DataFrame, name: str) -> Dict:
        """클래스 분포 분석"""
        class_counts = df['mapping_code'].value_counts()
        
        stats = {
            "total_samples": len(df),
            "unique_classes": len(class_counts),
            "mean_samples_per_class": class_counts.mean(),
            "median_samples_per_class": class_counts.median(),
            "min_samples": class_counts.min(),
            "max_samples": class_counts.max(),
            "std_samples": class_counts.std()
        }
        
        # 상위/하위 5개 클래스
        top5 = class_counts.head(5).to_dict()
        bottom5 = class_counts.tail(5).to_dict()
        
        return {
            "stats": stats,
            "top5": top5,
            "bottom5": bottom5
        }
    
    def run(self):
        """메인 실행"""
        self.console.print("[bold cyan]🔧 Val-only Class Handler[/bold cyan]")
        self.console.print(f"Mode: {'DRY RUN' if self.args.dry_run else 'ACTUAL EXECUTION'}")
        self.console.print(f"Strategy: {self.args.mode}")
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # 클래스 불일치 찾기
        train_only, val_only = self.find_class_mismatches(train_df, val_df)
        self.val_only_classes = val_only
        self.train_only_classes = train_only
        
        # Val-only 클래스 상세 정보
        if val_only:
            self.console.print(f"\n[yellow]Found {len(val_only)} val-only classes:[/yellow]")
            val_only_details = {}
            for class_code in sorted(val_only)[:10]:  # 처음 10개만 표시
                count = len(val_df[val_df['mapping_code'] == class_code])
                val_only_details[class_code] = count
                self.console.print(f"  - {class_code}: {count} samples")
            if len(val_only) > 10:
                self.console.print(f"  ... and {len(val_only) - 10} more")
        
        # 전처리 통계
        before_stats = {
            "Train samples": len(train_df),
            "Val samples": len(val_df),
            "Train classes": len(train_df['mapping_code'].unique()),
            "Val classes": len(val_df['mapping_code'].unique()),
            "Val-only classes": len(val_only),
            "Train-only classes": len(train_only)
        }
        
        # 처리 모드에 따른 실행
        if self.args.mode == "remove":
            # Val에서 제거
            val_df_clean = self.remove_val_only_classes(val_df, val_only)
            train_df_clean = train_df
            
        elif self.args.mode == "add-to-train":
            # Train에 추가
            train_df_clean = self.add_to_train(train_df, val_df, val_only, self.args.max_per_class)
            val_df_clean = val_df
            
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
        
        # 후처리 통계
        after_stats = {
            "Train samples": len(train_df_clean),
            "Val samples": len(val_df_clean),
            "Train classes": len(train_df_clean['mapping_code'].unique()),
            "Val classes": len(val_df_clean['mapping_code'].unique()),
            "Val-only classes": 0,
            "Train-only classes": len(train_only)
        }
        
        # 전후 비교
        self.print_before_after_table(before_stats, after_stats, "Class Mismatch Resolution")
        
        # 클래스 일치성 검증
        final_train_classes = set(train_df_clean['mapping_code'].unique())
        final_val_classes = set(val_df_clean['mapping_code'].unique())
        final_val_only = final_val_classes - final_train_classes
        
        # Manifest 저장
        if not self.args.dry_run:
            self.save_manifest(train_df_clean, self.train_manifest_path, suffix=f".{self.args.mode}")
            self.save_manifest(val_df_clean, self.val_manifest_path, suffix=f".{self.args.mode}")
        
        # 리포트 생성
        report_data = {
            "before": before_stats,
            "after": after_stats,
            "val_only_classes": list(val_only),
            "train_only_classes": list(train_only),
            "mode": self.args.mode,
            "val_only_details": val_only_details if val_only else {},
            "train_distribution": self.analyze_class_distribution(train_df_clean, "train"),
            "val_distribution": self.analyze_class_distribution(val_df_clean, "val")
        }
        self.save_report(report_data, "val_only_class_handling")
        
        # 최종 결론
        if len(final_val_only) == 0:
            self.print_conclusion("PASS", "✅ All classes are now consistent between train and val!")
        else:
            self.print_conclusion("FAIL", f"❌ Still have {len(final_val_only)} val-only classes")
        
        return 0 if len(final_val_only) == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Handle val-only classes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    DataQualityBase.add_common_arguments(parser)
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['remove', 'add-to-train'],
        default='remove',
        help='How to handle val-only classes (default: remove)'
    )
    
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=10,
        help='Max samples per class to add to train (for add-to-train mode, default: 10)'
    )
    
    # 사용 예시
    parser.epilog = """
Examples:
  # Remove val-only classes (default, dry run)
  python fix_val_only_classes.py
  
  # Actually remove val-only classes
  python fix_val_only_classes.py --no-dry-run
  
  # Add val-only samples to train instead
  python fix_val_only_classes.py --mode add-to-train --max-per-class 5
  
  # Custom manifests
  python fix_val_only_classes.py --train-manifest /path/to/train.csv --val-manifest /path/to/val.csv
"""
    
    args = parser.parse_args()
    handler = ValOnlyClassHandler(args)
    return handler.run()


if __name__ == "__main__":
    exit(main())