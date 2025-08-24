#!/usr/bin/env python3
"""
최종 품질 검증 유틸리티
- 모든 데이터 품질 지표를 종합적으로 검증
- PASS/FAIL 판정 및 상세 리포트 생성
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from base import DataQualityBase


class FinalQualityChecker(DataQualityBase):
    """최종 품질 검증"""
    
    def __init__(self, args):
        super().__init__(args)
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def check_class_consistency(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Train/Val 클래스 일치성 검사"""
        train_classes = set(train_df['mapping_code'].unique())
        val_classes = set(val_df['mapping_code'].unique())
        
        val_only = val_classes - train_classes
        train_only = train_classes - val_classes
        
        result = {
            "train_classes": len(train_classes),
            "val_classes": len(val_classes),
            "common_classes": len(train_classes & val_classes),
            "val_only_classes": list(val_only),
            "train_only_classes": list(train_only)[:10]  # 처음 10개만
        }
        
        passed = len(val_only) == 0
        
        if passed:
            self.checks_passed.append("Class Consistency")
        else:
            self.checks_failed.append(f"Class Consistency ({len(val_only)} val-only classes)")
            
        return passed, result
    
    def check_combination_ratio(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               target_min: float = 0.15, target_max: float = 0.35) -> Tuple[bool, Dict]:
        """Combination 비율 검사"""
        train_combo_ratio = (train_df['image_type'] == 'combination').mean()
        val_combo_ratio = (val_df['image_type'] == 'combination').mean()
        
        result = {
            "train_combo_ratio": train_combo_ratio,
            "val_combo_ratio": val_combo_ratio,
            "target_range": f"{target_min:.0%}-{target_max:.0%}"
        }
        
        train_in_range = target_min <= train_combo_ratio <= target_max
        val_in_range = target_min <= val_combo_ratio <= target_max
        
        passed = train_in_range and val_in_range
        
        if passed:
            self.checks_passed.append("Combination Ratio")
        elif train_combo_ratio < 0.10 or val_combo_ratio < 0.10:
            self.checks_failed.append(f"Combination Ratio (Train: {train_combo_ratio:.1%}, Val: {val_combo_ratio:.1%})")
        else:
            self.warnings.append(f"Combination Ratio suboptimal (Train: {train_combo_ratio:.1%}, Val: {val_combo_ratio:.1%})")
            
        return passed, result
    
    def check_corrupted_files(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """손상 파일 검사"""
        blacklist_path = self.report_dir / "blacklist.txt"
        
        if blacklist_path.exists():
            with open(blacklist_path, 'r') as f:
                blacklist = set(line.strip() for line in f if line.strip())
        else:
            blacklist = set()
        
        # K-001900 관련 파일 체크 (알려진 문제)
        k001900_count = sum(1 for path in blacklist if "K-001900" in path)
        
        result = {
            "blacklisted_files": len(blacklist),
            "k001900_affected": k001900_count,
            "blacklist_exists": blacklist_path.exists()
        }
        
        # 현재 manifest에 블랙리스트 파일이 있는지 확인
        all_paths = set(train_df['image_path'].tolist() + val_df['image_path'].tolist())
        remaining_corrupted = len(all_paths & blacklist)
        
        result['remaining_in_manifest'] = remaining_corrupted
        
        passed = remaining_corrupted == 0
        
        if passed:
            self.checks_passed.append("No Corrupted Files")
        else:
            self.checks_failed.append(f"Corrupted Files ({remaining_corrupted} still in manifest)")
            
        return passed, result
    
    def check_class_balance(self, train_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """클래스 균형 검사"""
        class_counts = train_df['mapping_code'].value_counts()
        
        # Gini 계수 계산
        sorted_counts = sorted(class_counts.values)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        # 불균형 비율
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        result = {
            "num_classes": len(class_counts),
            "min_samples": class_counts.min(),
            "max_samples": class_counts.max(),
            "mean_samples": class_counts.mean(),
            "gini_coefficient": gini,
            "imbalance_ratio": imbalance_ratio
        }
        
        # 임계값 기준
        if gini < 0.5:
            status = "good"
            self.checks_passed.append("Class Balance")
            passed = True
        elif gini < 0.7:
            status = "moderate"
            self.warnings.append(f"Class imbalance moderate (Gini: {gini:.3f})")
            passed = True
        else:
            status = "severe"
            self.checks_failed.append(f"Severe class imbalance (Gini: {gini:.3f})")
            passed = False
            
        result['status'] = status
        
        return passed, result
    
    def check_yolo_consistency(self) -> Tuple[bool, Dict]:
        """YOLO 데이터셋 일관성 검사"""
        yolo_path = self.data_root / "yolo_configs" / "yolo_dataset"
        
        if not yolo_path.exists():
            return True, {"yolo_dataset_exists": False}
        
        images_path = yolo_path / "images"
        labels_path = yolo_path / "labels"
        
        # 이미지와 라벨 파일 개수 확인
        image_files = set(f.stem for f in images_path.glob("*.png")) if images_path.exists() else set()
        label_files = set(f.stem for f in labels_path.glob("*.txt")) if labels_path.exists() else set()
        
        # 불일치 찾기
        images_only = image_files - label_files
        labels_only = label_files - image_files
        
        result = {
            "yolo_dataset_exists": True,
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "images_without_labels": len(images_only),
            "labels_without_images": len(labels_only)
        }
        
        passed = len(images_only) == 0 and len(labels_only) == 0
        
        if passed:
            self.checks_passed.append("YOLO Consistency")
        else:
            self.warnings.append(f"YOLO dataset mismatch (images: {len(image_files)}, labels: {len(label_files)})")
            
        return passed, result
    
    def check_sample_counts(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """샘플 수 적정성 검사"""
        train_count = len(train_df)
        val_count = len(val_df)
        total_count = train_count + val_count
        
        val_ratio = val_count / total_count if total_count > 0 else 0
        
        result = {
            "train_samples": train_count,
            "val_samples": val_count,
            "total_samples": total_count,
            "val_ratio": val_ratio
        }
        
        # Stage별 예상 샘플 수와 비교
        expected_totals = {
            "stage1": 5000,
            "stage2": 25000,
            "stage3": 100000,
            "stage4": 500000
        }
        
        # 현재 stage 추정
        for stage, expected in expected_totals.items():
            if total_count <= expected * 1.1:  # 10% 허용 오차
                result['estimated_stage'] = stage
                break
        
        # Val 비율 체크 (15-25% 권장)
        if 0.15 <= val_ratio <= 0.25:
            self.checks_passed.append("Sample Split Ratio")
            passed = True
        else:
            self.warnings.append(f"Val ratio {val_ratio:.1%} (recommended: 15-25%)")
            passed = True  # Warning only
            
        return passed, result
    
    def generate_summary_report(self, all_results: Dict) -> str:
        """요약 리포트 생성"""
        lines = []
        lines.append("=" * 60)
        lines.append("FINAL QUALITY CHECK SUMMARY")
        lines.append("=" * 60)
        
        # 전체 상태
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        lines.append(f"\n✅ Passed: {len(self.checks_passed)}/{total_checks}")
        lines.append(f"❌ Failed: {len(self.checks_failed)}/{total_checks}")
        lines.append(f"⚠️  Warnings: {len(self.warnings)}")
        
        # 세부 결과
        lines.append("\n" + "-" * 40)
        lines.append("DETAILED RESULTS:")
        lines.append("-" * 40)
        
        for check_name, result in all_results.items():
            if check_name == "class_consistency":
                lines.append(f"\n📋 Class Consistency:")
                lines.append(f"   Train: {result['train_classes']} classes")
                lines.append(f"   Val: {result['val_classes']} classes")
                if result['val_only_classes']:
                    lines.append(f"   ❌ Val-only: {result['val_only_classes']}")
                    
            elif check_name == "combination_ratio":
                lines.append(f"\n🔄 Combination Ratio:")
                lines.append(f"   Train: {result['train_combo_ratio']:.1%}")
                lines.append(f"   Val: {result['val_combo_ratio']:.1%}")
                lines.append(f"   Target: {result['target_range']}")
                
            elif check_name == "corrupted_files":
                lines.append(f"\n🗑️ Corrupted Files:")
                lines.append(f"   Blacklisted: {result['blacklisted_files']}")
                lines.append(f"   Remaining: {result['remaining_in_manifest']}")
                
            elif check_name == "class_balance":
                lines.append(f"\n⚖️ Class Balance:")
                lines.append(f"   Gini: {result['gini_coefficient']:.3f}")
                lines.append(f"   Imbalance: {result['imbalance_ratio']:.1f}x")
                lines.append(f"   Status: {result['status']}")
                
        # 권장사항
        lines.append("\n" + "=" * 60)
        lines.append("RECOMMENDATIONS:")
        lines.append("=" * 60)
        
        if self.checks_failed:
            lines.append("\n🚨 Critical Issues to Fix:")
            for issue in self.checks_failed:
                lines.append(f"   - {issue}")
                
        if self.warnings:
            lines.append("\n⚠️ Warnings to Consider:")
            for warning in self.warnings:
                lines.append(f"   - {warning}")
                
        if not self.checks_failed and not self.warnings:
            lines.append("\n✅ All checks passed! Dataset is ready for training.")
            
        return "\n".join(lines)
    
    def run(self):
        """메인 실행"""
        self.console.print("[bold cyan]🔍 Final Quality Check[/bold cyan]")
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # 모든 검사 수행
        all_results = {}
        
        # 1. 클래스 일치성
        passed, result = self.check_class_consistency(train_df, val_df)
        all_results['class_consistency'] = result
        
        # 2. Combination 비율
        passed, result = self.check_combination_ratio(train_df, val_df)
        all_results['combination_ratio'] = result
        
        # 3. 손상 파일
        passed, result = self.check_corrupted_files(train_df, val_df)
        all_results['corrupted_files'] = result
        
        # 4. 클래스 균형
        passed, result = self.check_class_balance(train_df)
        all_results['class_balance'] = result
        
        # 5. YOLO 일관성
        passed, result = self.check_yolo_consistency()
        all_results['yolo_consistency'] = result
        
        # 6. 샘플 수
        passed, result = self.check_sample_counts(train_df, val_df)
        all_results['sample_counts'] = result
        
        # 요약 리포트 생성
        summary = self.generate_summary_report(all_results)
        self.console.print(summary)
        
        # 리포트 저장
        report_data = {
            "summary": {
                "passed_checks": self.checks_passed,
                "failed_checks": self.checks_failed,
                "warnings": self.warnings,
                "overall_status": "PASS" if not self.checks_failed else "FAIL"
            },
            "detailed_results": all_results,
            "manifest_paths": {
                "train": str(self.train_manifest_path),
                "val": str(self.val_manifest_path)
            }
        }
        
        report_path = self.save_report(report_data, "final_quality_check")
        
        # 텍스트 리포트도 저장
        summary_path = self.report_dir / f"quality_summary_{self.timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        self.logger.info(f"Saved summary: {summary_path}")
        
        # 최종 결론
        if not self.checks_failed:
            self.print_conclusion("PASS", 
                f"✅ All critical checks passed!\n"
                f"{len(self.warnings)} warnings to consider")
        else:
            self.print_conclusion("FAIL",
                f"❌ {len(self.checks_failed)} critical issues found\n"
                f"Please run the appropriate fix utilities")
        
        return 0 if not self.checks_failed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Final quality check for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    DataQualityBase.add_common_arguments(parser)
    
    # 사용 예시
    parser.epilog = """
Examples:
  # Run final quality check
  python final_quality_check.py
  
  # Check cleaned manifests
  python final_quality_check.py --train-manifest artifacts/stage3/manifest_train.cleaned.csv \\
                                --val-manifest artifacts/stage3/manifest_val.cleaned.csv
  
  # With custom report path
  python final_quality_check.py --report-path /path/to/report.json
"""
    
    args = parser.parse_args()
    checker = FinalQualityChecker(args)
    return checker.run()


if __name__ == "__main__":
    exit(main())