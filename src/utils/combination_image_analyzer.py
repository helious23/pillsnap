#!/usr/bin/env python3
"""
Combination Image Analysis Utility

Analyzes combination pill images and their corresponding YOLO labels,
providing statistics on image-label matching and data completeness.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.core import PillSnapLogger


class CombinationImageAnalyzer:
    """Analyzer for combination pill images and labels"""
    
    def __init__(self, 
                 images_base: str = "/home/max16/pillsnap_data/train/images/combination",
                 labels_base: str = "/home/max16/pillsnap_data/train/labels/combination_yolo"):
        """
        Initialize analyzer
        
        Args:
            images_base: Base path for combination images
            labels_base: Base path for YOLO labels
        """
        self.images_base = Path(images_base)
        self.labels_base = Path(labels_base)
        self.logger = PillSnapLogger(__name__)
        
    def analyze_complete_dataset(self) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
        """
        Analyze complete combination dataset
        
        Returns:
            (total_images, total_with_labels, per_ts_stats)
        """
        total_images = 0
        total_with_labels = 0
        stats = defaultdict(lambda: [0, 0])  # [images, labels]
        
        self.logger.info("🔍 Analyzing combination dataset...")
        
        # Iterate through TS_combo directories
        for ts_dir in self.images_base.glob("TS_*_combo"):
            if not ts_dir.is_dir():
                continue
                
            ts_image_count = 0
            ts_with_labels = 0
            
            # Iterate through K-code directories
            for k_dir in ts_dir.iterdir():
                if not k_dir.is_dir():
                    continue
                    
                # Find combination images (not index images)
                for img_file in k_dir.glob("*_0_2_0_2_*.png"):
                    total_images += 1
                    ts_image_count += 1
                    
                    # Check for corresponding label
                    label_file = self.labels_base / f"{img_file.stem}.txt"
                    if label_file.exists():
                        total_with_labels += 1
                        ts_with_labels += 1
            
            stats[ts_dir.name] = (ts_image_count, ts_with_labels)
            self.logger.info(f"📂 {ts_dir.name}: {ts_image_count} images, {ts_with_labels} labels")
        
        return total_images, total_with_labels, dict(stats)
    
    def find_missing_labels(self) -> List[Path]:
        """Find images that don't have corresponding labels"""
        
        missing = []
        
        for ts_dir in self.images_base.glob("TS_*_combo"):
            if not ts_dir.is_dir():
                continue
                
            for k_dir in ts_dir.iterdir():
                if not k_dir.is_dir():
                    continue
                    
                for img_file in k_dir.glob("*_0_2_0_2_*.png"):
                    label_file = self.labels_base / f"{img_file.stem}.txt"
                    if not label_file.exists():
                        missing.append(img_file)
        
        return missing
    
    def validate_label_files(self, sample_size: int = 100) -> Dict[str, int]:
        """
        Validate label file contents
        
        Args:
            sample_size: Number of label files to sample for validation
            
        Returns:
            Validation statistics
        """
        stats = {
            'valid_labels': 0,
            'empty_labels': 0,
            'invalid_format': 0,
            'total_checked': 0
        }
        
        label_files = list(self.labels_base.glob("*.txt"))
        if len(label_files) > sample_size:
            import random
            label_files = random.sample(label_files, sample_size)
        
        for label_file in label_files:
            stats['total_checked'] += 1
            
            try:
                content = label_file.read_text().strip()
                if not content:
                    stats['empty_labels'] += 1
                    continue
                
                # Check YOLO format: class x_center y_center width height
                lines = content.split('\n')
                valid = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        valid = False
                        break
                    try:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                        if not (0 <= class_id <= 1000 and all(0 <= c <= 1 for c in coords)):
                            valid = False
                            break
                    except ValueError:
                        valid = False
                        break
                
                if valid:
                    stats['valid_labels'] += 1
                else:
                    stats['invalid_format'] += 1
                    
            except Exception:
                stats['invalid_format'] += 1
        
        return stats
    
    def print_analysis_report(self):
        """Print comprehensive analysis report"""
        
        # Dataset overview
        total_images, total_with_labels, ts_stats = self.analyze_complete_dataset()
        
        print("\n📊 Combination Dataset Analysis Report")
        print("=" * 50)
        print(f"Total Images: {total_images:,}")
        print(f"Images with Labels: {total_with_labels:,}")
        print(f"Missing Labels: {total_images - total_with_labels:,}")
        print(f"Matching Rate: {(total_with_labels/total_images*100):.1f}%")
        
        # Per-TS breakdown
        print(f"\n📂 Per-TS Statistics:")
        for ts_name, (img_count, label_count) in ts_stats.items():
            match_rate = (label_count / img_count * 100) if img_count > 0 else 0
            print(f"  {ts_name}: {img_count:,} images, {label_count:,} labels ({match_rate:.1f}%)")
        
        # Label validation
        print(f"\n🔍 Label Validation (sample):")
        validation_stats = self.validate_label_files()
        for key, value in validation_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Missing files
        missing = self.find_missing_labels()
        if missing:
            print(f"\n❌ Missing Labels ({len(missing)} files):")
            for i, missing_file in enumerate(missing[:5]):  # Show first 5
                print(f"  {missing_file}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
        else:
            print(f"\n✅ All images have corresponding labels")


def analyze_combination_dataset():
    """Standalone function to analyze combination dataset"""
    
    analyzer = CombinationImageAnalyzer()
    analyzer.print_analysis_report()


if __name__ == "__main__":
    analyze_combination_dataset()