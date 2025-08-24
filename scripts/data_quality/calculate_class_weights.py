#!/usr/bin/env python3
"""
클래스 가중치 계산 유틸리티
- 클래스 불균형 분석 및 가중치 계산
- Inverse frequency, Effective number 등 다양한 방법 지원
"""

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from base import DataQualityBase


class ClassWeightCalculator(DataQualityBase):
    """클래스 가중치 계산"""
    
    def __init__(self, args):
        super().__init__(args)
        self.method = args.method
        self.beta = args.beta
        self.normalize = args.normalize
        self.clip_max = args.clip_max
        
    def analyze_class_distribution(self, df: pd.DataFrame) -> Dict:
        """클래스 분포 분석"""
        class_counts = df['mapping_code'].value_counts()
        
        # 기본 통계
        stats_dict = {
            "num_classes": len(class_counts),
            "total_samples": len(df),
            "mean": class_counts.mean(),
            "std": class_counts.std(),
            "median": class_counts.median(),
            "min": class_counts.min(),
            "max": class_counts.max(),
            "q1": class_counts.quantile(0.25),
            "q3": class_counts.quantile(0.75)
        }
        
        # 불균형 지표
        # Gini 계수 계산
        sorted_counts = sorted(class_counts.values)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        # Imbalance ratio
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        stats_dict.update({
            "gini_coefficient": gini,
            "imbalance_ratio": imbalance_ratio,
            "cv": class_counts.std() / class_counts.mean()  # Coefficient of variation
        })
        
        return stats_dict, class_counts
    
    def calculate_inverse_frequency_weights(self, class_counts: pd.Series) -> Dict[str, float]:
        """Inverse frequency 가중치 계산"""
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        
        weights = {}
        for class_code, count in class_counts.items():
            # 기본 inverse frequency
            weight = total_samples / (num_classes * count)
            weights[class_code] = weight
            
        return weights
    
    def calculate_effective_number_weights(self, class_counts: pd.Series) -> Dict[str, float]:
        """Effective number 기반 가중치 계산"""
        weights = {}
        
        for class_code, count in class_counts.items():
            # Effective number: (1 - β^n) / (1 - β)
            if self.beta == 1:
                effective_num = count
            else:
                effective_num = (1 - self.beta ** count) / (1 - self.beta)
            
            # Weight는 effective number의 역수에 비례
            weight = 1.0 / effective_num
            weights[class_code] = weight
            
        return weights
    
    def calculate_balanced_weights(self, class_counts: pd.Series) -> Dict[str, float]:
        """Scikit-learn style balanced weights"""
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        
        weights = {}
        for class_code, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            weights[class_code] = weight
            
        return weights
    
    def calculate_sqrt_weights(self, class_counts: pd.Series) -> Dict[str, float]:
        """Square root 기반 가중치 (moderate balancing)"""
        weights = {}
        max_count = class_counts.max()
        
        for class_code, count in class_counts.items():
            # sqrt 역수로 moderate하게 조정
            weight = np.sqrt(max_count / count)
            weights[class_code] = weight
            
        return weights
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 정규화"""
        if not self.normalize:
            return weights
            
        values = list(weights.values())
        
        if self.normalize == "mean":
            # 평균이 1이 되도록
            mean_weight = np.mean(values)
            return {k: v / mean_weight for k, v in weights.items()}
            
        elif self.normalize == "max":
            # 최대값이 1이 되도록
            max_weight = max(values)
            return {k: v / max_weight for k, v in weights.items()}
            
        elif self.normalize == "sum":
            # 합이 클래스 수가 되도록
            sum_weight = sum(values)
            num_classes = len(weights)
            return {k: v * num_classes / sum_weight for k, v in weights.items()}
            
        return weights
    
    def clip_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """극단값 클리핑"""
        if not self.clip_max:
            return weights
            
        return {k: min(v, self.clip_max) for k, v in weights.items()}
    
    def plot_weight_distribution(self, class_counts: pd.Series, weights: Dict[str, float], save_path: str = None):
        """가중치 분포 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 클래스 분포
        ax = axes[0, 0]
        counts_sorted = class_counts.sort_values(ascending=False)
        ax.bar(range(len(counts_sorted)), counts_sorted.values)
        ax.set_xlabel('Class Rank')
        ax.set_ylabel('Sample Count')
        ax.set_title('Class Distribution (sorted)')
        ax.set_yscale('log')
        
        # 2. 가중치 분포
        ax = axes[0, 1]
        weight_values = [weights[c] for c in counts_sorted.index]
        ax.bar(range(len(weight_values)), weight_values)
        ax.set_xlabel('Class Rank (by count)')
        ax.set_ylabel('Weight')
        ax.set_title('Weight Distribution')
        
        # 3. Count vs Weight scatter
        ax = axes[1, 0]
        counts_list = []
        weights_list = []
        for class_code in class_counts.index:
            counts_list.append(class_counts[class_code])
            weights_list.append(weights[class_code])
        ax.scatter(counts_list, weights_list, alpha=0.5)
        ax.set_xlabel('Sample Count')
        ax.set_ylabel('Weight')
        ax.set_title('Count vs Weight Relationship')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # 4. Weight histogram
        ax = axes[1, 1]
        ax.hist(list(weights.values()), bins=50, edgecolor='black')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Weight Value Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            self.logger.info(f"Saved weight distribution plot: {save_path}")
        
        plt.close()
    
    def validate_weights(self, weights: Dict[str, float], class_counts: pd.Series) -> Dict:
        """가중치 검증"""
        weight_values = list(weights.values())
        
        validation = {
            "num_weights": len(weights),
            "min_weight": min(weight_values),
            "max_weight": max(weight_values),
            "mean_weight": np.mean(weight_values),
            "std_weight": np.std(weight_values),
            "weight_range": max(weight_values) / min(weight_values)
        }
        
        # Effective sample size 계산
        effective_samples = {}
        for class_code, count in class_counts.items():
            if class_code in weights:
                effective_samples[class_code] = count * weights[class_code]
        
        eff_values = list(effective_samples.values())
        validation.update({
            "effective_min": min(eff_values),
            "effective_max": max(eff_values),
            "effective_std": np.std(eff_values),
            "effective_cv": np.std(eff_values) / np.mean(eff_values)
        })
        
        return validation
    
    def run(self):
        """메인 실행"""
        self.console.print("[bold cyan]⚖️ Class Weight Calculator[/bold cyan]")
        self.console.print(f"Method: {self.method}")
        self.console.print(f"Beta: {self.beta}")
        self.console.print(f"Normalization: {self.normalize}")
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # 클래스 분포 분석
        train_stats, train_counts = self.analyze_class_distribution(train_df)
        val_stats, val_counts = self.analyze_class_distribution(val_df)
        
        # 분포 통계 표시
        self.print_summary_table("Train Class Distribution", {
            f"Classes": train_stats['num_classes'],
            f"Samples": train_stats['total_samples'],
            f"Mean±Std": f"{train_stats['mean']:.1f}±{train_stats['std']:.1f}",
            f"Min/Max": f"{train_stats['min']}/{train_stats['max']}",
            f"Gini": f"{train_stats['gini_coefficient']:.3f}",
            f"Imbalance": f"{train_stats['imbalance_ratio']:.1f}x"
        })
        
        # 상위/하위 클래스 표시
        self.console.print("\n[yellow]Top 5 Most Frequent Classes:[/yellow]")
        for class_code, count in train_counts.head(5).items():
            self.console.print(f"  {class_code}: {count:,} samples")
            
        self.console.print("\n[yellow]Bottom 5 Least Frequent Classes:[/yellow]")
        for class_code, count in train_counts.tail(5).items():
            self.console.print(f"  {class_code}: {count:,} samples")
        
        # 가중치 계산
        if self.method == "inverse":
            weights = self.calculate_inverse_frequency_weights(train_counts)
        elif self.method == "effective":
            weights = self.calculate_effective_number_weights(train_counts)
        elif self.method == "balanced":
            weights = self.calculate_balanced_weights(train_counts)
        elif self.method == "sqrt":
            weights = self.calculate_sqrt_weights(train_counts)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 정규화 및 클리핑
        weights = self.normalize_weights(weights)
        weights = self.clip_weights(weights)
        
        # 가중치 검증
        validation = self.validate_weights(weights, train_counts)
        
        self.print_summary_table("Weight Statistics", {
            "Min weight": f"{validation['min_weight']:.4f}",
            "Max weight": f"{validation['max_weight']:.4f}",
            "Mean weight": f"{validation['mean_weight']:.4f}",
            "Weight range": f"{validation['weight_range']:.1f}x",
            "Effective CV": f"{validation['effective_cv']:.3f}"
        })
        
        # 가중치 파일 저장
        weight_file = self.report_dir / f"class_weights_{self.method}_{self.timestamp}.json"
        
        if not self.args.dry_run:
            # JSON으로 저장
            with open(weight_file, 'w') as f:
                json.dump(weights, f, indent=2)
            self.logger.info(f"Saved weights to: {weight_file}")
            
            # NumPy 배열로도 저장 (학습 코드용)
            weight_array_file = weight_file.with_suffix('.npy')
            # 클래스 순서대로 정렬
            sorted_classes = sorted(weights.keys())
            weight_array = np.array([weights[c] for c in sorted_classes])
            np.save(weight_array_file, weight_array)
            
            # 클래스 인덱스 매핑 저장
            class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
            mapping_file = weight_file.with_suffix('.mapping.json')
            with open(mapping_file, 'w') as f:
                json.dump(class_to_idx, f, indent=2)
        
        # 시각화
        plot_path = self.report_dir / f"weight_distribution_{self.timestamp}.png"
        self.plot_weight_distribution(train_counts, weights, plot_path if not self.args.dry_run else None)
        
        # 리포트 저장
        report_data = {
            "method": self.method,
            "parameters": {
                "beta": self.beta,
                "normalize": self.normalize,
                "clip_max": self.clip_max
            },
            "distribution_stats": {
                "train": train_stats,
                "val": val_stats
            },
            "weight_stats": validation,
            "weight_file": str(weight_file),
            "top_weights": {k: weights[k] for k in list(train_counts.tail(10).index)},  # 희귀 클래스들
            "bottom_weights": {k: weights[k] for k in list(train_counts.head(10).index)}  # 빈번한 클래스들
        }
        self.save_report(report_data, "class_weight_calculation")
        
        # 사용 가이드 출력
        self.console.print("\n[green]📝 Usage Guide:[/green]")
        self.console.print("```python")
        self.console.print("# In your training script:")
        self.console.print(f"weights = json.load(open('{weight_file}'))")
        self.console.print("# Or for array:")
        self.console.print(f"weight_array = np.load('{weight_file.with_suffix('.npy')}')")
        self.console.print("criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_array))")
        self.console.print("```")
        
        # 최종 결론
        if validation['effective_cv'] < 0.5:
            self.print_conclusion("PASS", 
                f"✅ Weights successfully calculated!\n"
                f"Effective CV reduced to {validation['effective_cv']:.3f}")
        else:
            self.print_conclusion("WARNING",
                f"⚠️ Weights calculated but imbalance still high\n"
                f"Effective CV: {validation['effective_cv']:.3f}")
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Calculate class weights for imbalanced dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    DataQualityBase.add_common_arguments(parser)
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['inverse', 'effective', 'balanced', 'sqrt'],
        default='balanced',
        help='Weight calculation method (default: balanced)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.999,
        help='Beta parameter for effective number method (default: 0.999)'
    )
    
    parser.add_argument(
        '--normalize',
        type=str,
        choices=['none', 'mean', 'max', 'sum'],
        default='mean',
        help='Weight normalization method (default: mean)'
    )
    
    parser.add_argument(
        '--clip-max',
        type=float,
        default=10.0,
        help='Maximum weight value for clipping (default: 10.0)'
    )
    
    # 사용 예시
    parser.epilog = """
Examples:
  # Calculate balanced weights (default)
  python calculate_class_weights.py
  
  # Use effective number method
  python calculate_class_weights.py --method effective --beta 0.999
  
  # Inverse frequency with max normalization
  python calculate_class_weights.py --method inverse --normalize max
  
  # Moderate balancing with sqrt
  python calculate_class_weights.py --method sqrt --clip-max 5.0
"""
    
    args = parser.parse_args()
    calculator = ClassWeightCalculator(args)
    return calculator.run()


if __name__ == "__main__":
    exit(main())