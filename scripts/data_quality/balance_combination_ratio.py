#!/usr/bin/env python3
"""
Combination 비율 조정 유틸리티
- Single/Combination 비율을 목표 수준으로 조정
- 오버샘플링 또는 언더샘플링 전략 제공
"""

import argparse
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from base import DataQualityBase


class CombinationRatioBalancer(DataQualityBase):
    """Combination 비율 조정"""
    
    def __init__(self, args):
        super().__init__(args)
        self.target_ratio = args.target_ratio
        self.strategy = args.strategy
        
    def analyze_current_ratio(self, df: pd.DataFrame) -> Dict:
        """현재 Single/Combination 비율 분석"""
        type_counts = df['image_type'].value_counts()
        
        single_count = type_counts.get('single', 0)
        combo_count = type_counts.get('combination', 0)
        total = single_count + combo_count
        
        stats = {
            "total_samples": total,
            "single_count": single_count,
            "combination_count": combo_count,
            "single_ratio": single_count / total if total > 0 else 0,
            "combination_ratio": combo_count / total if total > 0 else 0
        }
        
        return stats
    
    def calculate_sampling_params(self, current_stats: Dict) -> Dict:
        """샘플링 파라미터 계산"""
        current_combo_ratio = current_stats['combination_ratio']
        total_samples = current_stats['total_samples']
        
        # 목표 달성을 위한 샘플 수 계산
        target_combo_count = int(total_samples * self.target_ratio)
        current_combo_count = current_stats['combination_count']
        current_single_count = current_stats['single_count']
        
        params = {
            "current_ratio": current_combo_ratio,
            "target_ratio": self.target_ratio,
            "gap": self.target_ratio - current_combo_ratio
        }
        
        if self.strategy == "oversample":
            # Combination 오버샘플링
            if current_combo_ratio < self.target_ratio:
                # 필요한 추가 combo 샘플 수
                additional_combo = target_combo_count - current_combo_count
                oversample_factor = target_combo_count / current_combo_count if current_combo_count > 0 else 1
                
                params.update({
                    "method": "oversample_combination",
                    "additional_samples": additional_combo,
                    "oversample_factor": oversample_factor,
                    "final_total": total_samples + additional_combo
                })
            else:
                params["method"] = "no_change_needed"
                
        elif self.strategy == "undersample":
            # Single 언더샘플링
            if current_combo_ratio < self.target_ratio:
                # Single을 줄여서 비율 맞추기
                target_single_count = int(current_combo_count / self.target_ratio * (1 - self.target_ratio))
                samples_to_remove = current_single_count - target_single_count
                
                params.update({
                    "method": "undersample_single",
                    "samples_to_remove": samples_to_remove,
                    "keep_ratio": target_single_count / current_single_count if current_single_count > 0 else 1,
                    "final_total": target_single_count + current_combo_count
                })
            else:
                params["method"] = "no_change_needed"
                
        elif self.strategy == "mixed":
            # 혼합 전략 (오버샘플링 + 언더샘플링)
            if current_combo_ratio < self.target_ratio:
                # Combo는 1.5배, Single은 적절히 줄이기
                combo_factor = min(1.5, self.target_ratio / current_combo_ratio)
                new_combo_count = int(current_combo_count * combo_factor)
                new_single_count = int(new_combo_count / self.target_ratio * (1 - self.target_ratio))
                
                params.update({
                    "method": "mixed_sampling",
                    "combo_factor": combo_factor,
                    "single_keep_ratio": new_single_count / current_single_count if current_single_count > 0 else 1,
                    "final_combo": new_combo_count,
                    "final_single": new_single_count,
                    "final_total": new_combo_count + new_single_count
                })
            else:
                params["method"] = "no_change_needed"
        
        return params
    
    def apply_oversampling(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """오버샘플링 적용"""
        combo_df = df[df['image_type'] == 'combination']
        single_df = df[df['image_type'] == 'single']
        
        if params['method'] == 'oversample_combination':
            # Combination 샘플 복제
            factor = params['oversample_factor']
            
            # 정수 부분만큼 전체 복제
            full_copies = int(factor)
            partial = factor - full_copies
            
            dfs = [single_df]  # Single은 그대로
            
            # 전체 복제
            for _ in range(full_copies):
                dfs.append(combo_df)
            
            # 부분 복제 (랜덤 샘플링)
            if partial > 0:
                n_partial = int(len(combo_df) * partial)
                if n_partial > 0:
                    dfs.append(combo_df.sample(n=n_partial, random_state=42, replace=True))
            
            return pd.concat(dfs, ignore_index=True)
        
        return df
    
    def apply_undersampling(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """언더샘플링 적용"""
        combo_df = df[df['image_type'] == 'combination']
        single_df = df[df['image_type'] == 'single']
        
        if params['method'] == 'undersample_single':
            # Single 샘플 줄이기
            keep_ratio = params['keep_ratio']
            n_keep = int(len(single_df) * keep_ratio)
            
            single_df_sampled = single_df.sample(n=n_keep, random_state=42)
            return pd.concat([single_df_sampled, combo_df], ignore_index=True)
        
        return df
    
    def apply_mixed_strategy(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """혼합 전략 적용"""
        combo_df = df[df['image_type'] == 'combination']
        single_df = df[df['image_type'] == 'single']
        
        if params['method'] == 'mixed_sampling':
            # Combination 오버샘플링
            combo_factor = params['combo_factor']
            n_combo = int(len(combo_df) * combo_factor)
            combo_df_sampled = combo_df.sample(n=n_combo, random_state=42, replace=True)
            
            # Single 언더샘플링
            single_keep_ratio = params['single_keep_ratio']
            n_single = int(len(single_df) * single_keep_ratio)
            single_df_sampled = single_df.sample(n=n_single, random_state=42)
            
            return pd.concat([single_df_sampled, combo_df_sampled], ignore_index=True)
        
        return df
    
    def generate_sampler_config(self, params: Dict) -> Dict:
        """DataLoader용 샘플러 설정 생성"""
        config = {
            "type": "WeightedRandomSampler",
            "target_ratio": self.target_ratio,
            "strategy": self.strategy
        }
        
        if params.get('method') == 'oversample_combination':
            # Combination에 더 높은 가중치
            config['weights'] = {
                "single": 1.0,
                "combination": params['oversample_factor']
            }
        elif params.get('method') == 'undersample_single':
            # Single에 낮은 가중치
            config['weights'] = {
                "single": params['keep_ratio'],
                "combination": 1.0
            }
        elif params.get('method') == 'mixed_sampling':
            config['weights'] = {
                "single": params['single_keep_ratio'],
                "combination": params['combo_factor']
            }
        
        return config
    
    def run(self):
        """메인 실행"""
        self.console.print("[bold cyan]⚖️ Combination Ratio Balancer[/bold cyan]")
        self.console.print(f"Mode: {'DRY RUN' if self.args.dry_run else 'ACTUAL EXECUTION'}")
        self.console.print(f"Target ratio: {self.target_ratio:.1%}")
        self.console.print(f"Strategy: {self.strategy}")
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # 현재 비율 분석
        train_stats = self.analyze_current_ratio(train_df)
        val_stats = self.analyze_current_ratio(val_df)
        
        # 현재 상태 표시
        current_status = {
            "Train Single": f"{train_stats['single_count']:,} ({train_stats['single_ratio']:.1%})",
            "Train Combination": f"{train_stats['combination_count']:,} ({train_stats['combination_ratio']:.1%})",
            "Val Single": f"{val_stats['single_count']:,} ({val_stats['single_ratio']:.1%})",
            "Val Combination": f"{val_stats['combination_count']:,} ({val_stats['combination_ratio']:.1%})"
        }
        self.print_summary_table("Current Distribution", current_status)
        
        # 샘플링 파라미터 계산
        train_params = self.calculate_sampling_params(train_stats)
        val_params = self.calculate_sampling_params(val_stats)
        
        # 파라미터 표시
        self.console.print("\n[yellow]Sampling Parameters:[/yellow]")
        self.console.print(f"Train: {train_params['method']}")
        if train_params.get('oversample_factor'):
            self.console.print(f"  - Oversample factor: {train_params['oversample_factor']:.2f}x")
        if train_params.get('keep_ratio'):
            self.console.print(f"  - Keep ratio: {train_params['keep_ratio']:.2%}")
        
        # 전처리 통계
        before_stats = {
            "Train samples": len(train_df),
            "Train combo ratio": f"{train_stats['combination_ratio']:.1%}",
            "Val samples": len(val_df),
            "Val combo ratio": f"{val_stats['combination_ratio']:.1%}"
        }
        
        # 샘플링 적용
        if self.strategy == "oversample":
            train_df_balanced = self.apply_oversampling(train_df, train_params)
            val_df_balanced = self.apply_oversampling(val_df, val_params)
        elif self.strategy == "undersample":
            train_df_balanced = self.apply_undersampling(train_df, train_params)
            val_df_balanced = self.apply_undersampling(val_df, val_params)
        elif self.strategy == "mixed":
            train_df_balanced = self.apply_mixed_strategy(train_df, train_params)
            val_df_balanced = self.apply_mixed_strategy(val_df, val_params)
        else:
            train_df_balanced = train_df
            val_df_balanced = val_df
        
        # 후처리 통계
        train_stats_after = self.analyze_current_ratio(train_df_balanced)
        val_stats_after = self.analyze_current_ratio(val_df_balanced)
        
        after_stats = {
            "Train samples": len(train_df_balanced),
            "Train combo ratio": f"{train_stats_after['combination_ratio']:.1%}",
            "Val samples": len(val_df_balanced),
            "Val combo ratio": f"{val_stats_after['combination_ratio']:.1%}"
        }
        
        # 전후 비교
        self.print_before_after_table(before_stats, after_stats, "Ratio Balancing Results")
        
        # 목표 달성 여부 확인
        train_gap = abs(train_stats_after['combination_ratio'] - self.target_ratio)
        val_gap = abs(val_stats_after['combination_ratio'] - self.target_ratio)
        
        # Manifest 저장
        if not self.args.dry_run:
            self.save_manifest(train_df_balanced, self.train_manifest_path, suffix=f".balanced_{self.strategy}")
            self.save_manifest(val_df_balanced, self.val_manifest_path, suffix=f".balanced_{self.strategy}")
        
        # 샘플러 설정 저장
        sampler_config = self.generate_sampler_config(train_params)
        sampler_path = self.report_dir / f"sampler_config_{self.timestamp}.json"
        
        if not self.args.dry_run:
            with open(sampler_path, 'w') as f:
                json.dump(sampler_config, f, indent=2)
            self.logger.info(f"Saved sampler config: {sampler_path}")
        
        # 리포트 저장
        report_data = {
            "before": {
                "train": train_stats,
                "val": val_stats
            },
            "after": {
                "train": train_stats_after,
                "val": val_stats_after
            },
            "params": {
                "train": train_params,
                "val": val_params
            },
            "sampler_config": sampler_config,
            "target_ratio": self.target_ratio,
            "strategy": self.strategy
        }
        self.save_report(report_data, "combination_ratio_balancing")
        
        # 최종 결론
        tolerance = 0.05  # 5% 허용 오차
        if train_gap <= tolerance and val_gap <= tolerance:
            self.print_conclusion("PASS", 
                f"✅ Successfully balanced to target ratio {self.target_ratio:.1%}\n"
                f"Train: {train_stats_after['combination_ratio']:.1%}, Val: {val_stats_after['combination_ratio']:.1%}")
        else:
            self.print_conclusion("PARTIAL",
                f"⚠️ Partially achieved target ratio {self.target_ratio:.1%}\n"
                f"Train gap: {train_gap:.1%}, Val gap: {val_gap:.1%}")
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Balance single/combination ratio",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    DataQualityBase.add_common_arguments(parser)
    
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.25,
        help='Target combination ratio (default: 0.25 = 25%%)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['oversample', 'undersample', 'mixed'],
        default='oversample',
        help='Balancing strategy (default: oversample)'
    )
    
    # 사용 예시
    parser.epilog = """
Examples:
  # Oversample combination to 25% (default, dry run)
  python balance_combination_ratio.py
  
  # Actually apply oversampling
  python balance_combination_ratio.py --no-dry-run
  
  # Target 30% combination with undersampling
  python balance_combination_ratio.py --target-ratio 0.3 --strategy undersample
  
  # Mixed strategy (both over and under sampling)
  python balance_combination_ratio.py --strategy mixed --target-ratio 0.2
"""
    
    args = parser.parse_args()
    balancer = CombinationRatioBalancer(args)
    return balancer.run()


if __name__ == "__main__":
    exit(main())