#!/usr/bin/env python3
"""
Stage 3 Sanity Check Script
Val > Train 현상 분석을 위한 3가지 점검:
A) 데이터 누수 점검
B) 동일 전처리 스모크 평가 
C) 도메인별 F1 분해
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 체크포인트 로딩을 위한 클래스 import
from src.training.train_stage3_two_stage import TwoStageTrainingConfig

from src.models.classifier_efficientnetv2 import create_pillsnap_classifier
from src.data.dataloader_manifest_training import ManifestDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class SanityChecker:
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        checkpoint_path: Optional[str] = None,
        output_dir: str = "artifacts/stage3/reports",
        device: str = "cuda"
    ):
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.checkpoint_path = checkpoint_path or "artifacts/stage3/checkpoints/stage3_classification_best.pt"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 결과 저장용
        self.results = {
            'leakage': {},
            'smoke_eval': {},
            'domain': {}
        }
        
    def load_manifests(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Val manifest 로드"""
        print("📂 Loading manifests...")
        train_df = pd.read_csv(self.train_manifest)
        val_df = pd.read_csv(self.val_manifest)
        
        # 고유 키 생성 (경로 + mapping_code)
        train_df['key'] = train_df['image_path'] + '_' + train_df['mapping_code'].astype(str)
        val_df['key'] = val_df['image_path'] + '_' + val_df['mapping_code'].astype(str)
        
        # mapping_code를 정수 인덱스로 변환
        all_codes = pd.concat([train_df['mapping_code'], val_df['mapping_code']]).unique()
        self.code_to_idx = {code: idx for idx, code in enumerate(sorted(all_codes))}
        
        train_df['label'] = train_df['mapping_code'].map(self.code_to_idx)
        val_df['label'] = val_df['mapping_code'].map(self.code_to_idx)
        
        print(f"  Train: {len(train_df):,} samples, {train_df['mapping_code'].nunique()} classes")
        print(f"  Val: {len(val_df):,} samples, {val_df['mapping_code'].nunique()} classes")
        print(f"  Total unique classes: {len(self.code_to_idx)}")
        
        return train_df, val_df
    
    def check_data_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """A) 데이터 누수 점검"""
        print("\n" + "="*60)
        print("A) DATA LEAKAGE CHECK")
        print("="*60)
        
        results = {}
        
        # A-1) 교집합 샘플 체크
        train_keys = set(train_df['key'])
        val_keys = set(val_df['key'])
        intersect_keys = train_keys & val_keys
        
        results['intersect_count'] = len(intersect_keys)
        results['intersect_ratio'] = len(intersect_keys) / len(val_keys) * 100
        results['intersect_samples'] = list(intersect_keys)[:20]  # 상위 20개만
        
        # A-2) Val-only 클래스 체크
        train_classes = set(train_df['mapping_code'])
        val_classes = set(val_df['mapping_code'])
        val_only_classes = val_classes - train_classes
        
        results['val_only_classes'] = list(val_only_classes)
        results['val_only_count'] = len(val_only_classes)
        
        # A-3) 파일 중복 체크 (같은 파일명, 다른 라벨)
        all_df = pd.concat([train_df, val_df], ignore_index=True)
        duplicates = all_df.groupby('image_path').agg({
            'mapping_code': lambda x: list(set(x))
        })
        duplicates = duplicates[duplicates['mapping_code'].apply(len) > 1]
        
        results['duplicate_files'] = duplicates.to_dict()['mapping_code'] if len(duplicates) > 0 else {}
        results['duplicate_count'] = len(duplicates)
        
        # 결과 출력
        print(f"\n📊 Results:")
        print(f"  • Intersection: {results['intersect_count']:,} samples ({results['intersect_ratio']:.2f}%)")
        if results['intersect_count'] > 0:
            print(f"    🚨 WARNING: Found {results['intersect_count']} overlapping samples!")
            print(f"    First 5: {results['intersect_samples'][:5]}")
        else:
            print(f"    ✅ PASS: No overlapping samples")
            
        print(f"  • Val-only classes: {results['val_only_count']}")
        if results['val_only_count'] > 0:
            print(f"    🚨 WARNING: {results['val_only_count']} classes only in validation!")
            print(f"    Classes: {results['val_only_classes'][:10]}")
        else:
            print(f"    ✅ PASS: All val classes exist in train")
            
        print(f"  • Duplicate files: {results['duplicate_count']}")
        if results['duplicate_count'] > 0:
            print(f"    ⚠️ WARNING: {results['duplicate_count']} files with multiple labels!")
        else:
            print(f"    ✅ PASS: No duplicate files")
        
        # 보고서 저장
        self._save_leakage_report(results)
        
        return results
    
    def smoke_eval(self, train_df: pd.DataFrame, val_df: pd.DataFrame, max_train_samples: int = 1000) -> Dict:
        """B) 동일 전처리 스모크 평가 (증강 OFF)"""
        print("\n" + "="*60)
        print("B) SMOKE EVALUATION (No Augmentation)")
        print("="*60)
        
        # 모델 로드
        print(f"📦 Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 클래스 수 결정
        num_classes = len(self.code_to_idx)
        
        model = create_pillsnap_classifier(
            num_classes=num_classes,
            model_name="efficientnetv2_l",
            pretrained=False,
            device=self.device
        )
        
        # 체크포인트 로드 (다양한 형식 지원)
        print(f"  체크포인트 키: {list(checkpoint.keys())[:5]}...")
        
        if 'classifier_state_dict' in checkpoint:
            # Stage 3 체크포인트 형식 (classifier_state_dict 사용)
            print("  📦 Loading classifier_state_dict from checkpoint")
            state_dict = checkpoint['classifier_state_dict']
        elif 'model_state_dict' in checkpoint:
            # 일반 체크포인트 형식
            print("  📦 Loading model_state_dict from checkpoint")
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print("  📦 Loading state_dict from checkpoint")
            state_dict = checkpoint['state_dict']
        else:
            # 직접 state_dict인 경우
            print("  📦 Using checkpoint as direct state_dict")
            state_dict = checkpoint
        
        # 다양한 prefix 제거 (_orig_mod, module, ema 등)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['_orig_mod.', 'module.', 'ema.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
        
        # 키 매칭 확인
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(new_state_dict.keys())
        matched_keys = model_keys & loaded_keys
        
        print(f"  키 매칭: {len(matched_keys)}/{len(model_keys)} 매칭")
        
        # 모델에 로드
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("  ✅ 체크포인트 완전 로드 성공")
        except RuntimeError:
            model.load_state_dict(new_state_dict, strict=False)
            print("  ⚠️ 체크포인트 부분 로드 (strict=False)")
        
        # 체크포인트 메타데이터 출력
        print(f"  에포크: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Top-1: {checkpoint.get('best_top1', 0):.4f}")
        print(f"  Best Top-5: {checkpoint.get('best_top5', 0):.4f}")
        
        model.eval()
        
        # 순수 전처리만 (증강 OFF)
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = {}
        
        # Train 평가 (샘플 제한)
        if len(train_df) > max_train_samples:
            print(f"  Sampling {max_train_samples:,} from {len(train_df):,} train samples")
            train_eval_df = train_df.sample(n=max_train_samples, random_state=42)
        else:
            train_eval_df = train_df
            
        print(f"\n📊 Evaluating Train ({len(train_eval_df):,} samples)...")
        train_metrics = self._evaluate_split(
            model, train_eval_df, transform, "Train"
        )
        results['train'] = train_metrics
        
        # Val 평가 (샘플링)
        val_eval_df = val_df.sample(n=min(1000, len(val_df)), random_state=42)
        print(f"\n📊 Evaluating Val ({len(val_eval_df):,} samples)...")
        val_metrics = self._evaluate_split(
            model, val_eval_df, transform, "Val"
        )
        results['val'] = val_metrics
        
        # Gap 계산
        results['gap_top1'] = val_metrics['top1'] - train_metrics['top1']
        results['gap_top5'] = val_metrics['top5'] - train_metrics['top5']
        results['gap_macro_f1'] = val_metrics['macro_f1'] - train_metrics['macro_f1']
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 SMOKE EVAL RESULTS:")
        print(f"  Train: Top-1={train_metrics['top1']:.2%}, Top-5={train_metrics['top5']:.2%}, Macro-F1={train_metrics['macro_f1']:.4f}")
        print(f"  Val:   Top-1={val_metrics['top1']:.2%}, Top-5={val_metrics['top5']:.2%}, Macro-F1={val_metrics['macro_f1']:.4f}")
        print(f"  GAP:   Top-1={results['gap_top1']*100:.1f}%p, Top-5={results['gap_top5']*100:.1f}%p, Macro-F1={results['gap_macro_f1']:.4f}")
        
        # 경고 체크
        if results['gap_top1'] > 0:
            print(f"  🎯 Val > Train by {results['gap_top1']*100:.1f}%p!")
            if results['gap_top1'] > 0.08:
                print(f"  ⚠️ WARNING: Unusual gap (>8%p)")
        
        # 보고서 저장
        self._save_smoke_report(results)
        
        return results
    
    def domain_breakdown(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """C) 도메인별 F1 분해 (single vs combination)"""
        print("\n" + "="*60)
        print("C) DOMAIN BREAKDOWN ANALYSIS")
        print("="*60)
        
        # 모델 로드 (smoke_eval과 동일)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        num_classes = len(self.code_to_idx)
        
        model = create_pillsnap_classifier(
            num_classes=num_classes,
            model_name="efficientnetv2_l",
            pretrained=False,
            device=self.device
        )
        
        # 체크포인트 로드 (다양한 형식 지원)
        print(f"  체크포인트 키: {list(checkpoint.keys())[:5]}...")
        
        if 'classifier_state_dict' in checkpoint:
            # Stage 3 체크포인트 형식 (classifier_state_dict 사용)
            print("  📦 Loading classifier_state_dict from checkpoint")
            state_dict = checkpoint['classifier_state_dict']
        elif 'model_state_dict' in checkpoint:
            # 일반 체크포인트 형식
            print("  📦 Loading model_state_dict from checkpoint")
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            print("  📦 Loading state_dict from checkpoint")
            state_dict = checkpoint['state_dict']
        else:
            # 직접 state_dict인 경우
            print("  📦 Using checkpoint as direct state_dict")
            state_dict = checkpoint
        
        # 다양한 prefix 제거 (_orig_mod, module, ema 등)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['_orig_mod.', 'module.', 'ema.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
        
        # 키 매칭 확인
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(new_state_dict.keys())
        matched_keys = model_keys & loaded_keys
        
        print(f"  키 매칭: {len(matched_keys)}/{len(model_keys)} 매칭")
        
        # 모델에 로드
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("  ✅ 체크포인트 완전 로드 성공")
        except RuntimeError:
            model.load_state_dict(new_state_dict, strict=False)
            print("  ⚠️ 체크포인트 부분 로드 (strict=False)")
        
        # 체크포인트 메타데이터 출력
        print(f"  에포크: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Top-1: {checkpoint.get('best_top1', 0):.4f}")
        print(f"  Best Top-5: {checkpoint.get('best_top5', 0):.4f}")
        
        model.eval()
        
        # 순수 전처리
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = {'train': {}, 'val': {}}
        
        # 각 split과 domain에 대해 평가
        for split_name, df in [('train', train_df), ('val', val_df)]:
            print(f"\n📊 {split_name.upper()} Domain Analysis:")
            
            for domain in ['single', 'combination']:
                domain_df = df[df['image_type'] == domain]
                
                if len(domain_df) == 0:
                    print(f"  ⚠️ No {domain} samples in {split_name}")
                    continue
                    
                print(f"  Evaluating {domain} ({len(domain_df):,} samples)...")
                
                # 도메인별 평가
                metrics = self._evaluate_split(
                    model, domain_df, transform, f"{split_name}_{domain}", 
                    max_samples=10000 if split_name == 'train' else None
                )
                
                results[split_name][domain] = metrics
                results[split_name][f'{domain}_support'] = len(domain_df)
        
        # Domain Gap 계산
        for split_name in ['train', 'val']:
            if 'single' in results[split_name] and 'combination' in results[split_name]:
                results[split_name]['domain_gap_f1'] = (
                    results[split_name]['single']['macro_f1'] - 
                    results[split_name]['combination']['macro_f1']
                )
                results[split_name]['domain_gap_top1'] = (
                    results[split_name]['single']['top1'] - 
                    results[split_name]['combination']['top1']
                )
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 DOMAIN BREAKDOWN RESULTS:")
        
        for split_name in ['train', 'val']:
            print(f"\n{split_name.upper()}:")
            if 'single' in results[split_name]:
                s = results[split_name]['single']
                print(f"  Single:      Top-1={s['top1']:.2%}, F1={s['macro_f1']:.4f} (n={results[split_name]['single_support']:,})")
            if 'combination' in results[split_name]:
                c = results[split_name]['combination']
                print(f"  Combination: Top-1={c['top1']:.2%}, F1={c['macro_f1']:.4f} (n={results[split_name]['combination_support']:,})")
            if 'domain_gap_f1' in results[split_name]:
                print(f"  Domain Gap:  Top-1={results[split_name]['domain_gap_top1']*100:.1f}%p, F1={results[split_name]['domain_gap_f1']:.4f}")
        
        # 경고 체크
        if 'val' in results and 'domain_gap_f1' in results['val']:
            if abs(results['val']['domain_gap_f1']) >= 0.10:
                print(f"\n  ⚠️ WARNING: Large domain gap in Val (F1 gap={results['val']['domain_gap_f1']:.3f})")
            
            combo_ratio = results['val'].get('combination_support', 0) / len(val_df)
            if combo_ratio < 0.15:
                print(f"  ⚠️ WARNING: Low combination support ({combo_ratio:.1%} < 15%)")
        
        # 보고서 저장
        self._save_domain_report(results)
        
        return results
    
    def _evaluate_split(
        self, 
        model: torch.nn.Module,
        df: pd.DataFrame,
        transform,
        split_name: str,
        max_samples: Optional[int] = None
    ) -> Dict:
        """특정 split 평가"""
        from PIL import Image
        
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        # 데이터로더 생성
        # label 컬럼이 있는지 확인하고 없으면 추가
        df_copy = df.copy()
        if 'label' not in df_copy.columns:
            df_copy['label'] = df_copy['mapping_code'].map(self.code_to_idx)
        
        # ManifestDataset은 label 컬럼 대신 mapping_code를 label로 사용
        df_copy['mapping_code'] = df_copy['label']
        
        dataset = ManifestDataset(
            manifest_df=df_copy,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {split_name}", leave=False):
                # ManifestDataset은 (images, labels) 튜플 반환
                if isinstance(batch, (list, tuple)):
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                else:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        # 메트릭 계산
        correct = np.array(all_preds) == np.array(all_labels)
        top1_acc = correct.mean()
        
        # Top-5 accuracy
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
        top5_correct = [label in top5 for label, top5 in zip(all_labels, top5_preds)]
        top5_acc = np.mean(top5_correct)
        
        # Macro F1
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'top1': top1_acc,
            'top5': top5_acc,
            'macro_f1': macro_f1,
            'n_samples': len(df)
        }
    
    def _save_leakage_report(self, results: Dict):
        """누수 리포트 저장"""
        # Text report
        report_path = self.output_dir / "leakage_summary.txt"
        with open(report_path, 'w') as f:
            f.write("DATA LEAKAGE CHECK REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Intersection samples: {results['intersect_count']:,} ({results['intersect_ratio']:.2f}%)\n")
            f.write(f"Val-only classes: {results['val_only_count']}\n")
            f.write(f"Duplicate files: {results['duplicate_count']}\n\n")
            
            if results['intersect_count'] > 0:
                f.write("⚠️ WARNING: Data leakage detected!\n")
                f.write(f"First 20 overlapping samples:\n")
                for sample in results['intersect_samples'][:20]:
                    f.write(f"  - {sample}\n")
            else:
                f.write("✅ PASS: No data leakage detected\n")
        
        # CSV report
        if results['intersect_count'] > 0:
            csv_path = self.output_dir / "leakage_pairs.csv"
            pd.DataFrame({
                'overlapping_samples': results['intersect_samples']
            }).to_csv(csv_path, index=False)
            
        print(f"  📝 Saved: {report_path}")
    
    def _save_smoke_report(self, results: Dict):
        """스모크 평가 리포트 저장"""
        # Text report
        report_path = self.output_dir / "smoke_eval_summary.txt"
        with open(report_path, 'w') as f:
            f.write("SMOKE EVALUATION REPORT (No Augmentation)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"Train: Top-1={results['train']['top1']:.4f}, Top-5={results['train']['top5']:.4f}, Macro-F1={results['train']['macro_f1']:.4f}\n")
            f.write(f"Val:   Top-1={results['val']['top1']:.4f}, Top-5={results['val']['top5']:.4f}, Macro-F1={results['val']['macro_f1']:.4f}\n\n")
            
            f.write("GAPS (Val - Train):\n")
            f.write(f"Top-1: {results['gap_top1']*100:+.1f}%p\n")
            f.write(f"Top-5: {results['gap_top5']*100:+.1f}%p\n")
            f.write(f"Macro-F1: {results['gap_macro_f1']:+.4f}\n\n")
            
            if results['gap_top1'] > 0:
                f.write("🎯 FINDING: Val > Train phenomenon detected!\n")
                if results['gap_top1'] > 0.08:
                    f.write("⚠️ WARNING: Unusual gap (>8%p) - investigate further\n")
                else:
                    f.write("✅ Gap within normal range (<8%p)\n")
        
        # CSV report
        csv_path = self.output_dir / "smoke_eval_metrics.csv"
        pd.DataFrame([
            {
                'split': 'train',
                'top1': results['train']['top1'],
                'top5': results['train']['top5'],
                'macro_f1': results['train']['macro_f1'],
                'n_samples': results['train']['n_samples']
            },
            {
                'split': 'val',
                'top1': results['val']['top1'],
                'top5': results['val']['top5'],
                'macro_f1': results['val']['macro_f1'],
                'n_samples': results['val']['n_samples']
            }
        ]).to_csv(csv_path, index=False)
        
        print(f"  📝 Saved: {report_path}")
        print(f"  📝 Saved: {csv_path}")
    
    def _save_domain_report(self, results: Dict):
        """도메인 분석 리포트 저장"""
        # CSV report
        csv_path = self.output_dir / "domain_breakdown.csv"
        rows = []
        for split in ['train', 'val']:
            if split not in results:
                continue
            for domain in ['single', 'combination']:
                if domain not in results[split]:
                    continue
                rows.append({
                    'split': split,
                    'domain': domain,
                    'top1': results[split][domain]['top1'],
                    'top5': results[split][domain]['top5'],
                    'macro_f1': results[split][domain]['macro_f1'],
                    'support': results[split].get(f'{domain}_support', 0)
                })
        
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        
        # Text report
        report_path = self.output_dir / "domain_summary.txt"
        with open(report_path, 'w') as f:
            f.write("DOMAIN BREAKDOWN REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for split in ['train', 'val']:
                if split not in results:
                    continue
                    
                f.write(f"{split.upper()}:\n")
                if 'single' in results[split]:
                    s = results[split]['single']
                    f.write(f"  Single:      Top-1={s['top1']:.4f}, F1={s['macro_f1']:.4f} (n={results[split].get('single_support', 0):,})\n")
                if 'combination' in results[split]:
                    c = results[split]['combination']
                    f.write(f"  Combination: Top-1={c['top1']:.4f}, F1={c['macro_f1']:.4f} (n={results[split].get('combination_support', 0):,})\n")
                if 'domain_gap_f1' in results[split]:
                    f.write(f"  Domain Gap:  Top-1={results[split]['domain_gap_top1']*100:.1f}%p, F1={results[split]['domain_gap_f1']:.4f}\n")
                f.write("\n")
            
            # 경고
            if 'val' in results and 'domain_gap_f1' in results['val']:
                if abs(results['val']['domain_gap_f1']) >= 0.10:
                    f.write("⚠️ WARNING: Large domain gap detected (F1 gap >= 0.10)\n")
                    f.write("   Possible domain shift or imbalanced evaluation\n")
        
        print(f"  📝 Saved: {csv_path}")
        print(f"  📝 Saved: {report_path}")
    
    def run_all_checks(self):
        """모든 체크 실행"""
        print("\n" + "="*80)
        print(" "*25 + "STAGE 3 SANITY CHECK")
        print(" "*20 + "Val > Train Phenomenon Analysis")
        print("="*80)
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # A) 데이터 누수 점검
        leakage_results = self.check_data_leakage(train_df, val_df)
        self.results['leakage'] = leakage_results
        
        # B) 스모크 평가
        smoke_results = self.smoke_eval(train_df, val_df)
        self.results['smoke_eval'] = smoke_results
        
        # C) 도메인 분석
        domain_results = self.domain_breakdown(train_df, val_df)
        self.results['domain'] = domain_results
        
        # 최종 요약
        self._print_final_summary()
        
        return self.results
    
    def _print_final_summary(self):
        """최종 요약 출력"""
        print("\n" + "="*80)
        print(" "*30 + "[SANITY SUMMARY]")
        print("="*80)
        
        warnings = []
        
        # Leakage 요약
        l = self.results['leakage']
        print(f"• Leakage: intersect={l['intersect_count']}, val_only_classes={l['val_only_count']}, duplicates={l['duplicate_count']}")
        if l['intersect_count'] > 0 or l['val_only_count'] > 0:
            warnings.append("DATA_LEAKAGE")
        
        # Smoke 요약
        s = self.results['smoke_eval']
        print(f"• Smoke (No-Aug): Train top1={s['train']['top1']:.1%}, Val top1={s['val']['top1']:.1%}, Gap={s['gap_top1']*100:+.1f}%p")
        print(f"                  Train F1={s['train']['macro_f1']:.3f}, Val F1={s['val']['macro_f1']:.3f}, Gap={s['gap_macro_f1']:+.3f}")
        if s['gap_top1'] > 0.08 or s['gap_macro_f1'] > 0.06:
            warnings.append("UNUSUAL_GAP")
        
        # Domain 요약
        if 'val' in self.results['domain']:
            d = self.results['domain']['val']
            if 'single' in d and 'combination' in d:
                print(f"• Domain (Val): single F1={d['single']['macro_f1']:.3f}, combo F1={d['combination']['macro_f1']:.3f}, Gap={d.get('domain_gap_f1', 0):.3f}")
                print(f"                supports: single={d.get('single_support', 0):,}, combo={d.get('combination_support', 0):,}")
                if abs(d.get('domain_gap_f1', 0)) >= 0.10:
                    warnings.append("DOMAIN_GAP")
        
        # Verdict
        print("\n• Verdict: ", end="")
        if warnings:
            print(f"⚠️ WARN ({', '.join(warnings)})")
        else:
            print("✅ OK")
        
        # 생성된 파일 목록
        print("\n📁 Generated Reports:")
        for report_file in sorted(self.output_dir.glob("*.txt")) + sorted(self.output_dir.glob("*.csv")):
            print(f"  - {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Sanity Check")
    parser.add_argument('--train-manifest', 
                       default='/home/max16/pillsnap/artifacts/stage3/manifest_train.remove.csv',
                       help='Path to training manifest')
    parser.add_argument('--val-manifest',
                       default='/home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv',
                       help='Path to validation manifest')
    parser.add_argument('--checkpoint',
                       default='artifacts/stage3/checkpoints/stage3_classification_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir',
                       default='artifacts/stage3/reports',
                       help='Output directory for reports')
    parser.add_argument('--check-leakage', action='store_true', help='Run leakage check only')
    parser.add_argument('--eval-smoke', action='store_true', help='Run smoke eval only')
    parser.add_argument('--eval-domain', action='store_true', help='Run domain analysis only')
    
    args = parser.parse_args()
    
    # Sanity Checker 초기화
    checker = SanityChecker(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # 체크 실행
    if args.check_leakage or args.eval_smoke or args.eval_domain:
        # 개별 실행
        train_df, val_df = checker.load_manifests()
        
        if args.check_leakage:
            checker.check_data_leakage(train_df, val_df)
        if args.eval_smoke:
            checker.smoke_eval(train_df, val_df)
        if args.eval_domain:
            checker.domain_breakdown(train_df, val_df)
    else:
        # 전체 실행
        checker.run_all_checks()


if __name__ == "__main__":
    main()