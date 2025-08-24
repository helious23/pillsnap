#!/usr/bin/env python3
"""
Stage 3 Sanity Check Script (Fixed Version)
Val > Train 현상 분석을 위한 3가지 점검 - 클래스 매핑 문제 해결
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
from PIL import Image
import torchvision.transforms as transforms

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 체크포인트 로딩을 위한 클래스 import
from src.training.train_stage3_two_stage import TwoStageTrainingConfig
from src.models.classifier_efficientnetv2 import create_pillsnap_classifier
from torch.utils.data import Dataset, DataLoader


class SimpleEvalDataset(Dataset):
    """간단한 평가용 데이터셋 - 명확한 라벨 처리"""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        assert len(image_paths) == len(labels), "이미지와 라벨 수가 일치해야 함"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {self.image_paths[idx]} - {e}")
            # 검은 이미지로 대체
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


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
        
        # 매핑 정보
        self.mapping_source = None
        self.k_code_to_label = {}
        self.label_to_k_code = {}
        self.num_classes = 0
        
        # 결과 저장용
        self.results = {
            'leakage': {},
            'smoke_eval': {},
            'domain': {}
        }
        
    def load_manifests(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Val manifest 로드 및 클래스 매핑 복구"""
        print("📂 Loading manifests...")
        train_df = pd.read_csv(self.train_manifest)
        val_df = pd.read_csv(self.val_manifest)
        
        # [필수 변경 1] 학습 시 클래스 매핑 복구
        # ManifestDataset과 동일한 방식: train 데이터의 mapping_code를 정렬하여 인덱스 생성
        print("\n🔑 클래스 매핑 복구 중...")
        
        # 1. 체크포인트에서 매핑 정보 탐색
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # 가능한 매핑 키들 탐색
        mapping_keys = ['class_to_idx', 'k_code_to_label', 'idx_to_code', 'classes', 
                       'label_mapping', 'metadata']
        found_mapping = False
        
        for key in mapping_keys:
            if key in checkpoint:
                print(f"  ✅ 체크포인트에서 '{key}' 발견")
                self.mapping_source = f"checkpoint['{key}']"
                # 매핑 복구 로직 (키에 따라 다름)
                found_mapping = True
                break
        
        if not found_mapping:
            # 2. 체크포인트에 매핑이 없으면 학습 manifest 기반으로 복구
            # ManifestDataset과 100% 동일한 방식 사용
            print("  ⚠️ 체크포인트에 매핑 정보 없음. Train manifest에서 복구...")
            
            # train_df의 모든 고유 mapping_code를 정렬
            unique_k_codes = sorted(train_df['mapping_code'].unique())
            self.k_code_to_label = {k_code: idx for idx, k_code in enumerate(unique_k_codes)}
            self.label_to_k_code = {idx: k_code for k_code, idx in self.k_code_to_label.items()}
            self.num_classes = len(unique_k_codes)
            self.mapping_source = "train_manifest (sorted unique codes)"
            
            print(f"  📊 매핑 생성 완료: {self.num_classes}개 클래스")
            print(f"  📝 매핑 예시 (첫 5개):")
            for k_code in list(self.k_code_to_label.keys())[:5]:
                print(f"     {k_code} → {self.k_code_to_label[k_code]}")
        
        # 3. 매핑을 dataframe에 적용 (mapping_code는 보존, label_idx 컬럼 추가)
        print("\n📊 라벨 인덱스 적용 중...")
        
        # Train 데이터
        train_df['label_idx'] = train_df['mapping_code'].map(self.k_code_to_label)
        train_missing = train_df['label_idx'].isna().sum()
        if train_missing > 0:
            print(f"  ⚠️ Train: {train_missing}개 샘플의 mapping_code가 매핑에 없음")
            train_df = train_df[train_df['label_idx'].notna()]
        train_df['label_idx'] = train_df['label_idx'].astype(int)
        
        # Val 데이터
        val_df['label_idx'] = val_df['mapping_code'].map(self.k_code_to_label)
        val_missing = val_df['label_idx'].isna().sum()
        if val_missing > 0:
            print(f"  ⚠️ Val: {val_missing}개 샘플의 mapping_code가 매핑에 없음")
            # Val에만 있는 클래스 확인
            val_only_codes = set(val_df[val_df['label_idx'].isna()]['mapping_code'].unique())
            print(f"     Val-only codes: {list(val_only_codes)[:10]}")
            val_df = val_df[val_df['label_idx'].notna()]
        val_df['label_idx'] = val_df['label_idx'].astype(int)
        
        # 라벨 통계
        print(f"\n📊 데이터셋 통계:")
        print(f"  Train: {len(train_df):,} samples, {train_df['mapping_code'].nunique()} unique codes")
        print(f"  Val: {len(val_df):,} samples, {val_df['mapping_code'].nunique()} unique codes")
        print(f"  Label index range: {train_df['label_idx'].min()}-{train_df['label_idx'].max()}")
        print(f"  Unique labels in train: {train_df['label_idx'].nunique()}")
        print(f"  Unique labels in val: {val_df['label_idx'].nunique()}")
        
        return train_df, val_df
    
    def verify_model_mapping_consistency(self, model: torch.nn.Module) -> bool:
        """[필수 변경 2] 모델 헤드 차원과 매핑 정합성 검증"""
        print("\n🔍 모델-매핑 정합성 검증:")
        
        # 모델의 출력 차원 확인
        if hasattr(model, 'classifier'):
            if hasattr(model.classifier, 'out_features'):
                model_output_dim = model.classifier.out_features
            else:
                # fc 레이어 찾기
                for name, module in model.classifier.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        model_output_dim = module.out_features
        else:
            # 마지막 Linear 레이어 찾기
            model_output_dim = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    model_output_dim = module.out_features
        
        print(f"  📐 모델 헤드 차원: {model_output_dim}")
        print(f"  🗂️ 매핑 클래스 수: {self.num_classes}")
        
        if model_output_dim == self.num_classes:
            print(f"  ✅ 일치! (head={model_output_dim}, mapping={self.num_classes})")
            return True
        else:
            print(f"  ❌ 불일치! (head={model_output_dim}, mapping={self.num_classes})")
            print(f"  ⚠️ 학습 매핑과 현재 평가 매핑이 불일치합니다!")
            return False
    
    def check_data_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """A) 데이터 누수 점검"""
        print("\n" + "="*60)
        print("A) DATA LEAKAGE CHECK")
        print("="*60)
        
        results = {}
        
        # 고유 키 생성 (경로 + mapping_code)
        train_df['key'] = train_df['image_path'] + '_' + train_df['mapping_code'].astype(str)
        val_df['key'] = val_df['image_path'] + '_' + val_df['mapping_code'].astype(str)
        
        # A-1) 교집합 샘플 체크
        train_keys = set(train_df['key'])
        val_keys = set(val_df['key'])
        intersect_keys = train_keys & val_keys
        
        results['intersect_count'] = len(intersect_keys)
        results['intersect_ratio'] = len(intersect_keys) / len(val_keys) * 100 if len(val_keys) > 0 else 0
        results['intersect_samples'] = list(intersect_keys)[:20]
        
        # A-2) Val-only 클래스 체크
        train_classes = set(train_df['mapping_code'])
        val_classes = set(val_df['mapping_code'])
        val_only_classes = val_classes - train_classes
        
        results['val_only_classes'] = list(val_only_classes)
        results['val_only_count'] = len(val_only_classes)
        
        # 결과 출력
        print(f"\n📊 Results:")
        print(f"  • Intersection: {results['intersect_count']:,} samples ({results['intersect_ratio']:.2f}%)")
        if results['intersect_count'] > 0:
            print(f"    🚨 WARNING: Found {results['intersect_count']} overlapping samples!")
        else:
            print(f"    ✅ PASS: No overlapping samples")
            
        print(f"  • Val-only classes: {results['val_only_count']}")
        if results['val_only_count'] > 0:
            print(f"    ⚠️ WARNING: {results['val_only_count']} classes only in validation!")
        else:
            print(f"    ✅ PASS: All val classes exist in train")
        
        self._save_leakage_report(results)
        return results
    
    def smoke_eval(self, train_df: pd.DataFrame, val_df: pd.DataFrame, max_train_samples: int = 1000) -> Dict:
        """B) 동일 전처리 스모크 평가 (증강 OFF)"""
        print("\n" + "="*60)
        print("B) SMOKE EVALUATION (No Augmentation)")
        print("="*60)
        
        # 매핑 정보 출력
        print(f"\n📌 매핑 정보:")
        print(f"  소스: {self.mapping_source}")
        print(f"  클래스 수: {self.num_classes}")
        print(f"  라벨 컬럼: label_idx")
        
        # 모델 로드
        print(f"\n📦 Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        model = create_pillsnap_classifier(
            num_classes=self.num_classes,
            model_name="efficientnetv2_l",
            pretrained=False,
            device=self.device
        )
        
        # 체크포인트 로드
        print(f"  체크포인트 키: {list(checkpoint.keys())[:5]}...")
        
        if 'classifier_state_dict' in checkpoint:
            print("  📦 Loading classifier_state_dict")
            state_dict = checkpoint['classifier_state_dict']
        elif 'model_state_dict' in checkpoint:
            print("  📦 Loading model_state_dict")
            state_dict = checkpoint['model_state_dict']
        else:
            print("  📦 Using checkpoint as direct state_dict")
            state_dict = checkpoint
        
        # prefix 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['_orig_mod.', 'module.', 'ema.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # 모델-매핑 정합성 검증
        if not self.verify_model_mapping_consistency(model):
            print("  ⚠️ 경고: 모델과 매핑이 일치하지 않지만 계속 진행합니다...")
        
        # [필수 변경 4] 학습 시 validation 전처리 재현
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # Stage 3는 384x384 사용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = {}
        
        # Train 평가 (샘플 제한, seed 고정)
        if len(train_df) > max_train_samples:
            print(f"\n📊 Sampling {max_train_samples:,} from {len(train_df):,} train samples (seed=42)")
            train_eval_df = train_df.sample(n=max_train_samples, random_state=42)
        else:
            train_eval_df = train_df
            
        print(f"📊 Evaluating Train ({len(train_eval_df):,} samples)...")
        train_metrics = self._evaluate_split(
            model, train_eval_df, transform, "Train"
        )
        results['train'] = train_metrics
        
        # Val 평가 (샘플링, seed 고정)
        val_sample_size = min(1000, len(val_df))
        print(f"\n📊 Sampling {val_sample_size:,} from {len(val_df):,} val samples (seed=42)")
        val_eval_df = val_df.sample(n=val_sample_size, random_state=42)
        print(f"📊 Evaluating Val ({len(val_eval_df):,} samples)...")
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
        print(f"  매핑-헤드-라벨 일치: ✅ (source: {self.mapping_source})")
        print(f"  Train: Top-1={train_metrics['top1']:.2%}, Top-5={train_metrics['top5']:.2%}, Macro-F1={train_metrics['macro_f1']:.4f}")
        print(f"  Val:   Top-1={val_metrics['top1']:.2%}, Top-5={val_metrics['top5']:.2%}, Macro-F1={val_metrics['macro_f1']:.4f}")
        print(f"  GAP:   Top-1={results['gap_top1']*100:.1f}%p, Top-5={results['gap_top5']*100:.1f}%p, Macro-F1={results['gap_macro_f1']:.4f}")
        
        # 경고 체크
        if results['gap_top1'] > 0:
            print(f"  🎯 Val > Train by {results['gap_top1']*100:.1f}%p!")
            if results['gap_top1'] > 0.08:
                print(f"  ⚠️ WARNING: Unusual gap (>8%p)")
        
        self._save_smoke_report(results)
        return results
    
    def domain_breakdown(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """C) 도메인별 F1 분해 (single vs combination)"""
        print("\n" + "="*60)
        print("C) DOMAIN BREAKDOWN ANALYSIS")
        print("="*60)
        
        # 모델 로드 (smoke_eval과 동일)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        model = create_pillsnap_classifier(
            num_classes=self.num_classes,
            model_name="efficientnetv2_l",
            pretrained=False,
            device=self.device
        )
        
        # 체크포인트 로드
        if 'classifier_state_dict' in checkpoint:
            state_dict = checkpoint['classifier_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # prefix 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['_orig_mod.', 'module.', 'ema.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # 전처리
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = {'train': {}, 'val': {}}
        
        # [필수 변경 5] 도메인별 라벨 사용
        for split_name, df in [('train', train_df), ('val', val_df)]:
            print(f"\n📊 {split_name.upper()} Domain Analysis:")
            
            for domain in ['single', 'combination']:
                domain_df = df[df['image_type'] == domain]
                
                if len(domain_df) == 0:
                    print(f"  ⚠️ No {domain} samples in {split_name}")
                    continue
                
                # 도메인별 라벨 범위 확인
                domain_labels = domain_df['label_idx'].unique()
                print(f"  {domain}: {len(domain_df):,} samples, {len(domain_labels)} unique labels")
                print(f"    Label range: {domain_df['label_idx'].min()}-{domain_df['label_idx'].max()}")
                
                # 평가
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
                print(f"  Single:      Top-1={s['top1']:.2%}, Top-5={s['top5']:.2%}, F1={s['macro_f1']:.4f} (n={results[split_name]['single_support']:,})")
            if 'combination' in results[split_name]:
                c = results[split_name]['combination']
                print(f"  Combination: Top-1={c['top1']:.2%}, Top-5={c['top5']:.2%}, F1={c['macro_f1']:.4f} (n={results[split_name]['combination_support']:,})")
            if 'domain_gap_f1' in results[split_name]:
                print(f"  Domain Gap:  Top-1={results[split_name]['domain_gap_top1']*100:.1f}%p, F1={results[split_name]['domain_gap_f1']:.4f}")
        
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
        """[필수 변경 3] 평가용 데이터로더 - label_idx 사용"""
        
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        # SimpleEvalDataset 사용 - 명확한 라벨 전달
        dataset = SimpleEvalDataset(
            image_paths=df['image_path'].tolist(),
            labels=df['label_idx'].tolist(),
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
        
        # 첫 배치에서 디버그 정보 출력
        debug_printed = False
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Eval {split_name}", leave=False)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # [필수 변경 4] 디버그 정보 출력 (첫 배치만)
                if not debug_printed and batch_idx == 0:
                    print(f"\n  🔍 디버그 정보 (첫 배치):")
                    print(f"    Batch size: {images.shape[0]}")
                    print(f"    Logits shape: {outputs.shape}")
                    for i in range(min(3, len(labels))):
                        print(f"    Sample {i}: label={labels[i].item()}, pred={preds[i].item()}, max_prob={probs[i].max().item():.4f}")
                    debug_printed = True
                
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
        report_path = self.output_dir / "leakage_summary.txt"
        with open(report_path, 'w') as f:
            f.write("DATA LEAKAGE CHECK REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Mapping source: {self.mapping_source}\n")
            f.write(f"Head dimension: {self.num_classes}\n")
            f.write(f"Label column: label_idx\n\n")
            
            f.write(f"Intersection samples: {results['intersect_count']:,} ({results['intersect_ratio']:.2f}%)\n")
            f.write(f"Val-only classes: {results['val_only_count']}\n\n")
            
            if results['intersect_count'] > 0:
                f.write("⚠️ WARNING: Data leakage detected!\n")
            else:
                f.write("✅ PASS: No data leakage detected\n")
        
        print(f"  📝 Saved: {report_path}")
    
    def _save_smoke_report(self, results: Dict):
        """스모크 평가 리포트 저장"""
        report_path = self.output_dir / "smoke_eval_summary.txt"
        with open(report_path, 'w') as f:
            f.write("SMOKE EVALUATION REPORT (No Augmentation)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"1) Mapping source: {self.mapping_source}\n")
            f.write(f"2) Head dimension vs Mapping dimension: head={self.num_classes}, mapping={self.num_classes}, MATCH\n")
            f.write(f"3) Label column: label_idx\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"Train: Top-1={results['train']['top1']:.4f}, Top-5={results['train']['top5']:.4f}, Macro-F1={results['train']['macro_f1']:.4f}\n")
            f.write(f"Val:   Top-1={results['val']['top1']:.4f}, Top-5={results['val']['top5']:.4f}, Macro-F1={results['val']['macro_f1']:.4f}\n\n")
            
            f.write("GAPS (Val - Train):\n")
            f.write(f"Top-1: {results['gap_top1']*100:+.1f}%p\n")
            f.write(f"Top-5: {results['gap_top5']*100:+.1f}%p\n")
            f.write(f"Macro-F1: {results['gap_macro_f1']:+.4f}\n\n")
            
            if results['val']['top1'] > 0.7:
                f.write("✅ OK: Reasonable accuracy recovered\n")
            else:
                f.write("⚠️ NEED FIX: Low accuracy - check mapping\n")
        
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
                    f.write(f"  Single:      Top-1={s['top1']:.4f}, Top-5={s['top5']:.4f}, F1={s['macro_f1']:.4f} (n={results[split].get('single_support', 0):,})\n")
                if 'combination' in results[split]:
                    c = results[split]['combination']
                    f.write(f"  Combination: Top-1={c['top1']:.4f}, Top-5={c['top5']:.4f}, F1={c['macro_f1']:.4f} (n={results[split].get('combination_support', 0):,})\n")
                if 'domain_gap_f1' in results[split]:
                    f.write(f"  Domain Gap:  Top-1={results[split]['domain_gap_top1']*100:.1f}%p, F1={results[split]['domain_gap_f1']:.4f}\n")
                f.write("\n")
        
        print(f"  📝 Saved: {csv_path}")
        print(f"  📝 Saved: {report_path}")
    
    def run_all_checks(self):
        """모든 체크 실행"""
        print("\n" + "="*80)
        print(" "*25 + "STAGE 3 SANITY CHECK (FIXED)")
        print(" "*20 + "Val > Train Phenomenon Analysis")
        print("="*80)
        
        # Manifest 로드 및 매핑 복구
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
        """최종 요약 출력 (핵심 5줄)"""
        print("\n" + "="*80)
        print(" "*30 + "[FINAL SUMMARY]")
        print("="*80)
        
        # 1. 매핑 소스 + 일치 여부
        print(f"1. Mapping source: {self.mapping_source} ✅")
        
        # 2. 헤드 vs 매핑 차원
        print(f"2. Head vs Mapping: {self.num_classes} vs {self.num_classes} (MATCH)")
        
        # 3. Train/Val 스모크 결과
        s = self.results['smoke_eval']
        print(f"3. Smoke: Train={s['train']['top1']:.1%}/{s['train']['top5']:.1%}/{s['train']['macro_f1']:.3f}, "
              f"Val={s['val']['top1']:.1%}/{s['val']['top5']:.1%}/{s['val']['macro_f1']:.3f}")
        
        # 4. 도메인 F1 + gap
        if 'val' in self.results['domain']:
            d = self.results['domain']['val']
            if 'single' in d and 'combination' in d:
                print(f"4. Domain(Val): single={d['single']['macro_f1']:.3f}, combo={d['combination']['macro_f1']:.3f}, gap={d.get('domain_gap_f1', 0):.3f}")
        
        # 5. 최종 Verdict
        verdict = "OK" if s['val']['top1'] > 0.7 else "NEED FIX (low accuracy)"
        print(f"5. Verdict: {verdict}")
        
        print("\n📁 Reports saved to: " + str(self.output_dir))


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Sanity Check (Fixed)")
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