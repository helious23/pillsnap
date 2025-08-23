#!/usr/bin/env python3
"""
Stage 3 Classification 전용 학습기

Classification 성능 극대화 전략:
- EfficientNetV2-L 사용 (Stage 3+ Large 모델)
- Single 95% + Combination 5% 데이터
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- 목표: Classification Accuracy 85%
- Detection 학습 완전 생략
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd

from src.models.classifier_efficientnetv2 import PillSnapClassifier, create_pillsnap_classifier
from src.data.dataloader_manifest_training import ManifestDataset, ManifestTrainingDataLoader
from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor  
from src.evaluation.evaluate_classification_metrics import ClassificationMetricsEvaluator
from src.utils.core import PillSnapLogger, load_config


class Stage3ClassificationTrainer:
    """Stage 3 Classification 전용 학습기"""
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        manifest_train: str = "artifacts/stage3/manifest_train.csv",
        manifest_val: str = "artifacts/stage3/manifest_val.csv",
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.logger = PillSnapLogger(__name__)
        
        # 설정 로드
        self.config = load_config(config_path)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val)
        
        # Stage 3 설정 확인
        self.stage_config = self.config['progressive_validation']['stage_configs']['stage_3']
        if not self.stage_config.get('focus') == 'classification_only':
            self.logger.warning("Stage 3이 Classification 전용으로 설정되지 않음")
        
        # RTX 5080 최적화
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.logger.info("🚀 RTX 5080 GPU 최적화 활성화")
        
        # 모니터링 시스템
        self.memory_monitor = GPUMemoryMonitor()
        self.metrics_evaluator = ClassificationMetricsEvaluator(
            num_classes=self.config['data']['num_classes']
        )
        
        # 재현성 설정
        self.seed = 42
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # 학습 상태
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.best_accuracy = 0.0
        self.best_macro_f1 = 0.0
        self.training_history = []
        
        self.logger.info("Stage 3 Classification 전용 학습기 초기화 완료")
        self.logger.info(f"목표 정확도: {self.stage_config['target_metrics']['classification_accuracy']}")
        self.logger.info(f"Manifest: train={self.manifest_train}, val={self.manifest_val}")
    
    def setup_data_loaders(self) -> None:
        """Manifest 기반 데이터 로더 설정"""
        
        # Manifest 파일 존재 확인
        if not self.manifest_train.exists():
            raise FileNotFoundError(f"Train manifest 없음: {self.manifest_train}")
        if not self.manifest_val.exists():
            raise FileNotFoundError(f"Val manifest 없음: {self.manifest_val}")
        
        # Train/Val manifest 로드
        train_df = pd.read_csv(self.manifest_train)
        val_df = pd.read_csv(self.manifest_val)
        
        self.logger.info(f"Train 샘플: {len(train_df):,}개")
        self.logger.info(f"Val 샘플: {len(val_df):,}개")
        
        # 데이터 분포 확인
        train_single = (train_df['image_type'] == 'single').sum()
        train_combo = (train_df['image_type'] == 'combination').sum()
        single_ratio = train_single / len(train_df)
        
        self.logger.info(f"Train 분포: Single {single_ratio:.1%} ({train_single:,}), Combination {1-single_ratio:.1%} ({train_combo:,})")
        
        # 배치 크기 설정 (RTX 5080 최적화)
        batch_size = self.config.get('train', {}).get('batch_size', 16)  # 기본값 16
        if single_ratio >= 0.9:  # Classification 중심인 경우 배치 크기 증가 가능
            batch_size = min(batch_size + 8, 32)  # 최대 32
            self.logger.info(f"Classification 중심으로 배치 크기 증가: {batch_size}")
        
        # 데이터로더 생성 (ManifestDataset 직접 사용)
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # 변환 정의
        train_transform = transforms.Compose([
            transforms.Resize((self.config['data']['img_size']['classification'], 
                              self.config['data']['img_size']['classification'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['data']['img_size']['classification'], 
                              self.config['data']['img_size']['classification'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 데이터셋 생성
        train_dataset = ManifestDataset(train_df, transform=train_transform)
        val_dataset = ManifestDataset(val_df, transform=val_transform)
        
        # 데이터로더 생성
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=True
        )
        
        self.logger.success("데이터 로더 설정 완료")
    
    def setup_model_and_optimizers(self) -> None:
        """모델 및 옵티마이저 설정"""
        
        # EfficientNetV2-L 모델 생성
        model_name = self.config.get('classification', {}).get('backbone', 'tf_efficientnetv2_l')
        self.model = create_pillsnap_classifier(
            num_classes=self.config['data']['num_classes'],
            model_name=model_name,  # backbone -> model_name으로 수정
            device=str(self.device)
        )
        
        # torch.compile 적용 (RTX 5080 최적화)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("torch.compile 적용 성공")
        except Exception as e:
            self.logger.warning(f"torch.compile 실패, 기본 모델 사용: {e}")
        
        # 옵티마이저
        train_config = self.config.get('train', {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 1e-5),
            fused=True  # RTX 5080 최적화
        )
        
        # 스케줄러
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * train_config.get('epochs', 50)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Mixed Precision
        self.scaler = GradScaler()
        
        # 손실 함수
        loss_config = self.config.get('loss', {})
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
        
        self.logger.success("모델 및 옵티마이저 설정 완료")
        model_name = self.config.get('classification', {}).get('backbone', 'unknown')
        self.logger.info(f"모델: {model_name}")
        self.logger.info(f"파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 학습"""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision 학습
            with autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 진행률 로그
            if (batch_idx + 1) % 50 == 0:
                accuracy = 100.0 * correct / total
                self.logger.info(
                    f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%, "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        
        # 에포크 통계
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time
        }
        
        self.logger.info(
            f"Train Epoch {epoch+1} 완료: Loss={avg_loss:.4f}, "
            f"Acc={accuracy:.2f}%, Time={epoch_time:.1f}s"
        )
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 검증"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 메트릭 계산
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_evaluator.compute_metrics(
            y_true=all_labels,
            y_pred=all_predictions
        )
        
        metrics['loss'] = avg_loss
        
        # 최고 성능 업데이트
        if metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
        
        if metrics['macro_f1'] > self.best_macro_f1:
            self.best_macro_f1 = metrics['macro_f1']
        
        self.logger.info(
            f"Val Epoch {epoch+1}: Loss={avg_loss:.4f}, "
            f"Acc={metrics['accuracy']:.2f}%, F1={metrics['macro_f1']:.4f}"
        )
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """체크포인트 저장"""
        
        exp_dir = Path(self.config.get('paths', {}).get('exp_dir', 'exp/stage3_classification'))
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Last checkpoint
        last_path = ckpt_dir / "stage3_classification_last.pt"
        torch.save(checkpoint, last_path)
        
        # Best checkpoint
        if is_best:
            best_path = ckpt_dir / "stage3_classification_best.pt"
            torch.save(checkpoint, best_path)
            self.logger.success(f"Best 모델 저장: {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """전체 학습 실행"""
        
        self.logger.info("=" * 60)
        self.logger.info("Stage 3 Classification 학습 시작")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. 데이터로더 설정
        self.setup_data_loaders()
        
        # 2. 모델 설정
        self.setup_model_and_optimizers()
        
        # 3. 학습 루프
        target_accuracy = self.stage_config['target_metrics']['classification_accuracy']
        epochs = self.config['train']['epochs']
        
        for epoch in range(epochs):
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate_epoch(epoch)
            
            # 체크포인트 저장
            is_best = val_metrics['accuracy'] >= self.best_accuracy
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 이력 저장
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1']
            }
            self.training_history.append(epoch_history)
            
            # 목표 달성 확인
            if val_metrics['accuracy'] >= target_accuracy:
                self.logger.success(
                    f"🎯 목표 달성! Accuracy {val_metrics['accuracy']:.1%} >= {target_accuracy:.1%}"
                )
                break
            
            # 메모리 모니터링
            gpu_memory = self.memory_monitor.get_memory_info()
            if gpu_memory['used_gb'] > 14:
                self.logger.warning(f"⚠️ GPU 메모리 사용량 높음: {gpu_memory['used_gb']:.1f}GB")
        
        # 4. 학습 완료
        total_time = time.time() - start_time
        
        final_results = {
            'stage': 3,
            'focus': 'classification_only',
            'total_time_hours': total_time / 3600,
            'best_accuracy': self.best_accuracy,
            'best_macro_f1': self.best_macro_f1,
            'target_achieved': self.best_accuracy >= target_accuracy,
            'epochs_completed': len(self.training_history),
            'training_history': self.training_history
        }
        
        self.logger.info("=" * 60)
        self.logger.info("Stage 3 Classification 학습 완료")
        self.logger.info(f"최고 정확도: {self.best_accuracy:.2f}%")
        self.logger.info(f"최고 Macro F1: {self.best_macro_f1:.4f}")
        self.logger.info(f"목표 달성: {'✅' if final_results['target_achieved'] else '❌'}")
        self.logger.info(f"총 학습 시간: {total_time/3600:.1f}시간")
        self.logger.info("=" * 60)
        
        return final_results


def main():
    """CLI 엔트리포인트"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3 Classification 전용 학습")
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--train-manifest", type=str, default="artifacts/stage3/manifest_train.csv", help="Train manifest 경로")
    parser.add_argument("--val-manifest", type=str, default="artifacts/stage3/manifest_val.csv", help="Val manifest 경로")
    parser.add_argument("--device", type=str, default="cuda", help="디바이스")
    
    args = parser.parse_args()
    
    # 학습 실행
    trainer = Stage3ClassificationTrainer(
        config_path=args.config,
        manifest_train=args.train_manifest,
        manifest_val=args.val_manifest,
        device=args.device
    )
    
    results = trainer.train()
    
    print("\n📊 최종 결과:")
    print(f"  - 최고 정확도: {results['best_accuracy']:.2f}%")
    print(f"  - 최고 Macro F1: {results['best_macro_f1']:.4f}")
    print(f"  - 목표 달성: {'✅' if results['target_achieved'] else '❌'}")
    print(f"  - 학습 시간: {results['total_time_hours']:.1f}시간")


if __name__ == "__main__":
    main()