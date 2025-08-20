"""
Classification Stage Training Module
분류 모델 전용 학습 모듈

EfficientNetV2-S 분류기 Stage별 학습:
- Progressive Validation 지원 (Stage 1~4)
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- 목표 정확도 달성 자동 체크
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from src.models.classifier_efficientnetv2 import PillSnapClassifier, create_pillsnap_classifier
from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor
from src.evaluation.evaluate_classification_metrics import ClassificationMetricsEvaluator
from src.utils.core import PillSnapLogger


class ClassificationStageTrainer:
    """분류 모델 전용 학습기"""
    
    def __init__(
        self, 
        num_classes: int,
        target_accuracy: float = 0.40,
        device: str = "cuda"
    ):
        self.num_classes = num_classes
        self.target_accuracy = target_accuracy
        self.device = torch.device(device)
        self.logger = PillSnapLogger(__name__)
        
        # 모니터링 시스템
        self.memory_monitor = GPUMemoryMonitor()
        self.metrics_evaluator = ClassificationMetricsEvaluator(num_classes)
        
        # 학습 상태
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_accuracy = 0.0
        self.training_history = []
        
        self.logger.info(f"ClassificationStageTrainer 초기화")
        self.logger.info(f"클래스 수: {num_classes}, 목표 정확도: {target_accuracy:.1%}")
    
    def setup_model_and_optimizers(
        self, 
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        mixed_precision: bool = True
    ) -> None:
        """모델 및 옵티마이저 설정"""
        
        try:
            # 분류기 생성
            self.model = create_pillsnap_classifier(
                num_classes=self.num_classes,
                device=str(self.device)
            )
            
            # 옵티마이저 설정
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # 학습률 스케줄러
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50  # 최대 50 에포크 가정
            )
            
            # 손실 함수 설정
            self.criterion = nn.CrossEntropyLoss()
            
            # Mixed Precision 설정
            if mixed_precision and torch.cuda.is_available():
                self.scaler = GradScaler()
                self.logger.info("Mixed Precision 활성화")
            
            self.logger.success("모델 및 옵티마이저 설정 완료")
            
        except Exception as e:
            self.logger.error(f"모델 설정 실패: {e}")
            raise
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """한 에포크 학습"""
        
        if self.model is None:
            raise RuntimeError("모델이 설정되지 않음. setup_model_and_optimizers() 먼저 호출")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision 학습
            if self.scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 진행 상황 로깅 (더 자주 출력)
            if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
                current_acc = correct / total if total > 0 else 0
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.2%}")
            elif batch_idx % 100 == 0:
                self.logger.info(f"Epoch {epoch} Batch {batch_idx}: Loss {loss.item():.4f}")
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'correct': correct,
            'total': total
        }
    
    def validate_epoch(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """검증 에포크"""
        
        if self.model is None:
            raise RuntimeError("모델이 설정되지 않음")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        all_predictions = []
        all_labels = []
        
        print(f"  📊 검증 중... ({len(val_loader)} 배치)")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 검증 진행상황 출력
                if batch_idx % 3 == 0 or batch_idx == len(val_loader) - 1:
                    current_acc = correct / total if total > 0 else 0
                    print(f"    Val Batch {batch_idx+1}/{len(val_loader)}: Acc={current_acc:.2%}")
                
                all_predictions.extend(predicted.cpu())
                all_labels.extend(labels.cpu())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = correct / total
        
        # 상세 메트릭 계산
        try:
            y_true = torch.tensor(all_labels)
            y_pred_logits = torch.zeros(len(all_predictions), self.num_classes)
            # 간단한 원-핫 인코딩으로 로짓 시뮬레이션
            for i, pred in enumerate(all_predictions):
                y_pred_logits[i, pred] = 1.0
            
            detailed_metrics = self.metrics_evaluator.evaluate_predictions(y_true, y_pred_logits)
            
        except Exception as e:
            self.logger.warning(f"상세 메트릭 계산 실패: {e}")
            detailed_metrics = None
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'correct': correct,
            'total': total,
            'detailed_metrics': detailed_metrics
        }
    
    def train_stage(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 10,
        early_stopping_patience: int = 5
    ) -> Dict[str, Any]:
        """전체 Stage 학습"""
        
        self.logger.step("분류 Stage 학습", f"{max_epochs} 에포크 목표 정확도 {self.target_accuracy:.1%}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(1, max_epochs + 1):
            # GPU 메모리 모니터링
            memory_stats = self.memory_monitor.get_current_usage()
            self.logger.info(f"Epoch {epoch}/{max_epochs} - GPU: {memory_stats['used_gb']:.1f}GB")
            
            # 학습
            train_results = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_results = self.validate_epoch(val_loader)
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            # 최고 성능 업데이트
            if val_results['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_results['accuracy']
                patience_counter = 0
                self.logger.metric("best_accuracy", self.best_accuracy, "%")
                
                # 모델 저장
                self._save_best_model()
            else:
                patience_counter += 1
            
            # 학습 히스토리 기록
            epoch_history = {
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'train_accuracy': train_results['accuracy'],
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
            }
            self.training_history.append(epoch_history)
            
            self.logger.info(f"Epoch {epoch} - Train: {train_results['accuracy']:.1%}, "
                           f"Val: {val_results['accuracy']:.1%}")
            
            # 목표 달성 체크
            if val_results['accuracy'] >= self.target_accuracy:
                self.logger.success(f"🎉 목표 정확도 달성! {val_results['accuracy']:.1%} >= {self.target_accuracy:.1%}")
                break
            
            # Early Stopping
            if patience_counter >= early_stopping_patience:
                self.logger.warning(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        
        # 최종 결과
        final_results = {
            'best_accuracy': self.best_accuracy,
            'target_achieved': self.best_accuracy >= self.target_accuracy,
            'epochs_completed': len(self.training_history),
            'total_time_minutes': total_time / 60,
            'training_history': self.training_history,
            'final_val_results': val_results
        }
        
        self.logger.success(f"분류 학습 완료 - 최고 정확도: {self.best_accuracy:.1%}")
        return final_results
    
    def _save_best_model(self) -> None:
        """최고 성능 모델 저장"""
        try:
            save_dir = Path("artifacts/models/classification")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = save_dir / f"best_classifier_{self.num_classes}classes.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'best_accuracy': self.best_accuracy,
                'num_classes': self.num_classes
            }, model_path)
            
            self.logger.info(f"최고 성능 모델 저장: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
    


def main():
    """CLI를 통한 분류 Stage 학습 실행"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="PillSnap Classification Stage Training")
    parser.add_argument("--stage", type=int, default=1, help="Progressive Validation Stage (1-4)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without actual training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    
    args = parser.parse_args()
    
    print(f"🚀 Classification Stage {args.stage} Training")
    print("=" * 60)
    print(f"📊 Parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"🖥️  Device: {args.device}")
    
    # Stage별 클래스 수 설정
    stage_classes = {1: 50, 2: 250, 3: 1000, 4: 4523}
    num_classes = stage_classes.get(args.stage, 50)
    
    if args.dry_run:
        print("🔍 Dry Run Mode - 설정 검증만 수행")
        trainer = ClassificationStageTrainer(
            num_classes=num_classes, 
            target_accuracy=0.40,
            device=args.device
        )
        trainer.setup_model_and_optimizers(learning_rate=args.learning_rate)
        print("✅ 모든 컴포넌트 초기화 성공")
        print("실제 학습을 위해서는 --dry-run 없이 실행하세요.")
        return
    
    # Stage 1 샘플 데이터 확인
    stage1_sample_path = "artifacts/stage1/sampling/stage1_sample.json"
    if not os.path.exists(stage1_sample_path):
        print(f"❌ Stage 1 샘플 데이터가 없습니다: {stage1_sample_path}")
        print("먼저 Progressive Validation 샘플링을 실행하세요:")
        print("./scripts/python_safe.sh -m src.data.progressive_validation_sampler")
        return
    
    print("✅ Stage 1 샘플 데이터 확인됨")
    
    # 실제 학습 시작
    print("🚀 실제 학습 파이프라인 시작")
    
    # 트레이너 초기화
    trainer = ClassificationStageTrainer(
        num_classes=num_classes,
        target_accuracy=0.40,
        device=args.device
    )
    trainer.setup_model_and_optimizers(learning_rate=args.learning_rate)
    
    
    # 데이터로더 생성
    print("📊 데이터로더 생성 중...")
    from src.data.dataloader_single_pill_training import SinglePillTrainingDataLoader
    
    dataloader_manager = SinglePillTrainingDataLoader(
        stage=args.stage,
        batch_size=args.batch_size
        # num_workers는 시스템 최적화를 통해 자동 설정
    )
    
    train_loader, val_loader, metadata = dataloader_manager.get_stage_dataloaders()
    
    print(f"✅ 데이터로더 준비 완료")
    print(f"   클래스 수: {metadata['num_classes']}")
    print(f"   학습 데이터: {metadata['train_size']}개")
    print(f"   검증 데이터: {metadata['val_size']}개")
    
    # 실제 학습 실행
    print(f"🏋️ 학습 시작 - {args.epochs} epochs")
    
    try:
        # 시스템 최적화된 DataLoader 사용
        print("🔧 시스템 최적화된 DataLoader 재생성")
        dataloader_manager_optimized = SinglePillTrainingDataLoader(
            stage=args.stage,
            batch_size=args.batch_size
            # num_workers는 자동 최적화됨
        )
        
        train_loader, val_loader, metadata = dataloader_manager_optimized.get_stage_dataloaders()
        print(f"✅ 최적화된 데이터로더 준비 완료")
        
        # 원래 train_stage() 호출
        print("🚀 원래 train_stage() 메서드 호출")
        results = trainer.train_stage(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            early_stopping_patience=5
        )
        
        print(f"\n✅ 학습 완료!")
        print(f"   최고 정확도: {results['best_accuracy']:.1%}")
        print(f"   목표 달성: {results['target_achieved']}")
        print(f"   완료 에포크: {results['epochs_completed']}")
        print(f"   소요 시간: {results['total_time_minutes']:.1f}분")
            
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()