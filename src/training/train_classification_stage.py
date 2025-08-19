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
                with autocast():
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
            
            # 진행 상황 로깅
            if batch_idx % 100 == 0:
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
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
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
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'best_accuracy': self.best_accuracy,
                'num_classes': self.num_classes
            }, model_path)
            
            self.logger.info(f"최고 성능 모델 저장: {model_path}")
            
        except Exception as e:
            self.logger.warning(f"모델 저장 실패: {e}")


def main():
    """분류 Stage 학습 테스트"""
    print("🔧 Classification Stage Trainer Test")
    print("=" * 50)
    
    # 테스트 설정
    trainer = ClassificationStageTrainer(num_classes=50, target_accuracy=0.40)
    trainer.setup_model_and_optimizers()
    
    # 더미 데이터로 테스트 (실제로는 DataLoader 전달)
    print("✅ Classification Stage Trainer 초기화 완료")
    print("실제 학습을 위해서는 DataLoader가 필요합니다.")


if __name__ == "__main__":
    main()