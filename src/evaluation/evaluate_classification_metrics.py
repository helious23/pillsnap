"""
Classification Metrics Evaluator
분류 성능 메트릭 평가 시스템

Stage 1 목표:
- 분류 정확도: 40% (50개 클래스)
- Top-5 정확도, F1-Score, 혼동 행렬
- 클래스별 성능 분석
- ROC 곡선 및 PR 곡선
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.core import PillSnapLogger


@dataclass
class ClassificationMetrics:
    """분류 성능 메트릭"""
    
    # 기본 메트릭
    top1_accuracy: float
    top5_accuracy: float
    
    # 클래스별 메트릭
    precision_macro: float
    recall_macro: float
    f1_macro: float
    
    # 가중 평균 메트릭
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    
    # 추가 메트릭
    auc_macro: Optional[float] = None
    auc_weighted: Optional[float] = None
    
    # 메타데이터
    num_classes: int = 50
    num_samples: int = 0
    evaluation_time_seconds: float = 0.0


class ClassificationMetricsEvaluator:
    """분류 성능 메트릭 평가기"""
    
    def __init__(self, num_classes: int = 50, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.logger = PillSnapLogger(__name__)
        
        self.logger.info(f"분류 메트릭 평가기 초기화: {num_classes}개 클래스")
    
    def evaluate_predictions(
        self, 
        y_true: torch.Tensor, 
        y_pred_logits: torch.Tensor,
        y_pred_probs: Optional[torch.Tensor] = None
    ) -> ClassificationMetrics:
        """예측 결과 평가"""
        
        start_time = time.time()
        
        try:
            # 입력 검증
            if y_true.shape[0] != y_pred_logits.shape[0]:
                raise ValueError(f"샘플 수 불일치: {y_true.shape[0]} vs {y_pred_logits.shape[0]}")
            
            # CPU로 이동
            y_true_np = y_true.cpu().numpy()
            y_pred_logits_np = y_pred_logits.cpu().numpy()
            
            # 확률 계산
            if y_pred_probs is None:
                y_pred_probs = F.softmax(y_pred_logits, dim=1)
            y_pred_probs_np = y_pred_probs.cpu().numpy()
            
            # Top-1 예측
            y_pred_top1 = np.argmax(y_pred_logits_np, axis=1)
            
            # Top-5 예측
            top5_indices = np.argsort(y_pred_logits_np, axis=1)[:, -5:]
            
            # 기본 메트릭 계산
            top1_accuracy = accuracy_score(y_true_np, y_pred_top1)
            top5_accuracy = self._calculate_top_k_accuracy(y_true_np, top5_indices, k=5)
            
            # 정밀도, 재현율, F1 점수
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_np, y_pred_top1, average=None, zero_division=0
            )
            
            precision_macro = np.mean(precision)
            recall_macro = np.mean(recall)
            f1_macro = np.mean(f1)
            
            # 가중 평균
            precision_weighted = precision_recall_fscore_support(
                y_true_np, y_pred_top1, average='weighted', zero_division=0
            )[0]
            recall_weighted = precision_recall_fscore_support(
                y_true_np, y_pred_top1, average='weighted', zero_division=0
            )[1]
            f1_weighted = precision_recall_fscore_support(
                y_true_np, y_pred_top1, average='weighted', zero_division=0
            )[2]
            
            # AUC 계산 (다중 클래스)
            try:
                auc_macro = roc_auc_score(y_true_np, y_pred_probs_np, 
                                        multi_class='ovr', average='macro')
                auc_weighted = roc_auc_score(y_true_np, y_pred_probs_np, 
                                           multi_class='ovr', average='weighted')
            except Exception as e:
                self.logger.warning(f"AUC 계산 실패: {e}")
                auc_macro = None
                auc_weighted = None
            
            # 메트릭 객체 생성
            metrics = ClassificationMetrics(
                top1_accuracy=top1_accuracy,
                top5_accuracy=top5_accuracy,
                precision_macro=precision_macro,
                recall_macro=recall_macro,
                f1_macro=f1_macro,
                precision_weighted=precision_weighted,
                recall_weighted=recall_weighted,
                f1_weighted=f1_weighted,
                auc_macro=auc_macro,
                auc_weighted=auc_weighted,
                num_classes=self.num_classes,
                num_samples=len(y_true_np),
                evaluation_time_seconds=time.time() - start_time
            )
            
            # 결과 로깅
            self.logger.info(f"분류 평가 완료 ({len(y_true_np)}개 샘플)")
            self.logger.metric("top1_accuracy", top1_accuracy, "%")
            self.logger.metric("top5_accuracy", top5_accuracy, "%")
            self.logger.metric("f1_macro", f1_macro)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"분류 평가 실패: {e}")
            raise
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, top_k_indices: np.ndarray, k: int) -> float:
        """Top-K 정확도 계산"""
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_indices[i]:
                correct += 1
        return correct / len(y_true)
    
    def generate_confusion_matrix(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor,
        save_plot: bool = True
    ) -> np.ndarray:
        """혼동 행렬 생성"""
        
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            # 혼동 행렬 계산
            cm = confusion_matrix(y_true_np, y_pred_np)
            
            if save_plot:
                self._plot_confusion_matrix(cm)
            
            return cm
            
        except Exception as e:
            self.logger.error(f"혼동 행렬 생성 실패: {e}")
            raise
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> str:
        """혼동 행렬 시각화"""
        try:
            plt.figure(figsize=(12, 10))
            
            # 정규화된 혼동 행렬
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 히트맵 생성
            sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                       xticklabels=False, yticklabels=False)
            
            plt.title(f'Confusion Matrix (Normalized)\n{self.num_classes} Classes')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # 저장
            plot_dir = Path("artifacts/reports/validation_results")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_file = plot_dir / f"confusion_matrix_{timestamp}.png"
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"혼동 행렬 저장: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            self.logger.warning(f"혼동 행렬 시각화 실패: {e}")
            return ""
    
    def generate_classification_report(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ) -> Dict[str, Any]:
        """상세 분류 리포트 생성"""
        
        try:
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            # 클래스별 리포트
            report_dict = classification_report(
                y_true_np, y_pred_np, 
                target_names=self.class_names[:self.num_classes],
                output_dict=True,
                zero_division=0
            )
            
            # 클래스별 성능 분석
            class_performance = []
            for i, class_name in enumerate(self.class_names[:self.num_classes]):
                if str(i) in report_dict or class_name in report_dict:
                    class_stats = report_dict.get(str(i), report_dict.get(class_name, {}))
                    class_performance.append({
                        'class_id': i,
                        'class_name': class_name,
                        'precision': class_stats.get('precision', 0.0),
                        'recall': class_stats.get('recall', 0.0),
                        'f1_score': class_stats.get('f1-score', 0.0),
                        'support': class_stats.get('support', 0)
                    })
            
            # 최고/최저 성능 클래스
            if class_performance:
                sorted_by_f1 = sorted(class_performance, key=lambda x: x['f1_score'], reverse=True)
                best_classes = sorted_by_f1[:5]
                worst_classes = sorted_by_f1[-5:]
            else:
                best_classes = []
                worst_classes = []
            
            detailed_report = {
                'classification_report': report_dict,
                'class_performance': class_performance,
                'best_performing_classes': best_classes,
                'worst_performing_classes': worst_classes,
                'overall_stats': {
                    'macro_avg': report_dict.get('macro avg', {}),
                    'weighted_avg': report_dict.get('weighted avg', {}),
                    'accuracy': report_dict.get('accuracy', 0.0)
                }
            }
            
            return detailed_report
            
        except Exception as e:
            self.logger.error(f"분류 리포트 생성 실패: {e}")
            return {}
    
    def evaluate_stage1_target_achievement(self, metrics: ClassificationMetrics) -> Dict[str, bool]:
        """Stage 1 목표 달성 여부 평가"""
        
        # Stage 1 목표 기준
        STAGE1_TARGETS = {
            'classification_accuracy': 0.40,  # 40%
            'f1_macro': 0.35,                 # 35% (보조 지표)
            'top5_accuracy': 0.70             # 70% (관대한 기준)
        }
        
        try:
            achievement = {}
            
            # 분류 정확도 목표
            accuracy_achieved = metrics.top1_accuracy >= STAGE1_TARGETS['classification_accuracy']
            achievement['classification_accuracy_target_met'] = accuracy_achieved
            
            # F1 점수 목표
            f1_achieved = metrics.f1_macro >= STAGE1_TARGETS['f1_macro']
            achievement['f1_macro_target_met'] = f1_achieved
            
            # Top-5 정확도 목표
            top5_achieved = metrics.top5_accuracy >= STAGE1_TARGETS['top5_accuracy']
            achievement['top5_accuracy_target_met'] = top5_achieved
            
            # 전체 목표 달성
            all_achieved = accuracy_achieved and f1_achieved
            achievement['stage1_classification_completed'] = all_achieved
            
            # 결과 로깅
            self.logger.info("🎯 Stage 1 분류 목표 달성 평가:")
            self.logger.info(f"  분류 정확도: {metrics.top1_accuracy:.1%} "
                           f"(목표: {STAGE1_TARGETS['classification_accuracy']:.1%}) "
                           f"{'✅' if accuracy_achieved else '❌'}")
            self.logger.info(f"  F1 매크로: {metrics.f1_macro:.1%} "
                           f"(목표: {STAGE1_TARGETS['f1_macro']:.1%}) "
                           f"{'✅' if f1_achieved else '❌'}")
            self.logger.info(f"  Top-5 정확도: {metrics.top5_accuracy:.1%} "
                           f"(목표: {STAGE1_TARGETS['top5_accuracy']:.1%}) "
                           f"{'✅' if top5_achieved else '❌'}")
            
            if all_achieved:
                self.logger.success("🎉 Stage 1 분류 목표 달성!")
            else:
                self.logger.warning("⚠️ Stage 1 분류 목표 미달성 - 추가 학습 필요")
            
            return achievement
            
        except Exception as e:
            self.logger.error(f"목표 달성 평가 실패: {e}")
            return {}
    
    def save_evaluation_report(
        self, 
        metrics: ClassificationMetrics, 
        detailed_report: Optional[Dict] = None
    ) -> str:
        """평가 리포트 저장"""
        
        try:
            report_dir = Path("artifacts/reports/validation_results")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"classification_evaluation_{timestamp}.json"
            
            # 리포트 데이터 구성
            report_data = {
                'timestamp': timestamp,
                'evaluation_type': 'classification_metrics',
                'stage': 1,
                'metrics': {
                    'top1_accuracy': metrics.top1_accuracy,
                    'top5_accuracy': metrics.top5_accuracy,
                    'precision_macro': metrics.precision_macro,
                    'recall_macro': metrics.recall_macro,
                    'f1_macro': metrics.f1_macro,
                    'precision_weighted': metrics.precision_weighted,
                    'recall_weighted': metrics.recall_weighted,
                    'f1_weighted': metrics.f1_weighted,
                    'auc_macro': metrics.auc_macro,
                    'auc_weighted': metrics.auc_weighted
                },
                'metadata': {
                    'num_classes': metrics.num_classes,
                    'num_samples': metrics.num_samples,
                    'evaluation_time_seconds': metrics.evaluation_time_seconds
                },
                'target_achievement': self.evaluate_stage1_target_achievement(metrics)
            }
            
            # 상세 리포트 추가
            if detailed_report:
                report_data['detailed_analysis'] = detailed_report
            
            # 저장
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"분류 평가 리포트 저장: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
            return ""


def main():
    """분류 메트릭 평가 테스트"""
    print("📊 Classification Metrics Evaluator Test")
    print("=" * 60)
    
    # 테스트 데이터 생성 (시뮬레이션)
    num_classes = 50
    num_samples = 1000
    
    evaluator = ClassificationMetricsEvaluator(num_classes=num_classes)
    
    # 더미 데이터 생성
    torch.manual_seed(42)
    y_true = torch.randint(0, num_classes, (num_samples,))
    y_pred_logits = torch.randn(num_samples, num_classes)
    
    # 일부 예측을 의도적으로 맞춤 (40% 정확도 시뮬레이션)
    correct_mask = torch.rand(num_samples) < 0.4
    y_pred_logits[correct_mask, y_true[correct_mask]] += 3.0
    
    print(f"테스트 데이터: {num_samples}개 샘플, {num_classes}개 클래스")
    
    # 평가 수행
    metrics = evaluator.evaluate_predictions(y_true, y_pred_logits)
    
    print(f"\n📈 평가 결과:")
    print(f"  Top-1 정확도: {metrics.top1_accuracy:.1%}")
    print(f"  Top-5 정확도: {metrics.top5_accuracy:.1%}")
    print(f"  F1 매크로: {metrics.f1_macro:.3f}")
    print(f"  F1 가중: {metrics.f1_weighted:.3f}")
    
    # 목표 달성 평가
    achievement = evaluator.evaluate_stage1_target_achievement(metrics)
    stage1_completed = achievement.get('stage1_classification_completed', False)
    
    print(f"\n🎯 Stage 1 목표 달성: {'✅ 성공' if stage1_completed else '❌ 미달성'}")
    
    # 상세 리포트 생성
    y_pred_classes = torch.argmax(y_pred_logits, dim=1)
    detailed_report = evaluator.generate_classification_report(y_true, y_pred_classes)
    
    # 혼동 행렬 생성
    cm = evaluator.generate_confusion_matrix(y_true, y_pred_classes)
    print(f"혼동 행렬 크기: {cm.shape}")
    
    # 리포트 저장
    report_file = evaluator.save_evaluation_report(metrics, detailed_report)
    if report_file:
        print(f"\n💾 평가 리포트 저장: {report_file}")
    
    print("\n✅ 분류 메트릭 평가 테스트 완료")


if __name__ == "__main__":
    main()