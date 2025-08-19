"""
Detection Metrics Evaluator
검출 성능 메트릭 평가 시스템

YOLOv11m 검출 성능 평가:
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1-Score
- 클래스별 성능 분석
- Stage별 목표 달성 검증
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.core import PillSnapLogger


@dataclass
class DetectionMetrics:
    """검출 성능 메트릭"""
    
    # 주요 mAP 메트릭
    map_50: float              # mAP@0.5
    map_50_95: float           # mAP@0.5:0.95
    
    # 클래스별 메트릭
    precision: float           # 평균 정밀도
    recall: float              # 평균 재현율
    f1_score: float            # F1 점수
    
    # 세부 메트릭
    ap_per_class: List[float]  # 클래스별 AP
    precision_per_class: List[float]
    recall_per_class: List[float]
    
    # 메타데이터
    num_classes: int = 1
    num_images: int = 0
    evaluation_time_seconds: float = 0.0


class DetectionMetricsEvaluator:
    """검출 성능 메트릭 평가기"""
    
    def __init__(self, num_classes: int = 1, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.logger = PillSnapLogger(__name__)
        
        self.logger.info(f"검출 메트릭 평가기 초기화: {num_classes}개 클래스")
    
    def evaluate_yolo_results(
        self, 
        results_dict: Dict[str, Any]
    ) -> DetectionMetrics:
        """YOLO 결과 평가"""
        
        start_time = time.time()
        
        try:
            # YOLO 결과에서 메트릭 추출
            metrics_data = results_dict.get('metrics', {})
            
            # 주요 메트릭
            map_50 = metrics_data.get('mAP_0.5', 0.0)
            map_50_95 = metrics_data.get('mAP_0.5:0.95', 0.0)
            precision = metrics_data.get('precision', 0.0)
            recall = metrics_data.get('recall', 0.0)
            
            # F1 점수 계산
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
            
            # 클래스별 메트릭 (단일 클래스의 경우)
            ap_per_class = [map_50] * self.num_classes
            precision_per_class = [precision] * self.num_classes
            recall_per_class = [recall] * self.num_classes
            
            # 메트릭 객체 생성
            detection_metrics = DetectionMetrics(
                map_50=map_50,
                map_50_95=map_50_95,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                ap_per_class=ap_per_class,
                precision_per_class=precision_per_class,
                recall_per_class=recall_per_class,
                num_classes=self.num_classes,
                num_images=results_dict.get('num_images', 0),
                evaluation_time_seconds=time.time() - start_time
            )
            
            # 결과 로깅
            self.logger.info(f"검출 평가 완료 ({detection_metrics.num_images}개 이미지)")
            self.logger.metric("mAP_0.5", map_50)
            self.logger.metric("mAP_0.5:0.95", map_50_95)
            self.logger.metric("precision", precision)
            self.logger.metric("recall", recall)
            self.logger.metric("f1_score", f1_score)
            
            return detection_metrics
            
        except Exception as e:
            self.logger.error(f"검출 평가 실패: {e}")
            raise
    
    def evaluate_stage_target_achievement(
        self, 
        metrics: DetectionMetrics,
        stage: int = 1
    ) -> Dict[str, bool]:
        """Stage별 목표 달성 여부 평가"""
        
        # Stage별 목표 기준
        STAGE_TARGETS = {
            1: {'map_50': 0.30, 'precision': 0.35, 'recall': 0.30},
            2: {'map_50': 0.50, 'precision': 0.55, 'recall': 0.50},
            3: {'map_50': 0.70, 'precision': 0.75, 'recall': 0.70},
            4: {'map_50': 0.85, 'precision': 0.90, 'recall': 0.85}
        }
        
        targets = STAGE_TARGETS.get(stage, STAGE_TARGETS[1])
        
        try:
            achievement = {}
            
            # mAP@0.5 목표
            map_achieved = metrics.map_50 >= targets['map_50']
            achievement['map_50_target_met'] = map_achieved
            
            # 정밀도 목표
            precision_achieved = metrics.precision >= targets['precision']
            achievement['precision_target_met'] = precision_achieved
            
            # 재현율 목표
            recall_achieved = metrics.recall >= targets['recall']
            achievement['recall_target_met'] = recall_achieved
            
            # 전체 목표 달성
            all_achieved = map_achieved and precision_achieved and recall_achieved
            achievement[f'stage{stage}_detection_completed'] = all_achieved
            
            # 결과 로깅
            self.logger.info(f"🎯 Stage {stage} 검출 목표 달성 평가:")
            self.logger.info(f"  mAP@0.5: {metrics.map_50:.3f} "
                           f"(목표: {targets['map_50']:.3f}) "
                           f"{'✅' if map_achieved else '❌'}")
            self.logger.info(f"  정밀도: {metrics.precision:.3f} "
                           f"(목표: {targets['precision']:.3f}) "
                           f"{'✅' if precision_achieved else '❌'}")
            self.logger.info(f"  재현율: {metrics.recall:.3f} "
                           f"(목표: {targets['recall']:.3f}) "
                           f"{'✅' if recall_achieved else '❌'}")
            
            if all_achieved:
                self.logger.success(f"🎉 Stage {stage} 검출 목표 달성!")
            else:
                self.logger.warning(f"⚠️ Stage {stage} 검출 목표 미달성 - 추가 학습 필요")
            
            return achievement
            
        except Exception as e:
            self.logger.error(f"목표 달성 평가 실패: {e}")
            return {}
    
    def generate_detection_report(
        self, 
        metrics: DetectionMetrics,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """상세 검출 리포트 생성"""
        
        try:
            # 성능 분석
            performance_analysis = {
                'overall_performance': {
                    'map_50': metrics.map_50,
                    'map_50_95': metrics.map_50_95,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score
                },
                'class_performance': []
            }
            
            # 클래스별 성능 (단일 클래스의 경우 단순화)
            for i in range(self.num_classes):
                class_perf = {
                    'class_id': i,
                    'class_name': self.class_names[i],
                    'ap': metrics.ap_per_class[i] if i < len(metrics.ap_per_class) else 0.0,
                    'precision': metrics.precision_per_class[i] if i < len(metrics.precision_per_class) else 0.0,
                    'recall': metrics.recall_per_class[i] if i < len(metrics.recall_per_class) else 0.0
                }
                performance_analysis['class_performance'].append(class_perf)
            
            # 성능 등급 평가
            if metrics.map_50 >= 0.80:
                performance_grade = "우수 (Excellent)"
            elif metrics.map_50 >= 0.60:
                performance_grade = "양호 (Good)"
            elif metrics.map_50 >= 0.40:
                performance_grade = "보통 (Average)"
            elif metrics.map_50 >= 0.20:
                performance_grade = "미흡 (Below Average)"
            else:
                performance_grade = "부족 (Poor)"
            
            # 개선 권장사항
            recommendations = []
            if metrics.precision < 0.70:
                recommendations.append("False Positive 감소를 위한 NMS 임계값 조정")
            if metrics.recall < 0.70:
                recommendations.append("False Negative 감소를 위한 Confidence 임계값 하향 조정")
            if metrics.map_50 < 0.50:
                recommendations.append("데이터 증강 및 추가 학습 에포크 권장")
            
            detailed_report = {
                'performance_analysis': performance_analysis,
                'performance_grade': performance_grade,
                'recommendations': recommendations,
                'metadata': {
                    'num_classes': metrics.num_classes,
                    'num_images': metrics.num_images,
                    'evaluation_time_seconds': metrics.evaluation_time_seconds
                }
            }
            
            # 시각화 생성
            if save_plots:
                plot_files = self._create_detection_plots(metrics)
                detailed_report['plot_files'] = plot_files
            
            return detailed_report
            
        except Exception as e:
            self.logger.error(f"검출 리포트 생성 실패: {e}")
            return {}
    
    def _create_detection_plots(self, metrics: DetectionMetrics) -> List[str]:
        """검출 성능 시각화"""
        plot_files = []
        
        try:
            plot_dir = Path("artifacts/reports/validation_results")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 1. 메트릭 막대 그래프
            plt.figure(figsize=(10, 6))
            
            metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
            metrics_values = [
                metrics.map_50, metrics.map_50_95, 
                metrics.precision, metrics.recall, metrics.f1_score
            ]
            
            bars = plt.bar(metrics_names, metrics_values, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            
            # 값 표시
            for bar, value in zip(bars, metrics_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.title('Detection Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            
            plot_file = plot_dir / f"detection_metrics_bar_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files.append(str(plot_file))
            self.logger.info(f"검출 메트릭 그래프 저장: {plot_file}")
            
        except Exception as e:
            self.logger.warning(f"검출 시각화 실패: {e}")
        
        return plot_files
    
    def save_evaluation_report(
        self, 
        metrics: DetectionMetrics,
        detailed_report: Optional[Dict] = None,
        stage: int = 1
    ) -> str:
        """평가 리포트 저장"""
        
        try:
            report_dir = Path("artifacts/reports/validation_results")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"detection_evaluation_stage{stage}_{timestamp}.json"
            
            # 리포트 데이터 구성
            report_data = {
                'timestamp': timestamp,
                'evaluation_type': 'detection_metrics',
                'stage': stage,
                'metrics': {
                    'map_50': metrics.map_50,
                    'map_50_95': metrics.map_50_95,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'ap_per_class': metrics.ap_per_class,
                    'precision_per_class': metrics.precision_per_class,
                    'recall_per_class': metrics.recall_per_class
                },
                'metadata': {
                    'num_classes': metrics.num_classes,
                    'num_images': metrics.num_images,
                    'evaluation_time_seconds': metrics.evaluation_time_seconds
                },
                'target_achievement': self.evaluate_stage_target_achievement(metrics, stage)
            }
            
            # 상세 리포트 추가
            if detailed_report:
                report_data['detailed_analysis'] = detailed_report
            
            # 저장
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"검출 평가 리포트 저장: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
            return ""


def main():
    """검출 메트릭 평가 테스트"""
    print("📊 Detection Metrics Evaluator Test")
    print("=" * 60)
    
    # 테스트 데이터 생성 (시뮬레이션)
    evaluator = DetectionMetricsEvaluator(num_classes=1, class_names=["pill"])
    
    # 더미 YOLO 결과
    dummy_results = {
        'metrics': {
            'mAP_0.5': 0.32,
            'mAP_0.5:0.95': 0.22,
            'precision': 0.36,
            'recall': 0.30
        },
        'num_images': 1000
    }
    
    print(f"테스트 데이터: 1000개 이미지, 1개 클래스")
    
    # 평가 수행
    metrics = evaluator.evaluate_yolo_results(dummy_results)
    
    print(f"\n📈 평가 결과:")
    print(f"  mAP@0.5: {metrics.map_50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.map_50_95:.3f}")
    print(f"  정밀도: {metrics.precision:.3f}")
    print(f"  재현율: {metrics.recall:.3f}")
    print(f"  F1 점수: {metrics.f1_score:.3f}")
    
    # Stage 1 목표 달성 평가
    achievement = evaluator.evaluate_stage_target_achievement(metrics, stage=1)
    stage1_completed = achievement.get('stage1_detection_completed', False)
    
    print(f"\n🎯 Stage 1 목표 달성: {'✅ 성공' if stage1_completed else '❌ 미달성'}")
    
    # 상세 리포트 생성
    detailed_report = evaluator.generate_detection_report(metrics)
    
    # 리포트 저장
    report_file = evaluator.save_evaluation_report(metrics, detailed_report, stage=1)
    if report_file:
        print(f"\n💾 평가 리포트 저장: {report_file}")
    
    print("\n✅ 검출 메트릭 평가 테스트 완료")


if __name__ == "__main__":
    main()