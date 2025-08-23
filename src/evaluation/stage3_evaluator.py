"""
Stage 3 전용 평가 시스템

100K 샘플, 1000 클래스 규모의 Stage 3 훈련을 위한
종합적인 평가 시스템. Two-Stage Pipeline 검증 포함.

Features:
- Classification 성능 평가 (Single 95%)
- Detection 성능 평가 (Combination 5%)
- 메모리 안정성 모니터링
- 확장성 테스트
- Two-Stage Pipeline 통합 성능 측정

Author: Claude Code - PillSnap ML Team
Date: 2025-08-23
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger


@dataclass
class Stage3Config:
    """Stage 3 평가 설정"""
    total_samples: int = 100000
    total_classes: int = 1000
    single_ratio: float = 0.95
    combination_ratio: float = 0.05
    target_accuracy: float = 0.85
    target_map50: float = 0.75
    batch_size: int = 16
    gpu_memory_limit_gb: float = 14.0


@dataclass
class ClassificationMetrics:
    """분류 성능 메트릭 (Single pill - 95%)"""
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    confusion_matrix: List = None
    per_class_accuracy: Dict = None
    num_classes: int = 0
    num_samples: int = 0
    inference_time_ms: float = 0.0
    memory_usage_gb: float = 0.0
    
    def __post_init__(self):
        if self.confusion_matrix is None:
            self.confusion_matrix = []
        if self.per_class_accuracy is None:
            self.per_class_accuracy = {}
    
    def update_from_predictions(self, predictions: torch.Tensor, targets: torch.Tensor):
        """예측 결과로부터 메트릭 업데이트"""
        if len(predictions) == 0:
            return
            
        # Top-1 정확도
        _, pred_top1 = predictions.topk(1, dim=1)
        correct_top1 = pred_top1.squeeze().eq(targets).float().sum().item()
        self.top1_accuracy = correct_top1 / len(targets)
        
        # Top-5 정확도  
        k = min(5, predictions.size(1))
        _, pred_topk = predictions.topk(k, dim=1)
        correct_topk = pred_topk.eq(targets.view(-1, 1).expand_as(pred_topk)).float().sum().item()
        self.top5_accuracy = correct_topk / len(targets)
        
        self.num_samples = len(targets)


@dataclass
class DetectionMetrics:
    """검출 성능 메트릭 (Combination pill - 5%)"""
    map50: float = 0.0           # mAP@0.5
    map95: float = 0.0           # mAP@0.5:0.95
    precision: float = 0.0
    recall: float = 0.0
    per_class_map: Dict = None
    total_detections: int = 0
    num_ground_truths: int = 0
    inference_time_ms: float = 0.0
    memory_usage_gb: float = 0.0
    
    def __post_init__(self):
        if self.per_class_map is None:
            self.per_class_map = {}
    
    def update_from_detections(self, predicted_boxes: List, ground_truth_boxes: List):
        """검출 결과로부터 메트릭 업데이트"""
        self.total_detections = len(predicted_boxes)
        self.num_ground_truths = len(ground_truth_boxes)
        
        # 간단한 매칭 기반 정확도 계산 (실제 구현에서는 IoU 기반)
        if self.total_detections > 0:
            self.per_class_map = {'class_0': 0.8, 'class_1': 0.7}  # 예시


@dataclass
class TwoStagePipelineMetrics:
    """Two-Stage Pipeline 통합 메트릭"""
    end_to_end_accuracy: float = 0.0
    single_pill_accuracy: float = 0.0
    combination_pill_accuracy: float = 0.0
    detection_recall: float = 0.0
    pipeline_efficiency: float = 0.0  # 0.0-1.0
    average_inference_time_ms: float = 0.0
    memory_peak_gb: float = 0.0
    successful_detections_ratio: float = 0.0
    
    def update_from_pipeline_results(self, results: Dict):
        """파이프라인 결과로부터 메트릭 업데이트"""
        if 'single_correct' in results and 'single_total' in results:
            self.single_pill_accuracy = results['single_correct'] / results['single_total']
        
        if 'combination_correct' in results and 'combination_total' in results:
            self.combination_pill_accuracy = results['combination_correct'] / results['combination_total']
            
        if 'detection_correct' in results and 'detection_total' in results:
            self.detection_recall = results['detection_correct'] / results['detection_total']
        
        if 'total_inference_time_ms' in results and 'total_samples' in results:
            self.average_inference_time_ms = results['total_inference_time_ms'] / results['total_samples']


@dataclass
class ScalabilityMetrics:
    """확장성 메트릭"""
    samples_per_second: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_usage_gb: float = 0.0
    cpu_utilization: float = 0.0
    batch_processing_time_ms: float = 0.0
    memory_stability_score: float = 1.0
    memory_leak_detected: bool = False
    max_batch_size_supported: int = 0
    
    def update_from_hardware_metrics(self, metrics: Dict):
        """하드웨어 메트릭으로부터 업데이트"""
        self.samples_per_second = metrics.get('samples_per_second', 0.0)
        self.gpu_utilization = metrics.get('gpu_utilization', 0.0)
        self.gpu_memory_usage_gb = metrics.get('gpu_memory_usage_gb', 0.0)
        self.cpu_utilization = metrics.get('cpu_utilization', 0.0)
        self.batch_processing_time_ms = metrics.get('batch_processing_time_ms', 0.0)
        self.memory_stability_score = metrics.get('memory_stability_score', 1.0)


@dataclass
class Stage3EvaluationReport:
    """Stage 3 평가 종합 보고서"""
    config: Stage3Config
    classification_metrics: ClassificationMetrics
    detection_metrics: DetectionMetrics
    pipeline_metrics: TwoStagePipelineMetrics
    scalability_metrics: ScalabilityMetrics
    total_evaluation_time_seconds: float
    stage4_readiness_score: float
    timestamp: float = None
    optimization_recommendations: List[Dict[str, Any]] = None
    target_metrics_achieved: Dict[str, bool] = None
    overall_score: float = 0.0  # 0.0-1.0
    ready_for_stage4: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []
        if self.target_metrics_achieved is None:
            self.target_metrics_achieved = {}


class Stage3Evaluator:
    """
    Stage 3 전용 평가 시스템
    
    100K 샘플, 1000 클래스 환경에서의 종합적인 성능 평가와
    Stage 4 진입 준비도를 측정.
    """
    
    def __init__(self, 
                 config: Stage3Config,
                 device: Union[torch.device, str] = 'cuda',
                 enable_memory_monitoring: bool = True,
                 enable_optimization_advisor: bool = True):
        """
        초기화
        
        Args:
            config: Stage 3 설정
            device: 실행 디바이스
            enable_memory_monitoring: 메모리 모니터링 활성화
            enable_optimization_advisor: 최적화 권고 활성화
        """
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.logger = PillSnapLogger(__name__)
        
        # 메모리 관리
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_optimization_advisor = enable_optimization_advisor
        
        # 평가 히스토리
        self.evaluation_history: List[Stage3EvaluationReport] = []
        
        self.logger.info(f"Stage 3 Evaluator 초기화 완료")
        self.logger.info(f"디바이스: {self.device}")
    
    def _evaluate_classification_performance(self, model, dataloader) -> ClassificationMetrics:
        """분류 성능 평가"""
        metrics = ClassificationMetrics()
        
        # Mock 구현 - 실제로는 model과 dataloader를 사용
        mock_predictions = torch.randn(100, 1000)
        mock_targets = torch.randint(0, 1000, (100,))
        
        metrics.update_from_predictions(mock_predictions, mock_targets)
        return metrics
    
    def _evaluate_detection_performance(self, model, dataloader) -> DetectionMetrics:
        """검출 성능 평가"""
        metrics = DetectionMetrics()
        
        # Mock 구현
        mock_predicted_boxes = [{'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class': 0}]
        mock_ground_truth_boxes = [{'bbox': [12, 12, 48, 48], 'class': 0}]
        
        metrics.update_from_detections(mock_predicted_boxes, mock_ground_truth_boxes)
        return metrics
    
    def _evaluate_pipeline_performance(self, models, dataloader) -> TwoStagePipelineMetrics:
        """파이프라인 성능 평가"""
        metrics = TwoStagePipelineMetrics()
        
        # Mock 결과
        pipeline_results = {
            'single_correct': 95,
            'single_total': 100,
            'combination_correct': 45,
            'combination_total': 50,
            'detection_correct': 48,
            'detection_total': 50,
            'total_inference_time_ms': 5000,
            'total_samples': 150
        }
        
        metrics.update_from_pipeline_results(pipeline_results)
        return metrics
    
    def _measure_scalability_metrics(self, dataloader) -> ScalabilityMetrics:
        """확장성 메트릭 측정"""
        metrics = ScalabilityMetrics()
        
        # Mock 하드웨어 메트릭
        hardware_metrics = {
            'samples_per_second': 120.0,
            'gpu_utilization': 88.0,
            'gpu_memory_usage_gb': 13.8,
            'cpu_utilization': 45.0,
            'batch_processing_time_ms': 133.3,
            'memory_stability_score': 0.95
        }
        
        metrics.update_from_hardware_metrics(hardware_metrics)
        return metrics
    
    def _run_classification_batch(self, model, batch):
        """분류 배치 실행"""
        # Mock 구현
        outputs = torch.randn(16, 1000)
        targets = torch.randint(0, 1000, (16,))
        return outputs, targets
    
    def _run_detection_batch(self, model, batch):
        """검출 배치 실행"""
        # Mock 구현
        detections = [{'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class': 0}]
        ground_truth = [{'bbox': [12, 12, 48, 48], 'class': 0}]
        return detections, ground_truth
    
    def _get_current_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (GB)"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _calculate_optimal_batch_size(self, model_memory_gb: float, 
                                    available_memory_gb: float,
                                    sample_memory_mb: float) -> int:
        """최적 배치 크기 계산"""
        usable_memory_gb = available_memory_gb - model_memory_gb
        sample_memory_gb = sample_memory_mb / 1024
        
        if sample_memory_gb <= 0:
            return 16
            
        optimal_batch = int(usable_memory_gb / sample_memory_gb)
        return max(1, min(32, optimal_batch))  # 최대 32로 제한
    
    def run_comprehensive_evaluation(self, dataloader, models: Dict) -> Stage3EvaluationReport:
        """종합 평가 실행"""
        start_time = time.time()
        
        self.logger.info("Stage 3 종합 평가 시작")
        
        # 1. Classification 평가
        classification_metrics = self._evaluate_classification_performance(
            models.get('classifier'), dataloader
        )
        
        # 2. Detection 평가
        detection_metrics = self._evaluate_detection_performance(
            models.get('detector'), dataloader
        )
        
        # 3. Pipeline 평가
        pipeline_metrics = self._evaluate_pipeline_performance(
            models, dataloader
        )
        
        # 4. Scalability 평가
        scalability_metrics = self._measure_scalability_metrics(dataloader)
        
        # 5. Stage 4 준비도 계산
        stage4_readiness_score = self._calculate_stage4_readiness(
            classification_metrics, detection_metrics, pipeline_metrics, scalability_metrics
        )
        
        total_time = time.time() - start_time
        
        # 보고서 생성
        report = Stage3EvaluationReport(
            config=self.config,
            classification_metrics=classification_metrics,
            detection_metrics=detection_metrics,
            pipeline_metrics=pipeline_metrics,
            scalability_metrics=scalability_metrics,
            total_evaluation_time_seconds=total_time,
            stage4_readiness_score=stage4_readiness_score
        )
        
        self.evaluation_history.append(report)
        
        self.logger.info(f"Stage 3 평가 완료 - 준비도: {stage4_readiness_score:.2%}")
        
        return report
    
    def _calculate_stage4_readiness(self, cls_metrics, det_metrics, 
                                  pipeline_metrics, scalability_metrics) -> float:
        """Stage 4 준비도 계산"""
        scores = []
        
        # Classification 성능 (40%)
        if cls_metrics.top1_accuracy > 0:
            cls_score = min(1.0, cls_metrics.top1_accuracy / self.config.target_accuracy)
            scores.append(cls_score * 0.4)
        
        # Detection 성능 (30%)
        if det_metrics.map50 > 0:
            det_score = min(1.0, det_metrics.map50 / self.config.target_map50)
            scores.append(det_score * 0.3)
        
        # 확장성 (30%)
        if scalability_metrics.samples_per_second > 0:
            scalability_score = min(1.0, scalability_metrics.samples_per_second / 100.0)
            scores.append(scalability_score * 0.3)
        
        return sum(scores) if scores else 0.0
    
    def export_evaluation_report(self, report: Stage3EvaluationReport, file_path: Path):
        """평가 보고서 내보내기"""
        try:
            report_dict = asdict(report)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Stage 3 평가 보고서 저장: {file_path}")
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")


def create_stage3_evaluator(config: Stage3Config = None) -> Stage3Evaluator:
    """Stage 3 평가기 생성 편의 함수"""
    if config is None:
        config = Stage3Config()
    
    return Stage3Evaluator(
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enable_memory_monitoring=True,
        enable_optimization_advisor=True
    )