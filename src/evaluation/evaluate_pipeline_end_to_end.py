"""
End-to-End Pipeline Evaluator
전체 파이프라인 종합 평가 시스템

Two-Stage Conditional Pipeline 전체 성능 평가:
- Single Mode: 직접 분류 성능
- Combo Mode: 검출 → 분류 성능
- 전체 처리량 및 지연시간
- 상업적 서비스 준비도 평가
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# from src.models.pipeline_two_stage_conditional import TwoStageConditionalPipeline
from src.evaluation.evaluate_classification_metrics import ClassificationMetricsEvaluator
from src.evaluation.evaluate_detection_metrics import DetectionMetricsEvaluator
from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor
from src.utils.core import PillSnapLogger


@dataclass
class EndToEndMetrics:
    """전체 파이프라인 메트릭"""
    
    # 전체 성능
    overall_accuracy: float
    single_mode_accuracy: float
    combo_mode_accuracy: float
    
    # 검출 성능 (Combo Mode)
    detection_map_50: float
    detection_precision: float
    detection_recall: float
    
    # 처리량 및 지연시간
    avg_inference_time_ms: float
    throughput_images_per_sec: float
    p95_inference_time_ms: float
    
    # 자원 사용량
    peak_memory_usage_gb: float
    avg_memory_usage_gb: float
    
    # 모드별 분포
    single_mode_ratio: float
    combo_mode_ratio: float
    
    # 에러율
    preprocessing_error_rate: float
    inference_error_rate: float
    total_error_rate: float


@dataclass
class CommercialReadinessScore:
    """상업적 준비도 점수"""
    
    accuracy_score: float       # 정확도 점수 (0-100)
    performance_score: float    # 성능 점수 (0-100)
    reliability_score: float    # 안정성 점수 (0-100)
    scalability_score: float    # 확장성 점수 (0-100)
    
    overall_score: float        # 전체 점수 (0-100)
    readiness_level: str        # 준비 수준 (Alpha/Beta/Production)
    
    recommendations: List[str]  # 개선 권장사항


class EndToEndPipelineEvaluator:
    """전체 파이프라인 평가기"""
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.logger = PillSnapLogger(__name__)
        
        # 평가 도구들
        self.memory_monitor = GPUMemoryMonitor()
        self.classification_evaluator = ClassificationMetricsEvaluator()
        self.detection_evaluator = DetectionMetricsEvaluator()
        
        # 평가 데이터 수집
        self.evaluation_results = []
        self.error_log = []
        
        self.logger.info("EndToEndPipelineEvaluator 초기화 완료")
    
    def evaluate_complete_pipeline(
        self,
        test_images: List[Path],
        ground_truth_labels: List[int],
        mode_labels: List[str],  # 'single' or 'combo'
        batch_size: int = 16,
        save_detailed_results: bool = True
    ) -> EndToEndMetrics:
        """전체 파이프라인 완전 평가"""
        
        self.logger.step("전체 파이프라인 평가", f"{len(test_images)}개 이미지 평가")
        
        start_time = time.time()
        
        # 결과 수집용
        single_results = []
        combo_results = []
        inference_times = []
        memory_usages = []
        errors = []
        
        try:
            # 배치별 평가
            for i in range(0, len(test_images), batch_size):
                batch_images = test_images[i:i+batch_size]
                batch_labels = ground_truth_labels[i:i+batch_size]
                batch_modes = mode_labels[i:i+batch_size]
                
                # 배치 처리
                batch_results = self._evaluate_batch(
                    batch_images, batch_labels, batch_modes
                )
                
                # 결과 분류
                for result in batch_results:
                    if result['mode'] == 'single':
                        single_results.append(result)
                    else:
                        combo_results.append(result)
                    
                    inference_times.append(result['inference_time_ms'])
                    memory_usages.append(result['memory_usage_gb'])
                    
                    if result['error']:
                        errors.append(result)
                
                # 진행 상황 로깅
                progress = min(i + batch_size, len(test_images))
                self.logger.info(f"진행률: {progress}/{len(test_images)} ({progress/len(test_images):.1%})")
            
            # 전체 메트릭 계산
            metrics = self._calculate_end_to_end_metrics(
                single_results, combo_results, inference_times, memory_usages, errors
            )
            
            total_time = time.time() - start_time
            
            self.logger.success(f"전체 파이프라인 평가 완료 - 소요시간: {total_time:.1f}초")
            self.logger.info(f"전체 정확도: {metrics.overall_accuracy:.1%}")
            self.logger.info(f"평균 추론 시간: {metrics.avg_inference_time_ms:.1f}ms")
            self.logger.info(f"처리량: {metrics.throughput_images_per_sec:.1f} images/sec")
            
            # 상세 결과 저장
            if save_detailed_results:
                self._save_detailed_evaluation_results(metrics, single_results, combo_results)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"전체 파이프라인 평가 실패: {e}")
            raise
    
    def _evaluate_batch(
        self,
        batch_images: List[Path],
        batch_labels: List[int],
        batch_modes: List[str]
    ) -> List[Dict[str, Any]]:
        """배치 평가"""
        
        batch_results = []
        
        for img_path, true_label, expected_mode in zip(batch_images, batch_labels, batch_modes):
            result = {
                'image_path': str(img_path),
                'true_label': true_label,
                'expected_mode': expected_mode,
                'predicted_label': -1,
                'predicted_mode': '',
                'confidence': 0.0,
                'inference_time_ms': 0.0,
                'memory_usage_gb': 0.0,
                'error': None,
                'detection_boxes': [],
                'mode': expected_mode
            }
            
            try:
                # 메모리 사용량 측정 시작
                memory_before = self.memory_monitor.get_current_usage()
                
                # 추론 시작
                inference_start = time.time()
                
                # 파이프라인 실행 (시뮬레이션)
                inference_result = self._simulate_pipeline_inference(img_path, expected_mode)
                
                inference_time = (time.time() - inference_start) * 1000  # ms
                
                # 메모리 사용량 측정 종료
                memory_after = self.memory_monitor.get_current_usage()
                
                # 결과 업데이트
                result.update({
                    'predicted_label': inference_result['predicted_class'],
                    'predicted_mode': inference_result['detected_mode'],
                    'confidence': inference_result['confidence'],
                    'inference_time_ms': inference_time,
                    'memory_usage_gb': memory_after['used_gb'],
                    'detection_boxes': inference_result.get('detection_boxes', [])
                })
                
            except Exception as e:
                result['error'] = str(e)
                self.logger.warning(f"이미지 처리 실패 {img_path}: {e}")
            
            batch_results.append(result)
        
        return batch_results
    
    def _simulate_pipeline_inference(self, image_path: Path, expected_mode: str) -> Dict[str, Any]:
        """파이프라인 추론 시뮬레이션"""
        import random
        
        # 실제 구현에서는 self.pipeline.predict(image_path) 호출
        
        if expected_mode == 'single':
            # Single Mode 시뮬레이션
            return {
                'predicted_class': random.randint(0, 49),  # 50개 클래스
                'detected_mode': 'single',
                'confidence': random.uniform(0.7, 0.95),
                'detection_boxes': []
            }
        else:
            # Combo Mode 시뮬레이션
            return {
                'predicted_class': random.randint(0, 49),
                'detected_mode': 'combo',
                'confidence': random.uniform(0.6, 0.9),
                'detection_boxes': [
                    {'x': 0.3, 'y': 0.3, 'w': 0.4, 'h': 0.4, 'confidence': 0.8},
                    {'x': 0.6, 'y': 0.5, 'w': 0.3, 'h': 0.3, 'confidence': 0.7}
                ]
            }
    
    def _calculate_end_to_end_metrics(
        self,
        single_results: List[Dict],
        combo_results: List[Dict],
        inference_times: List[float],
        memory_usages: List[float],
        errors: List[Dict]
    ) -> EndToEndMetrics:
        """전체 메트릭 계산"""
        
        total_samples = len(single_results) + len(combo_results)
        
        if total_samples == 0:
            raise ValueError("평가 결과가 없음")
        
        # 정확도 계산
        single_correct = sum(1 for r in single_results 
                           if r['predicted_label'] == r['true_label'] and not r['error'])
        combo_correct = sum(1 for r in combo_results 
                          if r['predicted_label'] == r['true_label'] and not r['error'])
        
        total_correct = single_correct + combo_correct
        
        single_accuracy = single_correct / len(single_results) if single_results else 0.0
        combo_accuracy = combo_correct / len(combo_results) if combo_results else 0.0
        overall_accuracy = total_correct / total_samples
        
        # 검출 성능 (Combo Mode만)
        detection_map = 0.32 if combo_results else 0.0  # 시뮬레이션
        detection_precision = 0.36 if combo_results else 0.0
        detection_recall = 0.30 if combo_results else 0.0
        
        # 처리량 및 지연시간
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            p95_inference_time = np.percentile(inference_times, 95)
            throughput = 1000 / avg_inference_time  # images/sec
        else:
            avg_inference_time = 0.0
            p95_inference_time = 0.0
            throughput = 0.0
        
        # 메모리 사용량
        if memory_usages:
            peak_memory = max(memory_usages)
            avg_memory = np.mean(memory_usages)
        else:
            peak_memory = 0.0
            avg_memory = 0.0
        
        # 에러율
        preprocessing_errors = sum(1 for e in errors if 'preprocessing' in str(e.get('error', '')))
        inference_errors = sum(1 for e in errors if 'inference' in str(e.get('error', '')))
        
        preprocessing_error_rate = preprocessing_errors / total_samples
        inference_error_rate = inference_errors / total_samples
        total_error_rate = len(errors) / total_samples
        
        return EndToEndMetrics(
            overall_accuracy=overall_accuracy,
            single_mode_accuracy=single_accuracy,
            combo_mode_accuracy=combo_accuracy,
            detection_map_50=detection_map,
            detection_precision=detection_precision,
            detection_recall=detection_recall,
            avg_inference_time_ms=avg_inference_time,
            throughput_images_per_sec=throughput,
            p95_inference_time_ms=p95_inference_time,
            peak_memory_usage_gb=peak_memory,
            avg_memory_usage_gb=avg_memory,
            single_mode_ratio=len(single_results) / total_samples,
            combo_mode_ratio=len(combo_results) / total_samples,
            preprocessing_error_rate=preprocessing_error_rate,
            inference_error_rate=inference_error_rate,
            total_error_rate=total_error_rate
        )
    
    def evaluate_commercial_readiness(self, metrics: EndToEndMetrics) -> CommercialReadinessScore:
        """상업적 준비도 평가"""
        
        self.logger.step("상업적 준비도 평가", "정확도, 성능, 안정성, 확장성 평가")
        
        # 정확도 점수 (0-100)
        accuracy_score = min(metrics.overall_accuracy * 100, 100)
        
        # 성능 점수 (50ms 목표 기준)
        target_latency = 50.0  # ms
        if metrics.avg_inference_time_ms <= target_latency:
            performance_score = 100
        elif metrics.avg_inference_time_ms <= target_latency * 2:
            performance_score = 100 - ((metrics.avg_inference_time_ms - target_latency) / target_latency * 50)
        else:
            performance_score = 25
        
        # 안정성 점수 (에러율 기준)
        if metrics.total_error_rate <= 0.01:  # 1% 이하
            reliability_score = 100
        elif metrics.total_error_rate <= 0.05:  # 5% 이하
            reliability_score = 80
        elif metrics.total_error_rate <= 0.10:  # 10% 이하
            reliability_score = 60
        else:
            reliability_score = 30
        
        # 확장성 점수 (메모리 사용량 및 처리량 기준)
        if metrics.peak_memory_usage_gb <= 14.0 and metrics.throughput_images_per_sec >= 20:
            scalability_score = 100
        elif metrics.peak_memory_usage_gb <= 16.0 and metrics.throughput_images_per_sec >= 10:
            scalability_score = 80
        else:
            scalability_score = 50
        
        # 전체 점수 (가중 평균)
        overall_score = (
            accuracy_score * 0.4 +
            performance_score * 0.25 +
            reliability_score * 0.2 +
            scalability_score * 0.15
        )
        
        # 준비 수준 결정
        if overall_score >= 85:
            readiness_level = "Production Ready"
        elif overall_score >= 70:
            readiness_level = "Beta Ready"
        elif overall_score >= 50:
            readiness_level = "Alpha Ready"
        else:
            readiness_level = "Development"
        
        # 권장사항 생성
        recommendations = []
        if accuracy_score < 80:
            recommendations.append("정확도 향상: 추가 학습 데이터 확보 및 모델 튜닝 필요")
        if performance_score < 80:
            recommendations.append("성능 최적화: 모델 경량화 또는 하드웨어 업그레이드 필요")
        if reliability_score < 80:
            recommendations.append("안정성 개선: 에러 처리 강화 및 예외 상황 대응 로직 보완")
        if scalability_score < 80:
            recommendations.append("확장성 개선: 메모리 최적화 및 배치 처리 개선")
        
        score = CommercialReadinessScore(
            accuracy_score=accuracy_score,
            performance_score=performance_score,
            reliability_score=reliability_score,
            scalability_score=scalability_score,
            overall_score=overall_score,
            readiness_level=readiness_level,
            recommendations=recommendations
        )
        
        # 결과 로깅
        self.logger.info(f"🏢 상업적 준비도 평가 결과:")
        self.logger.info(f"  정확도: {accuracy_score:.1f}/100")
        self.logger.info(f"  성능: {performance_score:.1f}/100")
        self.logger.info(f"  안정성: {reliability_score:.1f}/100")
        self.logger.info(f"  확장성: {scalability_score:.1f}/100")
        self.logger.info(f"  전체 점수: {overall_score:.1f}/100")
        self.logger.info(f"  준비 수준: {readiness_level}")
        
        return score
    
    def _save_detailed_evaluation_results(
        self,
        metrics: EndToEndMetrics,
        single_results: List[Dict],
        combo_results: List[Dict]
    ) -> str:
        """상세 평가 결과 저장"""
        
        try:
            results_dir = Path("artifacts/reports/validation_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"end_to_end_evaluation_{timestamp}.json"
            
            report_data = {
                'timestamp': timestamp,
                'evaluation_type': 'end_to_end_pipeline',
                'overall_metrics': {
                    'overall_accuracy': metrics.overall_accuracy,
                    'single_mode_accuracy': metrics.single_mode_accuracy,
                    'combo_mode_accuracy': metrics.combo_mode_accuracy,
                    'avg_inference_time_ms': metrics.avg_inference_time_ms,
                    'throughput_images_per_sec': metrics.throughput_images_per_sec,
                    'peak_memory_usage_gb': metrics.peak_memory_usage_gb,
                    'total_error_rate': metrics.total_error_rate
                },
                'mode_distribution': {
                    'single_mode_ratio': metrics.single_mode_ratio,
                    'combo_mode_ratio': metrics.combo_mode_ratio,
                    'single_samples': len(single_results),
                    'combo_samples': len(combo_results)
                },
                'detailed_results': {
                    'single_mode_results': single_results[:100],  # 처음 100개만 저장
                    'combo_mode_results': combo_results[:100]
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"상세 평가 결과 저장: {results_file}")
            return str(results_file)
            
        except Exception as e:
            self.logger.error(f"평가 결과 저장 실패: {e}")
            return ""


def main():
    """End-to-End Pipeline Evaluator 테스트"""
    print("📊 End-to-End Pipeline Evaluator Test")
    print("=" * 60)
    
    try:
        # 더미 파이프라인 (실제로는 TwoStageConditionalPipeline 인스턴스)
        pipeline = None  # 시뮬레이션용
        
        # 평가기 생성
        evaluator = EndToEndPipelineEvaluator(pipeline)
        
        # 더미 테스트 데이터
        test_images = [Path(f"dummy_image_{i}.jpg") for i in range(100)]
        ground_truth_labels = [i % 50 for i in range(100)]  # 50개 클래스
        mode_labels = ['single' if i % 3 != 0 else 'combo' for i in range(100)]  # 2:1 비율
        
        print(f"테스트 데이터: {len(test_images)}개 이미지")
        print(f"Single/Combo 비율: {mode_labels.count('single')}/{mode_labels.count('combo')}")
        
        # 전체 파이프라인 평가
        metrics = evaluator.evaluate_complete_pipeline(
            test_images=test_images,
            ground_truth_labels=ground_truth_labels,
            mode_labels=mode_labels,
            batch_size=16,
            save_detailed_results=True
        )
        
        print(f"\n📈 전체 파이프라인 평가 결과:")
        print(f"  전체 정확도: {metrics.overall_accuracy:.1%}")
        print(f"  Single Mode 정확도: {metrics.single_mode_accuracy:.1%}")
        print(f"  Combo Mode 정확도: {metrics.combo_mode_accuracy:.1%}")
        print(f"  평균 추론 시간: {metrics.avg_inference_time_ms:.1f}ms")
        print(f"  처리량: {metrics.throughput_images_per_sec:.1f} images/sec")
        print(f"  피크 메모리: {metrics.peak_memory_usage_gb:.1f}GB")
        print(f"  에러율: {metrics.total_error_rate:.1%}")
        
        # 상업적 준비도 평가
        readiness_score = evaluator.evaluate_commercial_readiness(metrics)
        
        print(f"\n🏢 상업적 준비도 평가:")
        print(f"  전체 점수: {readiness_score.overall_score:.1f}/100")
        print(f"  준비 수준: {readiness_score.readiness_level}")
        print(f"  권장사항: {len(readiness_score.recommendations)}개")
        
        print("\n✅ End-to-End Pipeline 평가 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()