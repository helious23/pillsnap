"""
PillSnap ML 최소셋 로깅 시스템 (1단계 필수)

핵심 메트릭만 기록하여 용량/시간 절감:
- 분류: top-1/top-5/macro-F1(전체+도메인)
- 검출: mAP@0.5/Recall/Precision(도메인) + selected_confidence
- 파이프라인: det/crop/cls/total(ms) (+ p50/p95/p99)
- 시스템: VRAM current/peak, grad-norm(after_clipping), 3epoch마다 before_clipping 스냅샷

RTX 5080 최적화
"""

import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics

import torch
import numpy as np

from src.utils.core import PillSnapLogger


@dataclass
class MinimalLoggingConfig:
    """최소셋 로깅 설정 (1단계 필수)"""
    
    # 로깅 주기
    log_every_n_steps: int = 10           # N step마다 로깅
    log_validation_always: bool = True     # 검증시 항상 로깅
    
    # 메트릭 선택
    classification_metrics: List[str] = None  # ["top1", "top5", "macro_f1", "domain_f1"]
    detection_metrics: List[str] = None       # ["map_0_5", "recall", "precision", "confidence"]
    pipeline_metrics: List[str] = None        # ["det_ms", "crop_ms", "cls_ms", "total_ms"]
    system_metrics: List[str] = None          # ["vram_current", "vram_peak", "grad_norm"]
    
    # 시스템 로깅
    log_grad_norm_before_clipping_every: int = 3  # 3 epoch마다 clipping 전 grad norm
    track_percentiles: bool = True                 # p50/p95/p99 추적
    
    # 저장 옵션
    save_to_json: bool = True
    save_to_tensorboard: bool = False  # 최소셋이므로 기본 비활성화
    
    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = ["top1", "top5", "macro_f1", "domain_f1"]
        if self.detection_metrics is None:
            self.detection_metrics = ["map_0_5", "recall", "precision", "confidence"]
        if self.pipeline_metrics is None:
            self.pipeline_metrics = ["det_ms", "crop_ms", "cls_ms", "total_ms"]
        if self.system_metrics is None:
            self.system_metrics = ["vram_current", "vram_peak", "grad_norm"]


class PipelineTotalTimer:
    """파이프라인 total(ms) 보장 Context Manager"""
    
    def __init__(self, logger_instance, operation: str = "total"):
        self.logger = logger_instance
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            total_ms = (time.perf_counter() - self.start_time) * 1000
            self.logger.record_pipeline_timing(self.operation, total_ms)
        return False


class PerformanceTracker:
    """성능 추적기 (파이프라인 레이턴시용)"""
    
    def __init__(self, track_percentiles: bool = True, window_size: int = 100):
        self.track_percentiles = track_percentiles
        self.window_size = window_size
        
        # 시간 측정 저장소
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.current_timings = {}
        
    def start_timer(self, operation: str) -> None:
        """타이머 시작"""
        self.current_timings[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """타이머 종료 및 시간 기록"""
        if operation not in self.current_timings:
            return 0.0
        
        elapsed_ms = (time.perf_counter() - self.current_timings[operation]) * 1000
        self.timings[operation].append(elapsed_ms)
        
        return elapsed_ms
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """특정 작업의 타이밍 통계 반환"""
        if operation not in self.timings or len(self.timings[operation]) == 0:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        values = list(self.timings[operation])
        
        stats = {
            "mean": statistics.mean(values)
        }
        
        if self.track_percentiles and len(values) >= 3:
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            stats.update({
                "p50": sorted_values[int(n * 0.50)],
                "p95": sorted_values[int(n * 0.95)],
                "p99": sorted_values[int(n * 0.99)]
            })
        else:
            stats.update({"p50": stats["mean"], "p95": stats["mean"], "p99": stats["mean"]})
        
        return stats
    
    def get_all_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """모든 작업의 타이밍 통계 반환"""
        return {
            operation: self.get_timing_stats(operation)
            for operation in self.timings.keys()
        }


class MinimalLogger:
    """최소셋 로깅 시스템 (1단계 필수)"""
    
    def __init__(self, config: MinimalLoggingConfig, save_dir: Optional[str] = None):
        """
        Args:
            config: 최소셋 로깅 설정
            save_dir: 로그 저장 디렉토리
        """
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 저장 경로 설정
        if save_dir is None:
            save_dir = Path("artifacts/logs/minimal")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 성능 추적기
        self.performance_tracker = PerformanceTracker(
            track_percentiles=config.track_percentiles
        )
        
        # 로그 저장소
        self.log_history = []
        self.step_count = 0
        self.epoch_count = 0
        
        # VRAM 피크 추적
        self.vram_peak = 0.0
        
        # Gradient norm 히스토리
        self.grad_norm_history = deque(maxlen=1000)
        
        self.logger.info(f"📝 최소셋 로거 초기화 - 저장 위치: {self.save_dir}")
    
    def log_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, Any],
        force_log: bool = False
    ) -> None:
        """스텝별 로깅"""
        self.step_count = step
        self.epoch_count = epoch
        
        # 로깅 주기 확인
        should_log = (
            force_log or
            step % self.config.log_every_n_steps == 0 or
            step == 0
        )
        
        if not should_log:
            return
        
        # 로그 엔트리 생성
        log_entry = {
            "timestamp": time.time(),
            "step": step,
            "epoch": epoch,
            "metrics": self._filter_metrics(metrics)
        }
        
        # 시스템 메트릭 추가
        system_metrics = self._collect_system_metrics()
        log_entry["metrics"]["system"] = system_metrics
        
        # 파이프라인 메트릭 추가
        pipeline_metrics = self._collect_pipeline_metrics()
        if pipeline_metrics:
            log_entry["metrics"]["pipeline"] = pipeline_metrics
        
        # 저장 및 출력
        self.log_history.append(log_entry)
        self._log_to_console(log_entry)
        
        if self.config.save_to_json:
            self._save_to_json()
    
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        selected_confidences: Optional[Dict[str, float]] = None
    ) -> None:
        """검증 로깅 (항상 기록)"""
        log_entry = {
            "timestamp": time.time(),
            "type": "validation",
            "epoch": epoch,
            "metrics": self._filter_metrics(metrics)
        }
        
        # Selected confidence 추가
        if selected_confidences:
            log_entry["selected_confidences"] = selected_confidences
        
        # 시스템 메트릭 추가
        system_metrics = self._collect_system_metrics()
        log_entry["metrics"]["system"] = system_metrics
        
        # 파이프라인 메트릭 추가
        pipeline_metrics = self._collect_pipeline_metrics()
        if pipeline_metrics:
            log_entry["metrics"]["pipeline"] = pipeline_metrics
        
        # 저장 및 출력
        self.log_history.append(log_entry)
        self._log_to_console(log_entry, is_validation=True)
        
        if self.config.save_to_json:
            self._save_to_json()
    
    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """핵심 메트릭만 필터링 (도메인별 Top-1/Top-5 지원)"""
        filtered = {}
        
        # 분류 메트릭 필터링 (도메인별 지원)
        if "classification" in metrics:
            cls_metrics = {}
            for metric in self.config.classification_metrics:
                if metric in metrics["classification"]:
                    cls_metrics[metric] = metrics["classification"][metric]
                    
                # 도메인별 메트릭 추가 검사 (flat keys)
                elif metric in ["top1", "top5", "macro_f1", "f1"] and f"{metric}_single" in metrics["classification"]:
                    # Flat format: top1_single, top5_single, macro_f1_single, f1_single, etc.
                    for domain in ["single", "combination"]:
                        domain_key = f"{metric}_{domain}"
                        if domain_key in metrics["classification"]:
                            cls_metrics[domain_key] = metrics["classification"][domain_key]
                            
                # 중첩 형식도 지원 (nested format)
                elif metric in ["top1", "top5", "macro_f1", "f1"] and "domain_metrics" in metrics["classification"]:
                    domain_metrics = metrics["classification"]["domain_metrics"]
                    if isinstance(domain_metrics, dict):
                        for domain in ["single", "combination"]:
                            if domain in domain_metrics and metric in domain_metrics[domain]:
                                cls_metrics[f"{metric}_{domain}"] = domain_metrics[domain][metric]
                                
            if cls_metrics:
                filtered["classification"] = cls_metrics
        
        # 검출 메트릭 필터링
        if "detection" in metrics:
            det_metrics = {}
            for metric in self.config.detection_metrics:
                if metric in metrics["detection"]:
                    det_metrics[metric] = metrics["detection"][metric]
            if det_metrics:
                filtered["detection"] = det_metrics
        
        # 기타 메트릭 (loss 등)
        for key, value in metrics.items():
            if key not in ["classification", "detection", "pipeline", "system"]:
                if isinstance(value, (int, float, str)):
                    filtered[key] = value
        
        return filtered
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """시스템 메트릭 수집"""
        system_metrics = {}
        
        # VRAM 메트릭
        if torch.cuda.is_available() and "vram_current" in self.config.system_metrics:
            current_vram = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved_vram = torch.cuda.memory_reserved() / 1024**3  # GB
            
            # 피크 추적
            self.vram_peak = max(self.vram_peak, reserved_vram)
            
            system_metrics.update({
                "vram_current": round(current_vram, 2),
                "vram_reserved": round(reserved_vram, 2),
                "vram_peak": round(self.vram_peak, 2)
            })
        
        # Gradient norm (after clipping)
        if hasattr(self, '_last_grad_norm') and "grad_norm" in self.config.system_metrics:
            system_metrics["grad_norm_after_clipping"] = self._last_grad_norm
        
        return system_metrics
    
    def _collect_pipeline_metrics(self) -> Dict[str, Any]:
        """파이프라인 메트릭 수집"""
        pipeline_stats = self.performance_tracker.get_all_timing_stats()
        
        if not pipeline_stats:
            return {}
        
        # 필터링된 파이프라인 메트릭
        filtered_pipeline = {}
        for operation in self.config.pipeline_metrics:
            if operation.replace("_ms", "") in pipeline_stats:
                op_key = operation.replace("_ms", "")
                stats = pipeline_stats[op_key]
                
                filtered_pipeline[operation] = round(stats["mean"], 2)
                
                if self.config.track_percentiles:
                    filtered_pipeline[f"{operation}_p50"] = round(stats["p50"], 2)
                    filtered_pipeline[f"{operation}_p95"] = round(stats["p95"], 2)
                    filtered_pipeline[f"{operation}_p99"] = round(stats["p99"], 2)
        
        return filtered_pipeline
    
    def record_pipeline_timing(self, operation: str, elapsed_ms: float) -> None:
        """파이프라인 타이밍 기록"""
        # 성능 추적기에 직접 기록
        self.performance_tracker.timings[operation].append(elapsed_ms)
    
    def start_pipeline_timer(self, operation: str) -> None:
        """파이프라인 타이머 시작"""
        self.performance_tracker.start_timer(operation)
    
    def end_pipeline_timer(self, operation: str) -> float:
        """파이프라인 타이머 종료"""
        return self.performance_tracker.end_timer(operation)
    
    def record_grad_norm(self, grad_norm: float, before_clipping: bool = False) -> None:
        """Gradient norm 기록"""
        if before_clipping:
            # 3 epoch마다 before_clipping 기록
            if self.epoch_count % self.config.log_grad_norm_before_clipping_every == 0:
                self.grad_norm_history.append({
                    "epoch": self.epoch_count,
                    "step": self.step_count,
                    "grad_norm_before_clipping": grad_norm,
                    "timestamp": time.time()
                })
        else:
            # After clipping은 항상 기록
            self._last_grad_norm = grad_norm
    
    def pipeline_timer(self, operation: str = "total") -> PipelineTotalTimer:
        """파이프라인 total(ms) 보장 Context Manager 생성
        
        Usage:
            with logger.pipeline_timer("total") as timer:
                # 파이프라인 전체 실행
                detection_result = detect(image)
                crop_result = crop(detection_result) 
                classification_result = classify(crop_result)
        """
        return PipelineTotalTimer(self, operation)
    
    def _log_to_console(self, log_entry: Dict[str, Any], is_validation: bool = False) -> None:
        """콘솔 출력"""
        entry_type = "VAL" if is_validation else "TRAIN"
        step = log_entry.get("step", "")
        epoch = log_entry.get("epoch", "")
        
        # 핵심 메트릭만 간단히 출력
        metrics = log_entry.get("metrics", {})
        
        output_parts = [f"[{entry_type}] E{epoch}"]
        if step:
            output_parts.append(f"S{step}")
        
        # 분류 메트릭 (도메인별 표시 지원)
        if "classification" in metrics:
            cls_metrics = metrics["classification"]
            
            # 도메인별 메트릭이 있는지 확인
            has_domain_metrics = any(key.endswith('_single') or key.endswith('_combination') for key in cls_metrics.keys())
            
            if has_domain_metrics:
                # 도메인별 분리 출력: CLS[S: t1=0.43/t5=0.75/F1=0.39 | C: t1=0.31/t5=0.62/F1=0.28]
                single_parts = []
                combo_parts = []
                
                # Single domain metrics
                if "top1_single" in cls_metrics:
                    single_parts.append(f"t1={cls_metrics['top1_single']:.2f}")
                if "top5_single" in cls_metrics:
                    single_parts.append(f"t5={cls_metrics['top5_single']:.2f}")
                if "macro_f1_single" in cls_metrics:
                    single_parts.append(f"F1={cls_metrics['macro_f1_single']:.2f}")
                elif "f1_single" in cls_metrics:
                    single_parts.append(f"F1={cls_metrics['f1_single']:.2f}")
                    
                # Combination domain metrics
                if "top1_combination" in cls_metrics:
                    combo_parts.append(f"t1={cls_metrics['top1_combination']:.2f}")
                if "top5_combination" in cls_metrics:
                    combo_parts.append(f"t5={cls_metrics['top5_combination']:.2f}")
                if "macro_f1_combination" in cls_metrics:
                    combo_parts.append(f"F1={cls_metrics['macro_f1_combination']:.2f}")
                elif "f1_combination" in cls_metrics:
                    combo_parts.append(f"F1={cls_metrics['f1_combination']:.2f}")
                    
                # 도메인별 출력 조합
                domain_output = "CLS["
                if single_parts:
                    domain_output += f"S: {'/'.join(single_parts)}"
                if combo_parts:
                    if single_parts:
                        domain_output += " | "
                    domain_output += f"C: {'/'.join(combo_parts)}"
                domain_output += "]"
                
                if single_parts or combo_parts:
                    output_parts.append(domain_output)
            else:
                # 기존 전체 메트릭 출력
                if "top1" in cls_metrics:
                    output_parts.append(f"Top1:{cls_metrics['top1']:.3f}")
                if "macro_f1" in cls_metrics:
                    output_parts.append(f"F1:{cls_metrics['macro_f1']:.3f}")
        
        # 검출 메트릭
        if "detection" in metrics:
            det_metrics = metrics["detection"]
            if "map_0_5" in det_metrics:
                output_parts.append(f"mAP:{det_metrics['map_0_5']:.3f}")
        
        # 시스템 메트릭
        if "system" in metrics:
            sys_metrics = metrics["system"]
            if "vram_current" in sys_metrics:
                output_parts.append(f"VRAM:{sys_metrics['vram_current']:.1f}GB")
        
        # Selected confidence (검증시)
        if "selected_confidences" in log_entry:
            confs = log_entry["selected_confidences"]
            conf_strs = [f"{k}:{v:.2f}" for k, v in confs.items()]
            output_parts.append(f"Conf:[{','.join(conf_strs)}]")
        
        output = " | ".join(output_parts)
        self.logger.info(output)
    
    def _save_to_json(self) -> None:
        """JSON 파일로 저장"""
        timestamp = time.strftime("%Y%m%d")
        json_file = self.save_dir / f"minimal_log_{timestamp}.json"
        
        # 최신 100개만 유지하여 파일 크기 제한
        recent_logs = self.log_history[-100:] if len(self.log_history) > 100 else self.log_history
        
        log_data = {
            "config": asdict(self.config),
            "logs": recent_logs,
            "grad_norm_history": list(self.grad_norm_history)
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """요약 통계 반환"""
        if not self.log_history:
            return {}
        
        return {
            "total_steps": self.step_count,
            "total_epochs": self.epoch_count,
            "log_entries": len(self.log_history),
            "vram_peak_gb": self.vram_peak,
            "pipeline_stats": self.performance_tracker.get_all_timing_stats(),
            "recent_grad_norms": list(self.grad_norm_history)[-10:]  # 최근 10개
        }


if __name__ == "__main__":
    print("🧪 최소셋 로깅 시스템 테스트 (1단계 필수 + 도메인별 + total(ms) 보장)")
    print("=" * 80)
    
    # 설정 테스트
    config = MinimalLoggingConfig(
        log_every_n_steps=5,
        track_percentiles=True
    )
    
    logger = MinimalLogger(config, save_dir="/tmp/test_minimal_logs")
    print(f"✅ 로거 초기화: {len(config.classification_metrics)}개 분류 메트릭")
    
    # 파이프라인 타이밍 테스트
    logger.start_pipeline_timer("detection")
    time.sleep(0.01)  # 10ms 시뮬레이션
    det_time = logger.end_pipeline_timer("detection")
    
    logger.start_pipeline_timer("classification")
    time.sleep(0.005)  # 5ms 시뮬레이션
    cls_time = logger.end_pipeline_timer("classification")
    
    print(f"✅ 파이프라인 타이밍: det={det_time:.1f}ms, cls={cls_time:.1f}ms")
    
    # 도메인별 메트릭 로깅 테스트
    mock_metrics = {
        "classification": {
            "top1": 0.752,
            "top5": 0.934,
            "macro_f1": 0.681,
            # 도메인별 메트릭 추가
            "top1_single": 0.823,
            "top5_single": 0.956,
            "f1_single": 0.734,
            "top1_combination": 0.645,
            "top5_combination": 0.812,
            "f1_combination": 0.598
        },
        "detection": {
            "map_0_5": 0.456,
            "recall": 0.623,
            "precision": 0.578
        },
        "loss": 2.34
    }
    
    # 파이프라인 total(ms) Context Manager 테스트
    with logger.pipeline_timer("total"):
        time.sleep(0.02)  # 20ms 전체 파이프라인 시뮬레이션
    print("✅ 파이프라인 total(ms) Context Manager 테스트 완료")
    
    # 스텝 로깅 (도메인별 메트릭 포함)
    logger.log_step(step=10, epoch=1, metrics=mock_metrics)
    print("✅ 도메인별 스텝 로깅 완료")
    
    # 검증 로깅 (도메인별 메트릭 포함)
    logger.log_validation(
        epoch=1,
        metrics=mock_metrics,
        selected_confidences={"single": 0.24, "combination": 0.26}
    )
    print("✅ 도메인별 검증 로깅 완료")
    
    # Gradient norm 기록
    logger.record_grad_norm(1.23, before_clipping=False)
    logger.record_grad_norm(2.45, before_clipping=True)  # 3 epoch마다만 기록
    print("✅ Gradient norm 기록 완료")
    
    # 요약 통계
    summary = logger.get_summary_stats()
    print(f"✅ 요약 통계: {summary['log_entries']}개 로그 엔트리")
    
    # 파이프라인 통계 확인
    pipeline_stats = summary.get('pipeline_stats', {})
    if 'total' in pipeline_stats:
        total_stats = pipeline_stats['total']
        print(f"✅ Total 파이프라인 통계: mean={total_stats['mean']:.1f}ms, p95={total_stats['p95']:.1f}ms")
    
    print("🎉 최소셋 로깅 시스템 (도메인별 + total(ms) 보장) 테스트 완료!")
    print("📊 도메인별 출력 예시: CLS[S: t1=0.82/t5=0.96/F1=0.73 | C: t1=0.65/t5=0.81/F1=0.60]")