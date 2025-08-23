"""
OptimizationAdvisor 시스템

RTX 5080 16GB + 128GB RAM 환경에서 Stage 3-4 훈련을 위한
지능형 최적화 권고 시스템. 실시간 성능 분석과 동적 하이퍼파라미터 조정.

Features:
- 하드웨어 기반 최적화 권고
- 동적 배치 크기 / 학습률 조정
- 메모리 효율성 분석
- 성능 병목 지점 식별
- Stage별 맞춤형 최적화

Author: Claude Code - PillSnap ML Team  
Date: 2025-08-23
"""

import os
import sys
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.core import PillSnapLogger
from src.utils.memory_state_manager import MemoryStateManager, MemoryState


class OptimizationLevel(Enum):
    """최적화 수준"""
    CONSERVATIVE = "conservative"    # 안정성 우선
    BALANCED = "balanced"           # 균형잡힌 설정
    AGGRESSIVE = "aggressive"       # 성능 우선
    EXTREME = "extreme"            # 최대 성능


class BottleneckType(Enum):
    """성능 병목 유형"""
    GPU_MEMORY = "gpu_memory"       # GPU 메모리 부족
    GPU_COMPUTE = "gpu_compute"     # GPU 연산 능력 부족
    CPU_BOUND = "cpu_bound"         # CPU 병목
    IO_BOUND = "io_bound"           # I/O 병목
    NETWORK_BOUND = "network_bound" # 네트워크 병목
    NONE = "none"                   # 병목 없음


@dataclass
class HardwareProfile:
    """하드웨어 프로파일"""
    gpu_name: str
    gpu_memory_gb: float
    gpu_compute_capability: Tuple[int, int]
    cpu_cores: int
    system_memory_gb: float
    storage_type: str  # "ssd", "nvme", "hdd"
    cuda_version: str
    pytorch_version: str


@dataclass
class TrainingMetrics:
    """훈련 메트릭스"""
    batch_size: int
    learning_rate: float
    epoch_time_seconds: float
    samples_per_second: float
    gpu_utilization: float
    gpu_memory_usage_gb: float
    cpu_utilization: float
    io_wait_percent: float
    validation_accuracy: Optional[float] = None
    training_loss: Optional[float] = None


@dataclass
class OptimizationRecommendation:
    """최적화 권고사항"""
    category: str                    # "batch_size", "learning_rate", etc.
    current_value: Any
    recommended_value: Any
    expected_improvement: float      # % 개선 예상
    confidence: float               # 0.0-1.0 신뢰도
    reasoning: str                  # 권고 이유
    risk_level: str                 # "low", "medium", "high"
    implementation_priority: int     # 1-5 (1=최우선)


@dataclass
class OptimizationReport:
    """최적화 보고서"""
    timestamp: float
    hardware_profile: HardwareProfile
    current_metrics: TrainingMetrics
    bottleneck_type: BottleneckType
    bottleneck_severity: float      # 0.0-1.0
    recommendations: List[OptimizationRecommendation]
    overall_score: float            # 0.0-1.0 전체 최적화 점수
    estimated_speedup: float        # 예상 속도 향상 배수


class OptimizationAdvisor:
    """
    지능형 최적화 권고 시스템
    
    RTX 5080 + Stage 3-4 환경에 특화된 최적화 권고사항을 
    실시간으로 생성하고 동적 조정을 수행.
    """
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 enable_auto_tuning: bool = False,
                 monitoring_interval: float = 10.0):
        """
        초기화
        
        Args:
            optimization_level: 최적화 수준
            enable_auto_tuning: 자동 튜닝 활성화
            monitoring_interval: 모니터링 간격 (초)
        """
        self.logger = PillSnapLogger(__name__)
        self.optimization_level = optimization_level
        self.enable_auto_tuning = enable_auto_tuning
        self.monitoring_interval = monitoring_interval
        
        # 하드웨어 프로파일 생성
        self.hardware_profile = self._detect_hardware_profile()
        
        # 메모리 매니저 통합
        self.memory_manager = MemoryStateManager()
        
        # 메트릭 히스토리
        self.metrics_history: List[TrainingMetrics] = []
        self.recommendations_history: List[OptimizationReport] = []
        
        # 모니터링 상태
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Stage별 최적화 프리셋
        self.stage_presets = self._load_stage_presets()
        
        self.logger.info(f"OptimizationAdvisor 초기화 완료")
        self.logger.info(f"GPU: {self.hardware_profile.gpu_name} ({self.hardware_profile.gpu_memory_gb:.1f}GB)")
        self.logger.info(f"최적화 수준: {optimization_level.value}")
    
    def _detect_hardware_profile(self) -> HardwareProfile:
        """하드웨어 프로파일 자동 감지"""
        try:
            # GPU 정보
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                compute_capability = torch.cuda.get_device_capability()
            else:
                gpu_name = "CPU Only"
                gpu_memory_gb = 0.0
                compute_capability = (0, 0)
            
            # CPU 정보
            cpu_cores = psutil.cpu_count(logical=False)
            
            # 시스템 메모리
            memory = psutil.virtual_memory()
            system_memory_gb = memory.total / 1024**3
            
            # 스토리지 타입 추정 (간단한 벤치마크)
            storage_type = self._detect_storage_type()
            
            # CUDA & PyTorch 버전
            cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
            pytorch_version = torch.__version__
            
            return HardwareProfile(
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                gpu_compute_capability=compute_capability,
                cpu_cores=cpu_cores,
                system_memory_gb=system_memory_gb,
                storage_type=storage_type,
                cuda_version=cuda_version,
                pytorch_version=pytorch_version
            )
            
        except Exception as e:
            self.logger.error(f"하드웨어 프로파일 감지 실패: {e}")
            # 기본값 반환
            return HardwareProfile(
                gpu_name="Unknown GPU",
                gpu_memory_gb=16.0,
                gpu_compute_capability=(8, 0),
                cpu_cores=8,
                system_memory_gb=128.0,
                storage_type="ssd",
                cuda_version="12.0",
                pytorch_version="2.0.0"
            )
    
    def _detect_storage_type(self) -> str:
        """스토리지 타입 감지"""
        try:
            # 간단한 I/O 벤치마크로 SSD/HDD 구분
            import tempfile
            import random
            
            test_data = bytes([random.randint(0, 255) for _ in range(1024 * 1024)])  # 1MB
            
            with tempfile.NamedTemporaryFile() as f:
                start_time = time.time()
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
                write_time = time.time() - start_time
            
            # 1MB 쓰기 시간으로 판단 (대략적)
            if write_time < 0.01:  # 10ms 미만
                return "nvme"
            elif write_time < 0.05:  # 50ms 미만
                return "ssd"
            else:
                return "hdd"
                
        except Exception:
            return "ssd"  # 기본값
    
    def _load_stage_presets(self) -> Dict[str, Dict]:
        """Stage별 최적화 프리셋 로드"""
        return {
            "stage_1": {
                "max_batch_size": 64,
                "base_learning_rate": 1e-3,
                "memory_target": 0.7,  # GPU 메모리의 70%
                "dataloader_workers": 4,
                "mixed_precision": True
            },
            "stage_2": {
                "max_batch_size": 32,
                "base_learning_rate": 5e-4,
                "memory_target": 0.8,  # GPU 메모리의 80%
                "dataloader_workers": 6,
                "mixed_precision": True
            },
            "stage_3": {
                "max_batch_size": 16,
                "base_learning_rate": 3e-4,
                "memory_target": 0.85,  # GPU 메모리의 85%
                "dataloader_workers": 8,
                "mixed_precision": True
            },
            "stage_4": {
                "max_batch_size": 8,
                "base_learning_rate": 1e-4,
                "memory_target": 0.9,  # GPU 메모리의 90%
                "dataloader_workers": 12,
                "mixed_precision": True
            }
        }
    
    def analyze_current_performance(self, metrics: TrainingMetrics) -> OptimizationReport:
        """현재 성능 분석 및 최적화 권고 생성"""
        self.logger.info("성능 분석 시작...")
        
        # 메트릭 히스토리에 추가
        self.metrics_history.append(metrics)
        
        # 병목 지점 식별
        bottleneck_type, bottleneck_severity = self._identify_bottleneck(metrics)
        
        # 최적화 권고사항 생성
        recommendations = self._generate_recommendations(metrics, bottleneck_type)
        
        # 전체 최적화 점수 계산
        overall_score = self._calculate_optimization_score(metrics)
        
        # 예상 속도 향상 계산
        estimated_speedup = self._estimate_speedup(recommendations)
        
        report = OptimizationReport(
            timestamp=time.time(),
            hardware_profile=self.hardware_profile,
            current_metrics=metrics,
            bottleneck_type=bottleneck_type,
            bottleneck_severity=bottleneck_severity,
            recommendations=recommendations,
            overall_score=overall_score,
            estimated_speedup=estimated_speedup
        )
        
        # 히스토리에 추가
        self.recommendations_history.append(report)
        
        self.logger.info(f"성능 분석 완료 - 점수: {overall_score:.2f}, 예상 향상: {estimated_speedup:.1f}x")
        
        return report
    
    def _identify_bottleneck(self, metrics: TrainingMetrics) -> Tuple[BottleneckType, float]:
        """성능 병목 지점 식별"""
        # GPU 메모리 부족 확인
        memory_usage_ratio = metrics.gpu_memory_usage_gb / self.hardware_profile.gpu_memory_gb
        if memory_usage_ratio > 0.9:
            return BottleneckType.GPU_MEMORY, min(1.0, (memory_usage_ratio - 0.9) * 10)
        
        # GPU 연산 능력 부족 확인
        if metrics.gpu_utilization < 80.0 and metrics.cpu_utilization < 50.0:
            # GPU와 CPU 모두 여유있으면 I/O 병목 의심
            if metrics.io_wait_percent > 20.0:
                return BottleneckType.IO_BOUND, min(1.0, metrics.io_wait_percent / 50.0)
        
        # GPU 연산 병목 확인
        if metrics.gpu_utilization > 95.0:
            return BottleneckType.GPU_COMPUTE, min(1.0, (metrics.gpu_utilization - 95.0) / 5.0)
        
        # CPU 병목 확인
        if metrics.cpu_utilization > 90.0:
            return BottleneckType.CPU_BOUND, min(1.0, (metrics.cpu_utilization - 90.0) / 10.0)
        
        return BottleneckType.NONE, 0.0
    
    def _generate_recommendations(self, 
                                 metrics: TrainingMetrics, 
                                 bottleneck_type: BottleneckType) -> List[OptimizationRecommendation]:
        """최적화 권고사항 생성"""
        recommendations = []
        
        # 배치 크기 최적화
        batch_rec = self._recommend_batch_size(metrics, bottleneck_type)
        if batch_rec:
            recommendations.append(batch_rec)
        
        # 학습률 최적화
        lr_rec = self._recommend_learning_rate(metrics)
        if lr_rec:
            recommendations.append(lr_rec)
        
        # DataLoader 워커 수 최적화
        worker_rec = self._recommend_dataloader_workers(metrics, bottleneck_type)
        if worker_rec:
            recommendations.append(worker_rec)
        
        # 메모리 최적화
        memory_rec = self._recommend_memory_optimizations(metrics, bottleneck_type)
        if memory_rec:
            recommendations.extend(memory_rec)
        
        # 우선순위 정렬
        recommendations.sort(key=lambda x: x.implementation_priority)
        
        return recommendations
    
    def _recommend_batch_size(self, 
                             metrics: TrainingMetrics, 
                             bottleneck_type: BottleneckType) -> Optional[OptimizationRecommendation]:
        """배치 크기 권고"""
        current_batch = metrics.batch_size
        memory_ratio = metrics.gpu_memory_usage_gb / self.hardware_profile.gpu_memory_gb
        
        # GPU 메모리 부족인 경우
        if bottleneck_type == BottleneckType.GPU_MEMORY:
            recommended_batch = max(1, int(current_batch * 0.75))
            if recommended_batch != current_batch:
                return OptimizationRecommendation(
                    category="batch_size",
                    current_value=current_batch,
                    recommended_value=recommended_batch,
                    expected_improvement=15.0,
                    confidence=0.9,
                    reasoning="GPU 메모리 부족으로 배치 크기 감소 필요",
                    risk_level="low",
                    implementation_priority=1
                )
        
        # 메모리 여유가 있고 GPU 활용도가 낮은 경우
        elif memory_ratio < 0.7 and metrics.gpu_utilization < 80.0:
            recommended_batch = min(64, int(current_batch * 1.25))
            if recommended_batch != current_batch:
                return OptimizationRecommendation(
                    category="batch_size",
                    current_value=current_batch,
                    recommended_value=recommended_batch,
                    expected_improvement=10.0,
                    confidence=0.7,
                    reasoning="메모리 여유로 배치 크기 증가하여 GPU 활용도 향상",
                    risk_level="medium",
                    implementation_priority=3
                )
        
        return None
    
    def _recommend_learning_rate(self, metrics: TrainingMetrics) -> Optional[OptimizationRecommendation]:
        """학습률 권고"""
        # 간단한 휴리스틱 기반 학습률 조정
        # 실제로는 더 복잡한 로직이 필요
        if len(self.metrics_history) < 3:
            return None
        
        # 최근 3 epoch의 성능 추이 분석
        recent_metrics = self.metrics_history[-3:]
        if all(m.training_loss for m in recent_metrics):
            losses = [m.training_loss for m in recent_metrics]
            
            # Loss가 정체되어 있으면 학습률 증가
            if abs(losses[-1] - losses[0]) < 0.001:
                recommended_lr = metrics.learning_rate * 1.2
                return OptimizationRecommendation(
                    category="learning_rate",
                    current_value=metrics.learning_rate,
                    recommended_value=recommended_lr,
                    expected_improvement=8.0,
                    confidence=0.6,
                    reasoning="학습 정체로 학습률 증가 필요",
                    risk_level="medium",
                    implementation_priority=4
                )
        
        return None
    
    def _recommend_dataloader_workers(self, 
                                    metrics: TrainingMetrics,
                                    bottleneck_type: BottleneckType) -> Optional[OptimizationRecommendation]:
        """DataLoader 워커 수 권고"""
        if bottleneck_type == BottleneckType.IO_BOUND:
            # I/O 병목이면 워커 수 증가
            recommended_workers = min(self.hardware_profile.cpu_cores, 12)
            return OptimizationRecommendation(
                category="dataloader_workers",
                current_value="unknown",  # 현재값을 알 수 없음
                recommended_value=recommended_workers,
                expected_improvement=20.0,
                confidence=0.8,
                reasoning="I/O 병목으로 DataLoader 워커 수 증가 필요",
                risk_level="low",
                implementation_priority=2
            )
        
        return None
    
    def _recommend_memory_optimizations(self, 
                                       metrics: TrainingMetrics,
                                       bottleneck_type: BottleneckType) -> List[OptimizationRecommendation]:
        """메모리 최적화 권고"""
        recommendations = []
        
        if bottleneck_type == BottleneckType.GPU_MEMORY:
            # Gradient Checkpointing 권고
            recommendations.append(OptimizationRecommendation(
                category="gradient_checkpointing",
                current_value=False,
                recommended_value=True,
                expected_improvement=25.0,
                confidence=0.9,
                reasoning="메모리 부족으로 Gradient Checkpointing 활성화",
                risk_level="low",
                implementation_priority=2
            ))
            
            # Mixed Precision 권고
            recommendations.append(OptimizationRecommendation(
                category="mixed_precision",
                current_value=False,
                recommended_value=True,
                expected_improvement=30.0,
                confidence=0.95,
                reasoning="메모리 절약을 위한 FP16 Mixed Precision 활성화",
                risk_level="low",
                implementation_priority=1
            ))
        
        return recommendations
    
    def _calculate_optimization_score(self, metrics: TrainingMetrics) -> float:
        """전체 최적화 점수 계산 (0.0-1.0)"""
        score = 0.0
        weights = {
            'gpu_utilization': 0.3,
            'memory_efficiency': 0.25,
            'throughput': 0.25,
            'cpu_utilization': 0.2
        }
        
        # GPU 활용도 점수 (80% 이상이면 만점)
        gpu_score = min(1.0, metrics.gpu_utilization / 80.0)
        score += weights['gpu_utilization'] * gpu_score
        
        # 메모리 효율성 점수 (85% 사용이면 만점)
        memory_ratio = metrics.gpu_memory_usage_gb / self.hardware_profile.gpu_memory_gb
        memory_score = min(1.0, memory_ratio / 0.85) * (1.0 - max(0, memory_ratio - 0.9) * 10)
        score += weights['memory_efficiency'] * memory_score
        
        # 처리량 점수 (상대적 평가)
        throughput_score = min(1.0, metrics.samples_per_second / 100.0)  # 100 samples/sec이 만점
        score += weights['throughput'] * throughput_score
        
        # CPU 활용도 점수 (50% 정도가 적정)
        cpu_score = 1.0 - abs(metrics.cpu_utilization - 50.0) / 50.0
        score += weights['cpu_utilization'] * max(0.0, cpu_score)
        
        return min(1.0, max(0.0, score))
    
    def _estimate_speedup(self, recommendations: List[OptimizationRecommendation]) -> float:
        """권고사항 적용 시 예상 속도 향상 계산"""
        total_improvement = 1.0
        
        for rec in recommendations:
            # 각 권고사항의 기대 효과를 곱셈으로 누적
            improvement_factor = 1.0 + (rec.expected_improvement / 100.0) * rec.confidence
            total_improvement *= improvement_factor
        
        return total_improvement
    
    def get_stage_recommendations(self, stage: str) -> Dict[str, Any]:
        """특정 Stage에 대한 권고사항 반환"""
        if stage not in self.stage_presets:
            stage = "stage_3"  # 기본값
        
        preset = self.stage_presets[stage].copy()
        
        # 하드웨어 기반 조정
        if self.hardware_profile.gpu_memory_gb < 16.0:
            preset["max_batch_size"] = max(1, preset["max_batch_size"] // 2)
            preset["memory_target"] = min(0.8, preset["memory_target"])
        
        if self.hardware_profile.cpu_cores < 8:
            preset["dataloader_workers"] = min(4, preset["dataloader_workers"])
        
        return preset
    
    def export_report(self, report: OptimizationReport, file_path: str):
        """최적화 보고서를 파일로 내보내기"""
        try:
            # dataclass를 dict로 변환
            report_dict = asdict(report)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"최적화 보고서 저장: {file_path}")
            
        except Exception as e:
            self.logger.error(f"보고서 저장 실패: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """최적화 현황 요약"""
        if not self.recommendations_history:
            return {"status": "no_data"}
        
        latest_report = self.recommendations_history[-1]
        
        return {
            "current_score": latest_report.overall_score,
            "bottleneck": latest_report.bottleneck_type.value,
            "bottleneck_severity": latest_report.bottleneck_severity,
            "total_recommendations": len(latest_report.recommendations),
            "high_priority_recommendations": len([r for r in latest_report.recommendations if r.implementation_priority <= 2]),
            "estimated_speedup": latest_report.estimated_speedup,
            "hardware_profile": {
                "gpu": latest_report.hardware_profile.gpu_name,
                "gpu_memory": f"{latest_report.hardware_profile.gpu_memory_gb:.1f}GB",
                "cpu_cores": latest_report.hardware_profile.cpu_cores
            }
        }


# 편의 함수들
def create_rtx5080_advisor(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> OptimizationAdvisor:
    """RTX 5080 환경에 최적화된 Advisor 생성"""
    return OptimizationAdvisor(
        optimization_level=optimization_level,
        enable_auto_tuning=False,  # 안전을 위해 수동 모드
        monitoring_interval=10.0
    )


def quick_performance_check(batch_size: int, 
                          learning_rate: float, 
                          gpu_memory_gb: float,
                          samples_per_second: float) -> OptimizationReport:
    """빠른 성능 체크"""
    advisor = create_rtx5080_advisor()
    
    # 간단한 메트릭 생성
    metrics = TrainingMetrics(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epoch_time_seconds=60.0,
        samples_per_second=samples_per_second,
        gpu_utilization=75.0,
        gpu_memory_usage_gb=gpu_memory_gb,
        cpu_utilization=45.0,
        io_wait_percent=10.0
    )
    
    return advisor.analyze_current_performance(metrics)