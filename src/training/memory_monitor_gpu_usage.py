"""
GPU Memory Usage Monitor
GPU 메모리 사용량 실시간 모니터링

RTX 5080 16GB 최적화:
- 실시간 메모리 사용량 추적
- OOM 예방 및 경고 시스템
- 배치 크기 자동 조정 지원
- 메모리 효율성 분석
"""

import time
import gc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import psutil
import GPUtil

from src.utils.core import PillSnapLogger


@dataclass
class MemorySnapshot:
    """메모리 스냅샷 데이터"""
    timestamp: float
    gpu_used_gb: float
    gpu_total_gb: float
    gpu_utilization_percent: float
    system_ram_used_gb: float
    system_ram_total_gb: float
    torch_cache_gb: float
    torch_allocated_gb: float


class GPUMemoryMonitor:
    """GPU 메모리 모니터링 시스템"""
    
    def __init__(self, target_memory_gb: float = 14.0):
        self.target_memory_gb = target_memory_gb
        self.logger = PillSnapLogger(__name__)
        self.snapshots: List[MemorySnapshot] = []
        self.warning_threshold = 0.9  # 90% 사용 시 경고
        self.critical_threshold = 0.95  # 95% 사용 시 위험
        
        # GPU 정보 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if self.gpu_count > 0:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"GPU 모니터링 초기화: {self.gpu_name} ({self.gpu_total_memory:.1f}GB)")
        else:
            self.logger.warning("CUDA GPU를 찾을 수 없음 - CPU 모드")
    
    def get_current_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 조회"""
        try:
            if not torch.cuda.is_available():
                return self._get_cpu_memory_stats()
            
            # GPU 메모리 정보
            gpu_stats = torch.cuda.memory_stats(self.device)
            allocated_gb = gpu_stats['allocated_bytes.all.current'] / (1024**3)
            cached_gb = gpu_stats['reserved_bytes.all.current'] / (1024**3)
            
            # GPU 사용률 (GPUtil 사용)
            try:
                gpus = GPUtil.getGPUs()
                gpu = gpus[0] if gpus else None
                gpu_used_gb = gpu.memoryUsed / 1024 if gpu else allocated_gb
                gpu_total_gb = gpu.memoryTotal / 1024 if gpu else self.gpu_total_memory
                gpu_utilization = gpu.load * 100 if gpu else 0.0
            except:
                gpu_used_gb = allocated_gb
                gpu_total_gb = self.gpu_total_memory
                gpu_utilization = 0.0
            
            # 시스템 RAM 정보
            system_memory = psutil.virtual_memory()
            system_used_gb = (system_memory.total - system_memory.available) / (1024**3)
            system_total_gb = system_memory.total / (1024**3)
            
            # 스냅샷 생성
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                gpu_used_gb=gpu_used_gb,
                gpu_total_gb=gpu_total_gb,
                gpu_utilization_percent=gpu_utilization,
                system_ram_used_gb=system_used_gb,
                system_ram_total_gb=system_total_gb,
                torch_cache_gb=cached_gb,
                torch_allocated_gb=allocated_gb
            )
            
            # 스냅샷 저장 (최근 100개만)
            self.snapshots.append(snapshot)
            if len(self.snapshots) > 100:
                self.snapshots.pop(0)
            
            # 메모리 사용량 검사
            self._check_memory_thresholds(snapshot)
            
            return {
                'used_gb': gpu_used_gb,
                'total_gb': gpu_total_gb,
                'utilization_percent': gpu_utilization,
                'allocated_gb': allocated_gb,
                'cached_gb': cached_gb,
                'usage_ratio': gpu_used_gb / gpu_total_gb if gpu_total_gb > 0 else 0,
                'system_ram_used_gb': system_used_gb,
                'system_ram_total_gb': system_total_gb
            }
            
        except Exception as e:
            self.logger.warning(f"메모리 사용량 조회 실패: {e}")
            return self._get_fallback_stats()
    
    def _get_cpu_memory_stats(self) -> Dict[str, float]:
        """CPU 모드 메모리 통계"""
        system_memory = psutil.virtual_memory()
        return {
            'used_gb': 0.0,
            'total_gb': 0.0,
            'utilization_percent': 0.0,
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'usage_ratio': 0.0,
            'system_ram_used_gb': (system_memory.total - system_memory.available) / (1024**3),
            'system_ram_total_gb': system_memory.total / (1024**3)
        }
    
    def _get_fallback_stats(self) -> Dict[str, float]:
        """폴백 통계 (에러 시)"""
        return {
            'used_gb': 0.0,
            'total_gb': 16.0,  # RTX 5080 기본값
            'utilization_percent': 0.0,
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'usage_ratio': 0.0,
            'system_ram_used_gb': 0.0,
            'system_ram_total_gb': 128.0  # 시스템 기본값
        }
    
    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """메모리 임계값 검사 및 경고"""
        usage_ratio = snapshot.gpu_used_gb / snapshot.gpu_total_gb
        
        if usage_ratio >= self.critical_threshold:
            self.logger.error(f"🚨 GPU 메모리 위험 수준: {usage_ratio:.1%} "
                             f"({snapshot.gpu_used_gb:.1f}GB/{snapshot.gpu_total_gb:.1f}GB)")
            self.logger.error("즉시 배치 크기 감소 또는 메모리 정리 필요!")
            
        elif usage_ratio >= self.warning_threshold:
            self.logger.warning(f"⚠️ GPU 메모리 경고 수준: {usage_ratio:.1%} "
                               f"({snapshot.gpu_used_gb:.1f}GB/{snapshot.gpu_total_gb:.1f}GB)")
            self.logger.warning("배치 크기 조정 권장")
        
        # 목표 메모리 초과 확인
        if snapshot.gpu_used_gb > self.target_memory_gb:
            self.logger.warning(f"목표 메모리 초과: {snapshot.gpu_used_gb:.1f}GB > {self.target_memory_gb:.1f}GB")
    
    def suggest_optimal_batch_size(self, current_batch_size: int, model_memory_estimate: float) -> int:
        """최적 배치 크기 제안"""
        try:
            current_stats = self.get_current_usage()
            available_memory = current_stats['total_gb'] - current_stats['used_gb']
            
            # 안전 마진 (20% 여유 공간)
            safe_memory = available_memory * 0.8
            
            # 배치당 메모리 사용량 추정
            memory_per_batch = model_memory_estimate
            max_safe_batches = int(safe_memory / memory_per_batch)
            
            # 최소 1, 최대 현재 배치 크기의 2배
            suggested_batch_size = max(1, min(max_safe_batches, current_batch_size * 2))
            
            if suggested_batch_size != current_batch_size:
                self.logger.info(f"배치 크기 제안: {current_batch_size} → {suggested_batch_size}")
                self.logger.info(f"근거: 사용 가능 메모리 {available_memory:.1f}GB, "
                                f"배치당 {memory_per_batch:.1f}GB")
            
            return suggested_batch_size
            
        except Exception as e:
            self.logger.warning(f"배치 크기 최적화 실패: {e}")
            return current_batch_size
    
    def force_memory_cleanup(self) -> Dict[str, float]:
        """강제 메모리 정리"""
        self.logger.info("GPU 메모리 강제 정리 시작...")
        
        try:
            # 사전 상태 기록
            before_stats = self.get_current_usage()
            
            # PyTorch 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Python 가비지 컬렉션
            gc.collect()
            
            # 사후 상태 확인
            time.sleep(0.5)  # 정리 완료 대기
            after_stats = self.get_current_usage()
            
            freed_memory = before_stats['used_gb'] - after_stats['used_gb']
            
            if freed_memory > 0.1:  # 100MB 이상 정리됨
                self.logger.success(f"메모리 정리 완료: {freed_memory:.1f}GB 확보")
            else:
                self.logger.info("메모리 정리 완료 (변화 미미)")
            
            return {
                'before_used_gb': before_stats['used_gb'],
                'after_used_gb': after_stats['used_gb'],
                'freed_gb': freed_memory
            }
            
        except Exception as e:
            self.logger.error(f"메모리 정리 실패: {e}")
            return {'freed_gb': 0.0}
    
    def get_memory_efficiency_report(self) -> Dict[str, float]:
        """메모리 효율성 리포트"""
        if len(self.snapshots) < 10:
            return {"error": "충분한 데이터 없음 (최소 10개 스냅샷 필요)"}
        
        try:
            recent_snapshots = self.snapshots[-10:]  # 최근 10개
            
            # 평균 사용량
            avg_usage = sum(s.gpu_used_gb for s in recent_snapshots) / len(recent_snapshots)
            
            # 최대/최소 사용량
            max_usage = max(s.gpu_used_gb for s in recent_snapshots)
            min_usage = min(s.gpu_used_gb for s in recent_snapshots)
            
            # 사용량 변동성
            usage_variance = sum((s.gpu_used_gb - avg_usage) ** 2 for s in recent_snapshots) / len(recent_snapshots)
            
            # 목표 메모리 대비 효율성
            target_efficiency = avg_usage / self.target_memory_gb if self.target_memory_gb > 0 else 0
            
            # 전체 메모리 대비 효율성
            total_memory = recent_snapshots[0].gpu_total_gb
            total_efficiency = avg_usage / total_memory if total_memory > 0 else 0
            
            report = {
                'average_usage_gb': avg_usage,
                'max_usage_gb': max_usage,
                'min_usage_gb': min_usage,
                'usage_variance': usage_variance,
                'target_efficiency_ratio': target_efficiency,
                'total_efficiency_ratio': total_efficiency,
                'stability_score': 1.0 - (usage_variance / avg_usage) if avg_usage > 0 else 0
            }
            
            # 효율성 평가
            if target_efficiency < 0.7:
                efficiency_status = "우수 (여유 충분)"
            elif target_efficiency < 0.9:
                efficiency_status = "양호 (적정 수준)"
            elif target_efficiency < 1.0:
                efficiency_status = "주의 (목표 근접)"
            else:
                efficiency_status = "위험 (목표 초과)"
            
            report['efficiency_status'] = efficiency_status
            
            return report
            
        except Exception as e:
            self.logger.error(f"효율성 리포트 생성 실패: {e}")
            return {"error": str(e)}
    
    def save_monitoring_report(self) -> str:
        """모니터링 리포트 저장"""
        try:
            report_dir = Path("artifacts/reports/performance_benchmark_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"gpu_memory_monitoring_{timestamp}.json"
            
            # 리포트 데이터 생성
            efficiency_report = self.get_memory_efficiency_report()
            current_stats = self.get_current_usage()
            
            report_data = {
                'timestamp': timestamp,
                'gpu_name': getattr(self, 'gpu_name', 'Unknown'),
                'target_memory_gb': self.target_memory_gb,
                'current_stats': current_stats,
                'efficiency_report': efficiency_report,
                'snapshot_count': len(self.snapshots),
                'recent_snapshots': [
                    {
                        'timestamp': s.timestamp,
                        'gpu_used_gb': s.gpu_used_gb,
                        'gpu_utilization_percent': s.gpu_utilization_percent
                    } for s in self.snapshots[-20:]  # 최근 20개
                ]
            }
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"메모리 모니터링 리포트 저장: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
            return ""


def main():
    """GPU 메모리 모니터 테스트"""
    print("🔍 GPU Memory Monitor Test")
    print("=" * 50)
    
    monitor = GPUMemoryMonitor(target_memory_gb=14.0)
    
    # 현재 상태 확인
    stats = monitor.get_current_usage()
    print(f"GPU 메모리: {stats['used_gb']:.1f}GB / {stats['total_gb']:.1f}GB ({stats['usage_ratio']:.1%})")
    print(f"시스템 RAM: {stats['system_ram_used_gb']:.1f}GB / {stats['system_ram_total_gb']:.1f}GB")
    
    # 효율성 리포트 (10초간 모니터링)
    print("\n10초간 메모리 사용량 모니터링...")
    for i in range(10):
        monitor.get_current_usage()
        time.sleep(1)
    
    efficiency_report = monitor.get_memory_efficiency_report()
    if 'error' not in efficiency_report:
        print(f"\n📊 효율성 분석:")
        print(f"  평균 사용량: {efficiency_report['average_usage_gb']:.1f}GB")
        print(f"  효율성 상태: {efficiency_report['efficiency_status']}")
        print(f"  안정성 점수: {efficiency_report['stability_score']:.3f}")
    
    # 리포트 저장
    report_file = monitor.save_monitoring_report()
    if report_file:
        print(f"\n💾 리포트 저장됨: {report_file}")
    
    print("\n✅ GPU 메모리 모니터 테스트 완료")


if __name__ == "__main__":
    main()