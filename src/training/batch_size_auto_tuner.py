"""
Batch Size Auto Tuner
배치 크기 자동 조정 시스템

RTX 5080 16GB GPU 최적화:
- OOM 방지 자동 배치 크기 조정
- 메모리 사용량 기반 동적 튜닝
- 처리량 최적화 (throughput optimization)
"""

import time
import torch
import gc
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass

from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor
from src.utils.core import PillSnapLogger


@dataclass
class BatchSizeTuningResult:
    """배치 크기 튜닝 결과"""
    optimal_batch_size: int
    max_tested_batch_size: int
    memory_usage_at_optimal: float
    throughput_samples_per_sec: float
    tuning_time_seconds: float
    oom_threshold_batch_size: Optional[int] = None


class BatchSizeAutoTuner:
    """배치 크기 자동 조정기"""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        memory_threshold_gb: float = 14.0,
        safety_margin: float = 0.1  # 10% 안전 마진
    ):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold_gb = memory_threshold_gb
        self.safety_margin = safety_margin
        self.logger = PillSnapLogger(__name__)
        
        # 메모리 모니터
        self.memory_monitor = GPUMemoryMonitor(target_memory_gb=memory_threshold_gb)
        
        # 튜닝 히스토리
        self.tuning_history = []
        
        self.logger.info(f"BatchSizeAutoTuner 초기화")
        self.logger.info(f"  초기 배치: {initial_batch_size}")
        self.logger.info(f"  최대 배치: {max_batch_size}")
        self.logger.info(f"  메모리 임계: {memory_threshold_gb:.1f}GB")
    
    def find_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_batch_generator: Callable[[int], Tuple[torch.Tensor, ...]],
        device: torch.device = None
    ) -> BatchSizeTuningResult:
        """최적 배치 크기 탐색"""
        
        self.logger.step("배치 크기 최적화", "이진 탐색으로 최적 배치 크기 탐색")
        
        if device is None:
            device = next(model.parameters()).device
        
        start_time = time.time()
        
        try:
            # 초기 메모리 정리
            self._clear_gpu_memory()
            
            # 이진 탐색으로 최대 배치 크기 찾기
            max_working_batch = self._binary_search_max_batch(
                model, sample_batch_generator, device
            )
            
            # 처리량 기반 최적화
            optimal_batch = self._optimize_for_throughput(
                model, sample_batch_generator, device, max_working_batch
            )
            
            # 최적 배치에서 최종 메트릭 측정
            final_metrics = self._measure_batch_performance(
                model, sample_batch_generator, device, optimal_batch
            )
            
            tuning_time = time.time() - start_time
            
            result = BatchSizeTuningResult(
                optimal_batch_size=optimal_batch,
                max_tested_batch_size=max_working_batch,
                memory_usage_at_optimal=final_metrics['memory_usage_gb'],
                throughput_samples_per_sec=final_metrics['throughput'],
                tuning_time_seconds=tuning_time,
                oom_threshold_batch_size=max_working_batch + 16 if max_working_batch < self.max_batch_size else None
            )
            
            self.logger.success(f"배치 크기 최적화 완료")
            self.logger.info(f"  최적 배치: {optimal_batch}")
            self.logger.info(f"  메모리 사용: {final_metrics['memory_usage_gb']:.1f}GB")
            self.logger.info(f"  처리량: {final_metrics['throughput']:.1f} samples/sec")
            
            return result
            
        except Exception as e:
            self.logger.error(f"배치 크기 최적화 실패: {e}")
            raise
    
    def _binary_search_max_batch(
        self,
        model: torch.nn.Module,
        sample_batch_generator: Callable[[int], Tuple[torch.Tensor, ...]],
        device: torch.device
    ) -> int:
        """이진 탐색으로 최대 배치 크기 찾기"""
        
        low = 1
        high = self.max_batch_size
        max_working_batch = self.initial_batch_size
        
        self.logger.info(f"이진 탐색 시작: [{low}, {high}]")
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # 메모리 정리
                self._clear_gpu_memory()
                
                # 테스트 실행
                success, memory_used = self._test_batch_size(
                    model, sample_batch_generator, device, mid
                )
                
                if success and memory_used <= self.memory_threshold_gb * (1 - self.safety_margin):
                    max_working_batch = mid
                    low = mid + 1
                    self.logger.info(f"  배치 {mid}: ✅ 성공 ({memory_used:.1f}GB)")
                else:
                    high = mid - 1
                    if not success:
                        self.logger.info(f"  배치 {mid}: ❌ OOM")
                    else:
                        self.logger.info(f"  배치 {mid}: ⚠️ 메모리 임계 초과 ({memory_used:.1f}GB)")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    self.logger.info(f"  배치 {mid}: ❌ OOM")
                else:
                    raise
        
        return max_working_batch
    
    def _test_batch_size(
        self,
        model: torch.nn.Module,
        sample_batch_generator: Callable[[int], Tuple[torch.Tensor, ...]],
        device: torch.device,
        batch_size: int
    ) -> Tuple[bool, float]:
        """특정 배치 크기 테스트"""
        
        try:
            model.train()
            
            # 샘플 배치 생성
            batch_data = sample_batch_generator(batch_size)
            batch_data = tuple(tensor.to(device) for tensor in batch_data)
            
            # Forward pass
            if len(batch_data) == 2:
                inputs, targets = batch_data
                outputs = model(inputs)
                
                # 간단한 손실 계산 (실제 손실 함수는 모델에 따라 다름)
                if hasattr(torch.nn.functional, 'cross_entropy') and targets.dtype == torch.long:
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                else:
                    loss = outputs.mean()  # 더미 손실
            else:
                inputs = batch_data[0]
                outputs = model(inputs)
                loss = outputs.mean()  # 더미 손실
            
            # Backward pass
            loss.backward()
            
            # 메모리 사용량 측정
            memory_stats = self.memory_monitor.get_current_usage()
            memory_used = memory_stats['used_gb']
            
            return True, memory_used
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False, float('inf')
            else:
                raise
        finally:
            # 그래디언트 정리
            model.zero_grad()
    
    def _optimize_for_throughput(
        self,
        model: torch.nn.Module,
        sample_batch_generator: Callable[[int], Tuple[torch.Tensor, ...]],
        device: torch.device,
        max_batch: int
    ) -> int:
        """처리량 기반 최적 배치 크기 찾기"""
        
        self.logger.info(f"처리량 최적화: 최대 배치 {max_batch}에서 시작")
        
        # 테스트할 배치 크기들 (최대 배치에서 시작해서 감소)
        test_batches = []
        current = max_batch
        while current >= 8:  # 최소 8까지 테스트
            test_batches.append(current)
            current = max(current // 2, current - 16)
        
        if test_batches[-1] != 8:
            test_batches.append(8)
        
        best_batch = max_batch
        best_throughput = 0.0
        
        for batch_size in test_batches:
            try:
                self._clear_gpu_memory()
                
                # 처리량 측정
                metrics = self._measure_batch_performance(
                    model, sample_batch_generator, device, batch_size
                )
                
                throughput = metrics['throughput']
                memory_used = metrics['memory_usage_gb']
                
                self.logger.info(f"  배치 {batch_size}: {throughput:.1f} samples/sec, {memory_used:.1f}GB")
                
                # 처리량이 더 좋고 메모리 사용량이 적절한 경우 업데이트
                if throughput > best_throughput and memory_used <= self.memory_threshold_gb:
                    best_throughput = throughput
                    best_batch = batch_size
                
            except Exception as e:
                self.logger.warning(f"배치 {batch_size} 테스트 실패: {e}")
                continue
        
        return best_batch
    
    def _measure_batch_performance(
        self,
        model: torch.nn.Module,
        sample_batch_generator: Callable[[int], Tuple[torch.Tensor, ...]],
        device: torch.device,
        batch_size: int,
        num_iterations: int = 5
    ) -> Dict[str, float]:
        """배치 성능 측정"""
        
        model.train()
        
        # 워밍업
        batch_data = sample_batch_generator(batch_size)
        batch_data = tuple(tensor.to(device) for tensor in batch_data)
        _ = model(batch_data[0])
        
        # 실제 측정
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            batch_data = sample_batch_generator(batch_size)
            batch_data = tuple(tensor.to(device) for tensor in batch_data)
            
            outputs = model(batch_data[0])
            
            # 간단한 백워드 패스
            loss = outputs.mean()
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # 메트릭 계산
        total_time = end_time - start_time
        total_samples = batch_size * num_iterations
        throughput = total_samples / total_time
        
        # 메모리 사용량
        memory_stats = self.memory_monitor.get_current_usage()
        
        return {
            'throughput': throughput,
            'memory_usage_gb': memory_stats['used_gb'],
            'avg_time_per_batch': total_time / num_iterations
        }
    
    def _clear_gpu_memory(self) -> None:
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(0.1)  # 정리 완료 대기
    
    def suggest_batch_size_for_inference(
        self,
        model: torch.nn.Module,
        sample_input_generator: Callable[[int], torch.Tensor],
        target_latency_ms: float = 50.0
    ) -> int:
        """추론용 배치 크기 제안"""
        
        self.logger.step("추론 배치 최적화", f"목표 지연시간 {target_latency_ms}ms")
        
        model.eval()
        device = next(model.parameters()).device
        
        # 다양한 배치 크기 테스트
        test_batches = [1, 2, 4, 8, 16, 32, 64]
        suitable_batches = []
        
        with torch.no_grad():
            for batch_size in test_batches:
                try:
                    self._clear_gpu_memory()
                    
                    # 추론 시간 측정
                    sample_input = sample_input_generator(batch_size).to(device)
                    
                    # 워밍업
                    _ = model(sample_input)
                    
                    # 실제 측정
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    for _ in range(10):
                        _ = model(sample_input)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    avg_time_ms = (end_time - start_time) * 1000 / 10
                    per_sample_ms = avg_time_ms / batch_size
                    
                    self.logger.info(f"  배치 {batch_size}: {avg_time_ms:.1f}ms total, {per_sample_ms:.1f}ms/sample")
                    
                    if per_sample_ms <= target_latency_ms:
                        suitable_batches.append((batch_size, per_sample_ms))
                
                except Exception as e:
                    self.logger.warning(f"추론 배치 {batch_size} 테스트 실패: {e}")
        
        if suitable_batches:
            # 목표 지연시간을 만족하는 가장 큰 배치 크기 선택
            optimal_batch = max(suitable_batches, key=lambda x: x[0])[0]
            self.logger.success(f"추론 최적 배치: {optimal_batch}")
            return optimal_batch
        else:
            self.logger.warning("목표 지연시간을 만족하는 배치 크기 없음, 배치 1 사용")
            return 1


def create_sample_batch_generator(input_shape: Tuple[int, ...], num_classes: int):
    """샘플 배치 생성기 팩토리"""
    
    def generator(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 더미 입력 및 라벨 생성
        inputs = torch.randn(batch_size, *input_shape)
        labels = torch.randint(0, num_classes, (batch_size,))
        return inputs, labels
    
    return generator


def main():
    """배치 크기 자동 조정 테스트"""
    print("🔧 Batch Size Auto Tuner Test")
    print("=" * 50)
    
    try:
        # 더미 모델 생성
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 배치 생성기
        batch_generator = create_sample_batch_generator((784,), 10)
        
        # 자동 튜너 생성
        tuner = BatchSizeAutoTuner(
            initial_batch_size=32,
            max_batch_size=128,
            memory_threshold_gb=14.0
        )
        
        # 최적 배치 크기 찾기
        result = tuner.find_optimal_batch_size(model, batch_generator, device)
        
        print(f"✅ 배치 크기 최적화 완료")
        print(f"   최적 배치: {result.optimal_batch_size}")
        print(f"   최대 테스트 배치: {result.max_tested_batch_size}")
        print(f"   메모리 사용: {result.memory_usage_at_optimal:.1f}GB")
        print(f"   처리량: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   튜닝 시간: {result.tuning_time_seconds:.1f}초")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()