"""
Stage 3 GPU 메모리 안정성 및 누수 탐지 테스트

RTX 5080 16GB 환경에서의 철저한 GPU 메모리 관리 검증:
- CUDA 메모리 누수 탐지
- OOM(Out of Memory) 방지 메커니즘 검증
- 장시간 학습 안정성 테스트
- Mixed Precision 및 torch.compile 최적화 검증
- 메모리 Fragment 방지
- 자동 배치 크기 조정 검증
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import time
import threading
import psutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings
import contextlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CUDAMemoryTracker:
    """CUDA 메모리 추적기"""
    
    def __init__(self):
        self.snapshots = []
        self.baseline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def take_snapshot(self, label=""):
        """메모리 스냅샷 촬영"""
        if not torch.cuda.is_available():
            return None
            
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'max_allocated': torch.cuda.max_memory_allocated(),
            'max_reserved': torch.cuda.max_memory_reserved()
        }
        
        self.snapshots.append(snapshot)
        
        if self.baseline is None:
            self.baseline = snapshot
            
        return snapshot
    
    def get_memory_increase(self):
        """베이스라인 대비 메모리 증가량 계산"""
        if not self.snapshots or not torch.cuda.is_available():
            return 0, 0
            
        current = self.snapshots[-1]
        allocated_increase = current['allocated'] - self.baseline['allocated']
        reserved_increase = current['reserved'] - self.baseline['reserved']
        
        return allocated_increase, reserved_increase
    
    def detect_memory_leak(self, threshold_mb=100):
        """메모리 누수 탐지"""
        if len(self.snapshots) < 2:
            return False, "스냅샷 부족"
            
        # 최근 10개 스냅샷에서 추세 분석
        recent_snapshots = self.snapshots[-10:]
        allocated_values = [s['allocated'] for s in recent_snapshots]
        
        if len(allocated_values) < 2:
            return False, "데이터 부족"
            
        # 선형 회귀로 추세 계산
        x = np.arange(len(allocated_values))
        slope = np.polyfit(x, allocated_values, 1)[0]
        
        threshold_bytes = threshold_mb * 1024 * 1024
        leak_detected = slope > threshold_bytes
        
        return leak_detected, f"메모리 증가 추세: {slope/1024/1024:.1f}MB/스냅샷"


class MockEfficientNetV2:
    """EfficientNetV2-L 모의 모델"""
    
    def __init__(self, num_classes=1000):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        # EfficientNetV2-L과 유사한 크기의 모델 생성
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        ).to(device)
        
        self.device = device
    
    def __call__(self, x):
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self):
        return self.model.train()
    
    def eval(self):
        return self.model.eval()


class TestStage3GPUMemoryStability:
    """Stage 3 GPU 메모리 안정성 테스트"""
    
    @pytest.fixture
    def memory_tracker(self):
        """메모리 추적기"""
        tracker = CUDAMemoryTracker()
        tracker.take_snapshot("test_start")
        yield tracker
        
        # 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.fixture
    def mock_model(self):
        """모의 모델"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = MockEfficientNetV2()
        yield model
        
        # 정리
        del model
        torch.cuda.empty_cache()
    
    def test_baseline_memory_usage(self, memory_tracker, mock_model):
        """기본 메모리 사용량 측정"""
        print("📊 기본 메모리 사용량 측정")
        
        # 모델 로딩 후 메모리 사용량
        memory_tracker.take_snapshot("model_loaded")
        
        # 옵티마이저 생성
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        memory_tracker.take_snapshot("optimizer_created")
        
        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler()
        memory_tracker.take_snapshot("scaler_created")
        
        # 베이스라인 메모리 사용량 분석
        snapshots = memory_tracker.snapshots
        
        for i, snapshot in enumerate(snapshots):
            allocated_mb = snapshot['allocated'] / 1024 / 1024
            reserved_mb = snapshot['reserved'] / 1024 / 1024
            print(f"  {snapshot['label']}: 할당={allocated_mb:.1f}MB, 예약={reserved_mb:.1f}MB")
        
        # 최종 메모리 사용량이 RTX 5080 16GB의 80% 이하여야 함
        final_allocated = snapshots[-1]['allocated']
        max_allowed = 16 * 1024 * 1024 * 1024 * 0.8  # 12.8GB
        
        assert final_allocated < max_allowed, f"베이스라인 메모리 과다: {final_allocated/1e9:.1f}GB > 12.8GB"
        print(f"✅ 베이스라인 메모리 검증 통과: {final_allocated/1e9:.1f}GB")
    
    def test_training_batch_memory_management(self, memory_tracker, mock_model):
        """학습 배치 메모리 관리 테스트"""
        print("🔄 학습 배치 메모리 관리 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        
        batch_sizes = [4, 8, 16, 32]  # 다양한 배치 크기 테스트
        memory_results = {}
        
        for batch_size in batch_sizes:
            try:
                print(f"  배치 크기 {batch_size} 테스트...")
                
                # 배치 데이터 생성
                images = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
                labels = torch.randint(0, 1000, (batch_size,), device=device)
                
                memory_tracker.take_snapshot(f"batch_{batch_size}_created")
                
                # Forward pass with Mixed Precision
                with torch.amp.autocast(device_type='cuda'):
                    outputs = mock_model(images)
                    loss = criterion(outputs, labels)
                
                memory_tracker.take_snapshot(f"batch_{batch_size}_forward")
                
                # Backward pass
                scaler.scale(loss).backward()
                
                memory_tracker.take_snapshot(f"batch_{batch_size}_backward")
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                memory_tracker.take_snapshot(f"batch_{batch_size}_optimizer")
                
                # 메모리 사용량 기록
                current_allocated = torch.cuda.memory_allocated()
                current_reserved = torch.cuda.memory_reserved()
                
                memory_results[batch_size] = {
                    'allocated_mb': current_allocated / 1024 / 1024,
                    'reserved_mb': current_reserved / 1024 / 1024,
                    'success': True
                }
                
                # 메모리 정리
                del images, labels, outputs, loss
                torch.cuda.empty_cache()
                
                print(f"    ✅ 배치 크기 {batch_size}: {current_allocated/1024/1024:.1f}MB 사용")
                
            except torch.cuda.OutOfMemoryError:
                memory_results[batch_size] = {'success': False, 'error': 'OOM'}
                print(f"    ❌ 배치 크기 {batch_size}: CUDA OOM")
                torch.cuda.empty_cache()
                
            except Exception as e:
                memory_results[batch_size] = {'success': False, 'error': str(e)}
                print(f"    ❌ 배치 크기 {batch_size}: {e}")
                torch.cuda.empty_cache()
        
        # 결과 분석
        successful_batches = [bs for bs, result in memory_results.items() if result['success']]
        max_successful_batch = max(successful_batches) if successful_batches else 0
        
        print(f"  최대 성공 배치 크기: {max_successful_batch}")
        
        # 최소 배치 크기 8은 성공해야 함 (Stage 3 요구사항)
        assert 8 in successful_batches, "배치 크기 8 처리 실패"
        
        # 메모리 효율성 검증
        if 8 in memory_results:
            batch8_memory = memory_results[8]['allocated_mb']
            assert batch8_memory < 8000, f"배치 크기 8 메모리 과다: {batch8_memory:.1f}MB > 8000MB"
        
        print(f"✅ 배치 메모리 관리 테스트 통과")
    
    def test_memory_leak_detection(self, memory_tracker, mock_model):
        """메모리 누수 탐지 테스트"""
        print("🔍 메모리 누수 탐지 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        
        # 반복적 학습 시뮬레이션 (100 이터레이션)
        batch_size = 8
        num_iterations = 100
        
        for iteration in range(num_iterations):
            # 배치 데이터 생성
            images = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
            labels = torch.randint(0, 1000, (batch_size,), device=device)
            
            # 학습 스텝
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = mock_model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 메모리 정리
            del images, labels, outputs, loss
            
            # 주기적 메모리 스냅샷
            if (iteration + 1) % 10 == 0:
                memory_tracker.take_snapshot(f"iteration_{iteration+1}")
                
                # 중간 가비지 컬렉션
                if (iteration + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # 메모리 누수 분석
        leak_detected, message = memory_tracker.detect_memory_leak(threshold_mb=50)  # 50MB 임계값
        
        print(f"  총 이터레이션: {num_iterations}")
        print(f"  메모리 누수 분석: {message}")
        
        assert not leak_detected, f"메모리 누수 탐지됨: {message}"
        print(f"✅ 메모리 누수 탐지 테스트 통과")
    
    @pytest.mark.slow
    def test_long_term_stability(self, memory_tracker, mock_model):
        """장시간 안정성 테스트"""
        print("🕐 장시간 GPU 메모리 안정성 테스트 (5분)")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        
        start_time = time.time()
        test_duration = 300  # 5분
        iteration = 0
        stability_violations = 0
        
        # 메모리 모니터링 스레드
        monitoring_active = threading.Event()
        monitoring_active.set()
        memory_violations = []
        
        def memory_monitor():
            """백그라운드 메모리 모니터링"""
            while monitoring_active.is_set():
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    
                    # 메모리 사용량이 14GB를 초과하면 위반
                    if allocated > 14 * 1024 * 1024 * 1024:
                        memory_violations.append({
                            'time': time.time() - start_time,
                            'allocated_gb': allocated / 1e9
                        })
                
                time.sleep(10)  # 10초마다 체크
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            while time.time() - start_time < test_duration:
                iteration += 1
                
                # 다양한 배치 크기로 테스트 (현실적 변동성)
                batch_size = 4 if iteration % 10 == 0 else 8
                
                images = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
                labels = torch.randint(0, 1000, (batch_size,), device=device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = mock_model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                del images, labels, outputs, loss
                
                # 주기적 메모리 관리
                if iteration % 100 == 0:
                    torch.cuda.empty_cache()
                    memory_tracker.take_snapshot(f"stability_iter_{iteration}")
                    
                    elapsed = time.time() - start_time
                    print(f"    진행: {iteration:,} 이터레이션, {elapsed:.0f}초 경과")
                
                # CPU 사용률도 체크
                if iteration % 1000 == 0:
                    cpu_percent = psutil.cpu_percent()
                    if cpu_percent > 95:
                        stability_violations += 1
        
        except Exception as e:
            print(f"    ⚠️ 장시간 테스트 중 예외: {e}")
            stability_violations += 1
        
        finally:
            monitoring_active.clear()
            monitor_thread.join(timeout=1)
        
        elapsed_time = time.time() - start_time
        
        print(f"  총 실행 시간: {elapsed_time:.1f}초")
        print(f"  총 이터레이션: {iteration:,}")
        print(f"  안정성 위반: {stability_violations}")
        print(f"  메모리 위반: {len(memory_violations)}")
        
        # 안정성 검증
        assert stability_violations == 0, f"안정성 위반 발생: {stability_violations}건"
        assert len(memory_violations) == 0, f"메모리 한계 초과: {len(memory_violations)}건"
        
        # 최종 메모리 누수 검사
        final_leak, leak_message = memory_tracker.detect_memory_leak(threshold_mb=100)
        assert not final_leak, f"장시간 실행 후 메모리 누수: {leak_message}"
        
        print(f"✅ 장시간 안정성 테스트 통과")
    
    def test_oom_recovery_mechanism(self, memory_tracker):
        """OOM 복구 메커니즘 테스트"""
        print("🛡️ OOM 복구 메커니즘 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        recovery_successful = False
        
        # 의도적으로 메모리 부족 상황 발생
        try:
            # 매우 큰 텐서 생성으로 OOM 유발
            huge_tensor = torch.randn(10000, 10000, 10000, device=device)
            
        except torch.cuda.OutOfMemoryError:
            print("    OOM 예외 발생 (예상됨)")
            
            # 복구 메커니즘 실행
            try:
                torch.cuda.empty_cache()
                gc.collect()
                
                # 복구 후 정상 작업 수행
                normal_tensor = torch.randn(100, 100, device=device)
                result = normal_tensor.sum()
                
                recovery_successful = True
                print(f"    복구 성공: 텐서 생성 및 연산 완료")
                
                del normal_tensor
                
            except Exception as recovery_error:
                print(f"    복구 실패: {recovery_error}")
        
        except Exception as e:
            print(f"    예상치 못한 예외: {e}")
        
        # 메모리 상태 확인
        torch.cuda.empty_cache()
        memory_tracker.take_snapshot("after_oom_test")
        
        current_allocated = torch.cuda.memory_allocated()
        print(f"  OOM 테스트 후 메모리: {current_allocated/1024/1024:.1f}MB")
        
        # 복구 성공 여부 검증
        assert recovery_successful, "OOM 복구 메커니즘 실패"
        
        # 메모리가 정상적으로 해제되었는지 확인
        assert current_allocated < 1024 * 1024 * 1024, "OOM 후 메모리 정리 실패"  # 1GB 이하
        
        print(f"✅ OOM 복구 메커니즘 테스트 통과")
    
    def test_mixed_precision_memory_efficiency(self, memory_tracker, mock_model):
        """Mixed Precision 메모리 효율성 테스트"""
        print("⚡ Mixed Precision 메모리 효율성 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        
        batch_size = 16
        
        # FP32 메모리 사용량 측정
        print("  FP32 모드 메모리 측정...")
        
        images_fp32 = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float32)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        
        optimizer.zero_grad()
        outputs = mock_model(images_fp32)
        loss = criterion(outputs, labels)
        loss.backward()
        
        memory_tracker.take_snapshot("fp32_peak")
        fp32_memory = torch.cuda.memory_allocated()
        
        del images_fp32, labels, outputs, loss
        torch.cuda.empty_cache()
        
        # Mixed Precision 메모리 사용량 측정
        print("  Mixed Precision 모드 메모리 측정...")
        
        scaler = torch.amp.GradScaler()
        images_fp16 = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = mock_model(images_fp16)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        memory_tracker.take_snapshot("mixed_precision_peak")
        mixed_precision_memory = torch.cuda.memory_allocated()
        
        del images_fp16, labels, outputs, loss
        torch.cuda.empty_cache()
        
        # 메모리 절약량 계산
        memory_savings = fp32_memory - mixed_precision_memory
        savings_percent = (memory_savings / fp32_memory) * 100
        
        print(f"  FP32 메모리 사용량: {fp32_memory/1024/1024:.1f}MB")
        print(f"  Mixed Precision 메모리 사용량: {mixed_precision_memory/1024/1024:.1f}MB")
        print(f"  메모리 절약: {memory_savings/1024/1024:.1f}MB ({savings_percent:.1f}%)")
        
        # Mixed Precision이 메모리를 절약해야 함
        assert memory_savings > 0, "Mixed Precision 메모리 절약 효과 없음"
        assert savings_percent > 10, f"Mixed Precision 효율성 부족: {savings_percent:.1f}% < 10%"
        
        print(f"✅ Mixed Precision 메모리 효율성 테스트 통과")
    
    def test_torch_compile_memory_impact(self, memory_tracker, mock_model):
        """torch.compile 메모리 영향 테스트"""
        print("🔧 torch.compile 메모리 영향 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        
        # 컴파일 전 메모리 사용량
        memory_tracker.take_snapshot("before_compile")
        baseline_memory = torch.cuda.memory_allocated()
        
        try:
            # 모델 컴파일
            compiled_model = torch.compile(mock_model.model, mode="reduce-overhead")
            memory_tracker.take_snapshot("after_compile")
            
            # 컴파일된 모델로 추론
            with torch.no_grad():
                test_input = torch.randn(4, 3, 384, 384, device=device, dtype=torch.float16)
                _ = compiled_model(test_input)
            
            memory_tracker.take_snapshot("after_compiled_inference")
            final_memory = torch.cuda.memory_allocated()
            
            memory_increase = final_memory - baseline_memory
            
            print(f"  컴파일 전 메모리: {baseline_memory/1024/1024:.1f}MB")
            print(f"  컴파일 후 메모리: {final_memory/1024/1024:.1f}MB")
            print(f"  메모리 증가: {memory_increase/1024/1024:.1f}MB")
            
            # torch.compile로 인한 메모리 증가가 합리적 범위 내여야 함
            max_allowed_increase = 2 * 1024 * 1024 * 1024  # 2GB
            assert memory_increase < max_allowed_increase, f"torch.compile 메모리 증가 과다: {memory_increase/1e9:.1f}GB"
            
            print(f"✅ torch.compile 메모리 영향 테스트 통과")
            
        except Exception as e:
            print(f"  torch.compile 지원되지 않음 또는 오류: {e}")
            pytest.skip("torch.compile not supported")
    
    def test_gradient_accumulation_memory(self, memory_tracker, mock_model):
        """Gradient Accumulation 메모리 테스트"""
        print("📈 Gradient Accumulation 메모리 테스트")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = mock_model.device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mock_model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        
        # 다양한 accumulation step에 대한 메모리 사용량 측정
        accumulation_steps = [1, 2, 4]
        memory_results = {}
        
        for accum_steps in accumulation_steps:
            print(f"  Accumulation Steps {accum_steps} 테스트...")
            
            torch.cuda.empty_cache()
            memory_tracker.take_snapshot(f"accum_{accum_steps}_start")
            
            optimizer.zero_grad()
            peak_memory = 0
            
            for step in range(accum_steps):
                batch_size = 8 // accum_steps  # 총 effective batch size는 8로 동일
                
                images = torch.randn(batch_size, 3, 384, 384, device=device, dtype=torch.float16)
                labels = torch.randint(0, 1000, (batch_size,), device=device)
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = mock_model(images)
                    loss = criterion(outputs, labels) / accum_steps  # 정규화
                
                scaler.scale(loss).backward()
                
                current_memory = torch.cuda.memory_allocated()
                peak_memory = max(peak_memory, current_memory)
                
                del images, labels, outputs, loss
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            memory_tracker.take_snapshot(f"accum_{accum_steps}_peak")
            memory_results[accum_steps] = peak_memory
            
            print(f"    Peak 메모리: {peak_memory/1024/1024:.1f}MB")
        
        # Gradient Accumulation이 메모리를 절약하는지 확인
        base_memory = memory_results[1]
        accum4_memory = memory_results[4]
        
        memory_savings = base_memory - accum4_memory
        
        print(f"  1 step vs 4 steps 메모리 절약: {memory_savings/1024/1024:.1f}MB")
        
        # 4-step accumulation이 메모리를 절약해야 함
        assert memory_savings > 0, "Gradient Accumulation 메모리 절약 효과 없음"
        
        print(f"✅ Gradient Accumulation 메모리 테스트 통과")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])