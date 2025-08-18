# PillSnap ML 구현 가이드

## 📋 개요

이 문서는 WORK ORDER에 따라 구현된 핵심 컴포넌트들의 사용법과 통합 가이드를 제공합니다.

## 🏗️ 구현 완료 컴포넌트

### 1. Pipeline Mode Resolver (파이프라인 모드 단일 소스)

**위치**: `/home/max16/pillsnap/src/core/pipeline_mode.py`

**목적**: 파이프라인 모드 결정을 위한 단일 진실 소스 제공

**사용법**:
```python
from src.core.pipeline_mode import get_pipeline_resolver

# 초기화 (첫 호출 시만 config 필요)
resolver = get_pipeline_resolver(config["data"])

# 모드 결정
mode, reason = resolver.resolve_mode(user_mode="single")
# Returns: ("single", "user_explicit_selection_single")

# 모드 요구사항 확인
requirements = resolver.get_mode_requirements("single")
# Returns: {"models_required": ["classification"], ...}
```

**핵심 특징**:
- 자동 판단 완전 제거 (`auto_fallback` 항상 False)
- 사용자 명시적 선택만 지원
- 싱글톤 패턴으로 일관성 보장

### 2. Detector Manager (검출 모델 지연 로딩)

**위치**: `/home/max16/pillsnap/src/core/detector_manager.py`

**목적**: YOLOv11m 검출 모델의 메모리 효율적 관리

**사용법**:
```python
from src.core.detector_manager import get_detector_manager

# 초기화
detector = get_detector_manager(config, model_path="/path/to/yolo.pt")

# 모델 사용 (자동 로드)
model = detector.get_model()
results = detector.predict(image_tensor)

# 통계 확인
stats = detector.get_stats()
print(f"Loaded: {stats['loaded']}, Idle: {stats['idle_time_seconds']}s")
```

**핵심 특징**:
- **Load Once Guard**: 첫 combo 요청 시만 로드
- **Idle TTL Reaper**: 10분 유휴 시 자동 언로드
- **Hysteresis**: 로드/언로드 사이 2분 최소 체류

### 3. OOM Recovery State Machine (OOM 복구)

**위치**: `/home/max16/pillsnap/src/core/oom_handler.py`

**목적**: 학습 중 OOM 발생 시 일관성 있는 복구

**사용법**:
```python
from src.core.oom_handler import OOMRecoveryStateMachine, handle_training_oom

# 초기화
oom_handler = OOMRecoveryStateMachine(config["train"]["oom"])

# 학습 루프 내
try:
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        action = handle_training_oom(e, oom_handler, {
            "batch_size": current_batch_size,
            "grad_accum": current_grad_accum
        })
        
        if action["action"] == "microbatching":
            # 새 배치 크기 적용
            dataloader = rebuild_dataloader(action["batch_size"])
```

**복구 단계**:
1. S1: GPU 캐시 정리 (1회)
2. S2: AMP fp16 강제 (1회)
3. S3: 마이크로배칭 (글로벌 배치 유지)
4. S4: 글로벌 배치 변경 (최후)

**가드레일**:
- `max_retries`: 4
- `max_grad_accum`: 8
- `min_batch`: 1

### 4. Memory Policy Manager (메모리 정책)

**위치**: `/home/max16/pillsnap/src/core/memory_policy.py`

**목적**: Stage별 128GB RAM 최적 활용 전략

**사용법**:
```python
from src.core.memory_policy import create_memory_policy

# 초기화
memory_manager = create_memory_policy(config)

# 현재 Stage 설정 확인
summary = memory_manager.get_stage_summary()
print(f"Stage {summary['stage']}: {summary['cache_policy']}")

# DataLoader 설정 획득
dl_config = memory_manager.get_dataloader_config()
# Returns: {"num_workers": 16, "prefetch_factor": 8, ...}

# 메모리 모니터링
stats = memory_manager.monitor_memory_usage()
if stats["percent"] > 85:
    suggestions = memory_manager.suggest_optimization()
```

**Stage별 기본 정책**:
- **Stage 1**: labels_only, 워커 8개
- **Stage 2**: hotset 2만장, 워커 12개  
- **Stage 3**: hotset 4만장, LMDB 활성화
- **Stage 4**: hotset 6만장, 최대 최적화

### 5. ONNX Export Manager (내보내기/검증)

**위치**: `/home/max16/pillsnap/src/core/onnx_export.py`

**목적**: 실용적 허용치 기반 ONNX 내보내기 및 검증

**사용법**:
```python
from src.core.onnx_export import export_and_validate

# 모델 내보내기 및 검증
onnx_path, validation = export_and_validate(
    model=classification_model,
    model_type="classification",
    config=config["export"],
    test_samples=validation_batch
)

if validation["passed"]:
    print(f"Export successful: {onnx_path}")
    print(f"MSE: {validation['mse_mean']:.2e}")
else:
    print(f"Validation failed: {validation['failures']}")
```

**실용적 허용치**:
- **FP32**: MSE ≤ 1e-4, Top-1 mismatch ≤ 1%
- **FP16**: MSE ≤ 5e-4, Top-1 mismatch ≤ 2%
- **Detection**: mAP Δ ≤ 0.01

### 6. Path Policy Validator (경로 정책)

**위치**: `/home/max16/pillsnap/src/core/path_policy.py`

**목적**: WSL/Windows 경로 정책 검증 및 자동 변환

**사용법**:
```python
from src.core.path_policy import validate_project_paths, get_wsl_safe_path

# 프로젝트 경로 검증
if not validate_project_paths(config):
    raise ValueError("Path policy violations detected")

# 안전한 WSL 경로 보장
safe_path = get_wsl_safe_path("C:\\Users\\max16\\data")
# Returns: "/mnt/c/Users/max16/data"
```

**정책**:
- WSL: `/mnt/` 경로만 사용
- Windows 도구: `C:\ProgramData\Cloudflare` 예외 허용
- 자동 변환 지원

## 🔧 통합 예제

### 완전한 학습 파이프라인 통합

```python
import torch
from src.core.pipeline_mode import get_pipeline_resolver
from src.core.detector_manager import get_detector_manager
from src.core.oom_handler import OOMRecoveryStateMachine
from src.core.memory_policy import create_memory_policy

def main_training_loop(config):
    # 1. 파이프라인 모드 설정
    resolver = get_pipeline_resolver(config["data"])
    mode, _ = resolver.resolve_mode(config["data"]["default_mode"])
    
    # 2. 메모리 정책 설정
    memory_manager = create_memory_policy(config)
    dl_config = memory_manager.get_dataloader_config()
    
    # 3. 검출기 매니저 초기화 (combo 모드만)
    detector = None
    if mode == "combo":
        detector = get_detector_manager(config)
    
    # 4. OOM 핸들러 초기화
    oom_handler = OOMRecoveryStateMachine(config["train"]["oom"])
    
    # 5. 데이터로더 생성
    train_loader = create_dataloader(
        batch_size=config["train"]["batch_size"],
        **dl_config
    )
    
    # 6. 학습 루프
    for epoch in range(config["train"]["epochs"]):
        for batch in train_loader:
            try:
                # Forward pass
                if mode == "single":
                    output = classification_model(batch)
                else:  # combo
                    detections = detector.predict(batch)
                    crops = extract_crops(detections)
                    output = classification_model(crops)
                
                # Backward pass
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM 복구
                    action = oom_handler.handle_oom({
                        "batch_size": train_loader.batch_size,
                        "grad_accum": grad_accum_steps
                    })
                    
                    if not action["continue_training"]:
                        save_checkpoint("emergency.pt")
                        return
                    
                    # 복구 액션 적용
                    apply_recovery_action(action)
```

## 📊 성능 모니터링

### 실시간 모니터링 대시보드

```python
def monitor_system_performance():
    """시스템 성능 실시간 모니터링"""
    
    # 메모리 상태
    memory_stats = memory_manager.monitor_memory_usage()
    
    # 검출기 상태 (combo 모드)
    if detector:
        detector_stats = detector.get_stats()
    
    # OOM 복구 이력
    oom_stats = oom_handler.get_stats()
    
    # 대시보드 출력
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                  시스템 성능 모니터링                      ║
    ╠══════════════════════════════════════════════════════════╣
    ║ 📊 메모리 사용량: {memory_stats['percent']:.1f}%         ║
    ║    - 캐시: {memory_stats['reserved_cache_gb']:.1f}GB     ║
    ║    - 워커: {memory_stats['reserved_workers_gb']:.1f}GB   ║
    ║                                                           ║
    ║ 🎯 검출기 상태: {'로드됨' if detector_stats['loaded'] else '언로드'}
    ║    - 추론 횟수: {detector_stats['total_inferences']}     ║
    ║    - 유휴 시간: {detector_stats.get('idle_time_seconds', 0):.0f}초
    ║                                                           ║
    ║ ⚠️ OOM 복구: {oom_stats['total_retries']}회              ║
    ║    - 현재 배치: {oom_stats['current_batch_size']}        ║
    ║    - Grad Accum: {oom_stats['current_grad_accum']}       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
```

## 🚀 다음 단계

### 구현 필요 컴포넌트

1. **src/data.py**: 데이터 파이프라인 구현
2. **src/train.py**: 학습 루프 통합
3. **src/api/service.py**: API 서비스 레이어
4. **scripts/**: 실행 스크립트

### 테스트 작성

```python
# tests/test_core_components.py
import pytest
from src.core.pipeline_mode import get_pipeline_resolver

def test_pipeline_mode_resolver():
    config = {"default_mode": "single", "auto_fallback": False}
    resolver = get_pipeline_resolver(config)
    
    # 사용자 명시적 선택
    mode, reason = resolver.resolve_mode("combo")
    assert mode == "combo"
    assert "user_explicit" in reason
    
    # 기본값 사용
    mode, _ = resolver.resolve_mode(None)
    assert mode == "single"
```

## 📝 주의사항

1. **메모리 관리**: Stage별 메모리 정책을 반드시 적용
2. **OOM 처리**: 학습 루프에 OOM 핸들러 통합 필수
3. **경로 정책**: 모든 경로는 PathPolicyValidator로 검증
4. **모드 결정**: PipelineModeResolver를 통해서만 결정

## ✅ 체크리스트

- [x] Pipeline Mode Resolver 구현
- [x] Detector Manager 구현  
- [x] OOM Recovery State Machine 구현
- [x] Memory Policy Manager 구현
- [x] ONNX Export Manager 구현
- [x] Path Policy Validator 구현
- [ ] 통합 테스트 작성
- [ ] 성능 벤치마크 실행
- [ ] 프로덕션 배포 준비

---
**작성일**: 2025-08-17  
**버전**: 1.0.0  
**다음 리뷰**: 통합 테스트 완료 후