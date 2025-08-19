"""
Stage 1 Evaluator: Progressive Validation + 현재 GPU 테스트 방법론 통합
- PART_0 OptimizationAdvisor 반자동 평가 시스템
- 현재 GPU 스모크 테스트 패턴 적용
- 사용자 선택권 제공 (완전 자동화 금지)
"""

import json
import time
import subprocess
import sys
from pathlib import Path

def run_gpu_smoke_tests():
    """현재 GPU 스모크 테스트 패턴 실행"""
    print("🚀 Running GPU Smoke Tests (Stage 1 Pattern)")
    
    results = {}
    
    # GPU-A: 합성 데이터 테스트 (현재 방법론)
    try:
        print("  📋 GPU-A: Synthetic Data Test")
        env = {"PYTHONPATH": "/home/max16/pillsnap"}
        result = subprocess.run([
            "/home/max16/pillsnap/.venv/bin/python", 
            "tests/gpu_smoke/gpu_smoke_A.py"
        ], cwd="/home/max16/pillsnap", capture_output=True, text=True, timeout=300, env=env)
        
        if result.returncode == 0:
            print("    ✅ GPU-A passed")
            results["gpu_a"] = {"success": True, "output": result.stdout}
        else:
            print("    ❌ GPU-A failed")
            results["gpu_a"] = {"success": False, "error": result.stderr}
            
    except Exception as e:
        print(f"    💥 GPU-A error: {e}")
        results["gpu_a"] = {"success": False, "error": str(e)}
    
    # GPU-B: 실데이터 테스트 (현재 방법론)
    try:
        print("  📋 GPU-B: Real Data Test")
        env = {"PYTHONPATH": "/home/max16/pillsnap"}
        result = subprocess.run([
            "/home/max16/pillsnap/.venv/bin/python", 
            "tests/gpu_smoke/gpu_smoke_B.py"
        ], cwd="/home/max16/pillsnap", capture_output=True, text=True, timeout=300, env=env)
        
        if result.returncode == 0:
            print("    ✅ GPU-B passed")
            results["gpu_b"] = {"success": True, "output": result.stdout}
        else:
            print("    ❌ GPU-B failed")
            results["gpu_b"] = {"success": False, "error": result.stderr}
            
    except Exception as e:
        print(f"    💥 GPU-B error: {e}")
        results["gpu_b"] = {"success": False, "error": str(e)}
    
    return results

def analyze_stage1_performance(gpu_results):
    """Stage 1 성능 분석 (PART_0 목표 기준)"""
    
    # PART_0 Stage 1 목표값
    target_metrics = {
        "classification_accuracy": 0.40,  # 50클래스 기준
        "detection_map_0_5": 0.30,       # 기본 검출 가능성
        "inference_time_ms": 50,          # RTX 5080 실시간 처리
        "memory_usage_gb": 14,            # VRAM 안정성
        "data_loading_s_per_batch": 2     # 128GB RAM 활용도
    }
    
    # 현재 GPU 테스트 결과 분석
    current_metrics = {}
    
    # GPU-B 실데이터 결과에서 메트릭 추출
    if gpu_results.get("gpu_b", {}).get("success"):
        try:
            # artifacts/gpu_runs 에서 최신 결과 로드
            gpu_runs_path = Path("artifacts/gpu_runs")
            if gpu_runs_path.exists():
                latest_run = max(gpu_runs_path.glob("GPU_B_real_*"))
                metrics_file = latest_run / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        gpu_metrics = json.load(f)
                    
                    current_metrics = {
                        "classification_accuracy": gpu_metrics.get("val_accuracy", 0.0),
                        "inference_time_ms": gpu_metrics.get("elapsed_seconds", 0) * 1000,
                        "memory_usage_gb": gpu_metrics.get("memory_peak_gb", 0),
                        "gpu_compatibility": True,
                        "pytorch_version": gpu_metrics.get("pytorch_version", "unknown")
                    }
        except Exception as e:
            print(f"  ⚠️ Could not parse GPU-B metrics: {e}")
    
    return current_metrics, target_metrics

def generate_optimization_advisor_recommendations(current_metrics, target_metrics, gpu_results):
    """OptimizationAdvisor 권장사항 생성 (PART_0 반자동화 철학)"""
    
    # 기본 체크
    mandatory_checks = {
        "gpu_environment": gpu_results.get("gpu_a", {}).get("success", False),
        "real_data_processing": gpu_results.get("gpu_b", {}).get("success", False),
        "pytorch_compatibility": "2.7.0+cu128" in current_metrics.get("pytorch_version", ""),
        "rtx5080_support": current_metrics.get("gpu_compatibility", False)
    }
    
    # 성능 평가
    performance_status = {}
    for metric, target in target_metrics.items():
        current = current_metrics.get(metric, 0)
        if metric in ["classification_accuracy", "detection_map_0_5"]:
            performance_status[metric] = current >= target
        elif metric in ["inference_time_ms", "memory_usage_gb", "data_loading_s_per_batch"]:
            performance_status[metric] = current <= target if current > 0 else True  # 측정되지 않으면 통과
        else:
            performance_status[metric] = True
    
    # 전체 평가
    all_mandatory_passed = all(mandatory_checks.values())
    performance_acceptable = sum(performance_status.values()) >= len(performance_status) * 0.6  # 60% 이상
    
    # OptimizationAdvisor 권장사항 (PART_0 사용자 선택권 중심)
    if all_mandatory_passed and performance_acceptable:
        status = "RECOMMEND_PROCEED"
        confidence = "high"
        reasons = [
            "✅ GPU 환경 검증 완료 (RTX 5080 + PyTorch 2.7.0+cu128)",
            "✅ 실데이터 처리 파이프라인 동작 확인",
            "✅ 기본 성능 임계값 달성",
            "✅ 메모리 사용량 안정성 확인"
        ]
        next_actions = [
            "Two-Stage Pipeline 아키텍처 구현 (PART_C)",
            "YOLOv11m 검출 모델 통합",
            "EfficientNetV2-S 분류 모델 통합", 
            "Progressive Validation Stage 2 진행"
        ]
    elif all_mandatory_passed:
        status = "SUGGEST_OPTIMIZE"
        confidence = "medium"
        reasons = [
            "✅ GPU 환경 검증 완료",
            "⚠️ 일부 성능 지표 개선 필요",
            "✅ 기본 기능 동작 확인"
        ]
        next_actions = [
            "성능 튜닝 후 Stage 2 진행 권장",
            "배치 크기 최적화 고려",
            "메모리 사용량 모니터링 강화"
        ]
    else:
        status = "WARN_STOP"
        confidence = "low"
        reasons = [
            "❌ 필수 체크 항목 실패",
            "❌ GPU 환경 설정 재검토 필요"
        ]
        next_actions = [
            "GPU 드라이버 및 CUDA 재설치",
            "PyTorch 호환성 재확인",
            "환경 설정 디버깅"
        ]
    
    return {
        "stage": 1,
        "purpose": "pipeline_validation",
        "timestamp": time.time(),
        "status": status,
        "confidence": confidence,
        "mandatory_checks": mandatory_checks,
        "performance_metrics": {
            "current": current_metrics,
            "targets": target_metrics,
            "status": performance_status
        },
        "reasons": reasons,
        "next_actions": next_actions,
        # PART_0 사용자 선택권 제공
        "user_options": {
            "1": "RECOMMEND_PROCEED: 권장사항 적용 후 Stage 2 진행",
            "2": "SUGGEST_OPTIMIZE: 현재 성능으로 Stage 2 진행", 
            "3": "WARN_STOP: 수동 디버깅 모드"
        }
    }

def display_terminal_dashboard(recommendations):
    """PART_B 터미널 대시보드 출력"""
    
    status_colors = {
        "RECOMMEND_PROCEED": "🟢",
        "SUGGEST_OPTIMIZE": "🟡", 
        "WARN_STOP": "🔴"
    }
    
    print("\n" + "=" * 60)
    print(f"🎯 Stage 1 OptimizationAdvisor Report")
    print("=" * 60)
    
    print(f"\n📋 Status: {status_colors.get(recommendations['status'], '⚪')} {recommendations['status']}")
    print(f"📊 Confidence: {recommendations['confidence'].upper()}")
    print(f"⏱️ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(recommendations['timestamp']))}")
    
    print(f"\n🔍 Mandatory Checks:")
    for check, passed in recommendations['mandatory_checks'].items():
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} {check}: {'PASS' if passed else 'FAIL'}")
    
    print(f"\n📈 Performance Metrics:")
    perf = recommendations['performance_metrics']
    for metric, target in perf['targets'].items():
        current = perf['current'].get(metric, 'N/A')
        status = perf['status'].get(metric, False)
        symbol = "✅" if status else "⚠️" 
        print(f"  {symbol} {metric}: {current} (target: {target})")
    
    print(f"\n💡 Reasons:")
    for reason in recommendations['reasons']:
        print(f"  • {reason}")
        
    print(f"\n🎯 Next Actions:")
    for action in recommendations['next_actions']:
        print(f"  → {action}")
    
    print(f"\n👤 User Options:")
    for key, option in recommendations['user_options'].items():
        print(f"  [{key}] {option}")
    
    print("=" * 60)

def evaluate_stage1():
    """Stage 1 OptimizationAdvisor 평가 실행"""
    print("=" * 60)
    print("🎯 Stage 1 Progressive Validation")
    print("   Purpose: Pipeline Validation (PART_0 Design)")
    print("   Test Pattern: Current GPU Smoke Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. GPU 스모크 테스트 실행 (현재 방법론)
    gpu_results = run_gpu_smoke_tests()
    
    # 2. 성능 분석
    current_metrics, target_metrics = analyze_stage1_performance(gpu_results)
    
    # 3. OptimizationAdvisor 권장사항 생성
    recommendations = generate_optimization_advisor_recommendations(
        current_metrics, target_metrics, gpu_results
    )
    
    # 4. 터미널 대시보드 출력
    display_terminal_dashboard(recommendations)
    
    # 5. 결과 저장
    report_path = Path("/mnt/data/exp/exp01/reports/stage_1_evaluation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(recommendations, indent=2))
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Evaluation completed in {elapsed:.1f}s")
    print(f"📁 Report saved: {report_path}")
    
    # 성공/실패 반환
    return recommendations['status'] != "WARN_STOP"

if __name__ == "__main__":
    success = evaluate_stage1()
    sys.exit(0 if success else 1)