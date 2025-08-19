#!/usr/bin/env python3
"""
GPU-Stage2 스모크 테스트: pillsnap.stage2.train_cls GPU 모드 검증
목적: 실제 Stage2 train_cls를 --device cuda로 실행하여 전체 파이프라인 검증
"""
import time, json, subprocess, sys
from pathlib import Path

def run_stage2_gpu_smoke():
    """Stage2 train_cls를 GPU 모드로 실행"""
    print("🚀 GPU-Stage2 Smoke Test (pillsnap.stage2.train_cls)")
    
    start_time = time.time()
    run_tag = f"GPU_Stage2_{int(time.time())}"
    output_dir = Path(f"artifacts/gpu_runs/{run_tag}")
    
    print(f"📁 Output directory: {output_dir}")
    
    # Stage2 GPU 학습 명령어 구성
    cmd = [
        sys.executable, "-m", "pillsnap.stage2.train_cls",
        "--manifest", "artifacts/manifest_enriched.csv",
        "--classes", "artifacts/classes_step11.json", 
        "--device", "cuda",           # GPU 사용
        "--epochs", "1",             # 1 에폭만 (스모크 테스트)
        "--batch-size", "4",         # 작은 배치
        "--limit", "16",             # 16개 샘플만
        "--outdir", str(output_dir)  # GPU 실행 결과 격리
    ]
    
    print("🏃 Executing Stage2 GPU training...")
    print(f"💻 Command: {' '.join(cmd)}")
    
    try:
        # 환경 설정
        env = {
            "PYTHONPATH": "/home/max16/pillsnap",
            "CUDA_VISIBLE_DEVICES": "0"  # GPU 0번 사용 명시
        }
        
        result = subprocess.run(
            cmd,
            cwd="/home/max16/pillsnap",
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        elapsed = time.time() - start_time
        
        # 결과 분석
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        
        print(f"⏱️ Execution time: {elapsed:.1f}s")
        print(f"📊 Return code: {result.returncode}")
        
        if success:
            print("✅ Stage2 GPU training completed successfully!")
            print("📝 Output:")
            print(stdout[-500:] if len(stdout) > 500 else stdout)  # 마지막 500자만
        else:
            print("❌ Stage2 GPU training failed!")
            print("🔍 STDOUT:")
            print(stdout[-1000:] if len(stdout) > 1000 else stdout)
            print("🔍 STDERR:")  
            print(stderr[-1000:] if len(stderr) > 1000 else stderr)
        
        # 출력 파일 확인
        checkpoint_files = list(output_dir.glob("*.pt")) if output_dir.exists() else []
        log_files = list(output_dir.glob("*.log")) if output_dir.exists() else []
        
        # 메트릭 저장
        metrics = {
            "success": success,
            "return_code": result.returncode,
            "elapsed_seconds": elapsed,
            "run_tag": run_tag,
            "command": " ".join(cmd),
            "checkpoint_files": [str(f) for f in checkpoint_files],
            "log_files": [str(f) for f in log_files],
            "stdout_length": len(stdout),
            "stderr_length": len(stderr),
            "test_type": "pillsnap.stage2.train_cls_gpu"
        }
        
        if output_dir.exists():
            (output_dir / "stage2_gpu_metrics.json").write_text(json.dumps(metrics, indent=2))
            
            # 상세 로그도 저장
            (output_dir / "stdout.log").write_text(stdout)
            (output_dir / "stderr.log").write_text(stderr)
        
        print(f"📁 Results saved to: {output_dir}")
        
        # 성공 시 추가 검증
        if success and checkpoint_files:
            print(f"🎯 Generated checkpoints: {len(checkpoint_files)}")
            for ckpt in checkpoint_files:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name}: {size_mb:.1f}MB")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ Stage2 GPU training timed out after {elapsed:.1f}s")
        return {"success": False, "error": "timeout", "elapsed_seconds": elapsed}
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 Stage2 GPU training error: {e}")
        return {"success": False, "error": str(e), "elapsed_seconds": elapsed}

def check_gpu_requirements():
    """GPU 요구사항 체크"""
    print("🔍 Checking GPU requirements...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.version.cuda}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ Compute capability: sm_{capability[0]}{capability[1]}")
            print(f"✅ GPU memory: {memory_gb:.1f}GB")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("=" * 60)
    print("🎯 GPU-Stage2 Integration Test")
    print("=" * 60)
    
    # GPU 환경 체크
    if not check_gpu_requirements():
        print("💥 GPU requirements not met")
        return {"success": False, "error": "gpu_requirements_not_met"}
    
    print("\n" + "=" * 60)
    
    # Stage2 GPU 실행
    result = run_stage2_gpu_smoke()
    
    print("\n" + "=" * 60)
    if result["success"]:
        print("🎉 GPU-Stage2 Integration Test PASSED!")
        print(f"⏱️ Total time: {result['elapsed_seconds']:.1f}s")
    else:
        print("💥 GPU-Stage2 Integration Test FAILED!")
        print(f"❌ Error: {result.get('error', 'unknown')}")
    
    print("=" * 60)
    return result

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["success"] else 1)