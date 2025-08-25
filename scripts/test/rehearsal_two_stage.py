#!/usr/bin/env python3
"""
Two-Stage Pipeline 리허설 학습
목적: Classification + Detection 통합 파이프라인 검증
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

import torch
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List
import subprocess

def get_gpu_memory():
    """GPU 메모리 사용량 조회"""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        )
        return f"{result.stdout.strip()}MB"
    except:
        return "N/A"

def format_file_info(filepath: Path) -> Dict:
    """파일 정보 포맷팅"""
    if not filepath.exists():
        return {"exists": False}
    
    stat = filepath.stat()
    return {
        "exists": True,
        "size_mb": round(stat.st_size / (1024*1024), 2),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }

def run_rehearsal():
    """리허설 학습 실행"""
    print("=" * 80)
    print("🎭 TWO-STAGE PIPELINE 리허설 학습 시작")
    print("=" * 80)
    
    # 설정
    EPOCHS = 3  # 짧은 테스트
    BATCH_SIZE = 8
    RESET_BEST = True
    
    # 초기 상태 기록
    initial_state = {
        "timestamp": datetime.now().isoformat(),
        "gpu_memory": get_gpu_memory()
    }
    
    # Detection state 확인
    state_file = Path("/home/max16/pillsnap/artifacts/yolo/stage3/state.json")
    initial_det_epochs = 0
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            initial_det_epochs = state.get("det_epochs_done", 0)
    
    print(f"\n📊 초기 상태:")
    print(f"  - Detection epochs 완료: {initial_det_epochs}")
    print(f"  - GPU 메모리: {initial_state['gpu_memory']}")
    print(f"  - Reset best: {RESET_BEST}")
    
    # Stage 3 Two-Stage 학습 명령어 구성
    cmd = [
        sys.executable, "-m", "src.training.train_stage3_two_stage",
        "--manifest-train", "/home/max16/pillsnap/artifacts/stage3/manifest_train.remove.csv",
        "--manifest-val", "/home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv",
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr-classifier", "5e-5",
        "--lr-detector", "1e-3",
        "--patience", "10",  # 조기종료 방지
        "--save-period", "1",  # 매 epoch 저장
        "--verbose"
    ]
    
    if RESET_BEST:
        cmd.append("--reset-best")
    
    # 로그 파일 준비
    log_file = Path(f"/home/max16/pillsnap/logs/rehearsal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 학습 시작:")
    print(f"  - 명령어: {' '.join(cmd)}")
    print(f"  - 로그: {log_file}")
    print("-" * 80)
    
    # 학습 실행
    start_time = time.time()
    epoch_metrics = []
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 실시간 로그 파싱
            for line in process.stdout:
                f.write(line)
                f.flush()
                
                # 핵심 메트릭 추출
                if "Epoch" in line and "/" in line:
                    print(line.strip())
                elif "val_top1" in line or "val_top5" in line:
                    print(f"  📈 {line.strip()}")
                elif "mAP" in line or "Best" in line:
                    print(f"  🎯 {line.strip()}")
                elif "Saving" in line or "checkpoint" in line:
                    print(f"  💾 {line.strip()}")
                elif "reset" in line.lower() and "best" in line.lower():
                    print(f"  🔄 {line.strip()}")
            
            process.wait()
            success = process.returncode == 0
            
    except Exception as e:
        print(f"❌ 학습 실행 중 오류: {e}")
        success = False
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 소요 시간: {elapsed_time/60:.1f}분")
    
    # 결과 검증
    print("\n" + "=" * 80)
    print("📋 결과 검증")
    print("=" * 80)
    
    # 1. Detection epochs 증가 확인
    final_det_epochs = initial_det_epochs
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            final_det_epochs = state.get("det_epochs_done", 0)
    
    det_increase = final_det_epochs - initial_det_epochs
    print(f"\n1️⃣ Detection Epochs:")
    print(f"   초기: {initial_det_epochs} → 최종: {final_det_epochs} (증가: +{det_increase})")
    print(f"   ✅ PASS" if det_increase == EPOCHS else f"   ❌ FAIL (기대값: +{EPOCHS})")
    
    # 2. Checkpoint 파일 확인
    print(f"\n2️⃣ Checkpoint 파일:")
    ckpt_dir = Path("/home/max16/pillsnap/artifacts/stage3/checkpoints")
    
    checkpoints = {
        "cls_last": ckpt_dir / "stage3_classification_last.pt",
        "cls_best": ckpt_dir / "stage3_classification_best.pt",
        "det_last": Path("/home/max16/pillsnap/runs/detect/train/weights/last.pt"),
        "det_best": Path("/home/max16/pillsnap/runs/detect/train/weights/best.pt")
    }
    
    print(f"   {'파일':<20} {'존재':<8} {'크기(MB)':<12} {'수정시간':<20}")
    print(f"   {'-'*60}")
    
    for name, path in checkpoints.items():
        info = format_file_info(path)
        if info["exists"]:
            print(f"   {name:<20} ✅      {info['size_mb']:<12.2f} {info['mtime']:<20}")
        else:
            print(f"   {name:<20} ❌")
    
    # 3. CSV 파일 행 수 확인
    print(f"\n3️⃣ Results CSV:")
    csv_files = {
        "classification": Path("/home/max16/pillsnap/artifacts/stage3/results.csv"),
        "detection": Path("/home/max16/pillsnap/runs/detect/train/results.csv")
    }
    
    for name, csv_path in csv_files.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"   {name}: {len(df)} 행")
            
            # 최근 3개 에폭 메트릭 표시
            if len(df) > 0:
                recent = df.tail(min(3, len(df)))
                if name == "classification":
                    if 'val_top1' in recent.columns:
                        print(f"     최근 val_top1: {recent['val_top1'].tolist()}")
                elif name == "detection":
                    if 'metrics/mAP50(B)' in recent.columns:
                        print(f"     최근 mAP50: {recent['metrics/mAP50(B)'].tolist()}")
        else:
            print(f"   {name}: 파일 없음")
    
    # 4. TensorBoard 로그 확인
    print(f"\n4️⃣ TensorBoard 로그:")
    tb_dirs = [
        Path("/home/max16/pillsnap_data/exp/exp01/tensorboard"),
        Path("/home/max16/pillsnap/runs/detect/train")
    ]
    
    for tb_dir in tb_dirs:
        if tb_dir.exists():
            event_files = list(tb_dir.glob("**/*tfevents*"))
            if event_files:
                # 최신 파일만 표시
                latest = max(event_files, key=lambda p: p.stat().st_mtime)
                print(f"   {tb_dir.name}: {len(event_files)}개 이벤트 파일")
                print(f"     최신: {latest.name}")
    
    # 5. 최종 요약
    print(f"\n" + "=" * 80)
    print(f"🏁 최종 요약")
    print(f"=" * 80)
    
    results = {
        "det_epochs_증가": det_increase == EPOCHS,
        "checkpoints_생성": all(format_file_info(p)["exists"] for p in checkpoints.values()),
        "csv_기록": all(p.exists() for p in csv_files.values()),
        "실행_성공": success
    }
    
    all_pass = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    print(f"\n{'🎉 모든 검증 통과!' if all_pass else '⚠️ 일부 검증 실패'}")
    print(f"GPU 메모리: {initial_state['gpu_memory']} → {get_gpu_memory()}")
    
    return all_pass

if __name__ == "__main__":
    success = run_rehearsal()
    sys.exit(0 if success else 1)