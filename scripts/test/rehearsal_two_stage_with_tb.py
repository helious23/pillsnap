#!/usr/bin/env python3
"""
Two-Stage Pipeline 리허설 학습 (TensorBoard 포함)
목적: Classification + Detection 통합 파이프라인 검증
Stage 3 학습과 동일한 파라미터 사용 (에폭만 축소)
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
from typing import Dict
from torch.utils.tensorboard import SummaryWriter

# Stage 3 학습과 동일한 설정
REHEARSAL_CONFIG = {
    'epochs': 3,  # 리허설용 축소 (원래 36)
    'batch_size': 8,
    'lr_classifier': 5e-5,
    'lr_detector': 1e-3,
    'weight_decay': 5e-4,
    'label_smoothing': 0.1,
    'patience': 10,  # 조기종료 방지
    'save_period': 1,  # 매 에폭 저장
    'reset_best': True,
    'verbose': True
}

class RehearsalMonitor:
    """리허설 진행상황 모니터링"""
    
    def __init__(self):
        self.start_time = time.time()
        self.tb_writer = None
        self.log_file = None
        self.initial_state = {}
        self.final_state = {}
        
    def setup(self):
        """초기 설정"""
        # TensorBoard 설정
        tb_dir = Path("/home/max16/pillsnap_data/exp/exp01/tensorboard/rehearsal")
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(tb_dir)
        print(f"✅ TensorBoard 로깅 시작: {tb_dir}")
        
        # 로그 파일 설정
        log_dir = Path("/home/max16/pillsnap/logs/rehearsal")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"rehearsal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 초기 상태 기록
        self.record_initial_state()
        
    def record_initial_state(self):
        """초기 상태 기록"""
        # Detection state 확인
        state_file = Path("/home/max16/pillsnap/artifacts/yolo/stage3/state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.initial_state['det_epochs_done'] = state.get("det_epochs_done", 0)
                self.initial_state['last_map50'] = state.get("last_metrics", {}).get("map50", 0)
        
        # Classification checkpoints 확인
        ckpt_dir = Path("/home/max16/pillsnap/artifacts/stage3/checkpoints")
        cls_best = ckpt_dir / "stage3_classification_best.pt"
        if cls_best.exists():
            self.initial_state['cls_best_exists'] = True
            self.initial_state['cls_best_mtime'] = cls_best.stat().st_mtime
        else:
            self.initial_state['cls_best_exists'] = False
            
        print(f"\n📊 초기 상태:")
        print(f"  Detection epochs 완료: {self.initial_state.get('det_epochs_done', 0)}")
        print(f"  Detection mAP@0.5: {self.initial_state.get('last_map50', 0):.4f}")
        print(f"  Classification best.pt: {'있음' if self.initial_state.get('cls_best_exists') else '없음'}")
        
    def log_to_tensorboard(self, epoch: int, metrics: Dict):
        """TensorBoard에 메트릭 기록"""
        if self.tb_writer:
            # Classification 메트릭
            if 'val_top1' in metrics:
                self.tb_writer.add_scalar('Classification/val_top1', metrics['val_top1'], epoch)
            if 'val_top5' in metrics:
                self.tb_writer.add_scalar('Classification/val_top5', metrics['val_top5'], epoch)
            if 'cls_loss' in metrics:
                self.tb_writer.add_scalar('Classification/loss', metrics['cls_loss'], epoch)
                
            # Detection 메트릭
            if 'det_map50' in metrics:
                self.tb_writer.add_scalar('Detection/mAP50', metrics['det_map50'], epoch)
            if 'det_precision' in metrics:
                self.tb_writer.add_scalar('Detection/precision', metrics['det_precision'], epoch)
            if 'det_recall' in metrics:
                self.tb_writer.add_scalar('Detection/recall', metrics['det_recall'], epoch)
            if 'box_loss' in metrics:
                self.tb_writer.add_scalar('Detection/box_loss', metrics['box_loss'], epoch)
                
            # 통합 메트릭
            if 'val_top1' in metrics and 'det_map50' in metrics:
                combined_score = (metrics['val_top1'] + metrics['det_map50']) / 2
                self.tb_writer.add_scalar('Combined/score', combined_score, epoch)
                
    def verify_results(self):
        """결과 검증"""
        print("\n" + "=" * 80)
        print("📋 리허설 결과 검증")
        print("=" * 80)
        
        results = {}
        
        # 1. Detection epochs 증가 확인
        state_file = Path("/home/max16/pillsnap/artifacts/yolo/stage3/state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                final_det_epochs = state.get("det_epochs_done", 0)
                det_increase = final_det_epochs - self.initial_state.get('det_epochs_done', 0)
                
                print(f"\n1️⃣ Detection Epochs 누적:")
                print(f"   초기: {self.initial_state.get('det_epochs_done', 0)} → 최종: {final_det_epochs}")
                print(f"   증가: +{det_increase} (기대값: +{REHEARSAL_CONFIG['epochs']})")
                
                results['det_epochs_ok'] = (det_increase == REHEARSAL_CONFIG['epochs'])
                
                # metrics_history 확인
                if 'metrics_history' in state:
                    print(f"   metrics_history: {len(state['metrics_history'])}개 에폭 기록")
                
        # 2. Checkpoint 파일 확인
        print(f"\n2️⃣ Checkpoint 파일 상태:")
        ckpt_dir = Path("/home/max16/pillsnap/artifacts/stage3/checkpoints")
        
        checkpoints = {
            "Classification last": ckpt_dir / "stage3_classification_last.pt",
            "Classification best": ckpt_dir / "stage3_classification_best.pt",
            "Detection last": Path("/home/max16/pillsnap/runs/detect/train/weights/last.pt"),
            "Detection best": Path("/home/max16/pillsnap/runs/detect/train/weights/best.pt")
        }
        
        all_ckpts_exist = True
        for name, path in checkpoints.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024*1024)
                mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
                print(f"   ✅ {name:<20}: {size_mb:>8.1f}MB  수정: {mtime}")
                
                # best.pt 갱신 확인
                if "best" in name and self.initial_state.get('cls_best_exists'):
                    if path.stat().st_mtime > self.initial_state.get('cls_best_mtime', 0):
                        print(f"      → best 갱신됨!")
            else:
                print(f"   ❌ {name:<20}: 없음")
                all_ckpts_exist = False
                
        results['checkpoints_ok'] = all_ckpts_exist
        
        # 3. CSV 파일 확인
        print(f"\n3️⃣ Results CSV 기록:")
        
        # Classification CSV
        cls_csv = Path("/home/max16/pillsnap/artifacts/stage3/results.csv")
        if cls_csv.exists():
            df_cls = pd.read_csv(cls_csv)
            print(f"   Classification: {len(df_cls)} 행")
            if 'val_top1' in df_cls.columns and len(df_cls) > 0:
                recent_top1 = df_cls['val_top1'].tail(3).tolist()
                print(f"     최근 val_top1: {[f'{v:.2f}%' for v in recent_top1]}")
                
        # Detection CSV  
        det_csv = Path("/home/max16/pillsnap/runs/detect/train/results.csv")
        if det_csv.exists():
            df_det = pd.read_csv(det_csv)
            print(f"   Detection: {len(df_det)} 행")
            if 'metrics/mAP50(B)' in df_det.columns and len(df_det) > 0:
                recent_map = df_det['metrics/mAP50(B)'].tail(3).tolist()
                print(f"     최근 mAP@0.5: {[f'{v:.3f}' for v in recent_map]}")
                
        results['csv_ok'] = cls_csv.exists() and det_csv.exists()
        
        # 4. TensorBoard 로그 확인
        print(f"\n4️⃣ TensorBoard 로그:")
        tb_dir = Path("/home/max16/pillsnap_data/exp/exp01/tensorboard")
        
        # 리허설 디렉토리 확인
        rehearsal_tb = tb_dir / "rehearsal"
        if rehearsal_tb.exists():
            event_files = list(rehearsal_tb.glob("*tfevents*"))
            print(f"   Rehearsal 이벤트 파일: {len(event_files)}개")
            if event_files:
                latest = max(event_files, key=lambda p: p.stat().st_mtime)
                print(f"   최신 파일: {latest.name}")
                
        results['tensorboard_ok'] = rehearsal_tb.exists() and len(list(rehearsal_tb.glob("*tfevents*"))) > 0
        
        # 5. 최종 요약
        print(f"\n" + "=" * 80)
        print(f"🏁 리허설 최종 요약")
        print(f"=" * 80)
        
        all_pass = all(results.values())
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
            
        elapsed = time.time() - self.start_time
        print(f"\n⏱️ 소요 시간: {elapsed/60:.1f}분")
        print(f"{'🎉 리허설 성공!' if all_pass else '⚠️ 일부 검증 실패'}")
        
        return all_pass
        
    def cleanup(self):
        """정리"""
        if self.tb_writer:
            self.tb_writer.close()
            print("✅ TensorBoard 로깅 종료")

def main():
    """메인 실행"""
    monitor = RehearsalMonitor()
    monitor.setup()
    
    print("\n" + "=" * 80)
    print("🎭 TWO-STAGE PIPELINE 리허설 시작")
    print("=" * 80)
    print(f"설정:")
    for key, value in REHEARSAL_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Stage 3 Two-Stage 학습 명령어
    cmd = [
        sys.executable, "-m", "src.training.train_stage3_two_stage",
        "--manifest-train", "/home/max16/pillsnap/artifacts/stage3/manifest_train.remove.csv",
        "--manifest-val", "/home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv",
        "--epochs", str(REHEARSAL_CONFIG['epochs']),
        "--batch-size", str(REHEARSAL_CONFIG['batch_size']),
        "--lr-classifier", str(REHEARSAL_CONFIG['lr_classifier']),
        "--lr-detector", str(REHEARSAL_CONFIG['lr_detector']),
        "--weight-decay", str(REHEARSAL_CONFIG['weight_decay']),
        "--label-smoothing", str(REHEARSAL_CONFIG['label_smoothing']),
        "--patience", str(REHEARSAL_CONFIG['patience']),
        "--save-period", str(REHEARSAL_CONFIG['save_period']),
        "--verbose"
    ]
    
    if REHEARSAL_CONFIG['reset_best']:
        cmd.append("--reset-best")
    
    # 명령어를 문자열로 변환
    cmd_str = " ".join(cmd)
    
    print(f"\n🚀 학습 명령어:")
    print(f"  {cmd_str}")
    print(f"\n로그 파일: {monitor.log_file}")
    print("-" * 80)
    
    # 여기서 실제 학습 수행 (백그라운드 실행을 위해 메인에서 처리)
    return monitor, cmd_str

if __name__ == "__main__":
    monitor, cmd = main()
    print(f"\n실행할 명령어가 준비되었습니다.")
    print(f"백그라운드 실행을 위해 상위 스크립트에서 호출하세요.")