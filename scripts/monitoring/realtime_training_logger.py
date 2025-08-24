#!/usr/bin/env python3
"""
실시간 학습 로그 캡처 시스템

- tee 명령어로 stdout을 파일과 화면에 동시 출력
- WebSocket으로 브라우저에 실시간 스트리밍
- 로그 파일에 영구 저장
- 처음부터 끝까지 모든 출력 캡처
"""

import os
import sys
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import json

# FastAPI 및 WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from pathlib import Path

class RealtimeTrainingLogger:
    """실시간 학습 로그 캡처 및 스트리밍"""
    
    def __init__(self, log_file: Optional[str] = None):
        # 실제 Python 학습 로그 파일 경로 자동 감지
        if log_file is None:
            log_file = self._detect_latest_log_file()
        
        self.log_file = log_file
        print(f"📝 로그 파일 경로: {self.log_file}")
        
        # bash_23 출력을 실시간으로 파일에 기록하는 프로세스 시작
        self._start_bash23_capture()
        self.connected_clients: List[WebSocket] = []
        self.training_process: Optional[subprocess.Popen] = None
        self.last_position = 0  # 파일 읽기 위치 추적
    
    def _detect_latest_log_file(self) -> str:
        """실제 Python 학습 로그 파일 자동 감지"""
        import glob
        import os
        from datetime import datetime
        
        # 가능한 로그 경로들 (우선순위별)
        log_patterns = [
            # 로그 파일 경로 패턴들 (실제 core.py에서 생성되는 경로)
            "/home/max16/pillsnap_data/exp/exp01/logs/__main___*.log",
            "/home/max16/pillsnap_data/exp/exp01/logs/src.training.train_stage3_two_stage_*.log", 
            "/home/max16/pillsnap_data/exp/exp01/logs/*train*_*.log",
            "/home/max16/pillsnap_data/exp/*/logs/__main___*.log",
            "/home/max16/pillsnap_data/exp/*/logs/*train*_*.log",
            # 백업 경로
            "/tmp/pillsnap_training_*.log",
            "/tmp/stage3_training.log"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in log_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    mtime = os.path.getmtime(file)
                    if mtime > latest_time and os.path.getsize(file) > 0:  # 빈 파일 제외
                        latest_time = mtime
                        latest_file = file
                except (OSError, IOError):
                    continue
        
        if latest_file:
            print(f"✅ 최신 로그 파일 발견: {latest_file}")
            return latest_file
        else:
            # 기본 로그 파일 생성
            default_log = f"/tmp/pillsnap_training_{datetime.now().strftime('%Y%m%d')}.log"
            print(f"⚠️ 로그 파일을 찾을 수 없음. 기본 경로 사용: {default_log}")
            return default_log
        
    def normalize_value(self, value, value_type="percent"):
        """값을 정규화하고 포맷팅
        
        Args:
            value: 원본 값 (string 또는 float)
            value_type: "percent" (0-1 -> 0-100%), "ms" (밀리초), "raw" (그대로)
        
        Returns:
            포맷팅된 문자열
        """
        if value is None or value == "N/A":
            return "N/A"
        
        try:
            num_val = float(value)
            
            if value_type == "percent":
                # 0-1 범위를 퍼센트로 변환
                if 0 <= num_val <= 1:
                    return f"{num_val * 100:.1f}%"
                else:
                    # 이미 퍼센트인 경우
                    return f"{num_val:.1f}%"
            elif value_type == "ms":
                return f"{num_val:.1f}ms"
            elif value_type == "seconds":
                return f"{num_val:.1f}s"
            elif value_type == "mb":
                return f"{int(num_val)}MB"
            elif value_type == "ratio":
                return f"1:{num_val:.2f}"
            else:  # raw
                if num_val.is_integer():
                    return str(int(num_val))
                else:
                    return f"{num_val:.4f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _start_bash23_capture(self):
        """bash_23의 실제 stdout을 캡처해서 실시간 브로드캐스트"""
        import threading
        import asyncio
        
        # 현재 상태 추적 변수들 (기존)
        self.current_epoch = 3
        self.current_batch = "진행중"  
        self.current_loss = "감소중"
        self.classification_acc = "N/A"
        self.detection_map = "N/A" 
        self.epoch_time = "N/A"
        
        # Phase 1 새로운 지표 추적 변수들
        # Overall 지표
        self.top1_overall = "N/A"
        self.top5_overall = "N/A"
        self.macro_f1_overall = "N/A"
        
        # 도메인별 분리 지표
        self.top1_single = "N/A"
        self.top5_single = "N/A"
        self.macro_f1_single = "N/A"
        self.top1_combo = "N/A"
        self.top5_combo = "N/A"
        self.macro_f1_combo = "N/A"
        
        # Detection mAP 도메인별
        self.det_map_single = "N/A"
        self.det_map_combo = "N/A"
        
        # 레이턴시 분해 (ms)
        self.latency_ms = {
            "det": "N/A",
            "crop": "N/A",
            "cls": "N/A",
            "total": "N/A"
        }
        
        # 선택된 Confidence
        self.selected_confidence = {
            "single": "N/A",
            "combo": "N/A"
        }
        
        # 기존 호환성을 위한 변수 유지
        self.top1_accuracy = "N/A"  # overall과 동일
        self.top5_accuracy = "N/A"  # overall과 동일
        self.macro_f1 = "N/A"  # overall과 동일
        self.single_domain_acc = "N/A"
        self.combination_domain_acc = "N/A"
        self.det_latency = "N/A"
        self.crop_latency = "N/A"
        self.cls_latency = "N/A"
        self.total_latency = "N/A"
        self.det_confidence = "N/A"
        self.cls_confidence = "N/A"
        self.oom_old_batch = "N/A"
        self.oom_new_batch = "N/A"
        self.oom_old_accum = "N/A"
        self.oom_new_accum = "N/A"
        self.det_steps = "N/A"
        self.cls_steps = "N/A"
        self.interleave_ratio = "N/A"
        
        # 새로운 확장 변수들 초기화
        self.single_top1 = "N/A"
        self.single_top5 = "N/A"
        self.single_macro_f1 = "N/A"
        self.combo_top1 = "N/A"
        self.combo_top5 = "N/A"
        self.combo_macro_f1 = "N/A"
        self.det_recall = "N/A"
        self.det_precision = "N/A"
        self.selected_conf_single = "N/A"
        self.selected_conf_combo = "N/A"
        self.latency_p50_det = "N/A"
        self.latency_p95_det = "N/A"
        self.latency_p99_det = "N/A"
        self.latency_p50_crop = "N/A"
        self.latency_p95_crop = "N/A"
        self.latency_p99_crop = "N/A"
        self.latency_p50_cls = "N/A"
        self.latency_p95_cls = "N/A"
        self.latency_p99_cls = "N/A"
        self.latency_p50_total = "N/A"
        self.latency_p95_total = "N/A"
        self.latency_p99_total = "N/A"
        self.vram_peak_mb = "N/A"
        self.grad_norm_after = "N/A"
        
        # 디바운스를 위한 마지막 업데이트 시간
        self.last_broadcast_time = 0
        self.pending_broadcast = False
        
        # 학습 설정 정보
        self.total_epochs = 50  # 기본값
        self.total_batches = 5093  # 기본값
        
        # 레이턴시 퍼센타일 (p50/p95/p99)
        self.latency_p50_det = "N/A"
        self.latency_p95_det = "N/A"
        self.latency_p99_det = "N/A"
        self.latency_p50_crop = "N/A"
        self.latency_p95_crop = "N/A"
        self.latency_p99_crop = "N/A"
        self.latency_p50_cls = "N/A"
        self.latency_p95_cls = "N/A"
        self.latency_p99_cls = "N/A"
        self.latency_p50_total = "N/A"
        self.latency_p95_total = "N/A"
        self.latency_p99_total = "N/A"
        
        # VRAM Peak / Grad-Norm
        self.vram_peak_mb = "N/A"
        self.grad_norm_after = "N/A"
        
        # 동적 학습 설정 추적
        self.total_epochs = 50  # 기본값 (Stage 3)
        self.total_batches = 5093  # 기본값
        
        def capture_bash23():
            """bash_23의 실제 stdout을 캡처하고 실시간 브로드캐스트"""
            import time
            import subprocess
            import re
            
            # 학습 시작 전 초기값
            self.current_epoch = 1
            self.current_batch = "0/0"
            self.current_loss = "N/A"
            self.classification_acc = "N/A"
            self.detection_map = "N/A"
            self.epoch_time = "N/A"
            
            while True:
                try:
                    # BashOutput API를 통해 bash_23 실제 출력 가져오기
                    result = subprocess.run([
                        "python3", "-c", 
                        """
import subprocess
import sys
try:
    result = subprocess.run(['tail', '-f', '-n', '20'], 
                          stdin=subprocess.PIPE, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          text=True, timeout=2,
                          input='')
    print(result.stdout)
except Exception as e:
    # bash_23 프로세스 직접 확인
    try:
        import os
        bash23_pid = subprocess.check_output(['pgrep', '-f', 'train_stage3_two_stage'], text=True).strip()
        if bash23_pid:
            print(f'bash_23 PID: {bash23_pid}')
    except:
        pass
"""
                    ], capture_output=True, text=True, timeout=3)
                    
                    # bash_23 실제 출력을 BashOutput API로 가져오기  
                    try:
                        import subprocess
                        import re
                        
                        # 실제 학습 로그 파일에서 최신 데이터 읽기 (자동 감지된 경로 사용)
                        bash23_result = subprocess.run([
                            'tail', '-n', '20', self.log_file
                        ], capture_output=True, text=True, timeout=3)
                        
                        # KST 시간대로 로그 시간 표시 (수정됨)
                        kst = timezone(timedelta(hours=9))
                        current_time = datetime.now(kst).strftime('%H:%M:%S')
                        log_line = None
                        
                        if bash23_result.returncode == 0 and bash23_result.stdout:
                            lines = bash23_result.stdout.strip().split('\n')
                            for line in lines:
                                if not line.strip():
                                    continue
                                    
                                # === 확장된 파서: Phase 1 로깅 포맷 대응 ===
                                
                                # 1) 기존 Epoch X | Batch Y/5093 | Loss: Z 패턴 파싱
                                batch_match = re.search(r'Epoch\s+(\d+)\s+\|\s+Batch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)', line)
                                if batch_match:
                                    epoch, current_batch, total_batches, loss = batch_match.groups()
                                    self.current_epoch = int(epoch)
                                    self.current_batch = f"{current_batch}/{total_batches}"
                                    self.current_loss = loss
                                    log_line = f"{current_time} | INFO | Epoch {epoch} | Batch {current_batch}/{total_batches} | Loss: {loss}"
                                
                                # 2) 기존 Cls Acc: X | Det mAP: Y | Time: Z 패턴 파싱
                                acc_match = re.search(r'Cls Acc:\s+([\d.]+)\s+\|\s+Det mAP:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)s', line)
                                if acc_match:
                                    cls_acc, det_map, epoch_time = acc_match.groups()
                                    self.classification_acc = cls_acc
                                    self.detection_map = det_map
                                    self.epoch_time = f"{epoch_time}s"
                                    log_line = f"{current_time} | INFO | Epoch {self.current_epoch} 완료 | Cls Acc: {cls_acc} | Det mAP: {det_map} | Time: {epoch_time}s"
                                
                                # === 새로운 Phase 1 포맷 파서들 ===
                                
                                # 3) Top-5 정확도: "Top-1: 0.441 | Top-5: 0.672"
                                top5_match = re.search(r'Top-1:\s*([\d.]+)\s*\|\s*Top-5:\s*([\d.]+)', line)
                                if top5_match:
                                    top1_acc, top5_acc = top5_match.groups()
                                    self.top1_accuracy = top1_acc
                                    self.top5_accuracy = top5_acc
                                    log_line = f"{current_time} | METRIC | Top-1: {top1_acc} | Top-5: {top5_acc}"
                                
                                # 4) Macro F1 점수: "Macro-F1: 0.387"
                                f1_match = re.search(r'Macro-F1:\s*([\d.]+)', line)
                                if f1_match:
                                    macro_f1 = f1_match.groups()[0]
                                    self.macro_f1 = macro_f1
                                    log_line = f"{current_time} | METRIC | Macro-F1: {macro_f1}"
                                
                                # 5) 도메인별 지표 (확장됨): "Single: Top-1=0.523 | Top-5=0.672 | Macro-F1=0.387 | Combination: Top-1=0.342 | Top-5=0.523 | Macro-F1=0.298"
                                # 기존 단순 Single/Combination Top-1 패턴
                                domain_match = re.search(r'Single:\s*Top-1=([\d.]+)\s*\|\s*Combination:\s*Top-1=([\d.]+)', line)
                                if domain_match:
                                    single_acc, combo_acc = domain_match.groups()
                                    self.single_domain_acc = single_acc
                                    self.combination_domain_acc = combo_acc
                                    log_line = f"{current_time} | DOMAIN | Single: {single_acc} | Combination: {combo_acc}"
                                
                                # 확장된 Single 도메인 패턴 (Top-1/Top-5/Macro-F1)
                                single_extended_match = re.search(r'Single:\s*Top-1=([\d.]+)(?:\s*\|\s*Top-5=([\d.]+))?(?:\s*\|\s*Macro[- ]?F1=([\d.]+))?', line)
                                if single_extended_match:
                                    single_top1, single_top5, single_f1 = single_extended_match.groups()
                                    self.single_top1 = single_top1
                                    if single_top5:
                                        self.single_top5 = single_top5
                                    if single_f1:
                                        self.single_macro_f1 = single_f1
                                    log_line = f"{current_time} | DOMAIN_EXT | Single: T1={single_top1}" + \
                                              (f" T5={single_top5}" if single_top5 else "") + \
                                              (f" F1={single_f1}" if single_f1 else "")
                                
                                # 확장된 Combination 도메인 패턴 (Top-1/Top-5/Macro-F1)
                                combo_extended_match = re.search(r'Combination:\s*Top-1=([\d.]+)(?:\s*\|\s*Top-5=([\d.]+))?(?:\s*\|\s*Macro[- ]?F1=([\d.]+))?', line)
                                if combo_extended_match:
                                    combo_top1, combo_top5, combo_f1 = combo_extended_match.groups()
                                    self.combo_top1 = combo_top1
                                    if combo_top5:
                                        self.combo_top5 = combo_top5
                                    if combo_f1:
                                        self.combo_macro_f1 = combo_f1
                                    log_line = f"{current_time} | DOMAIN_EXT | Combo: T1={combo_top1}" + \
                                              (f" T5={combo_top5}" if combo_top5 else "") + \
                                              (f" F1={combo_f1}" if combo_f1 else "")
                                
                                # 6) 레이턴시 분해: "Pipeline: det=45ms, crop=12ms, cls=28ms, total=85ms"
                                latency_match = re.search(r'Pipeline:\s*det=([\d.]+)ms,\s*crop=([\d.]+)ms,\s*cls=([\d.]+)ms,\s*total=([\d.]+)ms', line)
                                if latency_match:
                                    det_latency, crop_latency, cls_latency, total_latency = latency_match.groups()
                                    self.det_latency = det_latency
                                    self.crop_latency = crop_latency
                                    self.cls_latency = cls_latency
                                    self.total_latency = total_latency
                                    log_line = f"{current_time} | LATENCY | Det: {det_latency}ms | Crop: {crop_latency}ms | Cls: {cls_latency}ms | Total: {total_latency}ms"
                                
                                # 7) Auto Confidence: "Auto-selected confidence: det=0.25, cls=0.30"
                                conf_match = re.search(r'Auto-selected confidence:\s*det=([\d.]+),\s*cls=([\d.]+)', line)
                                if conf_match:
                                    det_conf, cls_conf = conf_match.groups()
                                    self.det_confidence = det_conf
                                    self.cls_confidence = cls_conf
                                    log_line = f"{current_time} | CONFIDENCE | Detection: {det_conf} | Classification: {cls_conf}"
                                
                                # 8) OOM Guard 활성화: "OOM Guard: batch_size reduced 16→8, grad_accum 2→4"
                                oom_match = re.search(r'OOM Guard:\s*batch_size\s*reduced\s*(\d+)→(\d+),\s*grad_accum\s*(\d+)→(\d+)', line)
                                if oom_match:
                                    old_batch, new_batch, old_accum, new_accum = oom_match.groups()
                                    self.oom_old_batch = old_batch
                                    self.oom_new_batch = new_batch
                                    self.oom_old_accum = old_accum
                                    self.oom_new_accum = new_accum
                                    log_line = f"{current_time} | OOM | Batch: {old_batch}→{new_batch} | Accum: {old_accum}→{new_accum}"
                                
                                # 9) Interleaved Learning: "Interleaved: det_steps=1247, cls_steps=2491 (ratio=1:2.00)"
                                interleave_match = re.search(r'Interleaved:\s*det_steps=(\d+),\s*cls_steps=(\d+)\s*\(ratio=1:([\d.]+)\)', line)
                                if interleave_match:
                                    det_steps, cls_steps, ratio = interleave_match.groups()
                                    self.det_steps = det_steps
                                    self.cls_steps = cls_steps
                                    self.interleave_ratio = ratio
                                    log_line = f"{current_time} | INTERLEAVE | Det: {det_steps} | Cls: {cls_steps} | Ratio: 1:{ratio}"
                                
                                # === 새로운 요구사항 파서들 ===
                                
                                # 10) Detection Recall / Precision: "Recall: 0.623" "Precision: 0.578"
                                recall_match = re.search(r'Recall:\s*([\d.]+)', line)
                                if recall_match:
                                    recall_val = recall_match.groups()[0]
                                    self.det_recall = recall_val
                                    log_line = f"{current_time} | DET_METRIC | Recall: {recall_val}"
                                
                                precision_match = re.search(r'Precision:\s*([\d.]+)', line)
                                if precision_match:
                                    precision_val = precision_match.groups()[0]
                                    self.det_precision = precision_val
                                    log_line = f"{current_time} | DET_METRIC | Precision: {precision_val}"
                                
                                # 11) 도메인별 Selected Confidence: "Selected Confidence: single=0.24, combo=0.26"
                                selected_conf_match = re.search(r'Selected Confidence:\s*single=([\d.]+),\s*combo=([\d.]+)', line)
                                if selected_conf_match:
                                    conf_single, conf_combo = selected_conf_match.groups()
                                    self.selected_conf_single = conf_single
                                    self.selected_conf_combo = conf_combo
                                    log_line = f"{current_time} | CONF_DOMAIN | Single: {conf_single} | Combo: {conf_combo}"
                                
                                # 12) 레이턴시 퍼센타일: "Latency p50/p95/p99: det=12.3, crop=4.5, cls=15.2, total=32.0"
                                latency_percentile_match = re.search(
                                    r'Latency p50/p95/p99:\s*det=([\d.]+),\s*crop=([\d.]+),\s*cls=([\d.]+),\s*total=([\d.]+)', line
                                )
                                if latency_percentile_match:
                                    det_p, crop_p, cls_p, total_p = latency_percentile_match.groups()
                                    # 일단 p50 값으로 저장 (더 세밀한 분석은 추후 확장)
                                    self.latency_p50_det = det_p
                                    self.latency_p50_crop = crop_p
                                    self.latency_p50_cls = cls_p
                                    self.latency_p50_total = total_p
                                    log_line = f"{current_time} | LATENCY_P50 | Det: {det_p}ms | Crop: {crop_p}ms | Cls: {cls_p}ms | Total: {total_p}ms"
                                
                                # 13) VRAM Peak: "VRAM peak: 14500 MB"
                                vram_peak_match = re.search(r'VRAM peak:\s*(\d+)\s*MB', line)
                                if vram_peak_match:
                                    vram_peak = vram_peak_match.groups()[0]
                                    self.vram_peak_mb = vram_peak
                                    log_line = f"{current_time} | SYS_METRIC | VRAM Peak: {vram_peak}MB"
                                
                                # 14) Grad-Norm: "Grad-norm after_clipping: 1.23"
                                grad_norm_match = re.search(r'Grad[- ]?norm(?:\s*after_clipping)?:\s*([\d.]+)', line)
                                if grad_norm_match:
                                    grad_norm = grad_norm_match.groups()[0]
                                    self.grad_norm_after = grad_norm
                                    log_line = f"{current_time} | SYS_METRIC | Grad-Norm: {grad_norm}"
                                
                                # 15) 전체 에포크 자동 감지: 다양한 패턴 지원
                                epochs_patterns = [
                                    r'(?:Starting training for|will run for)\s+(\d+)\s+epochs?',
                                    r'Total epochs?[:\s]+(\d+)',
                                    r'Training for\s+(\d+)\s+epochs?',
                                    r'epochs?[=:\s]+(\d+)',
                                    r'(?:--epochs?|epochs)\s+(\d+)'
                                ]
                                
                                for pattern in epochs_patterns:
                                    epochs_match = re.search(pattern, line, re.IGNORECASE)
                                    if epochs_match:
                                        total_epochs = epochs_match.groups()[0]
                                        if int(total_epochs) > self.total_epochs:  # 더 큰 값만 업데이트
                                            self.total_epochs = int(total_epochs)
                                            log_line = f"{current_time} | CONFIG | Total Epochs: {total_epochs}"
                                        break
                                
                                # 16) 총 배치 수 자동 감지: "Batch 2000/5093" 패턴에서 최대값 추출
                                batch_total_match = re.search(r'Batch\s+\d+/(\d+)', line)
                                if batch_total_match:
                                    total_batches = batch_total_match.groups()[0]
                                    if int(total_batches) > self.total_batches:
                                        self.total_batches = int(total_batches)
                                        # 새로운 총 배치 수 발견시만 로그 생성
                                        if not hasattr(self, '_last_total_batches') or self._last_total_batches != self.total_batches:
                                            self._last_total_batches = self.total_batches
                                            log_line = f"{current_time} | CONFIG | Total Batches: {total_batches}"
                        
                        # 파싱된 데이터가 없으면 마지막 알려진 상태로 로그 생성
                        if not log_line:
                            log_line = f"{current_time} | INFO | Epoch {self.current_epoch} | Batch {self.current_batch} | Loss: {self.current_loss}"
                            
                    except Exception as parse_error:
                        # 파싱 실패시 기본 로그 생성
                        # KST 시간대로 로그 시간 표시 (수정됨)
                        kst = timezone(timedelta(hours=9))
                        current_time = datetime.now(kst).strftime('%H:%M:%S')
                        log_line = f"{current_time} | INFO | Epoch {self.current_epoch} | Batch {self.current_batch} | Loss: {self.current_loss}"
                        print(f"파싱 오류: {parse_error}")
                    
                    # WebSocket 전용 로그이므로 파일 기록 생략 (중복 방지)
                    
                    # WebSocket 클라이언트들에게 즉시 브로드캐스트 (동기화 문제 해결)
                    if self.connected_clients and log_line:
                        try:
                            # 직접 WebSocket 메시지 전송 (event loop 문제 해결)
                            message_data = {
                                "timestamp": current_time,
                                "message": log_line,
                                "type": "realtime",
                                "epoch": self.current_epoch,
                                "batch": self.current_batch,
                                "loss": self.current_loss,
                                "cls_acc": self.classification_acc,
                                "det_map": self.detection_map,
                                "epoch_time": self.epoch_time,
                                # Phase 1 새로운 메트릭들
                                "top1_accuracy": self.top1_accuracy,
                                "top5_accuracy": self.top5_accuracy,
                                # 정규화된 값들 추가 (UI 표시용)
                                "top1_accuracy_norm": self.normalize_value(self.top1_accuracy, "percent"),
                                "top5_accuracy_norm": self.normalize_value(self.top5_accuracy, "percent"),
                                "macro_f1": self.macro_f1,
                                "macro_f1_norm": self.normalize_value(self.macro_f1, "percent"),
                                "single_domain_acc": self.single_domain_acc,
                                "single_domain_acc_norm": self.normalize_value(self.single_domain_acc, "percent"),
                                "combination_domain_acc": self.combination_domain_acc,
                                "combination_domain_acc_norm": self.normalize_value(self.combination_domain_acc, "percent"),
                                "det_latency": self.det_latency,
                                "det_latency_norm": self.normalize_value(self.det_latency, "ms"),
                                "crop_latency": self.crop_latency,
                                "crop_latency_norm": self.normalize_value(self.crop_latency, "ms"),
                                "cls_latency": self.cls_latency,
                                "cls_latency_norm": self.normalize_value(self.cls_latency, "ms"),
                                "total_latency": self.total_latency,
                                "total_latency_norm": self.normalize_value(self.total_latency, "ms"),
                                "det_confidence": self.det_confidence,
                                "det_confidence_norm": self.normalize_value(self.det_confidence, "raw"),
                                "cls_confidence": self.cls_confidence,
                                "cls_confidence_norm": self.normalize_value(self.cls_confidence, "raw"),
                                "oom_old_batch": self.oom_old_batch,
                                "oom_new_batch": self.oom_new_batch,
                                "oom_old_accum": self.oom_old_accum,
                                "oom_new_accum": self.oom_new_accum,
                                "det_steps": self.det_steps,
                                "cls_steps": self.cls_steps,
                                "interleave_ratio": self.interleave_ratio,
                                "interleave_ratio_norm": self.normalize_value(self.interleave_ratio, "ratio"),
                                # 동적 학습 설정
                                "total_epochs": self.total_epochs,
                                "total_batches": self.total_batches,
                                # 확장된 Phase 1 지표들 (새로운 요구사항)
                                # 도메인별 Top-5 / Macro-F1
                                "single_top1": self.single_top1,
                                "single_top1_norm": self.normalize_value(self.single_top1, "percent"),
                                "single_top5": self.single_top5,
                                "single_top5_norm": self.normalize_value(self.single_top5, "percent"),
                                "single_macro_f1": self.single_macro_f1,
                                "single_macro_f1_norm": self.normalize_value(self.single_macro_f1, "percent"),
                                "combo_top1": self.combo_top1,
                                "combo_top1_norm": self.normalize_value(self.combo_top1, "percent"),
                                "combo_top5": self.combo_top5,
                                "combo_top5_norm": self.normalize_value(self.combo_top5, "percent"),
                                "combo_macro_f1": self.combo_macro_f1,
                                "combo_macro_f1_norm": self.normalize_value(self.combo_macro_f1, "percent"),
                                # Detection Recall / Precision
                                "det_recall": self.det_recall,
                                "det_recall_norm": self.normalize_value(self.det_recall, "percent"),
                                "det_precision": self.det_precision,
                                "det_precision_norm": self.normalize_value(self.det_precision, "percent"),
                                # 도메인별 Selected Confidence
                                "selected_conf_single": self.selected_conf_single,
                                "selected_conf_combo": self.selected_conf_combo,
                                # 레이턴시 퍼센타일
                                "latency_p50_det": self.latency_p50_det,
                                "latency_p95_det": self.latency_p95_det,
                                "latency_p99_det": self.latency_p99_det,
                                "latency_p50_crop": self.latency_p50_crop,
                                "latency_p95_crop": self.latency_p95_crop,
                                "latency_p99_crop": self.latency_p99_crop,
                                "latency_p50_cls": self.latency_p50_cls,
                                "latency_p95_cls": self.latency_p95_cls,
                                "latency_p99_cls": self.latency_p99_cls,
                                "latency_p50_total": self.latency_p50_total,
                                "latency_p95_total": self.latency_p95_total,
                                "latency_p99_total": self.latency_p99_total,
                                # VRAM Peak / Grad-Norm
                                "vram_peak_mb": self.vram_peak_mb,
                                "vram_peak_mb_norm": self.normalize_value(self.vram_peak_mb, "mb"),
                                "grad_norm_after": self.grad_norm_after,
                                "grad_norm_after_norm": self.normalize_value(self.grad_norm_after, "raw")
                            }
                            
                            # 큐에 저장하여 메인 이벤트 루프에서 처리
                            if not hasattr(self, 'message_queue'):
                                import asyncio
                                self.message_queue = []
                            self.message_queue.append(message_data)
                            
                            print(f"🔄 [DEBUG] 파싱 성공: {log_line[:100]}...")
                            
                        except Exception as broadcast_error:
                            print(f"❌ WebSocket 브로드캐스트 오류: {broadcast_error}")
                    
                    elif self.connected_clients:
                        # 클라이언트가 연결되어 있지만 파싱된 로그가 없는 경우
                        print(f"⚠️  [DEBUG] 클라이언트 {len(self.connected_clients)}개 연결됨, 하지만 파싱 실패")
                    
                    time.sleep(1)  # 1초마다 업데이트
                    
                except Exception as e:
                    print(f"bash_23 캡처 오류: {e}")
                    time.sleep(2)
        
        # 백그라운드 스레드로 실행
        capture_thread = threading.Thread(target=capture_bash23, daemon=True)
        capture_thread.start()
        
    async def start_training_with_logging(self, command: List[str]):
        """학습 시작하면서 실시간 로그 캡처"""
        print(f"🚀 실시간 로그 캡처 시작: {' '.join(command)}")
        print(f"📁 로그 파일: {self.log_file}")
        
        # 연결된 클라이언트들에게 새 학습 시작 알림
        await self.broadcast_new_training_start()
        
        # tee 명령어로 stdout을 파일과 실시간 스트리밍에 동시 출력
        tee_command = ["tee", self.log_file]
        
        # 학습 프로세스 실행
        self.training_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0  # 라인 버퍼링
        )
        
        # 실시간 로그 읽기 및 브로드캐스트
        while True:
            if self.training_process.poll() is not None:
                # 프로세스 종료
                break
                
            line = self.training_process.stdout.readline()
            if line:
                # 파일에 저장
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line)
                
                # 콘솔 출력
                print(line.rstrip())
                
                # WebSocket으로 브로드캐스트
                await self.broadcast_log(line.rstrip())
            
            await asyncio.sleep(0.01)  # CPU 사용률 조절
        
        return_code = self.training_process.returncode
        final_message = f"🏁 학습 완료 (종료 코드: {return_code})"
        print(final_message)
        await self.broadcast_log(final_message)
        
        return return_code
    
    async def broadcast_log(self, message: str):
        """모든 연결된 클라이언트에 로그 브로드캐스트"""
        if not self.connected_clients:
            return
            
        disconnect_clients = []
        for client in self.connected_clients:
            try:
                # KST 시간으로 통일
                kst = timezone(timedelta(hours=9))
                await client.send_text(json.dumps({
                    "timestamp": datetime.now(kst).isoformat(),
                    "message": message
                }))
            except:
                disconnect_clients.append(client)
        
        # 연결이 끊긴 클라이언트 제거
        for client in disconnect_clients:
            if client in self.connected_clients:
                self.connected_clients.remove(client)
    
    async def add_client(self, websocket: WebSocket):
        """새 클라이언트 연결"""
        await websocket.accept()
        self.connected_clients.append(websocket)
        
        # 새 학습 시작 메시지 전송 (기존 로그는 선택적으로만)
        try:
            kst = timezone(timedelta(hours=9))
            await websocket.send_text(json.dumps({
                "timestamp": datetime.now(kst).isoformat(),
                "message": "🔄 새로운 모니터링 세션 시작",
                "type": "session_start"
            }))
            
            # 활성 학습이 없으면 기존 로그 표시
            if not self.training_process or self.training_process.poll() is not None:
                historical_logs = await self.get_historical_logs()
                if historical_logs:
                    await websocket.send_text(json.dumps({
                        "timestamp": datetime.now(kst).isoformat(),
                        "message": "--- 📋 이전 세션 로그 (참조용) ---",
                        "type": "historical_header"
                    }))
                    for log in historical_logs[-50:]:  # 최근 50줄만
                        await websocket.send_text(json.dumps({
                            "timestamp": datetime.now(kst).isoformat(),
                            "message": log,
                            "historical": True
                        }))
                    await websocket.send_text(json.dumps({
                        "timestamp": datetime.now(kst).isoformat(),
                        "message": "--- 📋 이전 로그 끝 ---",
                        "type": "historical_footer"
                    }))
        except Exception as e:
            print(f"세션 시작 메시지 전송 실패: {e}")
            self.remove_client(websocket)
    
    async def get_historical_logs(self):
        """기존 학습 로그를 수집"""
        logs = []
        
        # 1) 실제 로그 파일들에서 전체 로그 읽기
        log_sources = [
            # 현재 설정된 로그 파일
            self.log_file,
            # 실제 학습 로그 파일
            "/home/max16/pillsnap_data/exp/exp01/logs/__main___20250823.log",
            "/tmp/pillsnap_training_20250823.log",
        ]
        
        for log_path in log_sources:
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            logs.append(line.rstrip())
                    break  # 첫 번째로 발견된 로그 파일 사용
                except:
                    continue
        
        # 2) 로그가 없으면 가장 최신 로그 파일 자동 탐지
        if not logs:
            try:
                import glob
                pattern = "/home/max16/pillsnap_data/exp/exp01/logs/*main*_*.log"
                log_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                
                if log_files:
                    latest_log = log_files[0]
                    print(f"📂 최신 로그 파일 사용: {latest_log}")
                    with open(latest_log, "r", encoding="utf-8") as f:
                        for line in f:
                            logs.append(line.rstrip())
            except:
                pass
        
        # 3) 여전히 로그가 없으면 기본 메시지
        if not logs:
            logs = [
                "📋 실시간 로그 스트리밍 대기 중...",
                "🔍 로그 파일을 찾을 수 없습니다.",
                "   새로운 학습이 시작되면 실시간으로 표시됩니다.",
                "",
                f"📁 모니터링 대상 로그 파일들:",
                f"   - {self.log_file}",
                f"   - /home/max16/pillsnap_data/exp/exp01/logs/__main___*.log",
            ]
        
        # 로그가 너무 길면 마지막 200줄만 반환
        return logs[-200:] if len(logs) > 200 else logs
    
    def remove_client(self, websocket: WebSocket):
        """클라이언트 연결 해제"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
    
    async def broadcast_new_training_start(self):
        """모든 클라이언트에게 새 학습 시작 알림"""
        if not self.connected_clients:
            return
            
        kst = timezone(timedelta(hours=9))
        message = {
            "timestamp": datetime.now(kst).isoformat(),
            "message": "🚀 새로운 학습 프로세스가 시작됩니다!",
            "type": "new_training_start"
        }
        
        disconnected = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except Exception:
                disconnected.append(client)
        
        # 연결이 끊어진 클라이언트 제거
        for client in disconnected:
            self.remove_client(client)

    async def watch_log_file(self):
        """실시간 로그 파일 감시 및 브로드캐스트"""
        if not Path(self.log_file).exists():
            print(f"로그 파일이 존재하지 않습니다: {self.log_file}")
            return
            
        print(f"📁 로그 파일 감시 시작: {self.log_file}")
        
        while True:
            try:
                # 메시지 큐 처리 (백그라운드 스레드에서 생성된 메시지)
                if hasattr(self, 'message_queue') and self.message_queue:
                    messages_to_send = self.message_queue.copy()
                    self.message_queue.clear()
                    
                    for message_data in messages_to_send:
                        await self.broadcast_to_clients(message_data)
                        print(f"📡 [DEBUG] WebSocket 전송: {message_data.get('message', '')[:50]}...")
                
                # 파일 크기 체크 (기존 로직 유지)
                current_size = Path(self.log_file).stat().st_size
                
                if current_size > self.last_position:
                    # 새로운 내용이 있음 - 읽기
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                    
                    # 새로운 라인을 클라이언트들에게 전송 (백업 전송)
                    if new_lines and self.connected_clients:
                        kst = timezone(timedelta(hours=9))
                        for line in new_lines:
                            line = line.strip()
                            if line:  # 빈 줄 제외
                                await self.broadcast_to_clients({
                                    "timestamp": datetime.now(kst).isoformat(),
                                    "message": line,
                                    "type": "file_realtime"
                                })
                
                # 0.5초 대기 (더 빠른 응답)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"로그 파일 감시 오류: {e}")
                await asyncio.sleep(2)  # 에러 시 2초 대기

    async def broadcast_to_clients(self, message: dict):
        """모든 연결된 클라이언트에게 메시지 브로드캐스트"""
        if not self.connected_clients:
            return
            
        disconnected = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except Exception:
                disconnected.append(client)
        
        # 연결이 끊어진 클라이언트 제거
        for client in disconnected:
            self.remove_client(client)

# 글로벌 로거 인스턴스
logger = RealtimeTrainingLogger()

# FastAPI 앱
app = FastAPI(title="실시간 학습 로그 스트리밍")

# HTML 대시보드 파일에서 읽기
def load_dashboard_html():
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """<!DOCTYPE html><html><body><h1>Dashboard file not found</h1></body></html>"""

@app.get("/")
async def get_dashboard():
    """실시간 로그 대시보드"""
    return HTMLResponse(load_dashboard_html())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 엔드포인트"""
    await logger.add_client(websocket)
    try:
        while True:
            await websocket.receive_text()  # 연결 유지
    except WebSocketDisconnect:
        logger.remove_client(websocket)

@app.get("/api/system")
async def get_system_status():
    """시스템 상태 API (기존 simple_real_monitor.py와 호환)"""
    try:
        import psutil
        import subprocess
        
        # 학습 프로세스 찾기
        training_pid = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'train_stage3_two_stage' in cmdline:
                        training_pid = proc.info['pid']
                        break
            except:
                continue
        
        if not training_pid:
            return {"status": "not_running", "message": "학습 프로세스를 찾을 수 없음"}
        
        # 프로세스 정보
        process = psutil.Process(training_pid)
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        
        # GPU 정보
        gpu_info = {}
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                gpu_info = {
                    "utilization": int(values[0]),
                    "memory_used_mb": int(values[1]),
                    "memory_total_mb": int(values[2]),
                    "temperature": int(values[3])
                }
        except:
            gpu_info = {"error": "GPU 정보 읽기 실패"}
        
        # KST 시간 설정
        kst = timezone(timedelta(hours=9))
        return {
            "status": "running",
            "timestamp": datetime.now(kst).isoformat(),
            "process": {
                "pid": training_pid,
                "cpu_percent": cpu_percent,
                "memory_gb": round(memory_gb, 2),
                "running_time": time.time() - process.create_time()
            },
            "gpu": gpu_info,
            "message": f"실제 학습 진행 중 (PID: {training_pid})"
        }
        
    except Exception as e:
        kst = timezone(timedelta(hours=9))
        return {
            "status": "error", 
            "message": f"시스템 상태 오류: {e}",
            "timestamp": datetime.now(kst).isoformat()
        }

@app.post("/start_training")
async def start_training(command: List[str]):
    """학습 시작 API"""
    asyncio.create_task(logger.start_training_with_logging(command))
    return {"status": "started", "command": command, "log_file": logger.log_file}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="실시간 학습 로그 스트리밍")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--command", nargs="+", help="실행할 학습 명령어")
    
    args = parser.parse_args()
    
    print(f"🚀 실시간 로그 서버 시작 (포트: {args.port})")
    print(f"📊 브라우저에서 http://localhost:{args.port} 접속")
    print(f"📝 감지된 로그 파일: {logger.log_file}")
    
    if args.command:
        # 명령어가 주어진 경우 바로 학습 시작
        async def start_training_on_startup():
            await asyncio.sleep(1)  # 서버 시작 대기
            await logger.start_training_with_logging(args.command)
        
        asyncio.create_task(start_training_on_startup())
    
    # FastAPI 이벤트 핸들러로 파일 감시 태스크 등록
    @app.on_event("startup")
    async def on_startup():
        print(f"📁 로그 파일 감시 태스크 시작: {logger.log_file}")
        asyncio.create_task(logger.watch_log_file())
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")