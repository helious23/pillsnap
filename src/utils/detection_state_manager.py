#!/usr/bin/env python3
"""
Detection 학습 상태 관리 유틸리티
YOLO 누적 학습을 위한 상태 추적 및 관리
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import fcntl
import tempfile
import shutil
from datetime import datetime


class DetectionStateManager:
    """Detection 학습 상태 관리자"""
    
    def __init__(self, state_dir: Path = None):
        """
        Args:
            state_dir: 상태 파일을 저장할 디렉토리 (기본: artifacts/yolo/stage3)
        """
        if state_dir is None:
            state_dir = Path("/home/max16/pillsnap/artifacts/yolo/stage3")
        
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        
        # 기본 상태 스키마
        self.default_state = {
            "det_epochs_done": 0,
            "last_metrics": {
                "map50": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "box_loss": 0.0,
                "cls_loss": 0.0,
                "dfl_loss": 0.0
            },
            "last_updated": None,
            "last_pt_timestamp": None,
            "history": []  # 최근 10개 사이클 기록
        }
    
    def load_state(self) -> Dict[str, Any]:
        """상태 파일 로드 (atomic read)"""
        if not self.state_file.exists():
            return self.default_state.copy()
        
        try:
            with open(self.state_file, 'r') as f:
                # 파일 락 획득
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # 기본값으로 누락된 키 채우기
                for key, value in self.default_state.items():
                    if key not in state:
                        state[key] = value
                
                return state
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 상태 파일 로드 실패: {e}")
            return self.default_state.copy()
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """상태 파일 저장 (atomic write)"""
        try:
            # 타임스탬프 업데이트
            state["last_updated"] = datetime.now().isoformat()
            
            # 임시 파일에 먼저 쓰기
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                # 파일 락 획득
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(state, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # 원자적으로 교체
            shutil.move(str(temp_file), str(self.state_file))
            return True
            
        except Exception as e:
            print(f"❌ 상태 파일 저장 실패: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    def increment_epochs(self, state: Dict[str, Any]) -> int:
        """에폭 카운터 증가 및 다음 목표 반환"""
        current = state.get("det_epochs_done", 0)
        target = current + 1
        return target
    
    def update_metrics(self, state: Dict[str, Any], metrics: Dict[str, float]):
        """메트릭 업데이트"""
        if "last_metrics" not in state:
            state["last_metrics"] = {}
        
        state["last_metrics"].update(metrics)
        
        # 히스토리에 추가 (최근 10개만 유지)
        history_entry = {
            "epoch": state.get("det_epochs_done", 0),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        }
        
        if "history" not in state:
            state["history"] = []
        
        state["history"].append(history_entry)
        state["history"] = state["history"][-10:]  # 최근 10개만 유지
    
    def check_last_pt_updated(self, last_pt_path: Path, state: Dict[str, Any]) -> bool:
        """last.pt 파일이 업데이트되었는지 확인"""
        if not last_pt_path.exists():
            return False
        
        current_timestamp = last_pt_path.stat().st_mtime
        last_timestamp = state.get("last_pt_timestamp", 0)
        
        updated = current_timestamp > last_timestamp
        
        if updated:
            state["last_pt_timestamp"] = current_timestamp
        
        return updated
    
    def detect_stalled_training(self, state: Dict[str, Any], threshold: int = 3) -> bool:
        """학습이 정체되었는지 감지 (메트릭이 threshold 사이클 동안 변화 없음)"""
        history = state.get("history", [])
        
        if len(history) < threshold:
            return False
        
        # 최근 threshold개의 메트릭 비교
        recent = history[-threshold:]
        metrics_to_check = ["map50", "box_loss", "cls_loss", "dfl_loss"]
        
        for metric in metrics_to_check:
            values = [h.get("metrics", {}).get(metric, 0) for h in recent]
            
            # 모든 값이 동일한지 확인 (소수점 4자리까지)
            if len(set(round(v, 4) for v in values)) > 1:
                return False  # 변화가 있음
        
        return True  # 모든 메트릭이 동일 = 정체
    
    def format_summary(self, state: Dict[str, Any], updated: bool, deltas: Dict[str, float]) -> str:
        """한줄 요약 생성"""
        m = state.get("last_metrics", {})
        
        summary = (
            f"DET_CHECK | "
            f"done={state.get('det_epochs_done', 0)} | "
            f"last.pt_updated={'Yes' if updated else 'No'} | "
            f"map50={m.get('map50', 0):.3f} | "
            f"Δbox={deltas.get('box_loss', 0):.4f} "
            f"Δcls={deltas.get('cls_loss', 0):.4f} "
            f"Δdfl={deltas.get('dfl_loss', 0):.4f}"
        )
        
        return summary
    
    def calculate_deltas(self, state: Dict[str, Any]) -> Dict[str, float]:
        """이전 사이클 대비 변화량 계산"""
        history = state.get("history", [])
        
        if len(history) < 2:
            return {"box_loss": 0, "cls_loss": 0, "dfl_loss": 0}
        
        current = history[-1].get("metrics", {})
        previous = history[-2].get("metrics", {})
        
        deltas = {}
        for key in ["box_loss", "cls_loss", "dfl_loss"]:
            deltas[key] = current.get(key, 0) - previous.get(key, 0)
        
        return deltas