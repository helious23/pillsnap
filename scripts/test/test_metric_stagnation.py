#!/usr/bin/env python3
"""
메트릭 정체 감지 테스트
목적: 5 에폭 동안 mAP 개선이 1% 미만일 때 감지
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
import json
from datetime import datetime

def check_stagnation():
    """메트릭 정체 감지"""
    print("=== 메트릭 정체 감지 테스트 ===")
    
    # 1. state.json 읽기
    state_file = Path("/home/max16/pillsnap/artifacts/yolo/stage3/state.json")
    if not state_file.exists():
        print("❌ state.json 파일이 없습니다")
        return False
    
    with open(state_file) as f:
        state = json.load(f)
    
    # 2. metrics_history 확인
    history = state.get("metrics_history", [])
    if len(history) < 5:
        print(f"⚠️ 히스토리가 충분하지 않음: {len(history)} 에폭")
        return False
    
    # 3. 최근 5개 mAP 가져오기
    recent_5 = history[-5:]
    map_values = [h["metrics"].get("map50", 0) for h in recent_5]
    
    print(f"최근 5개 에폭 mAP@0.5: {map_values}")
    
    # 4. 개선율 계산
    if map_values[0] > 0:
        improvement = (map_values[-1] - map_values[0]) / map_values[0] * 100
    else:
        improvement = 0
    
    print(f"5 에폭간 개선율: {improvement:.2f}%")
    
    # 5. 정체 판단
    STAGNATION_THRESHOLD = 1.0  # 1% 미만 개선시 정체
    is_stagnant = abs(improvement) < STAGNATION_THRESHOLD
    
    if is_stagnant:
        print(f"⚠️ 메트릭 정체 감지됨! 개선율 {improvement:.2f}% < {STAGNATION_THRESHOLD}%")
        print("제안: 학습률 감소 또는 조기 종료 고려")
    else:
        print(f"✅ 정상적으로 학습 진행 중 (개선율: {improvement:.2f}%)")
    
    # 6. 표준편차로 변동성 체크
    import numpy as np
    std_dev = np.std(map_values)
    print(f"mAP 표준편차: {std_dev:.4f}")
    
    if std_dev < 0.005:  # 0.5% 미만 변동
        print("⚠️ 메트릭이 거의 변하지 않음 - 학습 정체 가능성")
    
    return not is_stagnant

if __name__ == "__main__":
    success = check_stagnation()
    sys.exit(0 if success else 1)