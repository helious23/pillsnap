#!/usr/bin/env python3
"""
Detection State Manager 및 Robust CSV Parser 테스트
YOLO 누적 학습 시스템 동작 확인
"""

import sys
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
from src.utils.detection_state_manager import DetectionStateManager
from src.utils.robust_csv_parser import RobustCSVParser
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def test_state_manager():
    """State Manager 테스트"""
    print("\n=== Detection State Manager 테스트 ===")
    
    # State Manager 초기화
    state_manager = DetectionStateManager()
    
    # 1. 초기 상태 로드
    state = state_manager.load_state()
    print(f"현재 완료된 Detection 에폭: {state.get('det_epochs_done', 0)}")
    
    # 2. 다음 목표 에폭 계산
    target = state_manager.increment_epochs(state)
    print(f"다음 목표 에폭: {target}")
    
    # 3. 메트릭 업데이트
    test_metrics = {
        'map50': 0.456,
        'precision': 0.678,
        'recall': 0.789,
        'box_loss': 0.123,
        'cls_loss': 0.234,
        'dfl_loss': 0.345
    }
    state_manager.update_metrics(state, test_metrics)
    print(f"메트릭 업데이트 완료")
    
    # 4. 변화량 계산
    deltas = state_manager.calculate_deltas(state)
    print(f"손실 변화량: box={deltas.get('box_loss', 0):.4f}, "
          f"cls={deltas.get('cls_loss', 0):.4f}, "
          f"dfl={deltas.get('dfl_loss', 0):.4f}")
    
    # 5. 요약 출력
    summary = state_manager.format_summary(state, updated=True, deltas=deltas)
    print(f"요약: {summary}")
    
    # 6. 상태 저장
    state['det_epochs_done'] = target
    if state_manager.save_state(state):
        print("✅ State 저장 성공")
    else:
        print("❌ State 저장 실패")
    
    # 7. 학습 정체 감지 테스트
    if state_manager.detect_stalled_training(state, threshold=3):
        print("⚠️ 학습 정체 감지됨")
    else:
        print("✅ 학습 정상 진행 중")

def test_csv_parser():
    """CSV Parser 테스트"""
    print("\n=== Robust CSV Parser 테스트 ===")
    
    csv_parser = RobustCSVParser(logger)
    
    # YOLO results.csv 경로
    csv_path = Path("/home/max16/pillsnap/artifacts/yolo/stage3/results.csv")
    
    if csv_path.exists():
        # CSV 파싱
        metrics = csv_parser.parse_results_csv(
            csv_path=csv_path,
            max_retries=3,
            retry_delay=1.0
        )
        
        print(f"파싱된 메트릭:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 유효성 검사
        if csv_parser.validate_metrics(metrics):
            print("✅ 메트릭 유효성 검사 통과")
        else:
            print("❌ 메트릭 유효성 검사 실패")
        
        # 히스토리 로드
        history_df = csv_parser.get_all_metrics_history(csv_path)
        if not history_df.empty:
            print(f"\n전체 히스토리: {len(history_df)} 에폭")
            print(history_df.tail(3))
    else:
        print(f"CSV 파일이 없습니다: {csv_path}")
        print("YOLO 학습을 먼저 실행해주세요.")

def check_yolo_artifacts():
    """YOLO 아티팩트 확인"""
    print("\n=== YOLO 아티팩트 확인 ===")
    
    yolo_dir = Path("/home/max16/pillsnap/artifacts/yolo/stage3")
    
    # weights 디렉토리
    weights_dir = yolo_dir / "weights"
    if weights_dir.exists():
        for pt_file in weights_dir.glob("*.pt"):
            size_mb = pt_file.stat().st_size / (1024 * 1024)
            print(f"  {pt_file.name}: {size_mb:.1f} MB")
    else:
        print("  weights 디렉토리가 없습니다")
    
    # results.csv
    results_csv = yolo_dir / "results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        print(f"  results.csv: {len(df)} 행")
    else:
        print("  results.csv가 없습니다")
    
    # state.json
    state_file = yolo_dir / "state.json"
    if state_file.exists():
        import json
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"  state.json: {state.get('det_epochs_done', 0)} 에폭 완료")
    else:
        print("  state.json이 없습니다 (새로 생성됨)")

if __name__ == "__main__":
    print("=" * 60)
    print("Detection State 관리 시스템 테스트")
    print("=" * 60)
    
    # 1. State Manager 테스트
    test_state_manager()
    
    # 2. CSV Parser 테스트
    test_csv_parser()
    
    # 3. YOLO 아티팩트 확인
    check_yolo_artifacts()
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)