#!/usr/bin/env python3
"""
TensorBoard 실시간 로깅 테스트
목적: TensorBoard 로깅 기능 확인 및 설정
"""

import sys
import os
sys.path.insert(0, '/home/max16/pillsnap')

from pathlib import Path
from datetime import datetime
import subprocess

def test_tensorboard_logging():
    """TensorBoard 로깅 테스트"""
    print("=== TensorBoard 실시간 로깅 테스트 ===")
    
    # 1. TensorBoard 로그 디렉토리 확인
    log_dirs = [
        Path("/home/max16/pillsnap/runs/detect/train"),
        Path("/home/max16/pillsnap/artifacts/tensorboard"),
        Path("/home/max16/pillsnap/logs/tensorboard")
    ]
    
    print("\nTensorBoard 로그 디렉토리 확인:")
    found_logs = False
    
    for log_dir in log_dirs:
        if log_dir.exists():
            # 이벤트 파일 찾기
            event_files = list(log_dir.glob("**/*tfevents*"))
            if event_files:
                print(f"✅ {log_dir}: {len(event_files)}개 이벤트 파일")
                for f in event_files[:3]:  # 처음 3개만 표시
                    print(f"    - {f.name}")
                found_logs = True
            else:
                print(f"⚠️ {log_dir}: 이벤트 파일 없음")
        else:
            print(f"❌ {log_dir}: 디렉토리 없음")
    
    # 2. YOLO 학습 로그 확인
    yolo_log_path = Path("/home/max16/pillsnap/runs/detect/train")
    if yolo_log_path.exists():
        print(f"\nYOLO 학습 로그 확인:")
        
        # results.csv 확인
        csv_path = yolo_log_path / "results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"  ✅ results.csv: {len(df)} 행")
            
            # TensorBoard 호환 형식으로 변환 가능 여부
            if 'epoch' in df.columns and 'metrics/mAP50(B)' in df.columns:
                print(f"  ✅ TensorBoard 변환 가능한 메트릭 존재")
        
        # args.yaml 확인 (TensorBoard 설정)
        args_path = yolo_log_path / "args.yaml"
        if args_path.exists():
            import yaml
            with open(args_path) as f:
                args = yaml.safe_load(f)
                if 'plots' in args:
                    print(f"  Plots 설정: {args['plots']}")
    
    # 3. TensorBoard 프로세스 확인
    print("\nTensorBoard 프로세스 확인:")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "tensorboard"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            print("✅ TensorBoard 프로세스 실행 중")
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"  PID: {pid}")
        else:
            print("⚠️ TensorBoard 프로세스 없음")
            print("\n시작 명령어:")
            print("  tensorboard --logdir=/home/max16/pillsnap/runs/detect/train --port=6006")
    except Exception as e:
        print(f"❌ 프로세스 확인 실패: {e}")
    
    # 4. 실시간 로깅 설정 제안
    print("\n실시간 로깅 개선 제안:")
    
    if not found_logs:
        print("1. YOLO 학습시 TensorBoard 활성화:")
        print("   model.train(..., plots=True)")
        print("")
        print("2. 수동 TensorBoard 로깅 추가:")
        print("   from torch.utils.tensorboard import SummaryWriter")
        print("   writer = SummaryWriter('runs/detect/train')")
        print("   writer.add_scalar('mAP50', value, epoch)")
    else:
        print("✅ TensorBoard 로그가 존재합니다")
        print("웹 브라우저에서 확인: http://localhost:6006")
    
    # 5. CSV to TensorBoard 변환 스크립트
    print("\nCSV → TensorBoard 변환:")
    csv_path = Path("/home/max16/pillsnap/runs/detect/train/results.csv")
    if csv_path.exists():
        print("✅ results.csv 존재 - TensorBoard 변환 가능")
        
        # 변환 코드 예시
        convert_script = Path("/home/max16/pillsnap/scripts/test/csv_to_tensorboard.py")
        with open(convert_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else '/home/max16/pillsnap/runs/detect/train/results.csv'
log_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/max16/pillsnap/runs/detect/train/tb_logs'

df = pd.read_csv(csv_path)
writer = SummaryWriter(log_dir)

for idx, row in df.iterrows():
    epoch = int(row['epoch'])
    
    # Detection 메트릭
    if 'metrics/mAP50(B)' in row:
        writer.add_scalar('Detection/mAP50', row['metrics/mAP50(B)'], epoch)
        writer.add_scalar('Detection/Precision', row['metrics/precision(B)'], epoch)
        writer.add_scalar('Detection/Recall', row['metrics/recall(B)'], epoch)
    
    # 학습 손실
    if 'train/box_loss' in row:
        writer.add_scalar('Loss/Box', row['train/box_loss'], epoch)
        writer.add_scalar('Loss/Class', row['train/cls_loss'], epoch)
        writer.add_scalar('Loss/DFL', row['train/dfl_loss'], epoch)

writer.close()
print(f'✅ TensorBoard 로그 생성: {log_dir}')
print(f'실행: tensorboard --logdir={log_dir} --port=6006')
""")
        convert_script.chmod(0o755)
        print(f"  변환 스크립트 생성: {convert_script}")
        print(f"  실행: python {convert_script}")
    
    print("\n✅ TensorBoard 로깅 테스트 완료")
    return True

if __name__ == "__main__":
    success = test_tensorboard_logging()
    sys.exit(0 if success else 1)