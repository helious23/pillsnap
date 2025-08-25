#!/usr/bin/env python3
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
