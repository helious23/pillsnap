#!/bin/bash
# 간단한 학습 모니터링 (문법 오류 수정)

watch -n 3 '
echo "=== $(date) ===";
echo "GPU:";
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits;
echo "TRAINING:";
ps aux | grep train_classification_stage | grep -v grep | head -1 | awk "{print \"CPU: \" \$3 \"% MEM: \" \$4 \"%\"}";
echo "REPORTS:";
ls -t /home/max16/pillsnap/artifacts/reports/training_progress_reports/*.json 2>/dev/null | wc -l;
echo "STATUS: Learning in progress...";
'