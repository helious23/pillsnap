#!/bin/bash
# 간단한 학습 모니터링

multitail \
  -l "bash -c 'while true; do 
    echo \"[GPU] $(date \"+%H:%M:%S\")\"; 
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | sed \"s/,/ | GPU: /g\" | sed \"s/^/GPU: /\"; 
    sleep 2; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[PROCESS] $(date \"+%H:%M:%S\")\"; 
    ps aux | grep python.*train_classification | grep -v grep | wc -l | xargs echo \"Active processes:\"; 
    ps aux | grep python.*train_classification | grep -v grep | head -1 | awk \"{print \"Main: PID \" \$2 \" CPU \" \$3 \"% MEM \" \$4 \"%\"}\"; 
    sleep 3; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[TRAINING] $(date \"+%H:%M:%S\")\"; 
    echo \"현재 학습 중...\";
    # 메모리 사용량으로 진행도 추정
    mem_usage=\$(ps aux | grep python.*train_classification | grep -v grep | head -1 | awk \"{print \$4}\");
    if [ -n \"\$mem_usage\" ]; then
      echo \"메모리 사용률: \${mem_usage}%\";
    fi;
    # 새로운 리포트 확인
    latest=\$(ls -t /home/max16/pillsnap/artifacts/reports/training_progress_reports/*.json 2>/dev/null | head -1);
    if [ -n \"\$latest\" ]; then
      echo \"Latest report: \$(basename \"\$latest\")\";
    else
      echo \"아직 리포트 없음 (학습 진행 중)\";
    fi;
    sleep 4; 
  done'"