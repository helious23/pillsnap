#!/bin/bash
# Stage 2 간단한 학습 모니터링

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
    echo \"[STAGE 2 TRAINING] $(date \"+%H:%M:%S\")\"; 
    echo \"현재 Stage 2 학습 중...\";
    # 메모리 사용량으로 진행도 추정
    mem_usage=\$(ps aux | grep python.*train_classification | grep -v grep | head -1 | awk \"{print \$4}\");
    if [ -n \"\$mem_usage\" ]; then
      echo \"메모리 사용률: \${mem_usage}%\";
    fi;
    # Stage 2 리포트 확인
    latest=\$(ls -t /home/max16/pillsnap/artifacts/stage2/reports/training_progress_reports/*.json 2>/dev/null | head -1);
    if [ -n \"\$latest\" ]; then
      echo \"Latest report: \$(basename \"\$latest\")\";
    else
      echo \"아직 리포트 없음 (학습 진행 중)\";
    fi;
    # Stage 2 manifest 정보
    manifest_file=\"/home/max16/pillsnap/artifacts/stage2/manifest_ssd.csv\";
    if [ -f \"\$manifest_file\" ]; then
      samples=\$(wc -l < \"\$manifest_file\");
      samples=\$((samples - 1));  # 헤더 제외
      echo \"Manifest: \${samples}개 샘플\";
    fi;
    sleep 4; 
  done'" \
  -l "bash -c 'while true; do 
    echo \"[SYSTEM] $(date \"+%H:%M:%S\")\"; 
    echo \"=== 메모리 사용량 ===\";
    free -h | grep Mem | awk \"{print \"메모리: \" \$3 \"/\" \$2}\";
    echo \"=== SSD 사용량 ===\";
    df -h /home/max16/ssd_pillsnap 2>/dev/null | tail -1 | awk \"{print \"SSD: \" \$3 \"/\" \$2 \" (\" \$5 \" 사용)\"}\" || echo \"SSD: 경로 없음\";
    echo \"=== Stage 2 데이터 ===\";
    if [ -d \"/home/max16/ssd_pillsnap/dataset/data/train/images/single\" ]; then
      k_codes=\$(ls /home/max16/ssd_pillsnap/dataset/data/train/images/single | wc -l);
      echo \"K-코드: \${k_codes}개\";
    fi;
    if [ -d \"/home/max16/ssd_pillsnap/dataset/data/train/labels/single\" ]; then
      echo \"라벨: 준비됨\";
    else
      echo \"라벨: 없음\";
    fi;
    sleep 8; 
  done'"