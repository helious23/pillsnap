#!/bin/bash
# Stage 2 학습 통합 모니터링 스크립트

multitail \
  -l "bash -c 'while true; do echo \"[$(date \"+%H:%M:%S\")] GPU STATUS\"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | sed \"s/,/ | /g\" | head -1; sleep 2; done'" \
  -l "bash -c 'while true; do echo; echo \"[$(date \"+%H:%M:%S\")] PROCESS STATUS\"; ps aux | grep train_classification | grep -v grep | head -3 | awk \"{printf \\\"%s %s CPU:%s MEM:%s CMD:%s\\n\\\", \\\$2, \\\$8, \\\$3, \\\$4, \\\$11}\"; echo; sleep 3; done'" \
  -l "bash -c 'while true; do 
      echo \"[$(date \"+%H:%M:%S\")] STAGE 2 TRAINING PROGRESS\"; 
      latest_report=\$(ls -t /home/max16/pillsnap/artifacts/stage2/reports/training_progress_reports/*.json 2>/dev/null | head -1);
      if [ -n \"\$latest_report\" ]; then
        echo \"Report: \$(basename \"\$latest_report\")\";
        python3 -c \"
import json, sys
try:
    with open('\$latest_report') as f: data = json.load(f)
    print(f'분류 정확도: {data[\\\"classification_metrics\\\"][\\\"final_accuracy\\\"]*100:.1f}% (목표: {data[\\\"config\\\"][\\\"target_classification_accuracy\\\"]*100:.1f}%)')
    if \\\"detection_metrics\\\" in data:
        print(f'검출 mAP: {data[\\\"detection_metrics\\\"][\\\"final_map\\\"]*100:.1f}% (목표: {data[\\\"config\\\"][\\\"target_detection_map\\\"]*100:.1f}%)')
    print(f'학습 시간: {data[\\\"total_training_time_minutes\\\"]*60:.0f}초')
    print(f'목표 달성: {data[\\\"validation_results\\\"][\\\"stage2_completed\\\"]}')
    print(f'클래스 수: {data[\\\"config\\\"][\\\"num_classes\\\"]}개')
except: print('리포트 파싱 실패')
\";
      else
        echo \"리포트 파일 없음\";
      fi;
      echo;
      sleep 5; 
    done'" \
  -l "bash -c 'while true; do 
      echo \"[$(date \"+%H:%M:%S\")] SYSTEM STATUS\"; 
      echo \"=== MEMORY ===\";
      free -h | grep -E \"Mem:|Swap:\" | awk \"{printf \\\"%s: %s/%s\\n\\\", \\\$1, \\\$3, \\\$2}\";
      echo \"=== DISK ===\";
      df -h /home/max16/ssd_pillsnap 2>/dev/null | tail -1 | awk \"{printf \\\"SSD: %s/%s (%s used)\\n\\\", \\\$3, \\\$2, \\\$5}\" || echo \"SSD: 경로 없음\";
      echo \"=== STAGE 2 MANIFEST ===\";
      manifest_file=\"/home/max16/pillsnap/artifacts/stage2/manifest_ssd.csv\";
      if [ -f \"\$manifest_file\" ]; then
        samples=\$(wc -l < \"\$manifest_file\");
        samples=\$((samples - 1));  # 헤더 제외
        echo \"Manifest 샘플: \${samples}개\";
      else
        echo \"Manifest 파일 없음\";
      fi;
      echo;
      sleep 10; 
    done'"