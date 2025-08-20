# Scripts 디렉토리 구조

## 📁 구조 개요

```
scripts/
├── core/                   # 핵심 유틸리티
│   ├── python_safe.sh     # 안전한 Python 실행
│   ├── setup_aliases.sh   # 편의 별칭 설정
│   ├── setup_venv.sh      # 가상환경 설정
│   └── update_docs.sh     # 문서 업데이트
│
├── stage1/                 # Stage 1 관련
│   ├── migrate_stage1_images_only.sh
│   └── migrate_stage1_to_ssd.sh
│
├── stage2/                 # Stage 2 관련
│   ├── run_stage2_sampling.py         # Stage 2 샘플링
│   ├── migrate_stage2_data.py         # Stage 2 데이터 이전
│   ├── monitor_stage2_migration.sh    # 실시간 모니터링
│   ├── quick_status.sh               # 빠른 상태 확인
│   └── check_stage_overlap.py        # Stage 중복 확인
│
├── monitoring/             # 모니터링 도구
│   ├── monitor_deadlock.sh
│   ├── monitor_simple.sh
│   ├── monitor_training.sh
│   ├── simple_monitor.sh
│   ├── simple_watch.sh
│   ├── live_log.sh
│   └── watch_training.sh
│
├── training/               # 학습 관련
│   ├── train_and_monitor.sh
│   └── train_with_monitor.sh
│
├── data/                   # 데이터 처리 (기존 유지)
├── deployment/             # 배포 관련 (기존 유지)
└── testing/               # 테스트 관련 (기존 유지)
```

## 🚀 빠른 사용법

### Stage 2 작업
```bash
# Stage 2 샘플링
./scripts/stage2/run_stage2_sampling.py

# Stage 2 데이터 이전
./scripts/stage2/migrate_stage2_data.py

# 진행 상황 모니터링
./scripts/stage2/quick_status.sh
./scripts/stage2/monitor_stage2_migration.sh
```

### 모니터링
```bash
# 학습 모니터링
./scripts/monitoring/monitor_training.sh

# 데드락 모니터링  
./scripts/monitoring/monitor_deadlock.sh
```

### 핵심 도구
```bash
# 안전한 Python 실행
./scripts/core/python_safe.sh [명령어]

# 환경 설정
./scripts/core/setup_venv.sh
```
