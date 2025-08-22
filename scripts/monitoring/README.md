# PillSnap Monitoring Scripts

PillSnap ML 학습 과정을 모니터링하기 위한 스크립트 모음입니다.

## 🚀 주요 스크립트

### 1. Universal Training Monitor (추천) ⭐
**파일:** `universal_training_monitor.sh`  
**용도:** 모든 Stage(1-4) 학습을 실시간으로 모니터링하는 통합 스크립트

```bash
# 기본 사용법 (자동 Stage 감지)
./scripts/monitoring/universal_training_monitor.sh
# 또는 별칭 사용 (추천)
monitor

# 특정 Stage 지정
./scripts/monitoring/universal_training_monitor.sh --stage 2
# 또는 별칭 사용
mon2

# 옵션 설정
./scripts/monitoring/universal_training_monitor.sh --interval 1 --lines 15
# 또는 별칭 사용 (빠른 새로고침)
monfast

# 도움말
./scripts/monitoring/universal_training_monitor.sh --help
```

**기능:**
- ✅ 실시간 학습 로그 출력
- ✅ GPU 상태 및 메모리 사용량
- ✅ 프로세스 상태 및 리소스 사용률  
- ✅ Stage별 진행상황 및 성능 지표
- ✅ 자동 Stage 감지 및 맞춤형 정보 표시
- ✅ 아름다운 컬러 출력

### 2. Quick Status Check
**파일:** `quick_status.sh`  
**용도:** 현재 학습 상태를 빠르게 확인

```bash
./scripts/monitoring/quick_status.sh
# 또는 별칭 사용 (추천)
status
```

**출력 예시:**
```
🔍 PillSnap 빠른 상태 확인
=================================
📊 학습 프로세스: 실행 중 ✅
PID: 12345, CPU: 85.2%, MEM: 12.3%

🎮 GPU 상태: 사용 가능 ✅
  NVIDIA GeForce RTX 5080: 95% 사용률, 8192/15469MB (53% 메모리)

🎯 완료된 Stage:
  Stage 1: 완료 ✅ (50 클래스)
    정확도: 74.9%
  Stage 2: 완료 ✅ (250 클래스)  
    정확도: 82.9%
  Stage 3: 미완료 ⏳
  Stage 4: 미완료 ⏳

💾 디스크 공간:
  사용률: 45% (850G 사용됨, 1.2T 사용 가능)
```

## ⌨️ 편리한 별칭 명령어

별칭이 설정되어 있다면 다음과 같이 간단하게 사용할 수 있습니다:

### 기본 모니터링
```bash
monitor     # 통합 모니터링 (실시간)
mon         # 통합 모니터링 (짧은 버전)
status      # 빠른 상태 확인
st          # 빠른 상태 확인 (짧은 버전)
```

### Stage별 모니터링  
```bash
mon1        # Stage 1 전용 모니터링
mon2        # Stage 2 전용 모니터링  
mon3        # Stage 3 전용 모니터링
mon4        # Stage 4 전용 모니터링
```

### 특별 기능
```bash
monfast     # 1초마다 빠른 새로고침 모니터링
gpu         # GPU 정보 (nvidia-smi)
gpuw        # GPU 정보 실시간 감시 (1초마다)
```

### 별칭 설정 방법
```bash
# 자동 설정 (한 번만)
./scripts/monitoring/setup_aliases.sh

# 수동으로 ~/.bashrc에 추가 후
source ~/.bashrc
# 또는 새 터미널 열기
```

## 📂 폴더 구조

```
scripts/monitoring/
├── universal_training_monitor.sh  # 🌟 메인 통합 모니터링 스크립트
├── quick_status.sh               # 빠른 상태 확인
├── live_log.sh                   # 실시간 로그만 출력
├── monitor_simple.sh             # 간단한 watch 기반 모니터링
├── monitor_training.sh           # 기본 모니터링
├── watch_current_training.sh     # 현재 프로세스 추적
├── simple_monitor.sh            # 레거시 (새 버전 권장)
├── _archived/                   # 아카이브된 스크립트들
└── README.md                    # 이 파일
```

## 🎯 Stage별 사용 예시

### Stage 1 학습 모니터링
```bash
# Stage 1 학습 시작
source .venv/bin/activate
python -m src.training.train_classification_stage --stage 1 --epochs 10 &

# 모니터링 시작 (별칭 사용 - 추천)
mon1
# 또는 전체 경로
./scripts/monitoring/universal_training_monitor.sh --stage 1
```

### Stage 2 학습 모니터링  
```bash
# Stage 2 학습 시작  
source .venv/bin/activate
python -m src.training.train_classification_stage --stage 2 --epochs 30 &

# 빠른 상태 확인 후 자동 감지 모니터링 (별칭 사용 - 추천)
status      # 현재 상태 확인
monitor     # 자동 감지 실시간 모니터링
```

### Stage 3-4 대용량 데이터 모니터링
```bash
# 높은 새로고침 빈도로 모니터링 (별칭 사용 - 추천)
mon3        # Stage 3 전용 모니터링  
monfast     # 1초마다 빠른 새로고침
gpuw        # GPU 상태 실시간 감시

# 또는 전체 경로
./scripts/monitoring/universal_training_monitor.sh --stage 3 --interval 1
```

## 🛠️ 문제 해결

### Q: "프로세스를 찾을 수 없습니다" 오류
**A:** 학습 프로세스가 실행 중인지 확인하세요:
```bash
ps aux | grep train_classification_stage
```

### Q: GPU 정보가 표시되지 않음
**A:** nvidia-smi 설치 상태를 확인하세요:
```bash
nvidia-smi --version
```

### Q: 로그가 표시되지 않음  
**A:** 로그 파일 경로를 확인하거나 직접 지정하세요. 스크립트가 다음 경로들을 자동으로 검색합니다:
- `/tmp/pillsnap_training_stage*/training.log`
- `/tmp/pillsnap_training/training.log`
- `./logs/training*.log`

## 🚀 고급 사용법

### 백그라운드 모니터링
```bash
# 터미널에서 분리하여 실행
nohup ./scripts/monitoring/universal_training_monitor.sh > monitor.out 2>&1 &
```

### 로그 파일로 저장
```bash
# 모니터링 결과를 파일로 저장
./scripts/monitoring/universal_training_monitor.sh | tee monitor_$(date +%Y%m%d_%H%M%S).log
```

### 여러 Stage 동시 모니터링
```bash
# 터미널을 여러 개 열어서 각각 다른 Stage 모니터링
tmux new-session -d -s stage1 './scripts/monitoring/universal_training_monitor.sh --stage 1'
tmux new-session -d -s stage2 './scripts/monitoring/universal_training_monitor.sh --stage 2'
```

## 🔄 업데이트 내역

- **v3.0** (2025-08-22): Universal Training Monitor 출시, 전체 폴더 정리
- **v2.x**: Stage별 개별 스크립트들 (아카이브됨)  
- **v1.x**: 초기 버전들 (아카이브됨)