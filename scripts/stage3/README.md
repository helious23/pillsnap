# 🚀 Stage 3 Two-Stage Pipeline 스크립트

**Stage 3**: 100,000개 샘플, 1,000개 클래스 Two-Stage Pipeline 학습 및 관리

## 📁 스크립트 목록

### 🎯 핵심 학습 스크립트

#### `train_stage3_two_stage.sh`
**메인 학습 실행 스크립트** - RTX 5080 최적화된 Two-Stage Pipeline 학습

```bash
# 기본 실행 (추천 설정)
./scripts/stage3/train_stage3_two_stage.sh

# 하이퍼파라미터 커스텀
./scripts/stage3/train_stage3_two_stage.sh --epochs 20 --batch-size 24

# 도움말
./scripts/stage3/train_stage3_two_stage.sh --help
```

**특징:**
- **RTX 5080 최적화**: 배치 32, Mixed Precision, torch.compile
- **Two-Stage 파이프라인**: Detection + Classification 통합 학습
- **실시간 검증**: GPU 메모리, Manifest 데이터, 시스템 리소스
- **에러 진단**: 실패 시 상세한 원인 분석 및 해결책 제시

**기본 하이퍼파라미터:**
- 에포크: 18 (RTX 5080 기준 최적)
- Classification 배치: 32
- Detection 배치: 16  
- 학습률: 3e-4
- 목표: Classification ≥85%, Detection mAP@0.5 ≥30%

---

### 📊 모니터링 스크립트

#### `monitor_stage3_realtime.py`
**실시간 학습 모니터링** - Stage 3 전용 상세 추적

```bash
# Stage 3 실시간 모니터링
python scripts/stage3/monitor_stage3_realtime.py

# 백그라운드 실행
python scripts/stage3/monitor_stage3_realtime.py --daemon
```

**기능:**
- GPU 메모리 사용률 실시간 추적
- Two-Stage 학습 진행률 별도 표시
- Classification/Detection 성능 지표 분리 모니터링
- WebSocket 기반 대시보드 (포트 8888)

#### `run_stage3_with_logs.sh`  
**로그 포함 학습 실행** - 학습과 동시에 실시간 로그 수집

```bash
# 학습 + 실시간 로깅
./scripts/stage3/run_stage3_with_logs.sh

# 로그 레벨 설정
./scripts/stage3/run_stage3_with_logs.sh --log-level DEBUG
```

---

## 🎯 Stage 3 목표 및 현황

### 📈 성능 목표
| 지표 | 목표 | 현재 상태 |
|------|------|-----------|
| **Classification 정확도** | ≥85% | 🔄 학습 중 |
| **Detection mAP@0.5** | ≥30% | 🔄 학습 중 |
| **학습 시간** | ≤2시간 | RTX 5080 기준 |
| **Two-Stage 파이프라인** | 완전 통합 | ✅ 구현 완료 |

### 📊 데이터 구성 (Manifest 기반)
- **총 샘플**: 100,000개 (학습 81,475개 + 검증 18,525개)
- **클래스 수**: 1,000개 (EDI 코드 기준)
- **Single/Combination 비율**: 95% / 5% (Classification 중심)
- **저장공간 절약**: 99.7% (73GB → 200MB manifest 파일)

---

## 🚀 빠른 시작 가이드

### 1단계: 환경 준비
```bash
# 가상환경 활성화
source .venv/bin/activate

# 환경 변수 설정
export PILLSNAP_DATA_ROOT="/home/max16/pillsnap_data"
```

### 2단계: Manifest 검증 (선택적)
```bash
# Stage 3 Manifest 무결성 확인
python -c "
import pandas as pd
train_df = pd.read_csv('artifacts/stage3/manifest_train.csv')
print(f'Train samples: {len(train_df):,}')
print(f'Unique classes: {train_df[\"mapping_code\"].nunique()}')
"
```

### 3단계: 학습 실행
```bash
# 기본 실행 (추천)
./scripts/stage3/train_stage3_two_stage.sh

# 또는 모니터링과 함께
./scripts/stage3/run_stage3_with_logs.sh
```

### 4단계: 결과 확인
```bash
# 체크포인트 확인
ls -la artifacts/stage3/checkpoints/

# 성능 평가
python -m tests.performance.stage_3_evaluator
```

---

## 🔧 트러블슈팅

### GPU 메모리 부족
```bash
# 배치 크기 감소
./scripts/stage3/train_stage3_two_stage.sh --batch-size 16

# GPU 메모리 정리  
nvidia-smi --gpu-reset
```

### Manifest 파일 문제
```bash
# Manifest 재생성
python -m src.data.create_stage3_manifest

# 수동 검증
python scripts/stage3/validate_stage3_manifest.py
```

### 학습 중단/재시작
```bash
# 체크포인트에서 재시작 (자동)
./scripts/stage3/train_stage3_two_stage.sh --resume

# 특정 체크포인트 지정
./scripts/stage3/train_stage3_two_stage.sh --checkpoint artifacts/stage3/checkpoints/stage3_classification_best.pt
```

---

## 📋 체크리스트

### ✅ 학습 전 확인사항
- [ ] RTX 5080 GPU 사용 가능
- [ ] `.venv/bin/activate` 가상환경 활성화
- [ ] `artifacts/stage3/manifest_train.csv` 존재 (81K+ 샘플)
- [ ] `artifacts/stage3/manifest_val.csv` 존재 (18K+ 샘플) 
- [ ] GPU 메모리 사용률 < 20%
- [ ] 디스크 여유 공간 > 10GB

### ✅ 학습 완료 후 확인사항  
- [ ] `artifacts/stage3/checkpoints/stage3_classification_best.pt` 생성
- [ ] `artifacts/stage3/checkpoints/stage3_detection_best.pt` 생성
- [ ] Classification 정확도 ≥85% 달성
- [ ] Detection mAP@0.5 ≥30% 달성
- [ ] 총 학습 시간 ≤2시간

### 🎯 Stage 4 준비사항
- [ ] Stage 3 성능 목표 달성 확인
- [ ] Two-Stage 파이프라인 안정성 검증  
- [ ] 체크포인트 무결성 검사
- [ ] Stage 4 데이터셋 (500K 샘플) 준비

---

## 📚 관련 문서

- **전체 프로젝트**: `README.md`
- **Stage 진행 현황**: `SESSION_STATUS.md`  
- **성능 평가**: `tests/performance/stage_3_evaluator.py`
- **Two-Stage 파이프라인**: `src/training/train_stage3_two_stage.py`
- **Manifest 생성**: `src/data/create_stage3_manifest.py`

---

**🏥 PillSnap ML - Stage 3 Two-Stage Pipeline 완성으로 프로덕션 준비 단계 진입**