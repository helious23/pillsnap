# 🚀 Stage 4 Production Training Plan

## 📋 개요

Stage 4는 PillSnap ML의 최종 프로덕션 학습 단계로, 500,000개 이미지와 4,523개 클래스를 대상으로 합니다.
Stage 3의 성공적인 완료를 바탕으로, 검증된 하이퍼파라미터와 개선된 시스템을 활용합니다.

---

## 🎯 목표 성능

| 메트릭 | Stage 3 달성 | Stage 4 목표 | 비고 |
|--------|-------------|-------------|------|
| **Classification Top-1** | 85.01% | **92%** | 프로덕션 목표 |
| **Classification Top-5** | 97.68% | **99%** | Near-perfect |
| **Detection mAP@0.5** | 32.73% | **85%** | 대폭 개선 필요 |
| **추론 시간** | - | **< 50ms** | 실시간 처리 |

---

## 🛠️ 사전 준비사항

### 1. ✅ Detection 버그 수정 (완료)

```python
# src/training/train_stage3_two_stage.py 수정 완료 (2025-08-24)

# 1. YOLO epochs 동적 설정 ✅
epochs=target_epochs,  # state.json 기반 누적 관리

# 2. State Management 추가 ✅
from src.utils.detection_state_manager import DetectionStateManager
state_manager = DetectionStateManager()
state = state_manager.load_state()

# 3. Robust CSV Parser 추가 ✅
from src.utils.robust_csv_parser import RobustCSVParser
csv_parser = RobustCSVParser()
metrics = csv_parser.parse_results_csv(csv_path)
```

### 2. 데이터 검증

```bash
# Manifest 파일 확인
wc -l /home/max16/pillsnap/artifacts/stage4/manifest_train.csv
wc -l /home/max16/pillsnap/artifacts/stage4/manifest_val.csv

# 손상 파일 체크
python -m src.utils.validate_manifest \
  --manifest artifacts/stage4/manifest_train.csv \
  --remove-corrupt
```

### 3. 시스템 준비

```bash
# 디스크 공간 확인 (최소 200GB 필요)
df -h /home/max16/pillsnap_data

# GPU 메모리 클리어
nvidia-smi
sudo fuser -v /dev/nvidia* | awk '{print $2}' | xargs -I {} kill {}

# TensorBoard 준비
pkill -f tensorboard
tensorboard --logdir=/home/max16/pillsnap_data/exp/exp01/tensorboard &
```

---

## 📊 학습 전략

### Phase 1: Baseline (Epoch 1-20)
**Stage 3 성공 레시피 그대로**

```bash
python -m src.training.train_stage3_two_stage \
  --manifest-train /home/max16/pillsnap/artifacts/stage4/manifest_train.csv \
  --manifest-val /home/max16/pillsnap/artifacts/stage4/manifest_val.csv \
  --epochs 50 \
  --batch-size 8 \
  --lr-classifier 5e-5 \
  --lr-detector 1e-3 \
  --weight-decay 5e-4 \
  --label-smoothing 0.1 \
  --validate-period 5 \
  --save-every 5 \
  --patience-cls 10 \
  --patience-det 8 \
  > /home/max16/pillsnap/artifacts/logs/stage4_$(date +%F_%H%M%S).log 2>&1 &
```

### Phase 2: Fine-tuning (Epoch 21-40)
**성능 정체 시 조정**

| 상황 | 조치 | 명령어 옵션 |
|------|------|------------|
| Classification < 60% | LR 증가 | `--lr-classifier 1e-4` |
| 과적합 (Train-Val > 20%) | Regularization 강화 | `--weight-decay 1e-3` |
| Detection 정체 | LR 감소 | `--lr-detector 5e-4` |
| OOM 발생 | Batch 크기 감소 | `--batch-size 4` |

### Phase 3: Final Push (Epoch 41-50+)
**목표 미달 시 추가 학습**

```python
# Discriminative Learning Rates
--lr-classifier 2e-5  # 미세조정
--lr-detector 1e-4    # 안정화
--epochs 100          # 필요시 연장
```

---

## 📈 모니터링 체크포인트

### 실시간 모니터링 (매 배치)
```bash
# 터미널 1: 로그 모니터링
tail -f /home/max16/pillsnap/artifacts/logs/stage4_*.log | grep -E "Loss:|Accuracy:|mAP"

# 터미널 2: GPU 모니터링
watch -n 1 nvidia-smi

# 터미널 3: 커스텀 모니터링
./scripts/monitoring/universal_training_monitor.sh --stage 4
```

### 주요 마일스톤 (매 5 에포크)
| Epoch | Classification 목표 | Detection 목표 | 체크포인트 |
|-------|-------------------|---------------|------------|
| 5 | 40% | 40% | 기본 학습 확인 |
| 10 | 60% | 50% | 중간 점검 |
| 15 | 75% | 60% | 성능 가속 확인 |
| 20 | 85% | 70% | Stage 3 수준 도달 |
| 30 | 90% | 80% | 목표 근접 |
| 40 | 92% | 85% | **최종 목표** |

---

## 🔧 트러블슈팅

### 문제 1: Classification 성능 정체
```python
# 해결책: Progressive Resize 활성화
--progressive-resize
--initial-size 128
--final-size 384
--resize-milestone 10
```

### 문제 2: Detection mAP 낮음
```python
# 해결책: Detection 집중 학습
--detector-epochs-per-cycle 2  # Detection 2배 학습
--classifier-epochs-per-cycle 1
```

### 문제 3: OOM (Out of Memory)
```python
# 해결책: Gradient Accumulation
--batch-size 4
--gradient-accumulation 2  # 효과적 batch size = 8
```

### 문제 4: 학습 불안정
```python
# 해결책: Gradient Clipping
--gradient-clip 1.0
--mixed-precision False  # FP32로 전환
```

---

## 📊 예상 결과

### 학습 시간
- **총 소요 시간**: 20-30시간
- **에포크당**: 30-40분
- **Early Stopping**: 30-40 에포크 예상

### 최종 성능 예측
| 메트릭 | 보수적 | 현실적 | 낙관적 |
|--------|--------|--------|---------|
| Classification | 88% | 92% | 95% |
| Detection | 70% | 85% | 90% |
| Top-5 Accuracy | 98% | 99% | 99.5% |

### 리소스 사용량
- **GPU 메모리**: 14-15GB / 16GB
- **시스템 RAM**: 40-60GB / 128GB
- **디스크 I/O**: 500MB/s 평균
- **체크포인트 크기**: ~3GB/epoch

---

## ✅ 체크리스트

### 학습 시작 전
- [ ] Detection 버그 수정 완료
- [ ] Manifest 파일 검증
- [ ] 200GB 디스크 공간 확보
- [ ] TensorBoard 실행
- [ ] 모니터링 스크립트 준비

### 학습 중
- [ ] 5 에포크마다 성능 확인
- [ ] 이상 징후 모니터링
- [ ] 체크포인트 백업
- [ ] 로그 분석

### 학습 완료 후
- [ ] 최종 메트릭 정리
- [ ] 모델 export (ONNX)
- [ ] 추론 속도 테스트
- [ ] API 통합 테스트
- [ ] 문서 업데이트

---

## 🚀 실행 명령어 (복사용)

```bash
# Stage 4 학습 시작 (권장)
nohup python -m src.training.train_stage3_two_stage \
  --manifest-train /home/max16/pillsnap/artifacts/stage4/manifest_train.csv \
  --manifest-val /home/max16/pillsnap/artifacts/stage4/manifest_val.csv \
  --epochs 50 \
  --batch-size 8 \
  --lr-classifier 5e-5 \
  --lr-detector 1e-3 \
  --weight-decay 5e-4 \
  --label-smoothing 0.1 \
  --validate-period 5 \
  --save-every 5 \
  > /home/max16/pillsnap/artifacts/logs/stage4_$(date +%F_%H%M%S).log 2>&1 &

# 모니터링
tail -f /home/max16/pillsnap/artifacts/logs/stage4_*.log
```

---

## 📝 참고사항

1. **Stage 3 교훈**
   - Classification은 lr=5e-5가 최적
   - Detection은 resume 버그 수정 필수
   - 22 에포크에서 조기종료 가능

2. **Stage 4 특이사항**
   - 4,523 클래스는 Stage 3의 4.5배
   - 500K 샘플은 더 robust한 학습 가능
   - 프로덕션 배포 전 최종 검증 필수

3. **성공 기준**
   - Classification 92% 달성 시 성공
   - Detection은 지속 개선 목표
   - 추론 속도 50ms 이하 필수

---

**🎯 Stage 4 성공을 위해 Stage 3의 검증된 레시피를 활용하되,**
**Detection 버그만 수정하면 프로덕션 레벨 달성 가능!**