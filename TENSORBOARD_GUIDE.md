# 📊 PillSnap TensorBoard 모니터링 가이드

## 🚀 빠른 시작

### 1. TensorBoard 실행
```bash
# 기본 실행 (포트 6006)
./scripts/monitoring/run_tensorboard.sh

# 다른 포트로 실행
./scripts/monitoring/run_tensorboard.sh -p 6007

# 자동 새로고침 모드
./scripts/monitoring/run_tensorboard.sh --reload
```

### 2. 웹 브라우저 접속
```
http://localhost:6006
```

## 📈 모니터링 가능한 메트릭

### Classification 메트릭
- **Loss**: 학습/검증 손실값
- **Accuracy**: Top-1 정확도
- **Top-5 Accuracy**: Top-5 정확도
- **Learning Rate**: 학습률 변화

### Detection 메트릭
- **Box Loss**: Bounding Box 회귀 손실
- **Class Loss**: 분류 손실
- **DFL Loss**: Distribution Focal Loss
- **mAP@0.5**: IoU 0.5 기준 mAP
- **mAP@0.5:0.95**: IoU 0.5~0.95 평균 mAP

### System 메트릭
- **GPU Memory Used**: 현재 GPU 메모리 사용량
- **GPU Memory Peak**: 최대 GPU 메모리 사용량
- **GPU Utilization**: GPU 사용률

## 🔧 학습 코드에 TensorBoard 통합하기

### 방법 1: 자동 패치 (권장)
```python
# train_stage3_two_stage.py 상단에 추가
from src.training.tensorboard_integration import patch_trainer_with_tensorboard

# TwoStageTrainer 클래스 정의 후 추가
patch_trainer_with_tensorboard(TwoStageTrainer)
```

### 방법 2: 수동 통합
```python
from src.utils.tensorboard_logger import TensorBoardLogger

# __init__에서 초기화
self.tb_logger = TensorBoardLogger(
    log_dir='runs',
    experiment_name='stage3_training'
)

# 배치 학습 중 로깅
self.tb_logger.log_scalar('train/loss', loss.item(), step)
self.tb_logger.log_scalar('train/accuracy', accuracy, step)

# 에포크 종료 시 로깅
self.tb_logger.log_classification_metrics(
    loss=epoch_loss,
    accuracy=epoch_accuracy,
    top5_accuracy=top5_accuracy,
    step=epoch,
    phase='train'
)

# 학습 종료 시
self.tb_logger.close()
```

## 📊 TensorBoard 주요 기능

### 1. Scalars 탭
- 시간에 따른 메트릭 변화 그래프
- 학습/검증 비교
- 스무딩 옵션 (노이즈 제거)

### 2. Histograms 탭
- 가중치 분포 변화
- 그래디언트 분포

### 3. Graphs 탭
- 모델 구조 시각화
- 연산 그래프

### 4. Text 탭
- 하이퍼파라미터
- 설정 정보

## 🎯 활용 팁

### 1. 여러 실험 비교
```bash
# 여러 실험을 한 번에 보기
tensorboard --logdir runs/
```

### 2. 특정 메트릭만 보기
- 왼쪽 사이드바에서 원하는 메트릭만 체크
- 정규식으로 필터링 가능

### 3. 다운로드
- 그래프 우측 상단 다운로드 버튼
- CSV, JSON, 이미지 형식 지원

### 4. 스무딩
- 우측 슬라이더로 스무딩 조절
- 노이즈가 많은 메트릭에 유용

## 🐛 문제 해결

### 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i:6006

# 기존 TensorBoard 종료
pkill -f tensorboard
```

### 로그가 안 보일 때
1. 로그 디렉토리 확인
2. 새로고침 (F5 또는 Shift+F5)
3. `--reload` 옵션으로 재시작

### 메모리 부족
- 큰 모델의 경우 Graph 탭 비활성화
- 히스토그램 로깅 빈도 줄이기

## 📝 현재 Stage 3 학습 상황 (2025-08-24)

### 🔄 재학습 진행 중
- **이전 학습**: Epoch 15/36에서 중단 (69.0% accuracy)
- **재학습 시작**: 2025-08-24
- **목표 Epochs**: 36
- **배치 크기**: 8

### 📊 개선된 하이퍼파라미터
```bash
--lr-classifier 5e-5      # 과적합 방지 (이전: 2e-4)
--lr-detector 1e-3        # Detection 학습률
--weight-decay 5e-4       # 정규화 강화
--label-smoothing 0.1     # 일반화 성능 향상
--validate-period 3       # 3 epochs마다 검증
--patience-cls 8          # Classification patience
--patience-det 6          # Detection patience
--reset-best             # Best 메트릭 초기화
```

### ✅ 코드 개선사항
- **YOLO Resume 수정**: 매 에포크 모델 지속 학습
- **체크포인트 정책**: Epsilon threshold + Patience 기반
- **TensorBoard 통합**: 실시간 메트릭 추적
- **Detection 실측치**: 시뮬레이션 제거, 실제 mAP 사용

## 📝 현재 구현 상태

✅ **완료됨**
- TensorBoard 로거 클래스 (`src/utils/tensorboard_logger.py`)
- 통합 헬퍼 모듈 (`src/training/tensorboard_integration.py`)
- 실행 스크립트 (`scripts/monitoring/run_tensorboard.sh`)
- Detection 학습 문제 수정 코드
- 체크포인트 저장 개선 코드
- `train_stage3_two_stage.py`에 TensorBoard 통합 ✅
- 중복 패치 방지 가드 추가 ✅
- 스모크 테스트 기능 구현 ✅

### 📊 기대 태그 목록
- **분류(학습)**: `train/loss`, `train/lr`, `train/grad_norm`
- **분류(검증)**: `val/top1`, `val/top5`, `val/macro_f1`, `val/single_f1`, `val/combo_f1`
- **검출**: `det/box_loss`, `det/cls_loss`, `det/dfl_loss`, `det/map50`, `det/precision`, `det/recall`
- **시스템/레이턴시**: `sys/vram_used`, `sys/vram_peak`, `latency/det`, `latency/crop`, `latency/cls`, `latency/total`

## 💡 예상 효과

기존 텍스트 로그 파싱 방식에서 벗어나:
- ✅ 실시간 메트릭 추적
- ✅ 여러 실험 비교
- ✅ 학습 곡선 시각화
- ✅ 과적합 조기 발견
- ✅ 하이퍼파라미터 영향 분석

---

**다음 학습 시작 전에 TensorBoard 통합 코드를 추가하면, 제대로 된 모니터링이 가능합니다!** 🚀