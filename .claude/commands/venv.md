# /venv — PillSnap ML 가상환경 자동 설정 명령어

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 가상환경을 자동으로 설정하고 검증합니다.
**모든 응답은 한국어로 작성**합니다.

---

## 🐍 Python 가상환경 설정 가이드

### 🔥 중요: Python 가상환경 사용법 (필수 숙지)

**모든 Python 실행 시 반드시 다음 방법만 사용:**

#### 1. 안전한 실행 스크립트 (권장)
```bash
# 기본 사용법
./scripts/core/python_safe.sh [Python 명령어와 인수들]

# 예시
./scripts/core/python_safe.sh --version
./scripts/core/python_safe.sh -m pytest tests/ -v
./scripts/core/python_safe.sh -m src.train
```

#### 2. 직접 경로 (대안)
```bash
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"
$VENV_PYTHON [명령어]
```

#### 3. 별칭 설정 (선택사항)
```bash
source scripts/core/setup_aliases.sh
pp --version              # Python 실행
ptest tests/ -v           # pytest 실행
ppip install numpy        # pip 실행
```

### ❌ 절대 금지사항
- `python`, `python3` 시스템 명령어 사용 금지 (Python 3.13 충돌)
- 환경 변수 없이 상대 경로 실행 금지
- 가상환경 비활성화 상태에서 직접 실행 금지

**현재 환경**: `.venv` (PyTorch 2.7.0+cu128, RTX 5080 호환)

---

## ⚡ 즉시 실행 가능한 환경 설정

### 1단계: 환경 확인
```bash
# 작업 디렉토리 확인
pwd
# 출력: /home/max16/pillsnap

# 가상환경 존재 확인
ls -la .venv/bin/python
# 출력: /home/max16/pillsnap/.venv/bin/python (존재해야 함)
```

### 2단계: 가상환경 검증
```bash
# Python 버전 확인
./scripts/core/python_safe.sh --version
# 예상 출력: Python 3.11.13

# PyTorch 및 CUDA 확인
./scripts/core/python_safe.sh -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
# 예상 출력: PyTorch: 2.7.0+cu128, CUDA available: True

# GPU 하드웨어 확인
./scripts/core/python_safe.sh -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')"
# 예상 출력: GPU: NVIDIA GeForce RTX 5080, Memory: 15GB
```

### 3단계: 데이터 환경 설정
```bash
# SSD 데이터 루트 설정
export PILLSNAP_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"
echo "데이터 루트: $PILLSNAP_DATA_ROOT"

# SSD 데이터 확인
ls -la $PILLSNAP_DATA_ROOT
du -sh $PILLSNAP_DATA_ROOT
# 예상: 7.0G, 5,000개 PNG 파일
```

### 4단계: 환경 완료 검증
```bash
# config.yaml SSD 경로 확인
grep -n "ssd_pillsnap" config.yaml
# 예상: SSD 경로가 설정되어 있어야 함

# 프로젝트 구조 확인
tree -L 2 src/
# 예상: models/, training/, evaluation/, data/ 등 디렉토리 존재

# 간단한 import 테스트
./scripts/core/python_safe.sh -c "from src.utils.core import ConfigLoader; print('✅ 환경 설정 완료')"
```

---

## 🚀 완료 후 즉시 실행 가능한 명령어

### 프로젝트 상태 확인
```bash
# Stage 1 파이프라인 테스트
./scripts/core/python_safe.sh tests/test_stage1_real_image.py

# 통합 테스트 실행
./scripts/core/python_safe.sh -m pytest tests/integration/ -v

# 모델 개별 테스트
./scripts/core/python_safe.sh -m src.models.detector_yolo11m
./scripts/core/python_safe.sh -m src.models.classifier_efficientnetv2
```

### 실제 학습 시작 (Ready!)
```bash
# Stage 1 분류 학습
./scripts/core/python_safe.sh -m src.training.train_classification_stage --stage 1 --epochs 10

# 배치 크기 자동 최적화
./scripts/core/python_safe.sh -m src.training.batch_size_auto_tuner --model-type classification
```

---

## 🔧 문제 해결

### 가상환경 오류 시
```bash
# 가상환경 재생성 (필요시)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CUDA 오류 시
```bash
# CUDA 호환성 확인
nvidia-smi
# PyTorch CUDA 버전 확인
./scripts/core/python_safe.sh -c "import torch; print(torch.version.cuda)"
```

### 데이터 경로 오류 시
```bash
# SSD 데이터 존재 확인
ls -la /home/max16/ssd_pillsnap/dataset/
# 없으면 HDD에서 복사 필요
```

---

## 📋 환경 설정 체크리스트

### ✅ 완료 확인 항목
- [ ] Python 3.11.13 가상환경 활성화
- [ ] PyTorch 2.7.0+cu128 설치 확인
- [ ] CUDA 사용 가능 (RTX 5080 감지)
- [ ] SSD 데이터 경로 설정 및 확인
- [ ] config.yaml SSD 경로 설정
- [ ] 기본 import 테스트 성공
- [ ] scripts/python_safe.sh 실행 가능

### 🎯 완료 후 상태
```
✅ Python: 3.11.13 (.venv)
✅ PyTorch: 2.7.0+cu128
✅ GPU: RTX 5080 (16GB, CUDA 활성)
✅ Data: /home/max16/ssd_pillsnap/dataset (7.0GB, 5,000장)
✅ Ready: Stage 1-4 학습 준비 완료
```

---

## 🚀 즉시 시작 가능

**환경 설정 완료 후 다음 단계**:

1. **Stage 1 파이프라인 테스트**: `./scripts/core/python_safe.sh tests/test_stage1_real_image.py`
2. **실제 학습 시작**: `./scripts/core/python_safe.sh -m src.training.train_classification_stage --stage 1`
3. **성능 평가**: `./scripts/core/python_safe.sh -m src.evaluation.evaluate_pipeline_end_to_end --stage 1`

**🎯 목표**: 92% 분류 정확도, 0.85 mAP@0.5 검출 성능 달성!

---

이 가이드를 따라 하면 **compact 이후에도 즉시 가상환경을 설정**하고 프로젝트를 진행할 수 있습니다.