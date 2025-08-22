# /initial-prompt — PillSnap ML 세션 초기화 스크립트

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 세션 초기화를 수행합니다.
**모든 응답은 한국어로 작성**합니다.

---

## 🎯 프로젝트 현재 상태 (2025-08-22)

### **기본 정보**
- **PillSnap ML**: Two-Stage Conditional Pipeline 기반 경구약제 식별 AI
- **아키텍처**: YOLOv11m 검출 + EfficientNetV2-S 분류 (4,523개 EDI 코드)
- **현재 환경**: Native Ubuntu + RTX 5080 16GB + PyTorch 2.8.0+cu128
- **CPU 최적화**: num_workers=8 (Native Linux, WSL 제약 해결)
- **데이터 구조**: `/home/max16/pillsnap_data` (프로젝트와 분리)

### **Progressive Validation 현황 (Manifest 기반)**
- ✅ **Stage 1**: 완료 (5K 샘플, 50 클래스, 74.9% 정확도, Native Linux)
- 🔄 **Stage 2**: 준비 완료 (25K 샘플, 250 클래스)
  - 데이터 구조: Linux SSD + Windows SSD 하이브리드
  - 심볼릭 링크: 81개 폴더 완전 설정
  - Albumentations 2.0.8 업그레이드 완료
- 🎯 **Stage 3**: **Manifest 기반 진행** (100K 샘플, 1K 클래스)
  - ⭐ **중요**: 물리적 데이터 복사 없이 manifest CSV로만 진행
  - 용량 절약: ~14.6GB → ~50MB (manifest 파일만)
  - 원본 하이브리드 스토리지에서 직접 로딩
- 🎯 **Stage 4**: **Manifest 기반 진행** (500K 샘플, 4.5K 클래스)
  - 동일한 manifest 방식으로 확장성 보장
  - 용량 절약: ~73GB → ~200MB (전체 절약)

### **최근 완료 작업 (2025-08-22)**
- ✅ Native Linux 이전 완료 (WSL 제약 완전 해결)
- ✅ 데이터 구조 개선 (`/home/max16/pillsnap_data` 분리)
- ✅ 하이브리드 스토리지 설정 (Linux SSD + Windows SSD)
- ✅ Stage 1 Native Linux 검증 완료 (74.9% 정확도, 1분)
- ✅ CPU 멀티프로세싱 활성화 (num_workers=8)
- ✅ Albumentations 2.0.8 업그레이드 (최신 버전)
- ✅ 문서 업데이트 (Native Linux 환경 반영)

---

## 🔥 Native Linux 환경 확인

**Python 환경:**
```bash
source .venv/bin/activate
```

**금지사항**: 시스템 Python 사용 금지 (가상환경 필수)

---

## 📋 세션 초기화 체크리스트

### **[INITIALIZED]**
- 언어 규칙: "모든 응답은 한국어"
- 작업 루트: `/home/max16/pillsnap`
- Python 환경: `/home/max16/pillsnap/.venv/bin/python` (PyTorch 2.8.0+cu128)
- 데이터 루트: `/home/max16/pillsnap_data` (프로젝트와 분리된 경로)

### **프롬프트 참조**
상세 설계는 다음 문서에 있으니 모든 문서를 반드시 읽는다:
- `Prompt/PART_0.md` - Progressive Validation Strategy
- `Prompt/PART_A.md` - 아키텍처 + 경로 정책
- `Prompt/PART_B.md` - 프로젝트 구조 + RTX 5080 최적화
- `Prompt/PART_C.md` - Two-Stage 데이터 파이프라인
- `Prompt/PART_D.md` - YOLOv11m 검출 모델
- `Prompt/PART_E.md` - EfficientNetV2-S 분류 모델
- `Prompt/PART_F.md` - API 서빙
- `Prompt/PART_G.md` - 최적화
- `Prompt/PART_H.md` - 배포

### **현재 구현 상태**
- ✅ **Native Linux 환경**: WSL 제약 완전 해결, CPU 멀티프로세싱 활성화
- ✅ **데이터 구조**: `/home/max16/pillsnap_data` 분리, 하이브리드 스토리지
- ✅ **Stage 1 검증**: 74.9% 정확도 (Native Linux, 1분 완료)
- ✅ **Albumentations 2.0.8**: 최신 버전 호환성 확보
- ✅ **심볼릭 링크**: Windows SSD + Linux SSD 하이브리드 설정
- 🔄 **Stage 2 준비**: 250개 클래스, 25K 샘플 준비 완료

### **핵심 설계 원칙**
1. **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo)
2. **Progressive Validation**: Stage 1-4 (5K→25K→100K→500K)
3. **Native Linux 최적화**: num_workers=8, CPU 멀티프로세싱 활용
4. **하이브리드 스토리지**: Linux SSD (3.5GB/s) + Windows SSD (1GB/s)
5. **RTX 5080 최적화**: Mixed Precision, torch.compile

### **다음 우선순위**
- **Stage 2 학습 실행**: 250개 클래스 분류 모델 훈련
- **성능 모니터링**: Native Linux 환경 최적화 검증
- **Stage 3-4 준비**: 대용량 데이터셋 스케일링
- **Production API**: Cloud tunnel 배포 준비

---

## 🚀 즉시 실행 가능 명령어

### **환경 확인**
```bash
# 기본 환경 확인
source .venv/bin/activate
python --version  # Python 3.11.13
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"  # PyTorch 2.8.0+cu128
```

### **학습 실행**
```bash
# Stage 1 테스트 (완료됨)
python -m src.training.train_classification_stage --stage 1 --epochs 1 --batch-size 32

# Stage 2 훈련 (준비 완료)
python -m src.training.train_classification_stage --stage 2 --epochs 30 --batch-size 32

# 통합 테스트
python -m pytest tests/integration/ -v
```

### **⌨️ 모니터링 별칭 (추천)**
```bash
# 현재 상태 빠른 확인
status       # GPU 사용률, 완료된 Stage, 디스크 공간

# 실시간 모니터링 
monitor      # 자동 Stage 감지 실시간 모니터링
mon2         # Stage 2 전용 모니터링  
mon3         # Stage 3 전용 모니터링
mon4         # Stage 4 전용 모니터링
monfast      # 1초마다 빠른 새로고침

# GPU 상태
gpu          # nvidia-smi 한 번 실행
gpuw         # nvidia-smi 실시간 감시 (1초마다)
```

### **별칭 설정되지 않은 경우**
```bash
# 빠른 상태 확인
./scripts/monitoring/quick_status.sh

# 실시간 모니터링
./scripts/monitoring/universal_training_monitor.sh
./scripts/monitoring/universal_training_monitor.sh --stage 2
./scripts/monitoring/universal_training_monitor.sh --interval 1  # 빠른 새로고침

# 별칭 자동 설정
./scripts/monitoring/setup_aliases.sh
```

---

## ⚠️ 중요 제약사항

- **Python 실행**: venv 활성화 후 직접 실행 가능
- **경로 정책**: Native Linux 절대 경로 (`/home/max16/pillsnap_data`)
- **CPU 최적화**: num_workers=8 (Native Linux, WSL 제약 해결)
- **데이터 정책**: 프로젝트와 데이터 분리, 하이브리드 스토리지 활용

---

**세션 초기화 완료**. 상세 컨텍스트는 `Prompt/PART_*.md` 파일들을 참조하세요.