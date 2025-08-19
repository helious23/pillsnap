# /initial-prompt — PillSnap ML 세션 초기화 스크립트

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 세션 초기화를 다음과 같이 수행합니다.
**모든 응답은 한국어로 작성**합니다. 모호하면 가장 단순한 해법을 우선합니다.

---

## 0) 프로젝트 개요 (Goal)
- **PillSnap ML**: Two-Stage Conditional Pipeline 기반 경구약제 식별 AI 시스템
- **아키텍처**: YOLOv11m 검출 + EfficientNetV2-S 분류 (5000개 EDI 코드)
- **환경**: WSL2 + RTX 5080 16GB + PyTorch 2.7.0+cu128
- **목표**: 92% 분류 정확도, 0.85 mAP@0.5 검출 성능

## 🔥 중요: Python 가상환경 사용법 (필수 숙지)
**모든 Python 실행 시 반드시 다음 방법만 사용:**

### 1. 안전한 실행 스크립트 (권장)
```bash
# 기본 사용법
./scripts/python_safe.sh [Python 명령어와 인수들]

# 예시
./scripts/python_safe.sh --version
./scripts/python_safe.sh -m pytest tests/ -v
./scripts/python_safe.sh -m src.train
```

### 2. 직접 경로 (대안)
```bash
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"
$VENV_PYTHON [명령어]
```

### 3. 별칭 설정 (선택사항)
```bash
source scripts/setup_aliases.sh
pp --version              # Python 실행
ptest tests/ -v           # pytest 실행
ppip install numpy        # pip 실행
```

### ❌ 금지사항
- `python`, `python3` 시스템 명령어 사용 금지 (Python 3.13 충돌)
- 환경 변수 없이 상대 경로 실행 금지
- 가상환경 비활성화 상태에서 직접 실행 금지

**현재 환경**: `.venv` (PyTorch 2.7.0+cu128, RTX 5080 호환)

---

## 1) 수집 대상 (읽기 순서 고정)
1. **프롬프트 사양 (필수)**:
   - `Prompt/PART_0.md` - Progressive Validation Strategy + OptimizationAdvisor
   - `Prompt/PART_A.md` - Two-Stage Conditional Pipeline 아키텍처
   - `Prompt/PART_B.md` - 프로젝트 구조 + RTX 5080 최적화
   - `Prompt/PART_C.md` - Two-Stage 데이터 파이프라인
   - `Prompt/PART_D.md` - YOLOv11m 검출 모델
   - `Prompt/PART_E.md` - EfficientNetV2-S 분류 모델
   - `Prompt/PART_F.md` - API 서빙 + FastAPI
   - `Prompt/PART_G.md` - 최적화 + 컴파일러
   - `Prompt/PART_H.md` - 배포 + ONNX 내보내기

2. **프로젝트 설정**:
   - `config.yaml` - PART_B 원래 설계 설정
   - `CLAUDE.md` - 프로젝트 가이드 + 세션 초기화 지침

3. **핵심 코드 (PART_C~F 구현)**:
   - `src/data.py` - Two-Stage 데이터 파이프라인
   - `src/models/detector.py` - YOLOv11m 래퍼
   - `src/models/classifier.py` - EfficientNetV2-S 구현
   - `src/models/pipeline.py` - 조건부 Two-Stage 파이프라인
   - `src/train.py` - Interleaved 학습 루프
   - `src/api/main.py` - FastAPI 서빙

4. **검증 시스템**:
   - `tests/stage_1_evaluator.py` - OptimizationAdvisor + GPU 테스트 통합
   - `tests/gpu_smoke/` - 현재 GPU 스모크 테스트 방법론

> 파일이 없거나 읽기 실패 시, 어떤 경로가 비어있는지 **명시적으로 경고**하고 계속 진행합니다.

---

## 2) 분석 (읽은 뒤 반드시 생성할 산출물)
다음 **섹션 헤더와 포맷**을 그대로 출력하세요. (없으면 빈 섹션으로 두지 말고 실패 원인을 표시)

### [INITIALIZED]
- 언어 규칙: "모든 응답은 한국어"
- 실행 시각, 작업 루트: `/home/max16/pillsnap`
- Python 환경: `/home/max16/pillsnap/.venv/bin/python` (PyTorch 2.7.0+cu128)

### 프롬프트 스캔 결과
- Prompt/ 읽은 파일: PART_0~H.md 목록 (순서 유지)
- 누락/오류: 경로·사유 요약

### 프로젝트 설정 스캔 결과  
- `config.yaml`: Progressive Validation Strategy + RTX 5080 최적화 설정
- `CLAUDE.md`: 세션 초기화 지침 + 프로젝트 가이드

### 코드 스캔 결과 (PART_C~F 구현 상태)
- `src/data/sampling.py`: Progressive Validation 샘플링 시스템 (✅ 구현 완료)
- `src/models/detector.py`: YOLOv11m 래퍼 (✅ 구현 완료) 
- `src/models/classifier.py`: EfficientNetV2-S (✅ 구현 완료)
- `src/models/pipeline.py`: 조건부 파이프라인 (✅ 구현 완료)
- `src/train.py`: Interleaved 학습 루프 (❌ 미구현)
- `src/api/main.py`: FastAPI 서빙 (⚠️ 기본 구조만)

### 환경 검증 시스템 상태 
- `tests/stage_1_evaluator.py`: OptimizationAdvisor + GPU 환경 테스트 완료
- GPU 환경 검증: 성공 (RTX 5080 + PyTorch 2.7.0+cu128 호환성 확인)
- **✅ Progressive Validation Stage 1 완료**: 5K 샘플, 50 클래스, Two-Stage Pipeline 실제 이미지 테스트 성공

### 컨텍스트 스냅샷 (핵심 설계 원칙)
1) **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo), 자동 판단 완전 제거
   - Single 모드 (기본): EfficientNetV2-S 직접 분류 (384px)
   - Combo 모드 (명시적): YOLOv11m 검출(640px) → 크롭 → 분류(384px)
2) **Progressive Validation Strategy**: Stage 1-4 (5K→25K→100K→500K), **Stage 1 완료**
   - Train 데이터만 사용 (247만개), Val은 최종 테스트 전용
   - 실제 클래스 수: 4,523개 (목표 5,000개에서 수정)
3) **OptimizationAdvisor**: 반자동화 평가 시스템, 사용자 선택권 제공 (PART_0 철학)  
4) **RTX 5080 최적화**: 
   - Mixed Precision (TF32), torch.compile 준비
   - channels_last (분류기만, YOLO는 호환성 문제로 제외)
   - 16 dataloader workers, batch prefetch
5) **메모리 최적화**: 128GB RAM 활용, hotset 캐싱, LMDB, prefetch
6) **경로 정책**: **SSD 기반 절대 경로** (/home/max16/ssd_pillsnap/dataset)
   - **Stage 1-2**: 내장 SSD (3,500MB/s, 35배 향상)
   - **Stage 3-4**: M.2 SSD 4TB 확장 예정 (7,450MB/s, 75배 향상)
   - **디스크 I/O 병목 해결 완료**: HDD(100MB/s) → SSD(3,500MB/s)
7) **Python 실행**: scripts/python_safe.sh 통한 가상환경 강제

### DoD (Definition of Done)
- [x] PART_0~H 프롬프트 전체 읽기 완료
- [x] config.yaml PART_B 설계 반영 확인  
- [x] GPU 환경 검증 시스템 동작 확인
- [x] PART_C~F 핵심 아키텍처 구현
- [x] **실제 Progressive Validation Stage 1 구현** (5K 샘플, 50 클래스, Two-Stage Pipeline)
- [ ] OptimizationAdvisor와 실제 Stage 1 성능 연동

### 위험·제약 및 폴백
- RTX 5080 sm_120 vs 기존 패키지 호환성 → PyTorch 2.7.0+cu128로 해결 완료
- 128GB RAM 최적화 → config.yaml stage_overrides로 단계별 조정
- 대용량 데이터셋 → Progressive Validation으로 단계적 확장

### 다음 행동 (현재 우선순위)
- ✅ PART_C Two-Stage 데이터 파이프라인 구현 완료 (`src/data/sampling.py`)
- ✅ YOLOv11m 검출 모델 래퍼 구현 완료 (`src/models/detector.py`)
- ✅ EfficientNetV2-S 분류 모델 구현 완료 (`src/models/classifier.py`)
- ✅ 조건부 Two-Stage 파이프라인 구현 완료 (`src/models/pipeline.py`)
- ✅ **실제 Progressive Validation Stage 1 구현 및 검증 완료** (5K 샘플, 50 클래스)
- **다음**: Stage 2 (25K 샘플) 및 학습 시스템 구현 (`src/train.py`)

---

## 3) 세션 핀 (고정)
- 한국어 응답 규칙
- PART_0~H 프롬프트 설계 컨텍스트  
- ✅ PART_C~F 핵심 모델 아키텍처 구현 완료
- RTX 5080 + PyTorch 2.7.0+cu128 환경
- Progressive Validation + OptimizationAdvisor 철학
- ✅ **Stage 1 완료 (5K 샘플, Two-Stage Pipeline 실제 이미지 테스트 성공)**

---

## 4) 실패 처리
- Prompt/PART_*.md 누락 시 **즉시 경고** 후 중단
- src/ 핵심 파일 누락 시 **구현 상태** 명시
- config.yaml 파싱 실패 시 **설정 문제** 지적
- 임의 추측으로 채우지 않고 **실제 상태** 보고

---

## 5) 주의
- 이 프롬프트는 **세션 초기화 전용**입니다 (코드 수정/생성은 다음 단계)
- 출력 섹션 헤더·형식을 변경하지 마세요
- PART_0~H 프롬프트 읽기는 **필수**입니다

## 🚀 Quick Start (새 세션 시작 시)

```bash
# 1. 세션 초기화 (필수)
/.claude/commands/initial-prompt.md

# 2. 환경 확인
./scripts/python_safe.sh -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"

# 3. Stage 1 파이프라인 테스트
./scripts/python_safe.sh tests/test_stage1_real_image.py

# 4. 단위 테스트 확인
./scripts/python_safe.sh -m pytest tests/unit/ -v --tb=short
```

# PillSnap ML 프로젝트 현재 상황 (세션 연속성용)

**프로젝트**: PillSnap ML - Two-Stage Conditional Pipeline AI 시스템
**목적**: 5000개 EDI 코드 경구약제 식별 (92% 정확도 목표)  
**환경**: WSL2 + RTX 5080 16GB + PyTorch 2.7.0+cu128 + 128GB RAM
**아키텍처**: YOLOv11m 검출 + EfficientNetV2-S 분류

---

## 현재까지 진행된 작업

### 1. 기초 인프라 구축 완료 ✅
- **Python 환경 정리**: `.venv_gpu` → `.venv` 직접 사용, Python 3.11.13 고정
- **안전 실행 시스템**: `scripts/python_safe.sh` 가상환경 강제 사용 (신규)
- **별칭 시스템**: `scripts/setup_aliases.sh` 편의성 향상 (신규)
- **설정 시스템**: `src/utils/core.py` ConfigLoader, PillSnapLogger 구현
- **로깅 시스템**: 콘솔+파일 로깅, 메트릭, 타이머, 진행상황 추적

### 2. 데이터 구조 스캔 및 검증 완료 ✅
- **실제 데이터 분석**: 263만개 이미지 (Train: 247만, Val: 16만)
- **🚨 중요: 데이터 사용 정책 확인**
  - **Train 데이터**: 학습/검증 분할용 (train:val = 85:15)
  - **Val 데이터**: 최종 test 전용 (학습에 절대 사용 금지)
- **데이터 분포**: Single 99.3%, Combo 0.7% (매우 불균형한 분포)
- **실제 클래스 수**: 4,523개 (기존 목표 5,000개보다 적음)
- **이미지 해상도**: 100% 동일한 976x1280 해상도 확인 (신규 발견)
- **Progressive Validation**: Train 데이터만 사용하여 Stage 1-4 진행

### 3. 프로젝트 구조 완전 정리 ✅
- **모듈 구조 정리**: `src/utils.py` → `src/utils/core.py` 기능별 분류
- **스크립트 정리**: `scripts/` 기능별 분류 (env, data, deployment, training)
- **테스트 정리**: `tests/` 기능별 분류 (unit, integration, smoke, stage_validation)
- **아티팩트 정리**: `artifacts/` 정리 (stage1, models, manifests, logs, wheels)

### 4. GPU 환경 검증 완료 ✅
- **RTX 5080 호환성**: PyTorch 2.7.0+cu128 완전 구축
- **가상환경 일원화**: `.venv` 직접 사용, 심볼릭 링크 제거
- **GPU 검증**: CUDA 11.8, 16GB VRAM, channels_last 최적화 확인

### 5. 데이터 파이프라인 핵심 구현 완료 ✅ (신규)
- **Stage 1 샘플링 시스템**: Progressive Validation 전략 구현
- **K-code → EDI-code 매핑**: 완전한 메타데이터 관리 시스템
- **이미지 전처리 파이프라인**: Two-Stage 최적화 (일반+특화 버전)
- **고정 해상도 최적화**: 976x1280 특화로 **76% 성능 향상** (58.5→103.0 images/sec)
- **COCO → YOLO 포맷 변환기**: Bounding box 정규화 완료
- **데이터 로더 시스템**: Single/Combo 파이프라인 구현, 텐서 형태 오류 수정

---

## 현재 상태 (2025-08-19 기준) - 새로운 아키텍처 컴포넌트 완료
- ✅ **1단계: 기초 인프라 구축 완료** (Python 환경, 설정시스템, 로깅)
- ✅ **데이터 구조 스캔 및 검증 완료** (263만 이미지, 올바른 Train/Val 분리 확인)
- ✅ **🚨 데이터 사용 정책 수정 완료**
  - Train 데이터만 학습/검증 분할 (247만개)
  - Val 데이터는 최종 test 전용 (16만개, 학습 금지)
  - Progressive Validation Stage 1-4는 Train 데이터만 사용
  - 실제 클래스 수 4,523개로 목표 수정
- ✅ **프로젝트 구조 완전 정리 완료** (모듈, 스크립트, 테스트, 아티팩트)
- ✅ **2단계: 데이터 파이프라인 핵심 구현 완료** (올바른 데이터 경로 확인)
- ✅ **3단계: 모델 아키텍처 구현 완료**
- ✅ **4단계: 새로운 아키텍처 컴포넌트 완료** (2025-08-19 신규)
  - YOLOv11m 검출기 구현 및 테스트 완료
  - EfficientNetV2-S 분류기 구현 및 테스트 완료
  - Two-Stage Pipeline 통합 및 실제 이미지 테스트 완료
  - Stage 1 (5K 샘플, 50 클래스) 검증 성공
- ✅ **5단계: 상업용 학습/평가 시스템 완료** (2025-08-19 신규)
  - **Training Components**: 분류/검출 Stage별 전용 학습기 구현
  - **BatchSizeAutoTuner**: RTX 5080 최적 배치 크기 자동 탐색
  - **TrainingStateManager**: 체크포인트, 배포용 모델 패키징
  - **GPU Memory Monitor**: 메모리 사용량 모니터링 및 최적화
  - **Detection/Classification Evaluators**: Stage별 목표 달성 검증
  - **End-to-End Pipeline Evaluator**: 상업적 준비도 평가
- ✅ **6단계: 코드베이스 정리 및 테스트 강화 완료** (2025-08-19 신규)
  - **코드 정리**: TODO 빈 파일 제거 (`src/data.py`, `src/infer.py`)
  - **통합 테스트 수정**: 3개 실패 테스트 해결 (체크포인트, 배포, Stage 기준)
  - **테스트 강화**: 22개 테스트 (18개 기존 + 4개 신규 엄격한 테스트)
  - **상업용 검증**: 성능, 메모리, 에러 처리, Stage1 목표 엄격 검증

## 🎯 다음 구현 계획 (7단계: 프로덕션 준비)

### 완료된 6단계 상업용 아키텍처 ✅

#### 모델 아키텍처 (완료)
1. ✅ **YOLOv11m 검출 모델** (`src/models/detector_yolo11m.py`)
   - Ultralytics YOLOv11m 래퍼 구현 완료
   - Combination 약품 검출용 (640px 입력)
   - RTX 5080 최적화 (Mixed Precision, torch.compile)

2. ✅ **EfficientNetV2-S 분류 모델** (`src/models/classifier_efficientnetv2.py`)
   - timm 기반 50개 클래스 분류기 (Stage 1용)
   - Single 약품 직접 분류용 (384px 입력)
   - Pre-trained weights 활용

3. ✅ **Two-Stage 조건부 파이프라인** (`src/models/pipeline_two_stage_conditional.py`)
   - 사용자 선택 기반 모드 전환
   - Single 모드: 직접 분류 (기본)
   - Combo 모드: 검출 → 크롭 → 분류

#### 상업용 학습/평가 시스템 (완료)
4. ✅ **Training Stage Components** (`src/training/`)
   - `train_classification_stage.py`: EfficientNetV2-S 전용 학습기
   - `train_detection_stage.py`: YOLOv11m 전용 학습기
   - `batch_size_auto_tuner.py`: RTX 5080 최적 배치 크기 자동 탐색
   - `training_state_manager.py`: 체크포인트, 배포용 모델 패키징
   - `memory_monitor_gpu_usage.py`: GPU 메모리 사용량 모니터링

5. ✅ **Evaluation Components** (`src/evaluation/`)
   - `evaluate_detection_metrics.py`: 검출 성능 평가 및 Stage별 목표 검증
   - `evaluate_classification_metrics.py`: 분류 성능 평가
   - `evaluate_pipeline_end_to_end.py`: 상업적 준비도 평가
   - `evaluate_stage1_targets.py`: Stage 1 완전 검증

6. ✅ **Data Loading Components** (`src/data/`)
   - `dataloader_single_pill_training.py`: 단일 약품 학습용 데이터로더
   - `dataloader_combination_pill_training.py`: 조합 약품 학습용 데이터로더
   - 기존 `dataloaders.py` 유지 (호환성)

#### 통합 테스트 및 검증 (완료)
7. ✅ **통합 테스트 시스템** (`tests/integration/test_new_architecture_components.py`)
   - 22개 테스트 (18개 기본 + 4개 엄격한 검증)
   - Training Components, Evaluation Components, Memory Monitoring
   - 성능, 메모리, 에러 처리, Stage1 목표 엄격 검증

8. ✅ **Stage 1 실제 검증** (5K 샘플, 50 클래스)
   - Progressive Validation 샘플러 실행 완료
   - 파이프라인 검증 성공 (Single: 254ms, Combo: 273ms)
   - 배치 처리 최적화 확인 (13.6ms/image)

### 다음 계획 (7단계: 프로덕션 배포)
- **FastAPI 서빙 완성** (`src/api/main.py` 개선)
- **ONNX 모델 내보내기** (`src/export.py` 구현)
- **Stage 2-4 Progressive Validation** (25K→100K→500K 샘플 확장)
- **실제 학습 파이프라인 실행** (새로운 Training Components 활용)
- **배포 환경 최적화** (Docker, Kubernetes 준비)

### 완성된 아키텍처 전체 개요
- **Total Files**: 45개 Python 파일 (정리 후)
- **Test Coverage**: 22개 통합 테스트 + 다수 단위 테스트
- **Core Components**: 모델, 학습, 평가, 데이터 로딩 시스템 완전 구현
- **Commercial Ready**: 상업용 수준의 테스트 및 검증 시스템

---

## 완료된 핵심 구성 요소 (6단계 완료)
1. ✅ **PART_B 프로젝트 구조**: PART_0~H 원래 설계 복원
2. ✅ **GPU 환경 준비**: RTX 5080 + PyTorch 2.7.0+cu128 완전 구축  
3. ✅ **config.yaml**: Two-Stage Pipeline + 128GB RAM 최적화 설정
4. ✅ **PART_C 데이터 파이프라인**: Two-Stage 데이터 처리 완전 구현
5. ✅ **최적화된 전처리**: 976x1280 고정 해상도 특화 (76% 성능 향상)
6. ✅ **PART_D~F 모델 아키텍처**: YOLOv11m + EfficientNetV2-S + Pipeline 완전 구현
7. ✅ **Stage 1 검증**: 5K 샘플, 50 클래스, 실제 이미지 테스트 성공
8. ✅ **상업용 Training System**: 분류/검출 전용 학습기, 배치 크기 자동 조정
9. ✅ **상업용 Evaluation System**: Stage별 목표 검증, End-to-End 평가
10. ✅ **통합 테스트 시스템**: 22개 테스트 (기본 + 엄격한 검증)
11. ✅ **코드베이스 정리**: TODO 파일 제거, 중복 코드 분석 완료

## 🎯 다음 목표 (7단계: 프로덕션 배포)
1. **실제 학습 파이프라인 실행** (새로운 Training Components 활용)
2. **Stage 2-4 Progressive Validation** (25K→100K→500K 샘플)
3. **FastAPI 서빙 완성** (`src/api/main.py` 개선)
4. **ONNX 모델 내보내기** (`src/export.py` 구현)
5. **배포 환경 최적화** (Docker, Kubernetes 준비)

---

## 프로젝트 구조 (최신 업데이트 2025-08-19)
```
/home/max16/pillsnap/
├── config.yaml        # Progressive Validation + RTX 5080 최적화 설정
├── CLAUDE.md          # 프로젝트 가이드 + 세션 초기화 지침
├── .claude/
│   └── commands/
│       └── initial-prompt.md  # 세션 초기화 스크립트
├── src/               # 핵심 구현 모듈 (45개 Python 파일)
│   ├── utils/           # 유틸리티 모듈
│   │   ├── core.py        # ConfigLoader, PillSnapLogger ✅
│   │   └── oom_guard.py   # OOM 방지 기능
│   ├── data/             # Two-Stage 데이터 파이프라인 ✅
│   │   ├── progressive_validation_sampler.py   # Progressive Validation 샘플러
│   │   ├── pharmaceutical_code_registry.py     # K-code → EDI-code 매핑
│   │   ├── image_preprocessing_factory.py      # 이미지 전처리 (일반)
│   │   ├── optimized_preprocessing.py          # 최적화된 전처리 (76% 향상)
│   │   ├── format_converter_coco_to_yolo.py    # COCO → YOLO 변환
│   │   ├── dataloaders.py                      # Single/Combo 데이터 로더 (기존)
│   │   ├── dataloader_single_pill_training.py # 단일 약품 전용 데이터로더 ✅
│   │   └── dataloader_combination_pill_training.py # 조합 약품 전용 데이터로더 ✅
│   ├── models/          # AI 모델 구현 ✅
│   │   ├── detector_yolo11m.py          # YOLOv11m 래퍼 ✅
│   │   ├── classifier_efficientnetv2.py # EfficientNetV2-S ✅
│   │   └── pipeline_two_stage_conditional.py # 조건부 파이프라인 ✅
│   ├── training/        # 상업용 학습 시스템 ✅ (신규)
│   │   ├── train_classification_stage.py   # 분류 Stage 전용 학습기
│   │   ├── train_detection_stage.py        # 검출 Stage 전용 학습기
│   │   ├── batch_size_auto_tuner.py        # RTX 5080 배치 크기 자동 조정
│   │   ├── training_state_manager.py       # 체크포인트, 배포용 모델 패키징
│   │   ├── memory_monitor_gpu_usage.py     # GPU 메모리 모니터링
│   │   └── train_interleaved_pipeline.py   # Interleaved 학습 루프
│   ├── evaluation/      # 상업용 평가 시스템 ✅ (신규)
│   │   ├── evaluate_detection_metrics.py     # 검출 성능 평가, Stage별 목표 검증
│   │   ├── evaluate_classification_metrics.py # 분류 성능 평가
│   │   ├── evaluate_pipeline_end_to_end.py   # 상업적 준비도 평가
│   │   └── evaluate_stage1_targets.py        # Stage 1 완전 검증
│   ├── infrastructure/ # 인프라 컴포넌트
│   ├── train.py         # Training 시스템 런처 ✅
│   ├── evaluate.py      # Evaluation 시스템 런처 ✅
│   └── api/             # FastAPI 서빙
├── tests/             # 테스트 시스템 (강화됨)
│   ├── unit/            # 단위 테스트 (80+ 테스트)
│   ├── integration/     # 통합 테스트 ✅
│   │   └── test_new_architecture_components.py # 22개 통합 테스트 (기본+엄격한)
│   ├── smoke/           # 스모크 테스트
│   └── performance/     # 성능 테스트
├── scripts/           # 운영 스크립트
│   ├── python_safe.sh   # 안전한 Python 실행 스크립트 ✅
│   ├── env/             # 환경 관리
│   ├── data/            # 데이터 처리
│   ├── deployment/      # 배포 및 운영
│   └── training/        # 학습 관련
└── artifacts/         # 실험 산출물
    ├── stage1/          # Stage 1 관련 결과물 ✅
    ├── models/          # 훈련된 모델 저장소
    ├── manifests/       # 데이터 매니페스트
    ├── reports/         # 평가 리포트 ✅
    └── logs/            # 실험 로그
```

### 주요 변경사항:
- ✅ **제거됨**: `src/data.py`, `src/infer.py` (TODO만 있던 빈 파일)
- ✅ **신규 추가**: `src/training/` 디렉토리 (6개 상업용 학습 컴포넌트)
- ✅ **신규 추가**: `src/evaluation/` 디렉토리 (4개 상업용 평가 컴포넌트)  
- ✅ **신규 추가**: 전용 데이터로더 2개 (single/combination)
- ✅ **강화됨**: 통합 테스트 22개 (18개 기본 + 4개 엄격한 검증)
- ✅ **업데이트됨**: 모델 파일명 정확한 반영

---

## 🛠️ 다음 실행 단계 (즉시 시작 가능)

### 🚨 필수 워크플로우 (모든 Stage 공통)

#### 1. Stage 최종 검증 전 필수 단계
**모든 Stage 마지막 검증 시에는 반드시 다음 순서를 준수:**

```bash
# 1단계: BatchSizeAutoTuner 최적 설정 탐색 (필수)
./scripts/python_safe.sh -m src.training.batch_size_auto_tuner --stage [1-4]

# 2단계: 최적 설정으로 학습률, epoch 수 계산
# - RTX 5080 최적 배치 크기 적용
# - 2시간(Stage1), 8시간(Stage2) 등 시간 제한 내 목표 달성 계산
```

#### 2. Stage 코드 완료 후 필수 검증 절차
**모든 Stage 코드가 완료된 다음에는 반드시 다음 순서로 검증:**

```bash
# 1단계: 모든 테스트 코드 실행 (필수)
./scripts/python_safe.sh -m pytest tests/unit/ -v --tb=short
./scripts/python_safe.sh -m pytest tests/integration/ -v --tb=short

# 2단계: 1 epoch 학습 실행으로 파이프라인 검증 (필수)
./scripts/python_safe.sh -m src.training.train_classification_stage --stage [1-4] --epochs 1 --dry-run
./scripts/python_safe.sh -m src.training.train_classification_stage --stage [1-4] --epochs 1

# 3단계: 파이프라인 정상 작동 확인 후 본격 학습 진행
```

#### 3. 적절한 Epoch 수 판단 및 학습 전략
**시간 제한보다 학습 품질 우선 원칙:**

- **Early Stopping 활용**: ValidationLoss 개선 없으면 자동 중단 (patience=5)
- **목표 달성 우선**: 목표 정확도 달성 시 즉시 완료
- **시간 제한은 참고용**: PART_0.md의 시간은 대략적 예상치, 품질 우선
- **충분한 max_epochs 설정**: 50+ epochs로 설정하되 Early Stopping으로 자동 중단

```bash
# 올바른 학습 전략 예시
./scripts/python_safe.sh -m src.training.train_classification_stage \
  --stage 1 \
  --epochs 50 \                    # 충분히 큰 수 설정  
  --batch-size 112 \               # BatchSizeAutoTuner 결과
  --early-stopping-patience 5     # 5 epoch 개선 없으면 중단
```

#### 4. 워크플로우 준수 이유
- **BatchSizeAutoTuner**: RTX 5080 하드웨어 특성에 맞는 최적 설정 보장
- **테스트 우선**: 코드 안정성 확보 후 학습 진행  
- **1 epoch 검증**: 긴 학습 전 파이프라인 오류 조기 발견
- **Early Stopping**: 과적합 방지 및 최적 수렴점 자동 탐지

### 핵심 구현 명령어 모음

#### 모델 테스트 및 검증
```bash
# YOLOv11m 검출기 단독 테스트
./scripts/python_safe.sh -m src.models.detector

# EfficientNetV2-S 분류기 단독 테스트  
./scripts/python_safe.sh -m src.models.classifier

# Two-Stage Pipeline 통합 테스트
./scripts/python_safe.sh -m src.models.pipeline

# 전체 단위 테스트 실행 (80개)
./scripts/python_safe.sh -m pytest tests/unit/ -v --tb=short
```

#### Stage 1 검증 명령어
```bash
# Stage 1 샘플 생성 (5K 이미지, 50 클래스)
./scripts/python_safe.sh -m src.data.progressive_validation_sampler

# 실제 이미지로 파이프라인 테스트
./scripts/python_safe.sh tests/test_stage1_real_image.py

# 새로운 아키텍처 컴포넌트 통합 테스트 (22개)
./scripts/python_safe.sh -m pytest tests/integration/test_new_architecture_components.py -v

# 상업용 학습 시스템 테스트
./scripts/python_safe.sh -m src.training.train_classification_stage
./scripts/python_safe.sh -m src.training.batch_size_auto_tuner

# End-to-End 평가 시스템 테스트
./scripts/python_safe.sh -m src.evaluation.evaluate_pipeline_end_to_end
```

### 4단계: Stage 1 실제 실행
```bash
# Progressive Validation Stage 1 샘플링
./scripts/python_safe.sh -m src.data.sampling

# Stage 1 실제 이미지 테스트
./scripts/python_safe.sh tests/test_stage1_real_image.py

# 단위 테스트 실행 (80개 테스트)
./scripts/python_safe.sh -m pytest tests/unit/ -v
```

### 완료된 구현 목록 ✅

#### 2단계: 데이터 파이프라인
- ✅ Progressive Validation 샘플링 (`src/data/sampling.py`)
  - Stage1SamplingStrategy: 5K 이미지, 50 클래스, 100개/클래스
  - ProgressiveValidationSampler: 자동 스캔 및 품질 검증
- ✅ 이미지 전처리 파이프라인 (`src/data/image_preprocessing.py`)
- ✅ 최적화된 전처리 (76% 성능 향상, `src/data/optimized_preprocessing.py`)
- ✅ COCO → YOLO 포맷 변환 (`src/data/format_converter.py`)
- ✅ Single/Combo 데이터 로더 (`src/data/dataloaders.py`)
- ✅ K-코드 매핑 관리자 (`src/data/metadata_manager.py`)

#### 3단계: 모델 아키텍처  
- ✅ YOLOv11m 검출기 (`src/models/detector.py`)
  - PillSnapYOLODetector: Ultralytics YOLO 래퍼
  - YOLOConfig: 640px 입력, conf=0.25, iou=0.45
  - RTX 5080 최적화: Mixed Precision 지원
- ✅ EfficientNetV2-S 분류기 (`src/models/classifier.py`)
  - PillSnapClassifier: timm 백본 활용
  - ClassifierConfig: 384px 입력, temperature scaling
  - Top-K 예측, 특징 추출, 배치 처리
- ✅ Two-Stage Pipeline (`src/models/pipeline.py`)
  - PillSnapPipeline: 사용자 제어 모드 선택
  - Single 모드: 직접 분류 (기본)
  - Combo 모드: 검출 → 크롭 → 분류

---

## 실행 환경 (현재 구축 완료)
```bash
# 🔥 가상환경 Python 실행 (권장)
./scripts/python_safe.sh --version
./scripts/python_safe.sh -m pytest tests/ -v
./scripts/python_safe.sh scripts/data/analyze_dataset_structure.py

# 별칭 설정 (선택사항)
source scripts/setup_aliases.sh
pp --version               # Python 실행
ptest tests/ -v           # pytest 실행

# 데이터 루트 설정 (SSD 이전 완료)
export PILLSNAP_DATA_ROOT="/home/max16/ssd_pillsnap/dataset"

# GPU 호환성 확인
./scripts/python_safe.sh -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# 출력: True 2.7.0+cu128

# 현재 상태: 6단계 상업용 아키텍처 완료, 7단계 프로덕션 배포 준비
```

**최신 변경사항 (2025-08-19)**:
- ✅ 프로젝트 구조 완전 정리 (모듈, 스크립트, 테스트, 아티팩트)
- ✅ 실제 데이터 구조 분석 완료 (263만 이미지, Single:Combo=143.6:1 불균형)
- ✅ Python 환경 일원화 + 안전 실행 스크립트 구축
- ✅ 데이터 파이프라인 핵심 구현 완료 (이미지 전처리 76% 성능 향상)
- ✅ 고정 해상도 (976x1280) 특화 최적화 완료
- ✅ **3단계 모델 아키텍처 구현 완료**:
  - YOLOv11m 검출기 (`src/models/detector_yolo11m.py`) + 단위 테스트 22개 통과
  - EfficientNetV2-S 분류기 (`src/models/classifier_efficientnetv2.py`) + 단위 테스트 31개 통과  
  - Two-Stage Pipeline (`src/models/pipeline_two_stage_conditional.py`) + 단위 테스트 27개 통과
  - Stage 1 실제 이미지 테스트 성공 (`tests/test_stage1_real_image.py`)
  - 테스트 결과: Single 254ms, Combo 273ms, 배치 처리 13.6ms/image
- ✅ **4-6단계 상업용 시스템 구현 완료** (신규):
  - **Training System**: 8개 핵심 컴포넌트 (`src/training/`)
  - **Evaluation System**: 4개 평가 모듈 (`src/evaluation/`)
  - **Data Loading**: 2개 전용 데이터로더 (`src/data/`)
  - **Integration Tests**: 22개 통합 테스트 (18개 기본 + 4개 엄격한 검증)
  - **Code Cleanup**: TODO 빈 파일 제거, 통합 테스트 실패 문제 해결
  - **Commercial Ready**: 상업용 수준의 성능/메모리/에러 처리 검증

**재현성 보장**: 새로운 세션에서는 `/.claude/commands/initial-prompt.md`를 실행하여 전체 컨텍스트를 복원할 수 있습니다.

**가상환경 사용법 상세**: `scripts/README.md` 참조

---
