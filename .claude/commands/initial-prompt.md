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
- `src/data.py`: Two-Stage 데이터 파이프라인 (구현 상태)
- `src/models/detector.py`: YOLOv11m 래퍼 (구현 상태) 
- `src/models/classifier.py`: EfficientNetV2-S (구현 상태)
- `src/models/pipeline.py`: 조건부 파이프라인 (구현 상태)
- `src/train.py`: Interleaved 학습 루프 (구현 상태)
- `src/api/main.py`: FastAPI 서빙 (구현 상태)

### 환경 검증 시스템 상태 
- `tests/stage_1_evaluator.py`: OptimizationAdvisor + GPU 환경 테스트 완료
- GPU 환경 검증: 성공 (RTX 5080 + PyTorch 2.7.0+cu128 호환성 확인)
- **✅ Progressive Validation Stage 1 완료**: 5K 샘플, 50 클래스, Two-Stage Pipeline 실제 이미지 테스트 성공

### 컨텍스트 스냅샷 (핵심 설계 원칙)
1) **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo), 자동 판단 완전 제거
2) **Progressive Validation Strategy**: Stage 1-4 (5K→25K→100K→500K), **Stage 1 완료**
3) **OptimizationAdvisor**: 반자동화 평가 시스템, 사용자 선택권 제공 (PART_0 철학)  
4) **RTX 5080 최적화**: Mixed Precision, torch.compile, channels_last, 16 workers
5) **메모리 최적화**: 128GB RAM 활용, hotset 캐싱, LMDB, prefetch
6) **경로 정책**: WSL 절대 경로만 사용 (/mnt/data/pillsnap_dataset)

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

## 현재 상태 (2025-08-19 기준) - 3단계 모델 아키텍처 완료
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
  - YOLOv11m 검출기 구현 및 테스트 완료
  - EfficientNetV2-S 분류기 구현 및 테스트 완료
  - Two-Stage Pipeline 통합 및 실제 이미지 테스트 완료
  - Stage 1 (5K 샘플, 50 클래스) 검증 성공

## 🎯 다음 구현 계획 (4단계: 학습 시스템)

### 완료된 3단계 모델 아키텍처 ✅
1. ✅ **YOLOv11m 검출 모델** (`src/models/detector.py`)
   - Ultralytics YOLOv11m 래퍼 구현 완료
   - Combination 약품 검출용 (640px 입력)
   - RTX 5080 최적화 (Mixed Precision, torch.compile)

2. ✅ **EfficientNetV2-S 분류 모델** (`src/models/classifier.py`)
   - timm 기반 50개 클래스 분류기 (Stage 1용)
   - Single 약품 직접 분류용 (384px 입력)
   - Pre-trained weights 활용

3. ✅ **Two-Stage 조건부 파이프라인** (`src/models/pipeline.py`)
   - 사용자 선택 기반 모드 전환
   - Single 모드: 직접 분류 (기본)
   - Combo 모드: 검출 → 크롭 → 분류

4. ✅ **Stage 1 실제 실행** (5K 샘플, 50 클래스)
   - Progressive Validation 샘플러 실행 완료
   - 파이프라인 검증 성공 (Single: 254ms, Combo: 273ms)
   - 배치 처리 최적화 확인 (13.6ms/image)

### 중기 계획 (4단계: 학습 시스템)
- **Interleaved 학습 루프** (`src/train.py`)
- **성능 평가 시스템** (`src/evaluate.py`)
- **OptimizationAdvisor 통합** (PART_0 평가 시스템)

### 장기 계획 (5-6단계: 서빙 및 배포)
- **FastAPI 서빙** (`src/api/main.py`)
- **ONNX 모델 내보내기** (`src/export.py`)
- **프로덕션 배포 준비**

---

## 완료된 핵심 구성 요소 
1. **PART_B 프로젝트 구조**: PART_0~H 원래 설계 복원
2. **GPU 환경 준비**: RTX 5080 + PyTorch 2.7.0+cu128 완전 구축  
3. **OptimizationAdvisor 준비**: PART_0 평가 시스템 환경 테스트 준비
4. **config.yaml**: Two-Stage Pipeline + 128GB RAM 최적화 설정
5. **환경 검증**: GPU 스모크 테스트를 통한 기본 환경 확인
6. **PART_C 데이터 파이프라인**: Two-Stage 데이터 처리 완전 구현 (신규)
7. **최적화된 전처리**: 976x1280 고정 해상도 특화 (76% 성능 향상, 신규)

## 🚧 다음 구현 예정 (4단계: 학습 시스템)
1. **Interleaved 학습 루프** (`src/train.py`)
   - PART_C 기반 검출기/분류기 교대 학습
   - RTX 5080 최적화 (Mixed Precision, Gradient Accumulation)
   
2. **성능 평가 시스템** (`src/evaluate.py`)
   - mAP@0.5 검출 성능 평가
   - Top-1/Top-5 분류 정확도 평가
   
3. **OptimizationAdvisor 통합** 
   - PART_0 자동 하이퍼파라미터 조정
   - 성능 메트릭 기반 학습률 스케줄링
   
4. **Progressive Validation Stage 2 준비**
   - 25K 샘플, 250 클래스로 확장
   - 메모리 사용량 및 처리 속도 모니터링
   
5. **학습 파이프라인 최적화**
   - Distributed Data Parallel (DDP) 지원
   - Automatic Mixed Precision (AMP) 최적화

---

## 프로젝트 구조 (정리 완료 2025-08-19)
```
/home/max16/pillsnap/
├── config.yaml        # Progressive Validation + RTX 5080 최적화 설정
├── CLAUDE.md          # 프로젝트 가이드 + 세션 초기화 지침
├── src/               # 핵심 구현 모듈
│   ├── utils/           # 유틸리티 모듈 (정리됨)
│   │   ├── core.py        # ConfigLoader, PillSnapLogger (완료)
│   │   └── oom_guard.py   # OOM 방지 기능
│   ├── data/             # Two-Stage 데이터 파이프라인 (완료)
│   │   ├── stage1_sampler.py      # Progressive Validation 샘플러
│   │   ├── metadata_manager.py    # K-code → EDI-code 매핑
│   │   ├── image_preprocessing.py # 이미지 전처리 (일반)
│   │   ├── optimized_preprocessing.py # 최적화된 전처리 (76% 향상)
│   │   ├── format_converter.py    # COCO → YOLO 변환
│   │   └── dataloaders.py         # Single/Combo 데이터 로더
│   ├── models/          # AI 모델 구현
│   │   ├── detector.py    # YOLOv11m 래퍼 (완료)
│   │   ├── classifier.py  # EfficientNetV2-S (완료)
│   │   └── pipeline.py    # 조건부 파이프라인 (완료)
│   ├── train.py         # Interleaved 학습 (TODO)  
│   └── api/             # FastAPI 서빙 (일부 구현됨)
├── tests/             # 기능별 테스트 (재구성됨)
│   ├── unit/            # 단위 테스트
│   ├── integration/     # 통합 테스트  
│   ├── smoke/           # 스모크 테스트
│   └── stage_validation/ # Progressive Validation 테스트
├── scripts/           # 운영 스크립트 (기능별 정리됨)
│   ├── env/             # 환경 관리
│   ├── data/            # 데이터 처리 (analyze_dataset_structure.py 포함)
│   ├── deployment/      # 배포 및 운영
│   └── training/        # 학습 관련
└── artifacts/         # 실험 산출물 (정리됨)
    ├── stage1/          # Stage 1 관련 결과물
    ├── models/          # 훈련된 모델 저장소
    ├── manifests/       # 데이터 매니페스트
    ├── logs/            # 실험 로그
    └── wheels/          # PyTorch CUDA 패키지 캐시
```

---

## 🛠️ 다음 실행 단계 (즉시 시작 가능)

### 1단계: YOLOv11m 검출 모델 구현
```bash
# PART_D 문서 기반 구현
./scripts/python_safe.sh -c "
import torch
from ultralytics import YOLO
print('Ultralytics YOLO 환경 검증:', torch.cuda.is_available())
"

# src/models/detector.py 구현 시작
# - YOLOv11m 래퍼 클래스
# - RTX 5080 최적화 (Mixed Precision)
# - Combination 약품 검출용 설정
```

### 2단계: EfficientNetV2-S 분류 모델 구현  
```bash
# PART_E 문서 기반 구현
./scripts/python_safe.sh -c "
import timm
model = timm.create_model('efficientnetv2_s', num_classes=4523)
print('timm EfficientNetV2-S 모델 생성 성공')
"

# src/models/classifier.py 구현
# - 4,523개 클래스 분류기
# - Pre-trained weights 활용
# - Single 약품 직접 분류용
```

### 3단계: Two-Stage 파이프라인 통합
```bash
# src/models/pipeline.py 구현
# - 사용자 선택 기반 모드 전환
# - Single 모드: 직접 분류  
# - Combo 모드: 검출 → 크롭 → 분류
```

### 4단계: Stage 1 실제 실행
```bash
# Progressive Validation Stage 1 샘플링 및 테스트
./scripts/python_safe.sh -m src.data.sampling
./scripts/python_safe.sh -m tests.stage_1_evaluator
```

### 완료된 데이터 파이프라인 (2단계) ✅
- ✅ Stage 1 데이터 샘플링 (`src/data/stage1_sampler.py`)
- ✅ 이미지 전처리 파이프라인 (`src/data/image_preprocessing.py`)
- ✅ 최적화된 전처리 (76% 성능 향상, `src/data/optimized_preprocessing.py`)
- ✅ COCO → YOLO 포맷 변환 (`src/data/format_converter.py`)
- ✅ Single/Combo 데이터 로더 (`src/data/dataloaders.py`)
- ✅ K-코드 매핑 관리자 (`src/data/metadata_manager.py`)

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

# 데이터 루트 설정 (자동)
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset"

# GPU 호환성 확인
./scripts/python_safe.sh -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# 출력: True 2.7.0+cu128

# 현재 상태: 3단계 모델 아키텍처 구현 완료, 4단계 학습 시스템 대기
```

**최신 변경사항 (2025-08-19)**:
- ✅ 프로젝트 구조 완전 정리 (모듈, 스크립트, 테스트, 아티팩트)
- ✅ 실제 데이터 구조 분석 완료 (263만 이미지, Single:Combo=143.6:1 불균형)
- ✅ Python 환경 일원화 + 안전 실행 스크립트 구축
- ✅ 데이터 파이프라인 핵심 구현 완료 (이미지 전처리 76% 성능 향상)
- ✅ 고정 해상도 (976x1280) 특화 최적화 완료
- ✅ **3단계 모델 아키텍처 구현 완료**:
  - YOLOv11m 검출기 + 단위 테스트 (22개 통과)
  - EfficientNetV2-S 분류기 + 단위 테스트 (31개 통과)  
  - Two-Stage Pipeline 통합 + 단위 테스트 (27개 통과)
  - Stage 1 실제 이미지 테스트 성공 (5K 샘플, 50 클래스)

**재현성 보장**: 새로운 세션에서는 `/.claude/commands/initial-prompt.md`를 실행하여 전체 컨텍스트를 복원할 수 있습니다.

**가상환경 사용법 상세**: `scripts/README.md` 참조

---
