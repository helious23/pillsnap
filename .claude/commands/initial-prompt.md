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
- **주의**: 아직 실제 Progressive Validation Stage 1 (5K 샘플, Two-Stage Pipeline) 미구현

### 컨텍스트 스냅샷 (핵심 설계 원칙)
1) **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo), 자동 판단 완전 제거
2) **Progressive Validation Strategy**: Stage 1-4 (5K→25K→100K→500K), **현재 환경 준비만 완료**
3) **OptimizationAdvisor**: 반자동화 평가 시스템, 사용자 선택권 제공 (PART_0 철학)  
4) **RTX 5080 최적화**: Mixed Precision, torch.compile, channels_last, 16 workers
5) **메모리 최적화**: 128GB RAM 활용, hotset 캐싱, LMDB, prefetch
6) **경로 정책**: WSL 절대 경로만 사용 (/mnt/data/pillsnap_dataset)

### DoD (Definition of Done)
- [ ] PART_0~H 프롬프트 전체 읽기 완료
- [ ] config.yaml PART_B 설계 반영 확인  
- [ ] GPU 환경 검증 시스템 동작 확인 (완료)
- [ ] PART_C~F 핵심 아키텍처 구현
- [ ] **실제 Progressive Validation Stage 1 구현** (5K 샘플, 50 클래스, Two-Stage Pipeline)
- [ ] OptimizationAdvisor와 실제 Stage 1 성능 연동

### 위험·제약 및 폴백
- RTX 5080 sm_120 vs 기존 패키지 호환성 → PyTorch 2.7.0+cu128로 해결 완료
- 128GB RAM 최적화 → config.yaml stage_overrides로 단계별 조정
- 대용량 데이터셋 → Progressive Validation으로 단계적 확장

### 다음 행동 (현재 우선순위)
- PART_C Two-Stage 데이터 파이프라인 구현 (`src/data.py`)
- YOLOv11m 검출 모델 래퍼 구현 (`src/models/detector.py`)
- EfficientNetV2-S 분류 모델 구현 (`src/models/classifier.py`)
- 조건부 Two-Stage 파이프라인 구현 (`src/models/pipeline.py`)
- **실제 Progressive Validation Stage 1 구현 및 검증** (5K 샘플, 50 클래스)

---

## 3) 세션 핀 (고정)
- 한국어 응답 규칙
- PART_0~H 프롬프트 설계 컨텍스트  
- 현재 환경 준비 완료 상태 + PART_C~F 구현 필요
- RTX 5080 + PyTorch 2.7.0+cu128 환경
- Progressive Validation + OptimizationAdvisor 철학
- **실제 Stage 1 (5K 샘플, Two-Stage Pipeline) 미시작**

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
- **실제 데이터 분석**: 263만개 이미지 (Single: 261만, Combo: 1.8만)
- **데이터 분포**: Single 99.3%, Combo 0.7% (매우 불균형한 분포)
- **이미지 해상도 분석**: 100% 동일한 976x1280 해상도 확인 (신규 발견)
- **K-코드 매핑**: EDI 코드 연결, 약품 메타데이터 추출
- **Progressive Validation**: Stage 1 요구사항 (5K 이미지, 50 클래스) 확인
- **데이터 무결성**: 이미지-라벨 매칭 검증 완료

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

## 현재 상태 (2025-08-19 기준)
- ✅ **1단계: 기초 인프라 구축 완료** (Python 환경, 설정시스템, 로깅)
- ✅ **데이터 구조 스캔 및 검증 완료** (263만 이미지, Single:Combo=143.6:1)
- ✅ **프로젝트 구조 완전 정리 완료** (모듈, 스크립트, 테스트, 아티팩트)
- ✅ **2단계: 데이터 파이프라인 핵심 구현 완료** (신규)
- ❌ **PART_D~F 모델 아키텍처 미구현** (YOLOv11m, EfficientNetV2-S, 통합 파이프라인)
- ❌ **실제 Progressive Validation Stage 1 미시작** (5K 샘플, 50 클래스)
- 🎯 **다음: 모델 아키텍처 구현 → Stage 1 통합 테스트**

---

## 완료된 핵심 구성 요소 
1. **PART_B 프로젝트 구조**: PART_0~H 원래 설계 복원
2. **GPU 환경 준비**: RTX 5080 + PyTorch 2.7.0+cu128 완전 구축  
3. **OptimizationAdvisor 준비**: PART_0 평가 시스템 환경 테스트 준비
4. **config.yaml**: Two-Stage Pipeline + 128GB RAM 최적화 설정
5. **환경 검증**: GPU 스모크 테스트를 통한 기본 환경 확인
6. **PART_C 데이터 파이프라인**: Two-Stage 데이터 처리 완전 구현 (신규)
7. **최적화된 전처리**: 976x1280 고정 해상도 특화 (76% 성능 향상, 신규)

## 미완료 핵심 구성 요소
1. **PART_D YOLOv11m 검출 모델**: 검출기 래퍼 및 훈련 파이프라인
2. **PART_E EfficientNetV2-S 분류 모델**: 분류기 및 훈련 파이프라인  
3. **PART_F 통합 Two-Stage Pipeline**: 조건부 파이프라인 구현
4. **실제 Progressive Validation**: Stage 1 (5K 샘플, 50 클래스) 미시작
5. **실제 모델 성능 검증**: YOLOv11m + EfficientNetV2-S 실제 동작 미확인

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
│   │   ├── detector.py    # YOLOv11m 래퍼 (TODO)
│   │   ├── classifier.py  # EfficientNetV2-S (TODO)
│   │   └── pipeline.py    # 조건부 파이프라인 (TODO)
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

## 다음 구현 우선순위 (3단계: 모델 아키텍처 구현)
1. **YOLOv11m 검출 모델**: PART_D 구현 (`src/models/detector.py`)
2. **EfficientNetV2-S 분류 모델**: PART_E 구현 (`src/models/classifier.py`)
3. **조건부 Two-Stage 파이프라인**: PART_F 통합 (`src/models/pipeline.py`)
4. **Interleaved 학습 루프**: PART_G 구현 (`src/train.py`)
5. **실제 Progressive Validation Stage 1**: 5K 샘플, 50 클래스 실행
6. **성능 검증 및 최적화**: OptimizationAdvisor 연동
7. **FastAPI 서빙**: PART_F API 엔드포인트 구현 (`src/api/main.py`)
8. **ONNX 내보내기**: PART_H 배포 준비 (`src/export.py`)

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

# 현재 상태: 데이터 파이프라인 완료, 모델 아키텍처 구현 대기
```

**최신 변경사항 (2025-08-19)**:
- ✅ 프로젝트 구조 완전 정리 (모듈, 스크립트, 테스트, 아티팩트)
- ✅ 실제 데이터 구조 분석 완료 (263만 이미지, Single:Combo=143.6:1 불균형)
- ✅ Python 환경 일원화 + 안전 실행 스크립트 구축
- ✅ 데이터 파이프라인 핵심 구현 완료 (이미지 전처리 76% 성능 향상)
- ✅ 고정 해상도 (976x1280) 특화 최적화 완료

**재현성 보장**: 새로운 세션에서는 `/.claude/commands/initial-prompt.md`를 실행하여 전체 컨텍스트를 복원할 수 있습니다.

**가상환경 사용법 상세**: `scripts/README.md` 참조

---
