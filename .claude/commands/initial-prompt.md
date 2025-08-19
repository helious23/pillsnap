# /initial-prompt — PillSnap ML 세션 초기화 스크립트

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 세션 초기화를 다음과 같이 수행합니다.
**모든 응답은 한국어로 작성**합니다. 모호하면 가장 단순한 해법을 우선합니다.

---

## 0) 프로젝트 개요 (Goal)
- **PillSnap ML**: Two-Stage Conditional Pipeline 기반 경구약제 식별 AI 시스템
- **아키텍처**: YOLOv11m 검출 + EfficientNetV2-S 분류 (5000개 EDI 코드)
- **환경**: WSL2 + RTX 5080 16GB + PyTorch 2.7.0+cu128
- **목표**: 92% 분류 정확도, 0.85 mAP@0.5 검출 성능

## 중요: Python 가상환경 경로
**모든 Python 실행 시 반드시 다음 경로를 사용:**
```bash
VENV_PYTHON="/home/max16/pillsnap/.venv/bin/python"
# 현재 .venv는 .venv_gpu로 심볼릭 링크 (RTX 5080 호환 PyTorch 2.7.0+cu128)
```
**시스템 python/python3 alias 사용 금지 (python3.13 충돌 방지)**

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
- **안전 실행 시스템**: `scripts/env/python_executor.sh` 환경 일관성 보장
- **설정 시스템**: `src/utils/core.py` ConfigLoader, PillSnapLogger 구현
- **로깅 시스템**: 콘솔+파일 로깅, 메트릭, 타이머, 진행상황 추적

### 2. 데이터 구조 스캔 및 검증 완료 ✅
- **실제 데이터 분석**: 526만개 이미지 (Single: 524만, Combo: 1.7만)
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

---

## 현재 상태 (2025-08-19 기준)
- ✅ **1단계: 기초 인프라 구축 완료** (Python 환경, 설정시스템, 로깅)
- ✅ **데이터 구조 스캔 및 검증 완료** (526만 이미지, K-코드 매핑)
- ✅ **프로젝트 구조 완전 정리 완료** (모듈, 스크립트, 테스트, 아티팩트)
- 🔄 **2단계: 데이터 파이프라인 구현 진행 중**
- ❌ **PART_C~F 아키텍처 미구현** (Two-Stage Pipeline 코드 없음)
- ❌ **실제 Progressive Validation Stage 1 미시작** (5K 샘플, 50 클래스)
- 🎯 **다음: 이미지 전처리 파이프라인 → COCO/YOLO 변환 → 데이터 로더**

---

## 완료된 핵심 구성 요소 
1. **PART_B 프로젝트 구조**: PART_0~H 원래 설계 복원
2. **GPU 환경 준비**: RTX 5080 + PyTorch 2.7.0+cu128 완전 구축  
3. **OptimizationAdvisor 준비**: PART_0 평가 시스템 환경 테스트 준비
4. **config.yaml**: Two-Stage Pipeline + 128GB RAM 최적화 설정
5. **환경 검증**: GPU 스모크 테스트를 통한 기본 환경 확인

## 미완료 핵심 구성 요소
1. **PART_C~F 아키텍처**: Two-Stage Pipeline 코드 미구현
2. **실제 Progressive Validation**: Stage 1 (5K 샘플, 50 클래스) 미시작
3. **실제 모델 성능 검증**: YOLOv11m + EfficientNetV2-S 실제 동작 미확인

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
│   ├── data.py          # Two-Stage 데이터 파이프라인 (TODO)
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

## 다음 구현 우선순위 (2단계: 데이터 파이프라인 구현)
1. **Stage 1 데이터 샘플링**: 526만 → 5K 이미지, 50 클래스 추출 (`src/data/stage1_sampler.py`)
2. **이미지 전처리 파이프라인**: Detection(640px), Classification(384px) 변환 (`src/data/image_transforms.py`)
3. **COCO → YOLO 포맷 변환**: Bounding box 정규화, 클래스 ID 매핑 (`src/data/coco_to_yolo_converter.py`)
4. **메모리 효율적 데이터 로더**: LMDB 캐싱, 배치 프리페칭 (`src/data/efficient_dataloader.py`)
5. **K-코드 매핑 관리자**: EDI 코드 매핑, 약품 메타데이터 (`src/data/drug_metadata_manager.py`)
6. **YOLOv11m 검출 모델**: PART_D 구현 (`src/models/detector.py`)
7. **EfficientNetV2-S 분류 모델**: PART_E 구현 (`src/models/classifier.py`)
8. **조건부 Two-Stage 파이프라인**: PART_C 통합 (`src/models/pipeline.py`)

---

## 실행 환경 (현재 구축 완료)
```bash
# RTX 5080 Python 환경 (정리됨)
bash scripts/env/activate_environment.sh  # 환경 활성화
bash scripts/env/python_executor.sh [스크립트]  # 안전한 Python 실행

# 데이터 루트 설정  
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset"

# 데이터 구조 분석 (완료됨)
bash scripts/env/python_executor.sh scripts/data/analyze_dataset_structure.py

# GPU 호환성 확인
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# 출력: True 2.7.0+cu128

# 현재 상태: 데이터 파이프라인 구현 진행 중
```

**최신 변경사항 (2025-08-19)**:
- ✅ 프로젝트 구조 완전 정리 (모듈, 스크립트, 테스트, 아티팩트)
- ✅ 실제 데이터 구조 분석 완료 (526만 이미지, K-코드 매핑)
- ✅ Python 환경 일원화 (.venv 직접 사용)
- 🔄 데이터 파이프라인 구현 시작

**재현성 보장**: 새로운 세션에서는 `/.claude/commands/initial-prompt.md`를 실행하여 전체 컨텍스트를 복원할 수 있습니다.

---
