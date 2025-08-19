# 🏥 PillSnap ML

**Two-Stage Conditional Pipeline 기반 경구약제 AI 식별 시스템**

---

## 🎯 프로젝트 개요

PillSnap ML은 **263만개 약품 이미지**를 활용하여 **4,523개 EDI 코드**를 식별하는 고성능 AI 시스템입니다.

### 🏗️ 아키텍처 - Two-Stage Conditional Pipeline

```
📷 입력 이미지 → 사용자 모드 선택
    ├─ Single 모드 → EfficientNetV2-S 직접 분류 (384px)
    └─ Combo 모드 → YOLOv11m 검출 (640px) → 크롭 → EfficientNetV2-S 분류
```

### 📊 성능 목표
- **Single 약품 정확도**: 92%
- **Combination 약품 mAP@0.5**: 0.85
- **추론 속도**: <100ms/이미지

---

## 🚀 Progressive Validation Strategy

단계별 확장을 통한 안정적인 시스템 구축:

| 단계 | 이미지 수 | 클래스 수 | 목적 |
|------|-----------|-----------|------|
| **Stage 1** | 5,000개 | 50개 | 파이프라인 검증 |
| **Stage 2** | 25,000개 | 250개 | 성능 기준선 |
| **Stage 3** | 100,000개 | 1,000개 | 확장성 테스트 |
| **Stage 4** | 2,000,000개 | 4,523개 | 프로덕션 배포 |

---

## 📁 프로젝트 구조

```
pillsnap/
├── 📁 src/                    # 핵심 구현 모듈
│   ├── utils/                 # 유틸리티 모듈
│   │   ├── core.py              # ConfigLoader, PillSnapLogger
│   │   └── oom_guard.py         # OOM 방지 시스템
│   ├── data.py                # Two-Stage 데이터 파이프라인 (TODO)
│   ├── models/                # AI 모델 구현
│   │   ├── detector.py          # YOLOv11m 검출 모델 (TODO)
│   │   ├── classifier.py        # EfficientNetV2-S 분류 모델 (TODO)
│   │   └── pipeline.py          # 조건부 파이프라인 (TODO)
│   ├── train.py               # 학습 파이프라인 (TODO)
│   └── api/                   # FastAPI 서빙 (일부 구현)
├── 📁 tests/                  # 기능별 테스트
│   ├── unit/                  # 단위 테스트
│   ├── integration/           # 통합 테스트
│   ├── smoke/                 # 스모크 테스트
│   └── stage_validation/      # Progressive Validation 테스트
├── 📁 scripts/                # 운영 스크립트
│   ├── env/                   # 환경 관리
│   ├── data/                  # 데이터 처리
│   ├── deployment/            # 배포 및 운영
│   └── training/              # 학습 관련
├── 📁 artifacts/              # 실험 산출물
│   ├── stage1/                # Stage 1 결과물
│   ├── models/                # 훈련된 모델
│   ├── manifests/             # 데이터 매니페스트
│   └── logs/                  # 실험 로그
├── config.yaml                # Progressive Validation + RTX 5080 최적화 설정
└── CLAUDE.md                  # 프로젝트 가이드 + 세션 초기화 지침
```

---

## 🔧 환경 설정

### 하드웨어 요구사항

**권장 사양**:
- **GPU**: RTX 5080 (16GB VRAM)
- **RAM**: 128GB 시스템 메모리
- **저장소**: NVMe SSD

**최소 사양**:
- **GPU**: RTX 3080 (10GB VRAM) 
- **RAM**: 32GB 시스템 메모리

### 소프트웨어 환경

```bash
# 환경 정보
OS: WSL2 (Ubuntu)
Python: 3.11.13
PyTorch: 2.7.0+cu128
CUDA: 11.8
```

---

## 🚀 빠른 시작

### 1. 환경 활성화

```bash
# 프로젝트 디렉토리로 이동
cd /home/max16/pillsnap

# 가상환경 활성화
bash scripts/env/activate_environment.sh

# 데이터 루트 설정
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset"
```

### 2. 데이터 구조 분석

```bash
# 실제 데이터 구조 스캔 (완료됨)
bash scripts/env/python_executor.sh scripts/data/analyze_dataset_structure.py

# 결과: 526만개 이미지, K-코드 매핑, 무결성 검증 완료
```

### 3. GPU 환경 검증

```bash
# PyTorch GPU 호환성 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, 버전: {torch.__version__}')"

# 예상 출력: CUDA: True, 버전: 2.7.0+cu128
```

---

## 📊 현재 구현 상태

### ✅ 완료된 단계

#### 1단계: 기초 인프라 구축 ✅
- **Python 환경**: 3.11.13 가상환경 구축
- **설정 시스템**: ConfigLoader (환경변수 오버라이드, 경로 검증)
- **로깅 시스템**: PillSnapLogger (콘솔+파일, 메트릭, 타이머)
- **안전 실행**: python_executor.sh (일관된 환경 보장)

#### 데이터 구조 스캔 및 검증 ✅
- **실제 데이터 분석**: 526만개 이미지 (Single: 524만, Combo: 1.7만)
- **데이터 매핑**: K-코드 → EDI 코드 연결, 약품 메타데이터 추출
- **무결성 검증**: 이미지-라벨 매칭 완료
- **Progressive Validation 준비**: Stage 1 요구사항 (5K 이미지, 50 클래스) 확인

#### 프로젝트 구조 정리 ✅
- **모듈 정리**: 기능별 분류 및 명확한 네이밍
- **스크립트 정리**: env, data, deployment, training 분류
- **테스트 정리**: unit, integration, smoke, stage_validation 분류
- **아티팩트 정리**: 실험 결과물 체계적 관리

### 🔄 진행 중인 단계

#### 2단계: 데이터 파이프라인 구현 (진행 중)
- **Stage 1 샘플링**: 526만 → 5K 이미지, 50 클래스 추출
- **이미지 전처리**: Detection(640px), Classification(384px) 최적화
- **포맷 변환**: COCO → YOLO 변환, 클래스 ID 매핑
- **메모리 최적화**: LMDB 캐싱, 128GB RAM 활용

### ❌ 미구현 단계

#### 3단계: 모델 아키텍처 구현
- YOLOv11m 검출 모델 구현
- EfficientNetV2-S 분류 모델 구현
- Two-Stage 조건부 파이프라인 통합

#### 4단계: 학습 파이프라인 구현
- Interleaved 학습 루프
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- OOM Guard 통합

#### 5단계: API 서비스 구현
- FastAPI REST 엔드포인트
- 이미지 업로드 및 처리
- Two-Stage 모드 선택

#### 6단계: 배포 및 모니터링
- ONNX 모델 내보내기
- Cloudflare Tunnel 배포
- 성능 모니터링

---

## 🔬 테스트 시스템

### 테스트 구조

```bash
tests/
├── unit/               # 단위 테스트
│   ├── test_config.py     # 설정 로딩 테스트
│   └── test_paths.py      # 경로 검증 테스트
├── integration/        # 통합 테스트
│   └── test_pipeline.py   # 파이프라인 전체 테스트
├── smoke/             # 스모크 테스트
│   └── gpu_smoke/        # GPU 기능 검증
└── stage_validation/  # Progressive Validation 테스트
    └── stage_*_evaluator.py  # 각 스테이지별 평가
```

### 테스트 실행

```bash
# 전체 테스트 실행
pytest tests/ -v

# 단위 테스트만
pytest tests/unit/ -v

# GPU 스모크 테스트
pytest tests/smoke/ -v
```

---

## ⚙️ 설정 파일

### config.yaml 주요 설정

```yaml
# Progressive Validation 설정
progressive_validation:
  enabled: true
  current_stage: 1           # 현재 Stage 1
  stages:
    stage1: {images: 5000, classes: 50}
    stage2: {images: 25000, classes: 250}
    stage3: {images: 100000, classes: 1000}
    stage4: {images: 500000, classes: 5000}

# Two-Stage Pipeline 설정
pipeline:
  mode: "user_controlled"     # 사용자 제어 모드
  detection_model: "yolov11m"
  classification_model: "efficientnetv2_s"
  input_sizes:
    detection: 640
    classification: 384

# RTX 5080 최적화
optimization:
  mixed_precision: true
  torch_compile: true
  channels_last: true
  dataloader_workers: 16
```

---

## 📈 성능 최적화

### RTX 5080 16GB 최적화

- **Mixed Precision (TF32)**: 메모리 효율성
- **torch.compile**: 학습 속도 최대 20% 향상
- **channels_last**: TensorCore 활용
- **LMDB 캐싱**: 128GB RAM 디스크 I/O 최적화

### 메모리 관리

- **OOM Guard**: 자동 배치 크기 조절
- **배치 프리페칭**: 16 workers로 GPU 대기시간 최소화
- **동적 할당**: VRAM 사용량 모니터링

---

## 🛠️ 주요 명령어

### 세션 초기화

```bash
# 새로운 세션에서 전체 컨텍스트 복원
/.claude/commands/initial-prompt.md
```

### 데이터 처리

```bash
# 데이터 구조 분석
bash scripts/env/python_executor.sh scripts/data/analyze_dataset_structure.py

# Progressive Validation Stage 1 샘플링 (TODO)
python -m src.data.stage1_sampler --output artifacts/stage1/
```

### 학습 (TODO)

```bash
# Single 약품 분류 학습
python -m src.train --mode single --stage 1 --epochs 100 --batch-size 128

# Combination 약품 검출 학습  
python -m src.train --mode combo --stage 1 --epochs 300 --batch-size 16
```

### API 서빙 (TODO)

```bash
# API 서버 시작
bash scripts/deployment/start_api_server.sh

# Cloudflare Tunnel 배포
powershell scripts/deployment/cloudflare_tunnel_start.ps1
```

---

## 📊 데이터셋 정보

### 전체 데이터 규모

- **총 이미지**: 263만개 (Train 데이터만 사용, Val은 최종 test 전용)
  - **Train 이미지**: 247만개 (학습 및 검증 분할용)
  - **Val 이미지**: 16만개 (최종 test 전용, 학습에 절대 사용 금지)
  - **Single 약품**: 261만개 (99.3%)
  - **Combination 약품**: 1.8만개 (0.7%)
- **K-코드**: 4,523개 (실제 식별 가능한 약품 코드)
- **EDI 코드**: 4,523개 (실제 분류 클래스 수)

### Stage 1 목표

- **이미지**: 5,000개 (Train 데이터의 0.2%)
- **클래스**: 50개 (전체의 1.1%)
- **목적**: 파이프라인 검증 및 기준선 설정

---

## 🤝 기여 가이드

### 개발 규칙

1. **경로 정책**: WSL 절대 경로만 사용 (`/mnt/data/`)
2. **Python 실행**: `scripts/env/python_executor.sh` 사용
3. **명명 규칙**: 구체적이고 기능적인 이름 사용
4. **테스트**: 모든 새 기능에 테스트 필수

### 코드 스타일

- **한국어 주석**: 모든 주석은 한국어로 작성
- **타입 힌트**: 함수 시그니처에 타입 명시
- **로깅**: PillSnapLogger 사용으로 일관된 로깅

---

## 📄 라이선스

[라이선스 정보 추가 예정]

---

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해주세요.

---

**PillSnap ML** - 차세대 약품 식별 AI 시스템  
*Claude Code와 함께 개발*

---

### 🔗 주요 링크

- **설정 가이드**: `CLAUDE.md`
- **세션 초기화**: `.claude/commands/initial-prompt.md`
- **데이터 분석 결과**: `artifacts/stage1/`
- **테스트 결과**: `tests/`

**마지막 업데이트**: 2025-08-19  
**현재 상태**: 2단계 - 데이터 파이프라인 구현 진행 중