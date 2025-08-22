# 🏥 PillSnap ML

**Commercial-Grade Two-Stage Conditional Pipeline 기반 경구약제 AI 식별 시스템**

[![Python](https://img.shields.io/badge/Python-3.11.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+cu128-orange.svg)](https://pytorch.org)
[![GPU](https://img.shields.io/badge/GPU-RTX%205080%2016GB-green.svg)](https://nvidia.com)
[![Architecture](https://img.shields.io/badge/Architecture-Commercial--Grade-purple.svg)](#)

---

## 🎯 프로젝트 개요

PillSnap ML은 **263만개 약품 이미지**를 활용하여 **4,523개 EDI 코드**를 식별하는 **상업용 수준 AI 시스템**입니다.

### 🏗️ Two-Stage Conditional Pipeline 아키텍처

```
📷 입력 이미지 → 사용자 모드 선택
    ├─ Single 모드 (기본) → EfficientNetV2-S 직접 분류 (384px) → EDI 코드
    └─ Combo 모드 (명시적) → YOLOv11m 검출 (640px) → 크롭 → 분류 → 다중 EDI 코드
```

### 🎯 성능 목표 & 현재 상태
- **Single 약품 정확도**: 92% (목표)
- **Combination 약품 mAP@0.5**: 0.85 (목표)
- **Stage 1 검증**: ✅ **완료** (74.9% 정확도, Native Linux)
- **Commercial 아키텍처**: ✅ **완료** (8개 상업용 컴포넌트 + 22개 통합 테스트)

---

## 🚀 Progressive Validation Strategy (Manifest 기반)

**안전한 단계별 확장**을 통한 프로덕션 준비:

| 단계 | 이미지 수 | 클래스 수 | 목적 | 상태 | 진행 방식 |
|------|-----------|-----------|------|------|-----------|
| **Stage 1** | 5,000개 | 50개 | 파이프라인 검증 | ✅ **완료** (74.9%) | Config 기반 |
| **Stage 2** | 25,000개 | 250개 | 성능 기준선 | 🔄 준비 완료 | Config 기반 |
| **Stage 3** | 100,000개 | 1,000개 | 확장성 테스트 | 🎯 **Manifest 기반** | **원본 직접로딩** |
| **Stage 4** | 500,000개 | 4,523개 | 프로덕션 배포 | 🎯 **Manifest 기반** | **원본 직접로딩** |

### ⭐ Stage 3-4 혁신적 접근법
- **물리적 복사 없음**: 73GB → 200MB 절약 (manifest CSV 파일만)
- **하이브리드 스토리지 최적화**: Linux SSD + Windows SSD 심볼릭 링크
- **Native Linux + 128GB RAM**: 실시간 고속 로딩으로 성능 손실 없음
- **용량 효율성**: SSD 공간 부족 문제 완전 해결

---

## 🖥️ 환경 구성 (Native Linux)

### 하드웨어 사양
- **GPU**: NVIDIA RTX 5080 16GB
- **CPU**: AMD Ryzen 7 7800X3D (8코어 16스레드)
- **RAM**: 128GB DDR5
- **Storage**: 4TB NVMe SSD
- **OS**: Native Ubuntu Linux

### 소프트웨어 환경
- **Python**: 3.11.13
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8

### 데이터 구조
```bash
/home/max16/
├── pillsnap/           # 프로젝트 코드
└── pillsnap_data/      # 데이터 전용 경로
    ├── train/
    │   ├── images/
    │   │   ├── single/  # 81개 폴더 (Linux + Windows SSD 하이브리드)
    │   │   └── combination/  # Windows SSD 심볼릭 링크
    │   └── labels/      # Linux SSD
    └── val/            # Windows SSD 심볼릭 링크
```

---

## 🚀 빠른 시작

### 1. 세션 초기화 (새 세션 시작 시 필수)

```bash
# 프로젝트 디렉토리로 이동
cd /home/max16/pillsnap

# 🔥 Claude Code 세션 초기화 (전체 컨텍스트 복원)
/.claude/commands/initial-prompt.md

# 환경 확인
source .venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"
# 예상 출력: CUDA: True, PyTorch: 2.8.0+cu128
```

### 2. Stage 1 학습 (완료)

```bash
# Stage 1 분류 학습 (74.9% 정확도 달성)
python -m src.training.train_classification_stage --stage 1 --epochs 1 --batch-size 32
```

### 3. Stage 2 학습 준비

```bash
# Stage 2 분류 학습 (250 클래스, 25K 샘플)
python -m src.training.train_classification_stage --stage 2 --epochs 30 --batch-size 32
```

### 4. 통합 테스트 실행

```bash
# 새로운 아키텍처 컴포넌트 통합 테스트
python -m pytest tests/integration/test_new_architecture_components.py -v

# 전체 단위 테스트 (80+ 테스트)
python -m pytest tests/unit/ -v --tb=short
```

---

## 📊 현재 구현 상태 (2025-08-22)

### ✅ **Native Linux 이전 완료**
- **WSL 제약 해결**: CPU 멀티프로세싱 활성화 (num_workers=8-12)
- **데이터 구조 개선**: `/home/max16/pillsnap_data` 분리
- **하이브리드 스토리지**: Linux SSD + Windows SSD 심볼릭 링크
- **Stage 1 검증**: 74.9% 정확도 (목표 40% 초과)

### ✅ **완료된 Commercial-Grade 아키텍처**

#### **데이터 파이프라인**
- **Progressive Validation**: Stage별 샘플링 시스템
- **최적화 전처리**: 976x1280 고정 해상도 특화
- **K-code → EDI-code 매핑**: 완전 구현

#### **AI 모델 아키텍처**
- **YOLOv11m 검출기**: 조합 약품 검출
- **EfficientNetV2-S 분류기**: 단일 약품 분류
- **Two-Stage Pipeline**: 조건부 파이프라인

#### **상업용 시스템**
- **Training Components**: 분류/검출 전용 학습기
- **Evaluation Components**: Stage별 평가 시스템
- **Data Loading Components**: 단일/조합 전용 로더

---

## 📁 프로젝트 구조

```
pillsnap/
├── 🔧 config.yaml              # 설정 파일
├── 📘 CLAUDE.md                # 프로젝트 가이드
├── 📁 src/                     # 핵심 구현
│   ├── data/                   # 데이터 파이프라인
│   ├── models/                 # AI 모델
│   ├── training/               # 학습 시스템
│   ├── evaluation/             # 평가 시스템
│   └── api/                    # API 서빙
├── 🧪 tests/                   # 테스트 시스템
└── 📜 scripts/                 # 운영 스크립트
```

---

## 🔧 설정 및 최적화

### GPU 최적화 (RTX 5080)
- Mixed Precision (TF32)
- channels_last 메모리 포맷
- torch.compile(mode='max-autotune')
- VRAM 사용량 모니터링 (14GB 제한)

### CPU 최적화 (Native Linux)
- num_workers=8-12 (16코어 활용)
- pin_memory=True
- persistent_workers=True
- prefetch_factor=6

---

## 📈 성능 메트릭

### Stage 1 결과 (2025-08-22)
- **학습 시간**: 1분
- **검증 정확도**: 74.9%
- **Top-5 정확도**: 76.7%
- **GPU 사용량**: 0.4GB
- **데이터 로딩**: 최적화됨

---

## 🚀 다음 단계

1. **Stage 2 학습 실행**: 250 클래스 분류
2. **검출 모델 학습**: YOLOv11m 조합 약품
3. **Stage 3-4 준비**: 대용량 데이터셋
4. **Production API**: Cloud tunnel 배포

---

## 📚 문서

- [CLAUDE.md](CLAUDE.md) - 프로젝트 종합 가이드
- [초기화 스크립트](.claude/commands/initial-prompt.md) - 세션 초기화
- [설계 문서](Prompt/) - PART_0 ~ PART_H 상세 설계

---

## 🤝 기여

프로젝트 기여 및 문의사항은 이슈를 통해 제출해주세요.

---

**🤖 Generated with [Claude Code](https://claude.ai/code)**

Co-Authored-By: Claude <noreply@anthropic.com>