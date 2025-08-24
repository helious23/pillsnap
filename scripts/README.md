# 📋 PillSnap ML - Scripts 디렉토리

**체계적으로 정리된 Stage별, 기능별 실행 스크립트 모음**

## 🗂️ 디렉토리 구조

```
scripts/
├── README.md                     # 📖 이 파일
│
├── core/                        # 🔧 핵심 유틸리티
│   ├── python_safe.sh           # 안전한 Python 실행
│   ├── setup_aliases.sh         # 편의 별칭 설정  
│   ├── setup_venv.sh            # 가상환경 구성
│   └── update_docs.sh           # 문서 업데이트
│
├── stage1/                      # 🎯 Stage 1 (5K 샘플, 50 클래스)
│   ├── migrate_stage1_images_only.sh
│   └── migrate_stage1_to_ssd.sh
│
├── stage2/                      # 🎯 Stage 2 (25K 샘플, 250 클래스)  
│   ├── run_stage2_sampling.py
│   ├── migrate_stage2_data.py
│   ├── monitor_stage2_migration.sh
│   ├── quick_status.sh
│   └── check_stage_overlap.py
│
├── stage3/                      # 🎯 Stage 3 (100K 샘플, 1K 클래스)
│   ├── train_stage3_two_stage.sh      # 🚀 메인 Two-Stage 학습
│   ├── monitor_stage3_realtime.py     # 📊 실시간 모니터링
│   ├── run_stage3_with_logs.sh        # 📝 로그 포함 실행
│   └── README.md                      # Stage 3 전용 가이드
│
├── monitoring/                  # 📊 모니터링 도구
│   ├── universal_training_monitor.sh  # 🔄 통합 모니터
│   ├── realtime_training_logger.py    # 📈 실시간 로거
│   ├── quick_status.sh               # ⚡ 빠른 상태 확인
│   └── _archived/                    # 구버전 보관
│
├── training/                    # 🏋️ 학습 관련
│   ├── run_model_training.sh
│   ├── train_with_monitor.sh
│   ├── train_and_monitor.sh  
│   └── reload_model_weights.sh
│
├── testing/                     # 🧪 테스트 실행
│   ├── run_all_tests.py               # ✅ 통합 테스트 실행기
│   ├── run_stage1_test_suite.py
│   ├── run_stage3_test_suite.py
│   ├── test_detection_state.py        # ✅ Detection State 테스트
│   ├── test_memory_manager.py         
│   └── test_optimization_advisor.py
│
├── backup/                      # 📦 백업 및 아카이빙 
│   └── freeze_stage_results.py        # ✅ Stage별 결과 동결
│
├── evaluation/                  # 📊 평가 및 검증
│   ├── sanity_check.py                # 기본 평가 스크립트
│   └── sanity_check_fixed.py          # ✅ 개선된 평가 스크립트
│
├── optimization/                # 🎯 최적화 및 튜닝
│   └── tune_detection_precision.py    # ✅ Detection 파라미터 튜닝
│
├── data_prep/                   # 🔧 데이터 준비
│   └── create_yolo_label_symlinks.py  # YOLO 라벨 심링크 생성
│
├── data/                        # 💾 데이터 처리
│   ├── analyze_dataset_structure.py
│   └── test_pharmaceutical_registry_builder.py
│   └── full_hash_verification.sh
│
├── deployment/                  # 🚀 배포 관련
│   ├── cloudflare_tunnel_*.ps1
│   ├── create_release_archive.sh
│   └── system_maintenance.sh
│
├── demo/                        # 🎪 데모 및 예제
│   ├── demo_progressive_resize.py     # 이동됨
│   └── demo_realtime_logs.py         # 이동됨
│
└── utils/                       # 🔧 유틸리티
    ├── simple_live_monitor.py         
    ├── simple_real_monitor.py        
    └── reorganize_scripts.py
```

## 🚀 빠른 사용 가이드

### Stage별 실행

```bash
# Stage 1 (완료)
# - 파이프라인 검증용, 이미 완료된 상태

# Stage 2 (완료) 
# - 기본 성능 확인, 이미 완료된 상태

# Stage 3 (완료) - Two-Stage Pipeline
./scripts/stage3/train_stage3_two_stage.sh                    # 기본 실행
./scripts/stage3/train_stage3_two_stage.sh --epochs 20        # 에포크 조정
./scripts/stage3/train_stage3_two_stage.sh --help             # 도움말

# Detection 상태 확인 및 테스트
python scripts/testing/test_detection_state.py                # State 관리 테스트

# Precision 튜닝
python scripts/optimization/tune_detection_precision.py       # conf/iou 최적화

# Stage 결과 백업 (Stage 1-4 지원)
python scripts/backup/freeze_stage_results.py --stage 3       # Stage 3 결과 동결
python scripts/backup/freeze_stage_results.py --stage 4       # Stage 4 결과 동결

# 평가 실행
python scripts/evaluation/sanity_check_fixed.py               # 개선된 평가

# 테스트 실행
python scripts/testing/run_all_tests.py                       # 모든 테스트
python scripts/testing/run_all_tests.py --category unit       # Unit 테스트만
python scripts/testing/run_all_tests.py --test tests/unit/test_classifier.py  # 특정 테스트
```

### 모니터링

```bash
# 통합 모니터링 (모든 Stage)
./scripts/monitoring/universal_training_monitor.sh

# Stage별 전용 모니터링
./scripts/monitoring/universal_training_monitor.sh --stage 3

# 빠른 상태 확인
./scripts/monitoring/quick_status.sh

# 실시간 로깅 (Stage 3 전용)
python scripts/stage3/monitor_stage3_realtime.py
```

### 핵심 유틸리티

```bash
# 환경 설정
./scripts/core/setup_venv.sh        # 가상환경 구성
./scripts/core/setup_aliases.sh     # 편의 별칭 설정

# 안전한 Python 실행
./scripts/core/python_safe.sh [명령어]
```

## 🎯 Stage 3 특화 기능

### 🚀 Two-Stage Pipeline 학습
```bash
# 기본 실행 (RTX 5080 최적화)
./scripts/stage3/train_stage3_two_stage.sh

# 하이퍼파라미터 커스터마이징  
./scripts/stage3/train_stage3_two_stage.sh \
  --epochs 18 \
  --batch-size 32 \
  --learning-rate 3e-4
```

**특징:**
- **RTX 5080 16GB 최적화**: Mixed Precision, torch.compile
- **Two-Stage 파이프라인**: YOLOv11x Detection + EfficientNetV2-L Classification
- **Manifest 기반**: 물리적 복사 없이 99.7% 저장공간 절약
- **실시간 검증**: GPU 메모리, 데이터 무결성, 시스템 리소스

### 📊 실시간 모니터링
```bash
# Stage 3 전용 모니터링
python scripts/stage3/monitor_stage3_realtime.py

# WebSocket 대시보드 (포트 8888)
python scripts/stage3/monitor_stage3_realtime.py --daemon --port 8888
```

## 🔧 컨벤션 및 규칙

### 파일 네이밍
```bash
# 실행 스크립트: 동사_명사_[상세].sh
train_stage3_two_stage.sh
monitor_realtime_training.sh

# Python 스크립트: 명사_[형용사]_명사.py
stage3_realtime_monitor.py
memory_usage_analyzer.py
```

### 헤더 표준
```bash
#!/bin/bash  
# PillSnap ML - [스크립트 제목]
# [한 줄 설명]
#
# 기능:
# - [주요 기능 1]
# - [주요 기능 2]
#
# 사용법:
#   ./scripts/[경로]/[스크립트명] [옵션]
```

### 로깅 표준
```bash
# 색상 코드 (PillSnap 표준)
RED='\033[0;31m'      # 에러
GREEN='\033[0;32m'    # 성공  
YELLOW='\033[1;33m'   # 경고
BLUE='\033[0;34m'     # 정보
PURPLE='\033[0;35m'   # Stage 표시
CYAN='\033[0;36m'     # 하이라이트

# 로그 함수 사용
log_info "정보 메시지"
log_success "성공 메시지"  
log_warning "경고 메시지"
log_error "에러 메시지"
log_stage "Stage 관련 메시지"
```

## 📈 성능 현황

### ✅ 완료된 Stage
- **Stage 1**: 74.9% 정확도 (Native Linux, 1분)
- **Stage 2**: 83.1% 정확도 (Native Linux, 하이브리드 스토리지)

### 🔄 현재 진행 중
- **Stage 3**: Two-Stage Pipeline 학습
  - 목표: Classification ≥85%, Detection mAP@0.5 ≥30%  
  - 데이터: 100K 샘플 (95% Single + 5% Combination)
  - 예상 시간: 1-2시간 (RTX 5080)

### 🎯 다음 단계
- **Stage 4**: 500K 샘플, 4.5K 클래스 프로덕션 학습
- **배포**: Cloudflare Tunnel 기반 API 서비스

## 🆘 트러블슈팅

### GPU 메모리 부족
```bash
# 배치 크기 감소
./scripts/stage3/train_stage3_two_stage.sh --batch-size 16

# GPU 리셋
nvidia-smi --gpu-reset
```

### 환경 문제
```bash  
# 가상환경 재설정
./scripts/core/setup_venv.sh

# Python 패키지 확인
./scripts/core/python_safe.sh -c "import torch; print(torch.__version__)"
```

### 권한 문제
```bash
# 스크립트 실행 권한 부여
find scripts/ -name "*.sh" -exec chmod +x {} \;
```

## 📚 관련 문서

- **프로젝트 개요**: `README.md`
- **진행 상황**: `SESSION_STATUS.md`  
- **코드 가이드**: `CLAUDE.md`
- **Stage 3 상세**: `scripts/stage3/README.md`
- **스크립트 재정리 계획**: `SCRIPTS_REORGANIZATION_PLAN.md`

---

**🏥 PillSnap ML - 체계적인 Stage별 스크립트 관리로 효율적인 개발 환경 구축**