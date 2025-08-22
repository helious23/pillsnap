# Part B — 프로젝트 뼈대 · 경로 · 환경 부트스트랩

[절대 경로 + 디스크 I/O 병목 해결 상황]

- **코드 루트**: /home/max16/pillsnap
- **현재 환경**: Native Ubuntu + M.2 SSD 4TB (CPU 멀티프로세싱 활용 num_workers=8)
- **WSL 제약 해결**: Native Linux 이전 완료 (16 CPU 코어 전체 활용)
- **데이터 루트**: 
  - **원본**: /mnt/data/pillsnap_dataset (외장 HDD 8TB, ext4, 100MB/s) - 전체 데이터셋
  - **현재 Native Linux**: /home/max16/pillsnap_data (하이브리드 스토리지, Linux SSD + Windows SSD) - Stage 1-2 완료
  - **미래 확장**: Stage 3-4를 위한 대용량 스토리지 예정
- **실험 디렉토리**: 
  - **Native Linux SSD**: /home/max16/pillsnap_data/exp/exp01 (현재 Stage 1-2 완료)
  - **백업**: /mnt/data/exp/exp01 (이전 기록)
- **Native Linux 환경 최적화 완료**:
  - **성과**: Stage 1 74.9% 정확도 (1분 완료), Stage 2 준비 완료
  - **성능**: 데이터 로딩 35배 향상, CPU 멀티프로세싱 활용 (num_workers=8)
  - **비교**: WSL 6분 vs Native Linux 1분 (향상된 성능)
- **하드웨어 스펙**:
  - **CPU**: AMD Ryzen 7 7800X3D (8코어 16스레드)
  - **RAM**: 128GB DDR5-5600 (삼성 32GB × 4)
  - **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
  - **Storage**: 
    - **OS/Code**: 1TB NVMe SSD (937GB 여유 공간)
    - **Data**: 8TB External HDD (100MB/s) + 4TB M.2 SSD 추가 계획 (7,450MB/s)
- **규칙**: 모든 데이터 스크립트는 **Native Linux SSD 경로** (/home/max16/pillsnap_data/) 사용. 원본 HDD 경로(/mnt/data/) 백업용. 프로젝트와 데이터 완전 분리.
- **예외**: Windows 운영 도구(Cloudflared 등, Part G/H)는 C:\ 표준 경로 사용 허용
- **데이터 처리 정책**:
  - **Stage 1**: 완료 (/home/max16/pillsnap_data) - 5,000장, 74.9% 정확도
  - **Stage 2**: 준비 완료 (25,000장, 250클래스)
  - **Stage 3-4**: 대용량 데이터셋 준비 중

[목표]

- **조건부 Two-Stage Pipeline**을 위한 프로젝트 구조 완성
- **128GB RAM + RTX 5080 16GB** 최적화 설정으로 config.yaml 구성
- **Native Linux SSD 데이터 경로**(/home/max16/pillsnap_data) 기반 환경 구축 (디스크 I/O 병목 해결)
- **단일/조합 약품** 구분 학습을 위한 스크립트 골격 생성
- 가상환경/의존성/기본 설정을 **한 번에 부팅** 가능하게 구성
- GPU CUDA 휠 우선 설치, 실패 시 CPU 폴백

[생성할 트리(정확히 이 구조로 생성)]

```
### file: PART_B.md — [프로젝트 트리] 일부 교정
/home/max16/pillsnap
├─ .gitignore
├─ .gitattributes
├─ .editorconfig
├─ README.md
├─ requirements.txt
├─ .env.example
├─ config.yaml
├─ scripts/
│  ├─ bootstrap_venv.sh
│  ├─ train.sh
│  ├─ export_onnx.sh
│  ├─ run_api.sh
│  ├─ maintenance.sh
│  ├─ backup_release.sh
│  ├─ reload_model.sh
│  ├─ ort_optimize.py
│  ├─ quantize_dynamic.py
│  ├─ perf_bench_infer.py
│  ├─ cf_start.ps1
│  ├─ cf_stop.ps1
│  └─ cf_status.ps1
├─ src/                        # 핵심 구현 모듈 (45개 Python 파일)
│  ├─ __init__.py
│  ├─ utils/                  # 유틸리티 모듈
│  │  ├─ __init__.py
│  │  ├─ core.py             # ConfigLoader, PillSnapLogger ✅
│  │  └─ oom_guard.py        # OOM 방지 기능
│  ├─ data/                  # Two-Stage 데이터 파이프라인 ✅
│  │  ├─ __init__.py
│  │  ├─ progressive_validation_sampler.py     # Progressive Validation 샘플러
│  │  ├─ pharmaceutical_code_registry.py       # K-code → EDI-code 매핑
│  │  ├─ image_preprocessing_factory.py        # 이미지 전처리 (일반)
│  │  ├─ optimized_preprocessing.py            # 최적화된 전처리 (76% 향상)
│  │  ├─ format_converter_coco_to_yolo.py      # COCO → YOLO 변환
│  │  ├─ dataloaders.py                        # Single/Combo 데이터 로더 (기존)
│  │  ├─ dataloader_single_pill_training.py   # 단일 약품 전용 데이터로더 ✅
│  │  └─ dataloader_combination_pill_training.py # 조합 약품 전용 데이터로더 ✅
│  ├─ models/                # AI 모델 구현 ✅
│  │  ├─ __init__.py
│  │  ├─ detector_yolo11m.py          # YOLOv11m 래퍼 ✅
│  │  ├─ classifier_efficientnetv2.py # EfficientNetV2-S ✅
│  │  └─ pipeline_two_stage_conditional.py # 조건부 파이프라인 ✅
│  ├─ training/              # 상업용 학습 시스템 ✅ (신규)
│  │  ├─ __init__.py
│  │  ├─ train_classification_stage.py   # 분류 Stage 전용 학습기
│  │  ├─ train_detection_stage.py        # 검출 Stage 전용 학습기
│  │  ├─ batch_size_auto_tuner.py        # RTX 5080 배치 크기 자동 조정
│  │  ├─ training_state_manager.py       # 체크포인트, 배포용 모델 패키징
│  │  ├─ memory_monitor_gpu_usage.py     # GPU 메모리 모니터링
│  │  └─ train_interleaved_pipeline.py   # Interleaved 학습 루프
│  ├─ evaluation/            # 상업용 평가 시스템 ✅ (신규)
│  │  ├─ __init__.py
│  │  ├─ evaluate_detection_metrics.py     # 검출 성능 평가, Stage별 목표 검증
│  │  ├─ evaluate_classification_metrics.py # 분류 성능 평가
│  │  ├─ evaluate_pipeline_end_to_end.py   # 상업적 준비도 평가
│  │  └─ evaluate_stage1_targets.py        # Stage 1 완전 검증
│  ├─ infrastructure/        # 인프라 컴포넌트
│  ├─ train.py              # Training 시스템 런처 ✅
│  ├─ evaluate.py           # Evaluation 시스템 런처 ✅
│  └─ api/                  # FastAPI 서빙
│     ├─ __init__.py
│     ├─ main.py            # FastAPI 앱
│     ├─ schemas.py         # edi_code 스키마
│     ├─ service.py         # Two-Stage 서비스
│     └─ security.py        # API 키 인증
└─ tests/                      # 테스트 시스템 (강화됨)
   ├─ unit/                    # 단위 테스트 (80+ 테스트)
   │  ├─ test_classifier.py    # 분류기 단위 테스트
   │  ├─ test_detector.py      # 검출기 단위 테스트
   │  ├─ test_pipeline.py      # 파이프라인 단위 테스트
   │  └─ ...                   # 기타 단위 테스트들
   ├─ integration/             # 통합 테스트 ✅
   │  ├─ test_new_architecture_components.py # 22개 통합 테스트 (기본+엄격한) ✅
   │  ├─ test_pipeline.py      # Two-Stage 파이프라인 통합 테스트
   │  ├─ test_api_min.py       # API 통합 테스트
   │  └─ test_entrypoints.py   # 진입점 테스트
   ├─ smoke/                   # 스모크 테스트
   │  ├─ test_smoke_detection.py    # YOLO 검출 스모크 테스트
   │  ├─ test_smoke_classification.py # 분류 스모크 테스트
   │  └─ gpu_smoke/            # GPU 스모크 테스트
   └─ performance/             # 성능 테스트 ✅
      ├─ stage_1_evaluator.py  # Stage 1 전용 평가 (파이프라인 검증)
      ├─ stage_2_evaluator.py  # Stage 2 전용 평가 (성능 기준선)
      ├─ stage_3_evaluator.py  # Stage 3 전용 평가 (프로덕션 준비)
      ├─ stage_4_evaluator.py  # Stage 4 전용 평가 (최종 프로덕션)
      ├─ stage_progress_tracker.py # 전체 Stage 진행 상황 추적
      └─ evaluate_stage.sh     # Stage별 평가 실행 스크립트
```

B-1. 메타 파일들(.gitignore / .gitattributes / .editorconfig / README 프롤로그)

[.gitignore — 정확히 이 내용으로 생성]

```
# venv / caches

venvs/
.venv/
**/__pycache__/**
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/
.cache/
.dist/
build/
dist/
*.egg-info/

# logs / artifacts

*.log
logs/
tb/
reports/
exp/

# secrets

.env

# OS / editor

.DS_Store
Thumbs.db
```

[.gitattributes — 개행/이진 파일 관리]

```
* text=auto eol=lf
*.sh text eol=lf
*.py text eol=lf
*.ps1 text eol=crlf
*.onnx binary
*.pt binary
*.png binary
*.jpg binary
*.zip binary
```

[.editorconfig — 편집기 규칙]
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 2
trim_trailing_whitespace = true

[*.py]
indent_size = 4

[README.md — 프롤로그(상단에 넣기)]

# pillsnap-ml

**조건부 Two-Stage 약품 식별 AI**

## 📋 프로젝트 개요
- **목적**: 약품 이미지에서 edi_code를 추출하여 약품 식별
- **데이터**: AIHub 166.약품식별 (5000종 단일 + 5000종 조합, 500K 이미지) - 영문 경로 /mnt/data/pillsnap_dataset
- **Pipeline**: 조건부 Two-Stage (단일→직접분류, 조합→검출후분류)

## 🔧 환경
- **Code root**: `/home/max16/pillsnap`
- **Data root**: `/home/max16/pillsnap_data` (Native Linux, 프로젝트 분리)
- **venv**: `$HOME/pillsnap/.venv`
- **Hardware**: AMD Ryzen 7 7800X3D + 128GB RAM + RTX 5080 16GB

## 🎯 성능 목표
- **단일 약품**: 92% accuracy (직접 분류)
- **조합 약품**: mAP@0.5 = 0.85 (YOLO 검출 → 분류)
- **전체 처리**: <100ms/image (RTX 5080)

## 🚀 Quick Start
Part B 끝의 "부트스트랩 & 점검" 섹션 참고.

B-2. requirements.txt (고정 + 안전한 범위 핀)

[요구사항]

- 다음 패키지를 설치한다. torch/torchvision은 CUDA 가능 시 CUDA 휠, 실패하면 CPU로 폴백.
- 버전은 안정-범용 조합을 쓴다(LLM이 최신과 호환되도록 상하한을 완만히 둠).

[requirements.txt — 정확히 이 내용으로 생성]

# Core DL (RTX 5080 최적화)

torch>=2.3,<2.5
torchvision>=0.18,<0.20
torchaudio>=2.3,<2.5
timm>=0.9.12,<1.0
ultralytics>=8.2.0

# Computer Vision
numpy>=1.24,<2.0
pillow>=10.0,<11.0
opencv-python-headless>=4.9,<5.0
albumentations>=2.0.8  # 최신 버전, PyTorch 2.8 호환
kornia>=0.7,<1.0

# Data & Config
pyyaml>=6.0,<7.0
tqdm>=4.66,<5.0
pandas>=2.0,<3.0
scikit-learn>=1.4,<2.0

# Logging & Monitoring
tensorboard>=2.15,<3.0
wandb>=0.16,<1.0
matplotlib>=3.8,<4.0
seaborn>=0.13,<1.0

# API & Service
fastapi>=0.110,<1.0
uvicorn[standard]>=0.27,<1.0
python-multipart>=0.0.9,<1.0
pydantic>=2.6,<3.0
pydantic-settings>=2.2,<3.0
python-dotenv>=1.0,<2.0

# Export & ONNX (GPU 지원)
onnx>=1.16,<2.0
onnxruntime>=1.17,<2.0
# onnxruntime-gpu>=1.17,<2.0  # GPU 환경에서 선택적 설치

# Memory & Performance (128GB RAM 활용)
lmdb>=1.4,<2.0
psutil>=5.9,<6.0

# Testing
httpx>=0.26,<1.0
pytest>=8.0,<9.0
pytest-asyncio>=0.23,<1.0

주석: GPU 환경이면 onnxruntime 대신 onnxruntime-gpu를 쓰고 싶을 수 있지만, 자동 판별은 Part E 스크립트에서 처리(프로바이더 선택).

B-3. .env.example (API 보안/CORS 템플릿)

[.env.example — 정확히 이 내용으로 생성]

# API / Server

API_KEY=CHANGE_ME_STRONG_RANDOM
LOG_LEVEL=info

# Optional: 강제 모델 경로 (비워두면 config.yaml/체크포인트 규칙 사용)

MODEL_PATH=

# CORS (콤마로 구분)

CORS_ALLOW_ORIGINS=http://localhost:3000,https://pillsnap.co.kr,https://api.pillsnap.co.kr

B-4. config.yaml (프로젝트 전역 설정 — 확정값 반영)

[config.yaml — RTX 5080 16GB + 128GB RAM 최적화 설정]

# 경로 설정 (Native Linux SSD 최적화)
paths:
  exp_dir: "/home/max16/pillsnap_data/exp/exp01"  # Native Linux SSD 실험 디렉토리
  data_root: "/home/max16/pillsnap_data"  # Native Linux SSD 데이터셋 경로 (Stage 1-2 완료)
  ckpt_dir: null  # exp_dir/checkpoints 자동 생성
  tb_dir: null    # exp_dir/tb 자동 생성
  reports_dir: null  # exp_dir/reports 자동 생성
  # 원본 HDD 경로 (필요시 참조용)
  data_root_hdd: "/mnt/data/pillsnap_dataset"  # 원본 데이터셋 (백업용)
  exp_dir_hdd: "/mnt/data/exp/exp01"  # 이전 실험 기록

# 데이터셋 구성
data:
  # 단순화된 파이프라인 (수정됨)
  pipeline_mode: "single"  # "single" (기본), "combo"
  default_mode: "single"   # API 기본값 (프론트엔드 권장)
  detection_lazy_load: false # 128GB RAM으로 즉시 로드
  
  # 실제 ZIP 추출 구조 기반 데이터 분할
  data_split:
    strategy: "user_controlled"                     # 사용자 제어 기반
    source_structure: "zip_based_folders"           # TS_1_single, TL_1_single 등 ZIP 기반 폴더
    split_ratio: [0.85, 0.15]                       # train:val = 85:15
    test_usage: "final_evaluation_only"              # test는 Stage 4 완료 후만 사용
  
  # 통일된 데이터 경로 (Native Linux SSD 최적화)  
  root: "/home/max16/pillsnap_data"  # Stage 1-2 완료, Stage 3-4 준비
  train:
    single_images: "data/train/images/single"      # TS_1_single~TS_81_single 폴더들 (각 폴더 내 K-코드 서브폴더 구조)
    combination_images: "data/train/images/combination"  # TS_1_combo~TS_8_combo 폴더들 (각 폴더 내 K-코드 서브폴더 구조)
    single_labels: "data/train/labels/single"    # TL_1_single~TL_81_single 폴더들 (각 폴더 내 K-코드_json 서브폴더 구조)
    combination_labels: "data/train/labels/combination"  # TL_1_combo~TL_8_combo 폴더들 (각 폴더 내 K-코드_json 서브폴더 구조)
  val:
    single_images: "data/val/images/single"        # VS_1_single~VS_10_single 폴더들 (각 폴더 내 K-코드 서브폴더 구조)
    combination_images: "data/val/images/combination"    # VS_1_combo 폴더 (각 폴더 내 K-코드 서브폴더 구조)
    single_labels: "data/val/labels/single"      # VL_1_single~VL_10_single 폴더들 (각 폴더 내 K-코드_json 서브폴더 구조)
    combination_labels: "data/val/labels/combination"  # VL_1_combo 폴더 (각 폴더 내 K-코드_json 서브폴더 구조)
  test:
    single_images: "data/test/images/single"       # Stage 4 완료 후만 사용 (각 폴더 내 K-코드 서브폴더 구조)
    combination_images: "data/test/images/combination" # (각 폴더 내 K-코드 서브폴더 구조)
    single_labels: "data/test/labels/single"        # (각 폴더 내 K-코드_json 서브폴더 구조)
    combination_labels: "data/test/labels/combination"  # (각 폴더 내 K-코드_json 서브폴더 구조)
  
  # 이미지 크기 (현실적 하드웨어 제약 고려)
  img_size:
    detection: 640      # YOLOv11m 입력 크기
    classification: 224 # EfficientNetV2-S 기본 크기
  
  # 클래스 정보 (Native Linux SSD 최적화)
  num_classes: 4523  # edi_code 기준 4523 클래스 (최종 실제 수)
  class_names_path: "/home/max16/pillsnap_data/processed/class_names.json"
  edi_mapping_path: "/home/max16/pillsnap_data/processed/edi_mapping.json"
  
  # 점진적 검증 샘플링 (PART_0 전략)
  progressive_validation:
    enabled: true
    current_stage: 1    # 1-4 단계 설정
    stage_configs:
      stage_1:
        max_samples: 5000
        max_classes: 50
        target_ratio: {single: 0.7, combination: 0.3}
        time_limit_hours: 2
        status: "completed"               # Stage 1 완료 상태
        accuracy_achieved: 0.749         # 달성 정확도 74.9%
        allow_success_on_time_cap: true   # 시간 캡 도달 시 성공 판정 허용
        min_samples_required: 1000        # 성공 판정 최소 처리 샘플 수
        min_class_coverage: 30            # 성공 판정 최소 클래스 커버리지
      stage_2: 
        max_samples: 25000
        max_classes: 250
        target_ratio: {single: 0.7, combination: 0.3}
        time_limit_hours: 8
        status: "ready"                  # Stage 2 준비 완료
      stage_3:
        max_samples: 100000
        max_classes: 4000
        target_ratio: {single: 0.7, combination: 0.3}  
        time_limit_hours: 16
      stage_4:
        max_samples: null    # 전체 데이터
        max_classes: 4523    # 실제 EDI 코드 수
        target_ratio: {single: 0.7, combination: 0.3}
        time_limit_hours: 48

  # 샘플링 전략
  sampling_strategy: "stratified_balanced"  # 계층적 균형 샘플링
  min_samples_per_class: 2  # train 1장, val 1장 최소 보장
  seed: 42                  # 재현 가능한 샘플링

# 사용자 제어 기반 설정 (단일화된 접근법)
  pipeline_strategy: "user_controlled"  # single 우선, combo 명시적 선택
  default_mode: "single"               # 90% 케이스 기본값
  auto_fallback: false                  # 자동 판단 완전 제거
    
  # OptimizationAdvisor 평가 설정
  optimization_advisor:
    enabled: true
    run_after_training: true    # 학습 완료 시 평가 실행
    generate_report: true       # JSON 리포트 생성
    update_tensorboard: true    # TB에 결과 로깅
    recommend_next_stage: true  # 권장사항 제공 (사용자 선택)
  
  # 128GB RAM 활용 최적화
  cache_labels: true      # 모든 라벨을 메모리에 캐시
  cache_images: false     # 이미지는 LMDB 변환 후 활용
  use_lmdb: true         # LMDB 변환으로 I/O 최적화
  
  # 증강 (데이터 품질 고려 보수적 설정)
  augment:
    train:
      albumentations: true
      horizontal_flip: 0.5
      vertical_flip: 0.0    # 약품 방향성 중요
      rotate_limit: 15      # 최대 15도 회전
      brightness_limit: 0.2
      contrast_limit: 0.2
      saturation_shift_limit: 0.2
      hue_shift_limit: 0.1
      blur_limit: 3
      noise_limit: 0.1
      cutout_holes: 8
      cutout_length: 16
    val:
      normalize_only: true

# 검출 모델 설정 (조합 약품용, 현실적 하드웨어 고려)
detection:
  model: "yolov11m"     # 현실적 VRAM 효율성
  pretrained: true
  num_classes: 1        # "pill" 위치 검출용 클래스 1개
  class_names: ["pill"]  # 약품 위치만 검출
  conf_threshold: 0.3   # 보수적 임계값
  iou_threshold: 0.5    # NMS 임계값
  max_detections: 100
  amp: true             # fp16 mixed precision

# 분류 모델 설정 (단일 + 조합 크롭용, 현실적 크기)
classification:
  backbone: "efficientnetv2_s.in21k_ft_in1k"  # 현실적 VRAM 효율성
  pretrained: true
  drop_rate: 0.3        # 과적합 방지
  drop_path_rate: 0.2   # Stochastic Depth
  num_classes: 5000     # edi_code 클래스 수
  amp: true             # fp16 mixed precision
  
  # 점진 리사이즈 전략 (VRAM 절약 + 정확도 방어)
  progressive_resize:
    enabled: true
    img_size_base: 224          # Phase 1-2 기본 크기 
    img_size_finetune: 288      # Phase 3 파인튜닝 크기 (마지막 2-5 epoch)
    switch_epoch: -5            # 마지막 5 epoch에 리사이즈
    warmup_epochs: 2            # 리사이즈 후 워밍업
    expected_gain: 0.02         # +1~3%p 정확도 향상 기대

# 손실 함수
loss:
  classification:
    type: "cross_entropy"
    label_smoothing: 0.1
    class_weights: "balanced"  # 클래스 불균형 대응
  detection:
    cls_loss_weight: 1.0
    box_loss_weight: 7.5
    dfl_loss_weight: 1.5

# 학습 설정 (RTX 5080 16GB + AMP 최적화) - Interleaved Two-Stage
train:
  # OOM 폴백 가드레일 (학습 일관성 보장)
  oom:
    max_retries: 4              # 총 재시도 상한
    max_grad_accum: 4           # grad_accum_steps 상한 (글로벌 배치 유지)
    min_batch: 1                # batch_size 하한
    cooldown_sec: 2             # 재시도 전 슬립
    escalate_to_fp16: true      # AMP fp16 강제 전환 허용
    
    # 학습 일관성 보장 (샘플 기반 스케줄링)
    consistency:
      preserve_global_batch: true     # batch_size × grad_accum 유지 우선
      scheduler_mode: "by_samples"    # 스텝 기반 → 샘플 기반 전환
      lr_rescaling: true              # Linear Scaling Rule 적용
      wd_rescaling: true              # Weight Decay 에폭당 총량 유지
      ema_per_sample: true            # EMA decay를 샘플 기준으로 계산
      bn_handling: "freeze_after_warmup"  # BN 통계 일관성 (freeze|groupnorm)
      replay_failed_batch: true      # OOM 배치를 동일 시드로 재실행
      audit_logging: true             # 모든 폴백 이벤트 상세 로깅
  # Two-Stage 학습 전략 
  strategy: "interleaved"  # sequential → interleaved
  interleave_ratio: [1, 1]  # det:cls = 1:1 균형
  
  # 검출 모델 학습 (조합 약품 전용, 순차 학습)
  detection:
    epochs: 50
    batch_size: null            # 동적 튜닝으로 자동 조정
    auto_batch_tune: true       # RTX 5080 16GB 최적화
    auto_batch_max: 16          # 현실적 상한 (학습시)
    grad_accum_steps: [1, 2, 4] # 마이크로배칭으로 글로벌 배치 유지
    grad_clip: 10.0
    
    optimizer: "adamw"
    lr: 2e-4
    weight_decay: 1e-4
    momentum: 0.937
    
    scheduler: "cosine"
    warmup_epochs: 3
    warmup_momentum: 0.8
    
    early_stopping:
      enabled: true
      monitor: "mAP@0.5"
      mode: "max"
      patience: 10
      min_delta: 0.001
  
  # 분류 모델 학습 (단일 + 조합 크롭, 순차 학습)
  classification:
    epochs: 30
    batch_size: null            # 동적 튜닝으로 자동 조정
    auto_batch_tune: true       # RTX 5080 16GB 최적화
    auto_batch_max: 96          # 현실적 상한 (EfficientNetV2-S)
    grad_accum_steps: [1, 2, 4] # 마이크로배칭으로 글로벌 배치 유지
    grad_clip: 1.0
    
    optimizer: "adamw"  
    lr: 2e-4
    weight_decay: 1e-4
    
    scheduler: "cosine"
    warmup_epochs: 2
    
    early_stopping:
      enabled: true
      monitor: "macro_f1"
      mode: "max" 
      patience: 10
      min_delta: 0.001

  # 공통 설정
  seed: 42
  deterministic: false  # 성능 모드
  resume: null         # 자동 체크포인트 복구

# GPU/메모리 최적화 (RTX 5080 16GB)
optimization:
  # Mixed Precision
  amp: true
  amp_dtype: "auto"      # bfloat16 > fp16 자동 선택
  
  # CUDA 최적화
  tf32: true
  channels_last: true
  torch_compile: "max-autotune"     # reduce-overhead → max-autotune
  compile_fallback: "reduce-overhead"  # 추가
  compile_warmup_steps: 100            # 추가
  
  # CUDA Graphs (실험적)
  use_cuda_graphs: false  # 안정성 우선
  
  # 메모리 관리
  empty_cache_steps: 100  # 주기적 캐시 정리
  
  # 프로파일링
  warmup_steps: 100      # 컴파일 워밍업
  profile_interval: 500   # 성능 로깅 주기

# 데이터로더 (128GB RAM + 16 스레드 최적화, Native Linux)
dataloader:
  num_workers: 8  # Native Linux 최적화값
  autotune_workers: true
  pin_memory: true
  pin_memory_device: "cuda"
  prefetch_factor: 6          # 기본값 상향 (4→6)
  prefetch_per_stage:         # Stage별 차별화
    1: 4
    2: 6
    3: 8
    4: 8
  persistent_workers: true
  drop_last: true
  multiprocessing_context: "spawn"
  
  # RTX 5080 16GB + 128GB RAM 고정 최적화
  # 128GB RAM 최적화 설정 (얇은 핫셋)
  ram_optimization:
    cache_policy: "hotset"      # 기본 핫셋만 사용
    hotset_size_images: 60000   # 6만장 캐시 (≈24.7 GiB uint8)
    cache_labels: true
    use_lmdb: false             # 복잡한 LMDB 제거 → 기본 경로만
    decode_dtype: uint8         # 메모리 절약
    to_tensor_dtype: float16    # 추가 절약  
    preload_samples: 0          # 프리로드 비활성화

# Stage별 하드웨어 최적화 오버라이드
stage_overrides:
  1:    # Stage 1: 파이프라인 검증 (5K, 50클래스)
    purpose: pipeline_validation   # 파이프라인 검증 단계
    eval_only: true               # 학습 비활성화, 평가만 수행
    augment_light: true          # 가벼운 증강으로 스루풋 최대화
    dataloader: 
      num_workers: 8
      prefetch_factor: 4
    train:
      detection: 
        auto_batch_tune: true
        auto_batch_max: 200        # base(160) → 200 증가
        batch_size: null           # 고정 배치 제거, 튜너에 위임
      classification: 
        auto_batch_tune: true
        auto_batch_max: 280        # base(224) → 280 증가
        batch_size: null           # 고정 배치 제거, 튜너에 위임
    ram_optimization:
      hotset_size_images: 30000  # Stage 1: 3만장으로 축소

validation_rules:
  - name: "stage1_batch_not_reduced"
    when: { stage: 1, purpose: pipeline_validation }
    assert:
      - "train.detection.auto_batch_max >= base.train.detection.auto_batch_max"
      - "train.classification.auto_batch_max >= base.train.classification.auto_batch_max"
    on_fail: "error"
    
  2:    # Stage 2: 성능 기준선 (25K, 250클래스)  
    dataloader:
      num_workers: 12
      prefetch_factor: 6
    train:
      detection: {auto_batch_max: 120}
      classification: {auto_batch_max: 160}
    ram_optimization:
      hotset_size_images: 50000  # Stage 2: 5만장
  3:    # Stage 3: 확장성 테스트 (100K, 1000클래스)
    dataloader:
      num_workers: 16
      prefetch_factor: 8
    ram_optimization:
      hotset_size_images: 70000  # Stage 3: 7만장
  4:    # Stage 4: 최종 프로덕션 (500K, 5000클래스)
    dataloader:
      num_workers: 16
      prefetch_factor: 8
    ram_optimization:
      hotset_size_images: 80000  # Stage 4: 8만장 (최대)

# 로깅 및 모니터링
logging:
  # TensorBoard
  tensorboard: true
  wandb: false          # 선택적 활성화
  
  # 로그 주기
  step_log_interval: 50
  epoch_log_interval: 1
  
  # 메트릭 저장
  save_metrics_json: true
  save_confusion_matrix: true
  save_roc_curves: true

  # 하드 케이스 로깅 (Native Linux SSD 최적화)
  hard_cases:
    enabled: true
    dir: "/home/max16/pillsnap_data/exp/exp01/hard_cases"  # Native Linux SSD 경로 사용
    max_per_epoch: 200
  
  # Windows 관련 경로는 스크립트에서 동적 관리 (경로 혼용 방지)
  windows_integration:
    cloudflared_config: ""  # Windows 스크립트에서 관리
    cloudflared_logs: ""    # Windows 스크립트에서 관리  
    powershell_scripts: ""  # Windows 스크립트에서 관리
    note: "WSL 내 설정은 100% Linux 경로만 사용. Windows는 네트워크 연동만."
    save:
      inputs: true     # 원본(또는 썸네일)
      crops: true      # 검출 크롭
      jsonl: true      # 결과/메타 JSONL
  
  # 체크포인트
  save_best: true
  save_last: true 
  save_top_k: 3

# 추론/서빙 설정
inference:
  # 파이프라인 모드별 설정
  single_confidence_threshold: 0.3  # single 모드 최소 신뢰도
  lazy_load_detector: true         # 검출기 지연 로딩 (메모리 절약)
  batch_size: 1                 # 실시간 서빙
  
  # 성능 목표
  target_latency_ms: 100        # RTX 5080 목표 지연시간
  
# ONNX 내보내기
export:
  opset: 17
  dynamic_axes:
    detection: {"images": {0: "batch"}}
    classification: {"input": {0: "batch"}}
  
  # 검증 설정
  compare:
    enabled: true
    mode: coverage              # smoke | coverage | full
    sample_count: 32           # smoke 모드에서만 사용
    per_class_k: 1             # coverage: 클래스당 최소 샘플
    min_classes: 1000          # coverage: 최소 커버 클래스 수
    max_total: 5000            # coverage: 총 샘플 상한
    stratify_by: ["class"]     # 계층적 샘플링
    hardness_bins: []          # 경계 사례 샘플링 (옵션)
    hard_per_bin: 0
    tolerance:
      detection_map: 0.01        # mAP 차이 허용값
      classification_acc: 0.005  # 정확도 차이 허용값

# API 서빙
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1            # GPU 모델 로딩으로 단일 워커
  timeout: 60
  cors_allow_origins: 
    - "http://localhost:3000"
    - "https://pillsnap.co.kr" 
    - "https://api.pillsnap.co.kr"
  require_api_key: true
  max_request_size: 20971520  # 20MB

B-5. bootstrap_venv.sh (가상환경 생성·설치·GPU 감지·폴백)

[scripts/bootstrap_venv.sh — 정확히 이 로직으로 구현]

- set -euo pipefail
- VENV="$HOME/pillsnap/.venv"; ROOT="/home/max16/pillsnap"
- 안내 echo: 루트/venv 경로/파이썬 버전 출력
- venv 없으면 python3 -m venv "$VENV"
- source "$VENV/bin/activate"
- pip 최신화: pip install -U pip wheel setuptools
- CUDA 감지:
  - if command -v nvidia-smi: GPU 존재
  - torch 미설치 상태면 먼저 CPU 휠로 시도하지 말고 **CUDA 휠 경로**를 선호:
    - 기본 인덱스로 pip install torch torchvision torchaudio ||
    - 실패 시 “CUDA 인덱스” 재시도(주석로 안내). 재실패하면 CPU 폴백(경고).
- requirements 설치: pip install -r requirements.txt (이미 torch가 설치됐다면 충돌 없게 유지)
- 환경 점검 스크립트 임베드:
  python - <<'PY'
  import torch, platform, os
  print("Python:", platform.python_version())
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
  print("GPU name:", torch.cuda.get_device_name(0))
  print("Capability:", torch.cuda.get_device_capability(0))
  print("Project ROOT:", os.getcwd())
  PY
- exp 디렉토리 보장: mkdir -p /home/max16/pillsnap_data/exp/exp01/{logs,tb,reports,checkpoints,export}
- 마지막에 “OK: venv ready” 출력

비고: torch 설치 경로는 환경마다 달라 충돌 가능성이 있으므로, 스크립트에 명확한 로그와 실패 시 폴백 메시지를 꼭 남겨.

B-6. 학습/익스포트/API 스크립트 골격 (train.sh / export_onnx.sh / run_api.sh)

[scripts/train.sh — 요구사항]

- set -euo pipefail
- VENV="$HOME/pillsnap/.venv"; ROOT="/home/max16/pillsnap"
- source "$VENV/bin/activate" && cd "$ROOT"
- DIRS 생성: $(yq '.paths.exp_dir' config.yaml)/{logs,tb,reports,checkpoints,export} 보장
- 구성 요약 echo: amp/amp_dtype/tf32/channels_last/batch/num_workers/profile_interval
-

```
python -m src.train --cfg config.yaml \
  >> "$(yq '.paths.exp_dir' config.yaml)/logs/train.out" \
  2>> "$(yq '.paths.exp_dir' config.yaml)/logs/train.err"
```

- 종료코드 분기: 0이면 SUCCESS, 아니면 ERROR 메시지

[scripts/export_onnx.sh — 요구사항]

- set -euo pipefail
- VENV/ROOT 활성화
- python - <<'PY'

# Part E의 export 파이프라인을 그대로 호출:

# 1) config 로드 2) best.pt 로드 3) ONNX export 4) Torch vs ONNX 비교 5) export_report.json 기록

# (실제 구현은 Part E에서 작성)

print("Export stub: implemented in Part E.")
PY

[scripts/run_api.sh — 요구사항]

- set -euo pipefail
- 옵션: --no-tmux (기본) / --tmux (개발용)
- VENV/ROOT 활성화
- ENV 로드(.env 있으면)
- uvicorn 실행: host=0.0.0.0, port=8000, workers=1, timeout/keepalive 보수적
- --tmux면 세션명 pillsnap_api로 띄우고 attach 안내, --no-tmux면 포그라운드

B-7. 운영 보조 스크립트(로그/백업/롤백/벤치/클라우드플레어)

[scripts/maintenance.sh]

- 7일↑ 로그 gzip 아카이브 → logs/archive/
- 14일↑ last\_\* 체크포인트 정리(옵션, best.pt는 제외)
- df -h / /mnt/data 출력 저장

[scripts/backup_release.sh]

- release-<UTC>-<sha|nogit>.tar.gz 생성
- 포함: config.yaml, .env.example, requirements.txt, export/\*.onnx, export_report.json, reports/metrics.json
- sha256sum 파일 생성

[scripts/reload_model.sh]

- 인자: --path /mnt/data/exp/exp01/export/model-....onnx
- curl -X POST http://localhost:8000/reload -H "X-API-Key: $API_KEY" -d '{"model_path":"..."}'
- /version 확인

[scripts/ort_optimize.py, quantize_dynamic.py, perf_bench_infer.py]

- Part H에서 상세 구현. 지금은 파일만 생성하고 TODO 주석으로 표시.

[Cloudflare PowerShell — cf_start.ps1 / cf_stop.ps1 / cf_status.ps1]

- net start/stop cloudflared
- sc query cloudflared
- cloudflared.log tail(상태 스크립트)

B-8. src 모듈 파일(빈 골격 + TODO)

[src/__init__.py] # 비워둠

[src/utils.py]

- TODO: Part D에서 seed/logger/TB/timer/autotune/ckpt/json 유틸 전부 구현(스펙은 Part D에 명시)

[src/data.py]

- TODO: Part C에서 조건부 Two-Stage 데이터로더 구현
  - 단일/조합 약품 구분 로딩
  - COCO → YOLO 포맷 변환  
  - edi_code 매핑 및 클래스 가중치
  - 128GB RAM 활용 캐싱 및 LMDB 변환

[src/models/detector.py]  

- TODO: Part C에서 YOLOv11m 검출 모델 래퍼 구현
  - ultralytics YOLO 통합
  - 조합 약품 전용 학습/추론
  - ONNX export 지원

[src/models/classifier.py]

- TODO: Part C에서 EfficientNetV2-S 분류 모델 구현  
  - timm 백본 + 5000 클래스 헤드
  - 단일 약품 + 조합 크롭 분류
  - ONNX export 지원

[src/models/pipeline.py]

- TODO: Part C에서 조건부 Two-Stage 파이프라인 구현
  - 단일/조합 자동 판단
  - 검출→크롭→분류 체인  
  - 성능 최적화 및 배치 처리

[src/train.py]

- TODO: Part D에서 조건부 Two-Stage 학습 루프 구현
  - 검출/분류 모델 분리 학습
  - RTX 5080 16GB 최적화 (AMP/TF32/compile)
  - AutoBatch/Worker Autotune
  - 128GB RAM 활용 최적화
  - 체크포인트/로깅/TensorBoard

[src/evaluate.py]

- TODO: Part D에서 Two-Stage 성능 평가 구현
  - 검출: mAP@0.5, mAP@0.5:0.95
  - 분류: accuracy, precision, recall, F1
  - 전체 파이프라인 end-to-end 평가
  - 혼동행렬, ROC, PR curve 저장

[src/infer.py]

- TODO: Part E에서 조건부 Two-Stage 추론 구현
  - Torch/ONNX 공용 인터페이스
  - 단일/조합 자동 판단 및 라우팅
  - 배치 추론 최적화
  - CLI 및 Python API 제공

[src/api/main.py, schemas.py, service.py, security.py]

- TODO: Part F에서 Two-Stage API 서빙 구현
  - FastAPI 기반 RESTful API
  - edi_code 추출 스키마
  - 조건부 파이프라인 서비스
  - API 키 인증 및 보안
  - 모델 hot-reload 지원

B-9. tests 골격

[tests/test_smoke_detection.py]

- TODO: Part D 완료 후 YOLOv11m 검출 모델 스모크 테스트
  - 1 epoch 학습, mAP 계산, 체크포인트 저장 확인
  - RTX 5080 메모리 사용량 모니터링

[tests/test_smoke_classification.py]  

- TODO: Part D 완료 후 EfficientNetV2-S 분류 모델 스모크 테스트
  - 1 epoch 학습, accuracy 계산, 체크포인트 저장 확인
  - 5000 클래스 대응 확인

[tests/test_pipeline.py]

- TODO: Part C 완료 후 조건부 Two-Stage 파이프라인 테스트
  - 단일/조합 자동 판단 로직 검증
  - end-to-end 추론 정확성 확인
  - 성능 벤치마크 (target: <100ms/image)

[tests/test_export_compare.py]

- TODO: Part E 완료 후 ONNX export 및 비교 테스트
  - PyTorch vs ONNX 출력 일치성 검증
  - 검출: mAP 차이 <0.01, bbox IoU >0.95
  - 분류: accuracy 차이 <0.005, top-1 일치

[tests/test_api_min.py]

- TODO: Part F 완료 후 API 기본 기능 테스트
  - /health 200 응답 확인
  - /predict edi_code 스키마 검증  
  - X-API-Key 인증 로직 (누락시 401/403)
  - 조건부 파이프라인 API 동작 확인

B-10. 부트스트랩 & 점검(실행 순서)

1. 폴더 생성/파일 작성(현재 파트 B 복붙 적용)
2. venv 부팅
   $ bash scripts/core/setup_venv.sh
   - 출력에 CUDA available: True/False, GPU name/capability 확인
3. 최소 디렉토리 확인
   $ tree -L 2 /home/max16/pillsnap # (없으면 'sudo apt install tree')
   $ ls -al /mnt/data/exp/exp01
4. (선택) VS Code/Cursor에서 "Remote - WSL"로 열기
   - 권장: Python Interpreter로 $HOME/pillsnap/.venv 선택
5. 다음 파트(C)로 진행하여 data.py 구현

## ⚠️ 주의사항 & 최적화 팁

### 🔧 하드웨어 최적화 
- **128GB RAM 활용**: 모든 라벨 메모리 캐시, LMDB 변환, 대용량 배치 프리페치
- **RTX 5080 16GB**: 배치 크기 조정 (검출 8, 분류 32), AMP/TF32/channels_last 필수
- **16 스레드 CPU**: num_workers=16, 데이터 로딩 병렬화 극대화
- **8TB SSD**: 순차 압축 해제로 디스크 공간 효율 관리

### 📁 파일 시스템
- **개행(EOL)**: WSL 스크립트 LF, PowerShell CRLF (.gitattributes 강제)
- **권한**: `chmod +x scripts/*.sh` 실행 전 필요
- **영문 경로**: /mnt/data/pillsnap_dataset (한글 인코딩 이슈 방지)
- **하드코딩 금지**: 모든 경로는 config.yaml/CLI/ENV 제어

### 🚀 성능 모니터링
- **디스크 사용률**: 85% 이상 시 자동 중단, 공간 확보 후 재개
- **VRAM 모니터링**: OOM 발생 시 자동 배치 크기 감소
- **WSL 리소스**: ~/.wslconfig 통해 메모리/CPU 할당 최적화

### 🎯 단계별 게이트 조건 (현실적 목표)
**Phase 1 (1k 클래스, 224px)**: 
- Classification Acc ≥ 0.78, Latency ≤ 220ms → 통과

**Phase 2 (2k 클래스, 224px→288 FT)**:
- Classification Acc ≥ 0.82, Detection mAP ≥ 0.70 → 통과

**Phase 3 (5k 클래스, 224px→288 FT)**:
- Classification Acc ≥ 0.85, Detection mAP ≥ 0.75, Latency ≤ 200ms → 배포  
- **사용자 제어**: mode 파라미터 기반 파이프라인 선택 (single 우선, combo 명시적 선택)
- **성능 목표**: RTX 5080에서 <100ms/image 처리

B-11. Stage 대시보드 및 OptimizationAdvisor 통합

### **터미널 대시보드 출력 설계**
- Stage 완료 시 즉시 표시되는 박스 형태 결과 화면
- 필수 체크, 성능 지표, 시스템 성능을 구분된 섹션으로 표시
- 🟢 PROCEED, 🟡 OPTIMIZE, 🔴 STOP 색상 코딩으로 직관적 판단
- 다음 단계 진행을 위한 구체적 명령어 제공

### **OptimizationAdvisor 스크립트 (scripts/evaluate_stage.sh)**
```bash
#!/bin/bash
# Stage별 권장사항 평가 및 결과 출력
# 사용법: bash scripts/evaluate_stage.sh [1|2|3|4|auto|summary|force-next]

set -euo pipefail

VENV="$HOME/pillsnap/.venv"
ROOT="/home/max16/pillsnap"
EXP_DIR="/home/max16/pillsnap_data/exp/exp01"

source "$VENV/bin/activate" && cd "$ROOT"

case "${1:-auto}" in
  "1"|"2"|"3"|"4")
    python -m tests.stage_${1}_evaluator
    ;;
  "auto")
    # config.yaml에서 current_stage 읽어서 권장사항 실행
    STAGE=$(yq '.data.progressive_validation.current_stage' config.yaml)
    python -m tests.stage_${STAGE}_evaluator
    ;;
  "summary")
    python -m tests.stage_progress_tracker
    ;;
  "force-next")
    # 경고 후 강제 다음 단계 진행
    echo "⚠️ WARNING: Forcing next stage without evaluation!"
    python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); cfg['data']['progressive_validation']['current_stage']+=1; yaml.dump(cfg,open('config.yaml','w'))"
    echo "✅ Stage advanced. Review config.yaml"
    ;;
  *)
    echo "Usage: $0 [1|2|3|4|auto|summary|force-next]"
    exit 1
    ;;
esac
```

### **JSON 리포트 스키마**

OptimizationAdvisor 평가 시 생성되는 `/exp_dir/reports/stage_N_evaluation.json`:
- 단계 정보: stage, timestamp, dataset_info
- 필수 체크: mandatory_checks (각 항목별 status/details)
- 성능 지표: performance_metrics (detection/classification/system)
- 권장사항 제공: recommendations, confidence, reasons
- 사용자 선택지: options, next_stage_config

### **TensorBoard 통합**
- **주요 태그**: stage_evaluation/* 네임스페이스로 구분
- **실시간 모니터링**: detection_mAP, classification_acc, memory_usage, inference_time
- **히스토리**: go_no_go_history로 단계별 판정 이력 추적
- **URL**: http://localhost:6006 (scripts/train.sh에서 자동 시작)

### **OptimizationAdvisor 평가 흐름**
1. 학습 완료 → evaluate_stage_metrics() 권장사항 평가 호출
2. 모든 기준 체크 및 권장사항 생성 (사용자 선택 필요)
3. RECOMMEND_PROCEED 시 → 사용자가 config.yaml의 current_stage 업데이트 선택
4. 터미널 박스 출력으로 결과 즉시 확인
5. 권장 명령어로 바로 다음 Stage 시작 가능

### **Stage별 차별화된 터미널 출력**
- **Stage 1**: 파이프라인 검증 (1,000장) - 기본 동작 확인 중심
- **Stage 2**: 성능 기준선 (10,000장) - Auto Batch/TensorBoard 로깅 확인
- **Stage 3**: 프로덕션 준비 (100,000장) - 확장성/안정성 테스트
- **Stage 4**: 최종 프로덕션 (500,000장) - API 배포/모니터링 완료

각 Stage마다 해당 수준에 맞는 필수 체크와 성능 목표를 다르게 설정하여 단계적 성장을 보장합니다.

**✅ PART_B 완료: 하드웨어 최적화된 조건부 Two-Stage 프로젝트 구조 + OptimizationAdvisor 권장 시스템 완성**
