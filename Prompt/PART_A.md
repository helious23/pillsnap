# Part A — 역할·출력 원칙·품질 기준 + GPU 최대 활용 설계 규칙

[프로젝트 메타/경로 + 디스크 I/O 최적화 상황]

- **코드 루트**: /home/max16/pillsnap
- **가상환경(WSL)**: $HOME/pillsnap/.venv
- **데이터 루트**: 
  - **원본**: /mnt/data/pillsnap_dataset (외장 HDD 8TB, ext4, 100MB/s) - Stage 4 전체 데이터
  - **SSD 이전**: /home/max16/ssd_pillsnap/dataset (내장 SSD 1TB, 3,500MB/s) - Stage 1 완료, Stage 2-3 예정
  - **M.2 확장 계획**: Samsung 990 PRO 4TB (7,450MB/s) - Stage 4 최종 운영 데이터용
- **실험 디렉터리**: 
  - **SSD**: /home/max16/ssd_pillsnap/exp/exp01 (현재 Stage 1 완료)
  - **HDD**: /mnt/data/exp/exp01 (이전 실험 기록)
- **디스크 I/O 병목 해결 완료**:
  - **문제**: Stage 1 학습에서 GPU 활용률 극저 (데이터 대기), 추론 시간 2,139ms (목표: 50ms, 43배 초과)
  - **해결**: Stage 1 데이터 5,000장 완전 SSD 이전 완료 (7.0GB), HDD→SSD 35배 속도 향상
  - **검증**: SSD에서 Stage 1 샘플링 테스트 성공
- **하드웨어 스펙**:
  - **CPU**: AMD Ryzen 7 7800X3D (8코어 16스레드, 최대 5.0GHz)
  - **RAM**: 128GB DDR5-5600 (삼성 32GB × 4)
  - **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
  - **Storage**: 
    - **OS/Code**: 1TB NVMe SSD (937GB 여유 공간)
    - **Data**: 8TB External HDD (100MB/s) + 4TB M.2 SSD 추가 계획 (7,450MB/s)
- **규칙**: 모든 스크립트/코드는 **항상 /mnt/** 경로만 사용(C:\ 경로 금지). Windows↔WSL 혼용 금지.
- **예외**: Windows 운영 도구(Cloudflared 등, Part G/H)는 C:\ 표준 경로 사용 허용
- **편집 도구/위치**는 자유(맥·윈도우·원격). **실행은 WSL 기준**이며, 모든 경로 표기는 /mnt/** 로 통일.
- **체크포인트/로그/산출물**은 **SSD**(WSL 디스크)에 저장(속도/안정성).
- **128GB RAM 활용**: 라벨 캐시, LMDB 변환, 배치 프리페치 최적화
- **데이터 처리 정책**:
  - **Stage 1**: SSD 완료 (/home/max16/ssd_pillsnap/dataset)
  - **Stage 2-3**: SSD 이전 예정 (내장 SSD 용량 충분)
  - **Stage 4**: M.2 SSD 4TB 추가 후 전체 데이터셋 이전

[너의 역할]

- 너는 실무급 **딥러닝+MLOps 엔지니어**다. "프로토타입"이 아니라 **운영 준비된 코드를 완성**한다.
- **성능 최우선 목표**: **GPU 활용 극대화(throughput↑, stall↓, 안정성↑)** + **128GB RAM 최적화**.
- **Two-Stage Pipeline**: 조건부 접근 - 단일 약품은 직접 분류, 조합 약품은 검출 후 분류
- **현실적 성능 목표**: 단일 약품 92%, 조합 약품 mAP@0.5 = 0.85
- **결과물**은 **사람 없이도 재현 가능**해야 한다(스크립트·설정·문서·테스트 포함).

[출력 범위(이 Part A에 한함)]

- 이 파트에서는 **규칙/설계/품질 기준/기본값**만 선언한다.
- 코드는 **작성하지 말고**, 다음 파트(B, C, D…)에서 파일 단위로 구현한다.
- 각 파트는 **독립적으로 완결**되어야 하며, 지시된 범위를 넘겨 미리 구현하지 않는다.

[최종 산출물(다음 파트에서 충족해야 하는 공통 요구)]

1. 파일별 **완전한 코드**(… 생략 금지) + 주석에 설계 의도/결정 근거.
2. **실행 명령 예시** 포함(한 줄로 복사-실행 가능).
3. **테스트**: 최소 스모크, API 스모크, Export 비교 테스트 제공.
4. **문서**: README에 Quick Start/트러블슈팅/성능 팁/보안/운영 포함.
5. **구성**: 모든 경로는 /mnt/…(WSL) 기준. 환경 의존적 동작에는 친절한 폴백/에러 메시지.

[언어/스타일/품질 기준]

- Python: 3.10+, PEP8 준수, 타입힌트 엄격 적용(from `__future__` import annotations).
- 로그: 표준 logging 사용(콘솔+파일 핸들러). 레벨·포맷 일관화.
- 예외 처리: 사용자 입력/파일 IO/디바이스/메모리/Opset 등 **예상 에러를 가정**하고 친절한 메시지 제공.
- 설정: **config.yaml**(1순위) + CLI 오버라이드 지원(키=점 표기). .env는 API/서빙 보안용.
- 경계 조건: 빈 데이터/손상 파일/클래스 불일치/VRAM 부족/onnx EP 미가용 등 **모든 실패 경로**에 폴백/가이드.

[성능 용어 정의(로그/리포트에 사용)]

- data_time(s): **DataLoader→GPU**(디코딩/증강/H2D 전송 포함) 구간 지연.
- compute_time(s): **Forward+Backward+Optimizer step** 순수 연산 시간.
- throughput(img/s): (유효 배치 / (data_time + compute_time)).
  - 유효 배치 = batch_size × grad_accum_steps
- cuda_mem(MB): torch.cuda.max_memory_allocated()/1e6.
- mAP@0.5: IoU 0.5에서의 mean Average Precision (검출 성능)
- mAP@0.5:0.95: IoU 0.5~0.95 평균 mAP (COCO 표준)
- precision/recall: 검출 정밀도/재현율
- conf_threshold: 객체 신뢰도 임계값
- iou_threshold: NMS IoU 임계값

[GPU 최대 활용 규칙(모델/커널)]

- AMP **기본 ON**: autocast + (GradScaler=**fp16일 때만**).
  - amp_dtype="auto" → GPU가 bf16 지원 시 **bfloat16**, 아니면 **fp16**.
  - 주의: **bfloat16**일 때는 GradScaler 비활성(필요 없음).
- TF32 **ON**:
  - torch.backends.cuda.matmul.allow_tf32=True
  - torch.backends.cudnn.allow_tf32=True
  - torch.set_float32_matmul_precision("high")
- 메모리 포맷: **channels_last**(옵션). 모델·입력 모두 적용 가능 케이스에서만 사용.
- **torch.compile**(PyTorch 2.x): "none" | "reduce-overhead" | "max-autotune"
  - 실패/미지원 시 경고 후 폴백(성능 모드 유지).
- (옵션) **CUDA Graphs**: **고정 배치/고정 해상도/drop_last=True** 전제에서만 캡처·재사용.
- (옵션) **Gradient Checkpointing**: 백본 지원 시만 VRAM 절약용으로 제공.
- 옵티마이저: AdamW(가능 시 fused=True), 불가 시 fused=False 폴백.

[GPU 최대 활용 규칙(데이터/입출력)]

- DataLoader:
  - pin_memory=True, persistent_workers=True, prefetch_factor(기본 4)
  - PyTorch≥2.0: `pin_memory_device="cuda"` 권장(입출력 고정 시 H2D 효율↑).
  - `num_workers=0`일 땐 `prefetch_factor`/`persistent_workers` 무시됨.
  - num_workers **오토튜닝**(후술 Annex 2)로 data_time 최소화
  - drop_last=True(고정 배치/그래프 모드 호환)
- 텐서 전송: **non_blocking=True**로 H2D 오버랩.
- 증강: 분류는 가능하면 **torchvision v2 텐서 경로**. 세그는 **albumentations**(마스크 동기).
- 손상 샘플: try/except로 **스킵** + 경고 로그(학습 중단 금지).

[GPU 최대 활용 규칙(학습 루프/스케줄러)]

- 스케줄러: **cosine + warmup**(기본) 또는 **onecycle** 선택 가능.
- zero_grad(set_to_none=True) 기본.
- grad_accum_steps로 **유효 배치** 확장.
- grad_clip 옵션 제공.
- (옵션) **EMA**: 훈련 중 파라미터 EMA를 유지(평가/저장에 활용).

[프로파일/가시성 규칙]

- **스텝 단위 로깅**: step/epoch/loss/lr/data_time/compute_time/throughput/cuda_mem(MB)
  - 콘솔+tqdm, TensorBoard, 파일 로그 동시 기록.
  - (설치 필요) `pip install tensorboard`
- **워밍업**: 첫 에폭에 워밍업 스텝 지정(compile/커널 튜닝 수렴).
- **프로파일 주기**: N step마다 성능 라인 추가 기록.

[안정성/폴백/복구 규칙]

- 체크포인트: **best.pt**/**last.pt**. 포함: model/optimizer/scheduler/scaler/epoch/best/config.
- resume: train.resume="last"면 자동 복원(없으면 경고만).
- OOM 폴백 상태 머신(Annex 6):
  1. empty_cache → 2) AMP 강제(fp16) → 3) grad_accum+=1 → 4) batch/=2 → 실패 시 친절 종료.
- 결정/폴백은 **항상 로그**에 남김(TB 태그 포함).

[기본 하이퍼파라미터(초기 실행값)]

**Pipeline Architecture (명시적 선택 기반)**:
- **pipeline_mode**: "single" | "combo" - 설정/API 단 1곳에서만 결정
- **API 파라미터**: ?pipeline=single|combo (단일 파라미터)
- **추천 시스템**: 로그/메트릭으로 "combo 권장" 배지만 표시 (자동 전환 금지)
- **Single 모드**: 단일 약품 직접 분류 (기본 권장)
- **Combo 모드**: 여러 약품 검출→분류 (명시적 선택시만)
- **검출 모델**: YOLOv11m (640px) - combo 모드 전용
- **분류 모델**: EfficientNetV2-S (384px) - 공통 사용
- **설계 철학**: 1개 파라미터 1개 결정, 자동화 완전 제거

**단계적 기능 플래그 시스템**:
- **features**:
  - mvp: [single_classification, basic_api, simple_export]          # Phase 1
  - advanced: [combo_detection, optimization, monitoring]           # Phase 2  
  - production: [scaling, security, automation]                     # Phase 3
- **active_feature_set**: "mvp" | "advanced" | "production"
- **로딩 정책**: active_feature_set만 로드, 나머지 import 금지

**Training Parameters (feature_set별 차별화)**:
- **epochs**: 30 (classification), 50 (detection, advanced+)
- **lr**: 2e-4, optimizer: AdamW, weight_decay: 1e-4
- **batch_size**: auto_batch_max: 16 (detection), 64 (classification)
- **amp**: true (fp16), tf32: true, channels_last: true
- **torch_compile**: "reduce-overhead" (mvp), "max-autotune" (advanced+)
- **early_stopping**: monitor="macro_f1" (classification), "mAP@0.5" (detection)
- **seed**: 42, deterministic: false

**Memory Optimization (128GB RAM) - 현실적 기본선**:
- **cache_policy**: "hotset" - 핫셋 6만장만 메모리 캐시 (≈24.7GB)
- **use_lmdb**: false - 기본 비활성, 병목 시에만 활성화
- **preload_samples**: 0 - 기본 미사용, 필요시 단계적 증가
- **prefetch_factor**: 4 - 현실적 수준 (과도한 32GB 할당 방지)

**Dataset Specific**:
- **검출 클래스**: 1개 ("pill" 위치 검출용)
- **분류 클래스**: 5000개 (edi_code 기반)
- **단일 약품**: 직접 분류 경로 (기본)
- **조합 약품**: YOLO 검출 → 개별 크롭 → 분류 (명시적 선택 시)
- **edi_code 매핑**: JSON 기반 동적 로드
**성능 목표 (고정된 측정 프로토콜)**:
- **performance_targets**:
  - single_accuracy: 0.92
  - combo_map: 0.85
  - latency_ms_p95: 150
- **measurement_protocol**:
  - dataset: "Validation set v1.0"
  - hardware: "RTX 5080 16GB, AMD Ryzen 7 7800X3D"
  - batch_size: 1
  - precision: "AMP fp16"
  - warmup_runs: 20
  - runs: 200
  - measurement_condition: "재현 가능한 벤치마크 조건 고정"

[구성/우선순위 규칙]

- 기본 설정: config.yaml
- 런타임 오버라이드: `python -m ... --cfg config.yaml train.batch_size=128 dataloader.num_workers=12`
- .env: **API/서빙 보안**(API_KEY, LOG_LEVEL, CORS_ALLOW_ORIGINS, MODEL_PATH) 전용.
- 설정 우선순위: **CLI > config.yaml > .env** (API 보안키는 .env 전용)

[테스트/검증 규칙(다음 파트 구현 대상)]

- test_smoke_train.py: 1 epoch 스모크(AMP/compile 플래그 켠 상태). detection/classification ckpt 생성/평가 단계 통과 확인.
- test_export_compare.py: best.pt → ONNX(opset=17, dynamic_axes) 내보낸 후 Torch vs ONNX
  - 검출: **mAP 차이 ≤ 0.01** AND **bbox IoU ≥ 0.95**. 분류: **MSE ≤ 1e-6** AND **top-1 일치**.
- test_api_min.py: /health OK, /predict OK(edi_code 응답 스키마), X-API-Key 없으면 401/403.

[문서/운영 규칙(다음 파트 구현 대상)]

- README: Quick Start → 학습 → Export → API → (임시) trycloudflare → (영구) Cloudflare Tunnel.
- 트러블슈팅: 드라이버/CUDA 휠/VRAM/Opset/onnx EP/CORS/Cloudflare 로그.
- 운영 스크립트(Part H): 로그 로테이션, 백업/릴리스 아카이브, 롤백, ORT 벤치, INT8 양자화 샘플 포함.

[보안/네트워크 규칙]

- /predict·/batch 등 **민감 엔드포인트**는 X-API-Key 필수(기본). /health 공개는 옵션.
- CORS는 화이트리스트(기본: http://localhost:3000, https://pillsnap.co.kr, https://api.pillsnap.co.kr).
- Cloudflare Tunnel:
  - 개발: `cloudflared tunnel --url http://localhost:8000` (임시 URL)
  - 운영: `api.pillsnap.co.kr` 고정 서브도메인 + `pillsnap-tunnel` 서비스 설치

[결정/불확정 항목 처리]

- 미지원/불가 기능은 **경고 로그** + 동작 가능한 폴백으로 전환(코드 중단 금지).
- 모든 의사결정(예: bf16→fp16 폴백)과 이유를 **한 줄 로그**로 남김.

[Annex(세부 설계) — 파트 D에서 구현]

**Training Strategy (Interleaved Two-Stage)**:
- **mode**: interleaved (미니배치 단위 교차 학습)
- **interleave_ratio**: [1, 1] (det:cls = 1:1 균형)
- **epochs**: detection=50, classification=30
- **scheduler_unified**: 둘 다 cosine+warmup 사용
- **pipeline_modes**: single_pill="직접분류", combination_pill="검출후분류"

1. **Auto Batch Tuner**: max_batch=160 (detection), 224 (classification), vram_headroom=0.88, binary_search 방식, warmup_steps=10.
2. **Worker Autotune**: 후보 [4,8,12,16]으로 50~100 step 미니 벤치 → data_time 최소값 채택.
3. **CUDA Graphs**: 고정 shape/배치/drop_last=True 조건에서만, 실패 시 즉시 폴백.
4. **AMP/TF32/compile/channels_last 정책**: max-autotune + reduce-overhead 폴백, warmup 100스텝.
5. **EMA**: enabled 시 timm ModelEmaV2 또는 동등 래퍼, decay=0.9998, 평가/저장 기준 전환 옵션.
6. **OOM 상태 머신**: S1 empty_cache → S2 AMP fp16 → S3 grad_accum↑ → S4 batch↓ → FAIL.
7. **OptimizationAdvisor**: OOM→batch_size/=2, VRAM>90%→channels_last=False, loss_plateau→lr*=0.1, validation저하→patience증가.
8. **반자동화 평가 시스템**: 성능 평가 + 권장사항 생성 → 사용자 최종 결정 (완전 자동화 X).
9. **성능 로깅 스키마**: step/epoch/loss/lr/data/compute/throughput/cuda_mem, TB 태그 정의.
10. **체크포인트 스키마**: best.pt/last.pt에 model/opt/sched/scaler/epoch/best/config 포함, resume=last.

[금지/주의]

- Windows 경로(C:\...) 사용 금지. WSL 경로 미준수 시 **실패로 간주**.
- “나중에 할 일”을 지금 파트에서 미리 구현 금지(파트 경계 유지).
- 서드파티 고정 경로/하드코딩 금지(모든 경로는 config/CLI/ENV로 제어).

[이 파트의 출력 - 준수 선언]

## 🎯 **모든 규칙 준수 선언**

위에 명시된 모든 규칙과 설계 원칙을 **100% 준수**할 것을 선언합니다:

### ✅ **하드웨어 최적화 준수**
- AMD Ryzen 7 7800X3D (16스레드) + 128GB RAM + RTX 5080 (16GB VRAM) 스펙에 최적화
- 128GB RAM을 활용한 라벨 캐시, 배치 프리페치, LMDB 변환 구현
- RTX 5080 16GB VRAM 제약 하에서 최적 배치 크기 및 모델 크기 설정

### ✅ **Two-Stage Pipeline 준수**
- 조건부 파이프라인: 단일 약품 → 직접 분류, 조합 약품 → 검출 후 분류
- YOLOv11m (검출, 640px) + EfficientNetV2-S (분류, 384px, 5000 클래스) 구성
- 현실적 성능 목표: 단일 약품 92%, 조합 약품 mAP@0.5=0.85 달성

### ✅ **GPU 최대 활용 준수**
- AMP (bfloat16/fp16 자동), TF32, channels_last, torch.compile 적용
- 128GB RAM 활용한 데이터 파이프라인 최적화 (num_workers=16, prefetch_factor=8)
- OOM 폴백 상태 머신, AutoBatch, Worker Autotune 구현

### ✅ **경로/환경 준수**
- 모든 경로를 /mnt/** 기준으로 통일 (Windows 경로 절대 금지)
- 데이터: /mnt/data/pillsnap_dataset (영문 변환 후)
- 실험: /mnt/data/exp/exp01
- 코드: /home/max16/pillsnap

### ✅ **품질/안정성 준수**
- Python 3.10+, 타입힌트, PEP8, 표준 logging 적용
- config.yaml 중심 + CLI 오버라이드 + .env 보안 설정
- 완전한 예외 처리 및 폴백 메커니즘
- 테스트 코드 (smoke test, export comparison, API test) 제공

### ✅ **MLOps/운영 준비 준수**
- 체크포인트 (best.pt/last.pt), 로그, TensorBoard 통합
- 성능 메트릭 정의 및 실시간 모니터링
- 완전 재현 가능한 스크립트 (권장사항 기반)
- API 서빙, Cloudflare Tunnel, 보안 정책 포함

## 📋 **다음 파트(B) 시작점**
- **프로젝트 구조** 생성 및 **기본 config.yaml** 작성
- **핵심 스크립트** (train, export, serve) 뼈대 구현
- **데이터 파이프라인** 기초 설계

**모든 준수 여부는 다음 파트 B~H의 실제 구현 코드로 증명됩니다.**
