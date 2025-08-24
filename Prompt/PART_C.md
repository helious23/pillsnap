# Part C — 조건부 Two-Stage 데이터 파이프라인 + 128GB RAM 최적화

[절대 경로/전제 + 디스크 I/O 병목 해결 상황]

- **원본 데이터**: /mnt/data/pillsnap_dataset (원본 보관)
- **데이터 구조**: 
  - **Native Linux**: /home/max16/pillsnap_data (Linux SSD, 주요 데이터)
  - **Windows SSD**: 심볼릭 링크로 연결 (하이브리드 스토리지)
  - **백업**: /mnt/data/pillsnap_dataset (원본 보관용)
- **Native Linux 이전 완료**: WSL 제약 해결, CPU 멀티프로세싱 활성화 (num_workers=8)
  - **✅ Stage 1**: 74.9% 정확도 달성 (Native Linux, 1분 완료)
  - **✅ Stage 2**: 83.1% 정확도 달성 (25K 샘플, 250클래스)
  - **✅ Stage 3**: 85.01% 정확도 달성 (100K 샘플, 1,000클래스)
  - **🎯 Stage 4**: 준비 완료 (500K 샘플, 4,523클래스)
- 기본 작업: 약품 검출+분류(Detection → Classification, Two-Stage). 순수 분류 모드도 지원.
- 모든 데이터 경로는 **/home/max16/pillsnap_data** 사용 (프로젝트와 분리).
- 코드는 /home/max16/pillsnap, **학습 산출물/체크포인트**는 /home/max16/pillsnap_data/exp/로 고정 (데이터 분리).

C-0) 목표 & 산출물

- 목표:
  1. COCO 포맷 약품 검출 데이터를 YOLO 포맷으로 변환하고 안전하게 로딩한다.
  2. 사용자 선택 기반 Two-Mode: 단일 분류용 + 조합 검출용 데이터로더를 구성한다.
  3. DataLoader 병목을 줄여 GPU로의 H2D 전송을 최대한 겹친다(non_blocking).
  4. 깨진 샘플/예외는 "스킵 + 로그"로 처리(학습 중단 금지).
  5. 약품 메타데이터 매핑 테이블을 구축하고 약품 ID와 연결한다.
  6. 자동 판단 로직 완전 제거 - 사용자 선택만 지원
- 이 파트에서 구현(파일: src/data.py):
  • PillsnapDetDataset(torch.utils.data.Dataset) # YOLO 검출용
  • PillsnapClsDataset(torch.utils.data.Dataset) # crop된 약품 분류용  
  • convert_coco_to_yolo(coco_json_path, output_dir) -> yolo_format_files
  • build_drug_metadata_mapping(root_path) -> dict[drug_id, complete_metadata]
  • build_transforms_detection(train: bool, img_size: int, augment_cfg: dict) -> callable
  • build_transforms_classification(train: bool, img_size: int, augment_cfg: dict) -> callable
  • discover_detection_data(root, coco_annotations) -> (images, annotations, classes)
  • make_splits(records, annotations, split_ratio, seed, stratified: bool, persist_path: str, max_samples: int|None) -> dict{train,val,test}
  • build_dataloaders_by_mode(cfg, mode="single"|"combo", logger, seed) -> (train_loader, val_loader, test_loader|None, meta)
  • safe_collate_detection(batch) # bbox 검증 포함
  • safe_collate_classification(batch) # SkipSample-safe
  • validate_bbox(bbox, img_w, img_h) -> bool
  • worker_init_fn(worker_id) # 시드 고정
  • open_image_safely(path) -> PIL.Image | dict(\_skip=True, error=...)
  • summarize_dataset(...) -> dict # 약품 분포/바운딩박스 통계/메타데이터 매핑 현황
  • build_drug_metadata_mapping(coco_annotations_dir: str, output_path: str) -> dict # edi_code → drug_metadata 매핑
  • get_class_id_from_edi_code(edi_code: str, mapping_data: dict) -> int # edi_code → class_id 변환
  • get_drug_metadata_from_class_id(class_id: int, mapping_data: dict) -> dict # class_id → drug_metadata 변환

C-1) config.yaml 확장(키 추가/설명)

```
data:
  task: "two_stage"              # "two_stage" (detection + classification)
  pipeline_strategy: "user_controlled"  # single 우선, combo 명시적 선택
  default_mode: "single"         # 90% 케이스 기본값
  auto_fallback: false           # 자동 판단 완전 제거
  root: "/home/max16/pillsnap_data"  # Native Linux SSD 데이터 경로 (Stage 1-2 완료)
  detection:
    img_size: 640
    coco_json_path: "data/train/labels"  # COCO annotation 경로
    yolo_output_dir: "/home/max16/pillsnap_data/exp/exp01/yolo_data"      # Native Linux SSD에 변환된 YOLO 포맷 저장
    conf_threshold: 0.3
    iou_threshold: 0.5
    max_detections: 100
  classification:
    img_size: 384                # EfficientNetV2-S 고해상도 크기
    crop_padding: 0.1            # 검출된 bbox 확장 비율
  # 데이터 분할 전략: AI Hub Training(159개)를 train/val로, AI Hub Validation(22개)를 test로
  split_ratio: [0.85, 0.15]      # AI Hub Training 데이터만 train:val로 분할
  test_data_source: "aihub_validation"  # AI Hub Validation 전체를 test로 사용 (Stage 학습 중 절대 사용 금지)
  test_usage_policy: "final_evaluation_only"  # test는 모든 Stage 완료 후 최종 평가시 1회만 사용
  max_samples: null              # 디버그 시 1000 같은 제한 지원(null이면 전체)
  drug_metadata_file: "/home/max16/pillsnap_data/exp/exp01/drug_metadata.json"  # Native Linux SSD에 drug_id → complete_metadata 매핑
  # 단순화된 Stage 평가 시스템 (Native Linux 업데이트)
  progressive_validation:
    current_stage: 2  # Stage 2 준비 완료
    stage_1: {max_samples: 5000, max_classes: 50, target_accuracy: 0.78, achieved_accuracy: 0.749, status: "completed"}
    stage_2: {max_samples: 25000, max_classes: 250, target_accuracy: 0.82, status: "ready"}
    stage_3: {max_samples: 100000, max_classes: 1000, target_accuracy: 0.85, max_latency_ms: 200}
    stage_4: {max_samples: 500000, max_classes: 4523, target_accuracy: 0.85, max_latency_ms: 200}
  extensions: [".jpg",".jpeg",".png",".bmp",".webp"]
  ignore_hidden: true            # ._* 숨김 파일 무시
  verify_on_build: true          # 스캔 시 이미지 오픈 검증(권장)
  cache_meta_path: "/home/max16/pillsnap_data/exp/exp01/splits.json"  # Native Linux SSD에 분할/메타 캐시
  broken_policy: "skip"          # "skip"|"fail" — 반드시 "skip"
  grayscale_policy: "rgb"        # "rgb"(3채널 변환) | "skip"
  rgba_policy: "drop_alpha"      # "drop_alpha"(RGB 변환) | "skip"
  augment:
    detection:
      randaugment: false         # YOLO 자체 증강 사용
      mosaic: 0.5
      mixup: 0.1
      hflip: true
      scale: [0.5, 1.5]          # 스케일 증강 범위
    classification:
      randaugment: true
      color_jitter: [0.2,0.2,0.2,0.05]   # brightness, contrast, saturation, hue
      hflip: true
      auto_augment: null         # "imagenet" | null
  imbalance:
    use_class_weights: true      # 분류 단계에서만 적용
    method: "inv_freq"           # "inv_freq" | "effective_num"
    beta: 0.9999                 # effective_num일 때만
    use_weighted_sampler: false  # true면 DataLoader sampler로 대체(충돌 주의)

dataloader:
  num_workers: 8  # Native Linux 최적화 값
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
  
  # 단계적 메모리 최적화 (기본 off, 단계적 On)
  ram_optimization:
    # Phase 1: 기본 캐시만 (보수적 시작점)
    cache_policy: "labels_only"    # 초기에는 레이블만 캐시
    cache_labels: true
    preload_samples: 0             # 프리로드 비활성화
    hotset_size_images: 0          # 핫셋 비활성화
    use_lmdb: false                # LMDB 비활성화
    
    # Phase 2: 병목 확인 후 가벼운 확장
    # cache_policy: "hotset"
    # hotset_size_images: 40000    # 40K 이미지 캐시 (~16GB)
    
    # Phase 3: 진짜 I/O 병목일 때만
    # use_lmdb: true 또는 WebDataset 중 하나만 선택
```

> 참고: Native Linux에서 num_workers=8로 최적화되었습니다.
> autotune_workers가 true이면 Part D의 오토튜너가 [4,8,12,16] 후보로 벤치 후 최적을 반영합니다.

```

C-2) 디렉토리/파일 구조(COCO → YOLO 변환)

- 통일된 입력 구조 (Native Linux SSD 이전 완료, 실제 ZIP 추출 구조):
```
/home/max16/pillsnap_data/  # Stage 1-2 완료, Stage 3-4 준비
├─ data/train/
│  ├─ labels/
│  │  ├─ combination/
│  │  │  ├─ TL_1_combo/
│  │  │  │  ├─ K-000250-000573-002483-006192_json/   # K-코드_json 폴더
│  │  │  │  │  └─ K-000250-000573-002483-006192_*.json  # JSON 라벨 파일들
│  │  │  │  ├─ K-000250-000573-002483-012778_json/
│  │  │  │  └─ ... (547개 조합 K-코드_json 폴더)
│  │  │  ├─ TL_2_combo/
│  │  │  └─ ... (TL_1_combo~TL_8_combo)
│  │  └─ single/
│  │     ├─ TL_1_single/
│  │     │  ├─ K-000001_json/                    # K-코드_json 폴더
│  │     │  │  └─ K-000001_*.json                 # JSON 라벨 파일들
│  │     │  ├─ K-000002_json/
│  │     │  └─ ... (50개 단일 K-코드_json 폴더)
│  │     ├─ TL_2_single/
│  │     └─ ... (TL_1_single~TL_81_single)
│  └─ images/
│     ├─ combination/
│     │  ├─ TS_1_combo/
│     │  │  ├─ K-000250-000573-002483-006192/   # K-코드 폴더
│     │  │  │  ├─ K-000250-000573-002483-006192_0_0_0_0_60_000_200.png
│     │  │  │  └─ ...
│     │  │  ├─ K-000250-000573-002483-012778/
│     │  │  └─ ... (547개 조합 K-코드 폴더)
│     │  ├─ TS_2_combo/
│     │  └─ ... (TS_1_combo~TS_8_combo)
│     └─ single/
│        ├─ TS_1_single/
│        │  ├─ K-000001/                         # K-코드 폴더
│        │  │  ├─ K-000001_0_0_0_0_60_000_200.png
│        │  │  └─ ...
│        │  ├─ K-000002/
│        │  └─ ... (50개 단일 K-코드 폴더)
│        ├─ TS_2_single/
│        └─ ... (TS_1_single~TS_81_single)
├─ data/val/
│  ├─ labels/
│  │  ├─ combination/
│  │  │  └─ VL_1_combo/
│  │  │     ├─ K-016235-027733-029667-031885_json/   # K-코드_json 폴더
│  │  │     │  └─ K-016235-027733-029667-031885_*.json  # JSON 라벨 파일들
│  │  │     └─ ... (500개 조합 K-코드_json 폴더)
│  │  └─ single/
│  │     ├─ VL_1_single/
│  │     │  ├─ K-039148_json/                    # K-코드_json 폴더
│  │     │  │  └─ K-039148_*.json                 # JSON 라벨 파일들
│  │     │  └─ ... (50개 단일 K-코드_json 폴더)
│  │     └─ ... (VL_1_single~VL_10_single)
│  └─ images/
│     ├─ combination/
│     │  └─ VS_1_combo/
│     │     ├─ K-016235-027733-029667-031885/   # K-코드 폴더
│     │     │  ├─ K-016235-027733-029667-031885_0_0_0_0_60_000_200.png
│     │     │  └─ ...
│     │     └─ ... (500개 조합 K-코드 폴더)
│     └─ single/
│        ├─ VS_1_single/
│        │  ├─ K-039148/                         # K-코드 폴더
│        │  │  ├─ K-039148_0_0_0_0_60_000_200.png
│        │  │  └─ ...
│        │  └─ ... (50개 단일 K-코드 폴더)
│        └─ ... (VS_1_single~VS_10_single)
└─ data/test/ (Stage 4 완료 후만 사용, 동일한 K-코드 폴더 구조)
```

- 출력 구조 (Native Linux SSD 최적화, YOLO 포맷):
```
/home/max16/pillsnap_data/exp/exp01/yolo_data/  # Native Linux SSD에 YOLO 포맷 저장
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/  
│  ├─ train/
│  └─ val/
├─ data.yaml        # YOLO 설정 파일
└─ drug_metadata.json # drug_id → complete_metadata 매핑
```

- 메타데이터 추출: JSON annotations에서 전체 47개 필드를 활용하여 완전한 약품 정보 매핑 테이블 생성.

C-3) COCO → YOLO 변환 및 검증 규칙(강제)

- convert_coco_to_yolo():
1. COCO JSON 파일들을 순회하여 이미지 경로와 annotations 추출.
2. bbox 좌표를 YOLO 포맷으로 변환: [x_center, y_center, width, height] (0~1 정규화).
3. 각 이미지에 대응하는 .txt 파일 생성 (YOLO 라벨 포맷).
4. data.yaml 파일 생성: train/val 경로, 클래스 수, 클래스 이름 정의.
5. drug_metadata.json 생성: drug_id → complete_metadata 매핑 테이블.
- validate_bbox(): 바운딩 박스 좌표 검증 (0≤x,y,w,h≤1, w>0, h>0).
- summarize_dataset():
• 샘플(최대 2~3천장)로 해상도/종횡비 히스토, 클래스 분포 상위 10, 손상/스킵 비율 로그.
- make_splits(records, labels, split_ratio, seed, stratified=True, persist_path, max_samples):
• AI Hub Training 데이터(159개)만 train/val로 분할 (85:15 비율)
• AI Hub Validation 데이터(22개)는 test로 완전 분리, 학습 과정에서 절대 사용 금지
• Stratified 분할(비율 보존). 클래스 수가 1개면 랜덤 분할로 폴백(경고).
• 각 클래스 최소 1개 val 보장(부족 시 경고).
• persist_path가 존재하면 캐시 재사용(무결성: 파일 존재/개수/라벨 합치). 실제 구현에서는 persist_path=cfg.data.cache_meta_path 를 사용한다.
• max_samples 설정 시 균일 서브샘플(전체에서 무작위 균일; 클래스별 다운샘플은 TODO 주석).
• 반환: {'train': train_records, 'val': val_records, 'test': test_records} # test는 최종 평가용
- 에지 케이스:
• 라벨 누락/구조 오류 파일: 즉시 경고 후 skip(broken_policy 고정).
• 중복 경로/깨진 심볼릭링크도 skip.

C-4) 전처리/증강(vision v2, 텐서 경로)

- build_transforms(train, img_size, augment_cfg):
공통: 입력을 RGB 3채널로 통일(그레이스케일/알파 정책 적용).
train:
• RandomResizedCrop(img_size, antialias=True)
• RandomHorizontalFlip(p=0.5) (cfg.augment.hflip)
• (옵션) RandAugment(num_ops=2~3, magnitude=7~9) (cfg.augment.randaugment)
• (옵션) ColorJitter(\*cfg.augment.color_jitter)
• ToTensor() → Normalize(IMAGENET mean/std)
val/test:
• Resize(shorter=img_size) + CenterCrop(img_size) 또는 Resize(img_size)
• ToTensor() → Normalize(...)
- 텐서는 float32, AMP는 모델측에서 처리. 모델은 channels_last 메모리 포맷(Part D에서 전송 시 설정).

C-5) Dataset/Collate/DataLoader 설계(성능 핵심)

- PillsnapClsDataset:
**getitem**(i):
1. img = open_image_safely(path) # 실패 시 dict(\_skip=True, error=...)
2. transforms 적용 → Tensor
3. 성공: {"image": Tensor, "label": int, "path": str} 반환
   **len**(): 샘플 수
- safe_collate(batch):
• \_skip=True 항목 필터. 남은 샘플 < 1이면 경고 후 빈 배치 예외 회피(상위 루프가 다음 스텝 진행).
• (images[N,C,H,W], labels[N], meta{paths}) 반환.
- DataLoader:
train_loader = DataLoader(
dataset=train_ds,
batch_size=cfg.train.batch_size,
shuffle=not cfg.data.imbalance.use_weighted_sampler,
sampler=WeightedRandomSampler(...) if cfg.data.imbalance.use_weighted_sampler else None,
num_workers=cfg.dataloader.num_workers,
pin_memory=cfg.dataloader.pin_memory,
prefetch_factor=cfg.dataloader.prefetch_factor,
persistent_workers=cfg.dataloader.persistent_workers,
drop_last=cfg.dataloader.drop_last,
collate_fn=safe_collate,
worker_init_fn=worker_init_fn
)
val_loader: shuffle=False, sampler=None, 나머지 동일.
- 워커 오토튜닝: Part D의 autotune_num_workers()로 후보 [4,8,12,16]에 대해 50~100 step 미니벤치 → data_time 최소값 선택 → 최종 로더 재생성.

C-6) 클래스 불균형 처리(가중치 vs 샘플러)

- compute_class_weights(labels, num_classes, method, smooth, beta):
• inv_freq: w_c = total/(count_c + smooth); 평균 1.0로 스케일.
• effective_num: w_c = (1 - β)/(1 - β^{count_c}); 평균 1.0로 스케일.
→ torch.Tensor[num_classes]을 반환하여 CrossEntropy(weight=...)에 사용.
- WeightedRandomSampler:
• 극단적 불균형에서만 사용(과보정 주의). 사용 시 train_loader.shuffle=False로 전환.
• loss 가중치와 동시 사용 비권장(둘 중 하나 우선).

C-7) 안전 로딩(open_image_safely) 정책

- EXIF Orientation 교정(transpose) 시도.
- 컬러 정책:
• grayscale_policy="rgb": convert("RGB")로 3채널화. "skip"이면 스킵.
• rgba_policy="drop_alpha": convert("RGB")로 알파 드롭. "skip"이면 스킵.
- 포맷: WebP/PNG/BMP/JPEG 지원. Pillow 오류 시 1회 OpenCV(cv2.imdecode) 폴백.
- 최종 실패는 예외를 던지지 말고 {"\_skip": True, "error": "..."} 반환 → collate에서 필터.
- 로깅 강화: 각 실패 단계별 구체적 에러 메시지 기록 (PIL실패/OpenCV폴백/최종실패)

C-8) 메타/로그 가시성

- summarize_dataset() 로그 항목:
• 총 파일/유효 파일 수, 클래스 수/상위 10 클래스 분포
• 해상도 통계(평균/중앙/분산, 샘플링), 종횡비 히스토 범례
• broken/skip 카운트(비율)
- build_dataloaders() 로그:
• 분할 비율/샘플 수(train/val/test), num_workers/pin_memory/prefetch/persistent/drop_last
• test 데이터는 로더 생성하되 Stage 학습 중 사용 금지 명시
• sampler/가중치 사용 여부, max_samples 적용 여부
• 오토튜닝을 돌렸다면 후보별 data_time 및 최종 선택값

C-9) 반환 메타(meta 딕셔너리) 스펙
meta = {
"class_names": list[str],
"class_to_idx": dict[str,int],
"train_count": int, "val_count": int, "test_count": int,
"img_size": int,
"splits_json": "/home/max16/pillsnap_data/exp/exp01/splits.json",
"test_usage_policy": "final_evaluation_only",
"weights_used": "inv_freq|effective_num|none",
"sampler_used": "weighted|none",
"skipped_files": int
}

C-10) 테스트(권장)

- tests/test_data_min.py 작성:
• discover_dataset(): 파일/클래스 수 일치, 숨김/확장자 필터 동작
• make_splits(): 캐시 저장/재사용/무결성 체크
• open_image_safely(): grayscale/RGBA/EXIF 처리 검증
• build_dataloaders(): safe_collate로 빈 배치 예외 없이 배치 생성
• compute_class_weights(): 값 범위/평균=1 스케일 검증
- 성능 스모크: num_workers 후보 [4,8]로 200 step 미니벤치 → data_time 개선 로그 확인.

C-11) 실행·검증 절차

1. config.yaml의 data.root/img_size/augment 확인.
2. (선택) classes.json 작성해 data.class_names에 지정, 없으면 폴더명 자동.
3. src/data.py를 본 명세대로 구현(주석에 설계 이유·예외 처리 근거 포함).
4. $ bash scripts/core/setup_venv.sh
5. (간이 점검) Python REPL:
 > > > from src import data as D, utils as U
 > > >
 > > > # cfg 로드 후:
 > > >
 > > > # train_loader, val_loader, meta = D.build_dataloaders(cfg, logger, seed=42)
6. Part D로 넘어가 $ bash scripts/train.sh 실행. 로그에 데이터 요약/분할/오토튜닝 결과가 찍히는지 확인.

C-12) 성능 튜닝 팁

- Pillow-SIMD: 설치 시 디코딩↑ (환경 충돌 시 보수적으로 패스, README에 주석 링크).
- ulimit -n 상향: 워커 파일핸들 부족 오류 예방.
- prefetch_factor: 2→4→8로 늘리며 data_time 추적(메모리 여유 필요).
- persistent_workers=True: 에폭 사이 재포크 비용↓.
- 입력 크기 다양 시 RandomResizedCrop이 resize+augmentation을 통합해 캐시 효율↑.

C-13) Drug Metadata 매핑 로직 구현

**build_drug_metadata_mapping 함수**:
```python
def build_drug_metadata_mapping(coco_annotations_dir: str, output_path: str) -> dict:
    """
    COCO annotations에서 edi_code → drug_metadata 매핑 테이블 구축
    
    Args:
        coco_annotations_dir: COCO JSON 파일들이 있는 디렉토리
        output_path: 매핑 테이블 저장 경로
    
    Returns:
        매핑 딕셔너리 {edi_code: {metadata}}
    """
    drug_metadata = {}
    edi_to_class_id = {}
    class_id_counter = 0
    
    # 모든 COCO JSON 파일 순회
    for json_file in glob.glob(f"{coco_annotations_dir}/**/*.json", recursive=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # images 섹션에서 약품 메타데이터 추출
        for image_info in coco_data.get('images', []):
            edi_code = image_info.get('di_edi_code')
            if not edi_code:
                continue
                
            # edi_code → class_id 매핑 (첫 등장시 새 ID 할당)
            if edi_code not in edi_to_class_id:
                edi_to_class_id[edi_code] = class_id_counter
                class_id_counter += 1
            
            # 완전한 약품 메타데이터 수집
            metadata = {
                'edi_code': edi_code,
                'class_id': edi_to_class_id[edi_code],
                'dl_name': image_info.get('dl_name', ''),
                'drug_shape': image_info.get('drug_shape', ''),
                'color_class1': image_info.get('color_class1', ''),
                'color_class2': image_info.get('color_class2', ''),
                'print_front': image_info.get('print_front', ''),
                'print_back': image_info.get('print_back', ''),
                'drug_type': image_info.get('drug_type', ''),
                'mark_code_front': image_info.get('mark_code_front', ''),
                'mark_code_back': image_info.get('mark_code_back', ''),
                # 추가 필드들...
            }
            
            drug_metadata[edi_code] = metadata
    
    # 매핑 테이블 저장
    mapping_data = {
        'edi_to_class_id': edi_to_class_id,
        'drug_metadata': drug_metadata,
        'num_classes': len(edi_to_class_id),
        'created_at': datetime.utcnow().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Drug metadata mapping saved: {len(drug_metadata)} drugs, {len(edi_to_class_id)} classes")
    return mapping_data

def get_class_id_from_edi_code(edi_code: str, mapping_data: dict) -> int:
    """edi_code → class_id 변환"""
    return mapping_data['edi_to_class_id'].get(edi_code, -1)

def get_drug_metadata_from_class_id(class_id: int, mapping_data: dict) -> dict:
    """class_id → drug_metadata 변환"""
    for edi_code, metadata in mapping_data['drug_metadata'].items():
        if metadata['class_id'] == class_id:
            return metadata
    return {'error': 'metadata_not_found', 'class_id': class_id}
```

C-14) 구체적 메모리 관리 구현

**LMDB 캐시 시스템**:
```python
# src/data/lmdb_cache.py
import lmdb
import pickle
from cachetools import LRUCache

class LMDBImageCache:
    def __init__(self, lmdb_path: str, readonly: bool = True, map_size: str = "500GB"):
        self.env = lmdb.open(lmdb_path, readonly=readonly, 
                            lock=False, readahead=True, 
                            max_readers=512, map_size=self._parse_size(map_size))
        
    def get(self, key: str) -> Optional[np.ndarray]:
        with self.env.begin() as txn:
            data = txn.get(key.encode())
            if data:
                return pickle.loads(data)
        return None
```

**LRU 캐시 래퍼**:
```python
# 32GB LRU 캐시 with size tracking
class SizedLRUCache:
    def __init__(self, max_bytes: int = 34359738368):  # 32GB
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
```

**프로세스간 공유 메모리**:
```python
# Dataset.__getitem__ 최적화
def __getitem__(self, idx: int):
    # 1. LMDB 조회
    if self.use_lmdb:
        image = self.lmdb_cache.get(f"image_{idx}")
        if image is not None:
            return self._process_cached(image, idx)
    
    # 2. LRU 캐시 조회  
    cache_key = f"{self.split}_{idx}"
    if cache_key in self.lru_cache:
        image = self.lru_cache[cache_key]
        return self._process_cached(image, idx)
    
    # 3. 디스크 로드
    image = self._load_from_disk(idx)
    
    # 4. 캐시 저장 (크기 제한)
    if self._should_cache(image):
        self.lru_cache[cache_key] = image
    
    return self._process_image(image, idx)

def _process_cached(self, image: np.ndarray, idx: int):
    # torch.from_numpy with share_memory_() for multiprocessing
    tensor = torch.from_numpy(image).share_memory_()
    if self.channels_last:
        tensor = tensor.to(memory_format=torch.channels_last)
    return {"image": tensor, "label": self.labels[idx]}
```

**Stage별 프리로드 전략**:
```python
# Stage별 데이터 사전 로드
def preload_stage_data(cfg, stage: int):
    preload_count = cfg.dataloader.ram_optimization.preload_samples.get(f"stage_{stage}", 0)
    
    if preload_count > 0:
        logger.info(f"Stage {stage}: Preloading {preload_count} samples to memory")
        # hotset에 자주 사용되는 샘플들 우선 로드
        for idx in range(min(preload_count, len(dataset))):
            _ = dataset[idx]  # 캐시에 저장됨
```

C-Seg) 세그멘테이션 파이프라인(옵션)

- data.task="segmentation"일 때만.
- 구조: images/_.jpg|png, masks/_.png(정수 클래스 인덱스, 파일명 매칭)
- 변환: albumentations로 이미지/마스크 동기(rotate/flip/scale/crop/colorjitter)
- 보간: 이미지=bilinear, 마스크=nearest
- Dataset.**getitem** → {"image": Tensor, "mask": LongTensor, "path": str}
- 손실/지표: CE+Dice, mIoU/Dice/PixelAcc
- DataLoader/오토튜닝/로깅 규칙은 분류와 동일

C-Det) YOLOv11 데이터(옵션, Ultralytics)

- 포맷:
root/images/train|val|test/_.jpg|png
root/labels/train|val|test/_.txt # <cls> <cx> <cy> <w> <h> (0~1)
- data.yaml 예:
path: /mnt/data/AIHub_576
train: images/train
val: images/val
test: images/test
nc: 3
names: ["pill","capsule","tablet"]
- 유효성 유틸(src/det/utils_det.py): 라벨 범위/클래스 인덱스/고아 파일 검사.
- 변환 보조 스크립트(scripts/prepare_yolo.sh): 분류→검출 변환(선택).

Annex) 체크리스트 요약
A1. discover_dataset(): verify_on_build=True면 PIL.verify()+EXIF transpose, 실패=skip
A2. open_image_safely(): grayscale/rgba 정책, OpenCV 폴백 1회, 실패는 {"\_skip":True}
A3. make_splits(): stratified+persist JSON, 무결성=경로/개수/라벨
A4. build_transforms(): vision v2 기반, Normalize(IMAGENET), train/val 구분
A5. Dataset: **getitem** 예외를 던지지 않고 "\_skip" 플래그로 상위에 통지
A6. safe_collate(): \_skip 필터, 빈 배치 방지, meta(paths) 반환
A7. build_dataloaders(): sampler/가중치/오토튜닝/로깅 반영
A8. compute_class_weights(): inv_freq/effective_num, 평균=1 스케일
A9. worker_init_fn(): numpy/torch/random 시드 고정
A10. summarize_dataset(): 분포/해상도/종횡비/손상 로그

## 🎯 **PART_C 핵심 업데이트 완료**

### ✅ **조건부 Two-Stage 데이터 파이프라인 설계**
- **단일 약품**: 직접 분류 (384px, EfficientNetV2-S 최적화)
- **조합 약품**: YOLO 검출 → 분류 (640px, YOLOv11m)  
- **사용자 제어**: mode="single"|"combo" 파라미터 기반 파이프라인 선택

### ✅ **128GB RAM + RTX 5080 16GB 최적화 (보수적 기본값)**
- **메모리 캐시**: 핫셋 6만장만 캐시 (≈25GB, 기본 off)
- **LMDB 변환**: 기본 비활성, data_time 병목 시에만 활성화 (Opt-in)
- **데이터로더**: num_workers=8, prefetch_factor=4, pin_memory_device="cuda"

### ✅ **영문 데이터 경로 구조**
- **입력**: /mnt/data/pillsnap_dataset (한글→영문 변환)
- **처리**: single/combination 분리 구조
- **출력**: LMDB + YOLO 형식 최적화

### 📋 **다음 파트에서 구현할 핵심 클래스**
- `UserControlledTwoStageDataset`: 사용자 제어 파이프라인 선택
- `SinglePillDataset`: 직접 분류용 (384px)  
- `CombinationPillDataset`: YOLO 검출용 (640px)
- `memory_efficient_loader`: 128GB RAM 최적화
- `lmdb_converter`: 대용량 데이터 I/O 최적화

**✅ PART_C 완료: 하드웨어 최적화된 조건부 Two-Stage 데이터 파이프라인 설계**

---

## 🎯 **Stage 3-4 Manifest 기반 접근법 (2025-08-22 업데이트)**

### **⭐ 중요한 정책 변경**
**Stage 3-4는 반드시 manifest 기반으로만 진행합니다.**

- **물리적 데이터 복사**: ❌ 금지 (SSD 용량 부족)
- **Manifest CSV 파일**: ✅ 권장 (용량 절약)
- **원본 직접 로딩**: ✅ 하이브리드 스토리지 활용

### **용량 절약 효과**
```
Stage 3 (100K 샘플): 14.6GB → 50MB (99.7% 절약)
Stage 4 (500K 샘플): 73.0GB → 200MB (99.7% 절약)
총 절약량: 87.6GB → 250MB (99.7% 절약)
```

### **기술적 근거**
1. **Native Linux + 128GB RAM**: 실시간 고속 로딩 가능
2. **하이브리드 스토리지**: Linux SSD (3.5GB/s) + Windows SSD (1GB/s)
3. **기존 코드 호환성**: `src/data.py` 데이터로더 그대로 사용
4. **성능 손실 없음**: 메모리 캐시 + 빠른 SSD I/O

### **구현 방향**
- **Stage 1-2**: 기존 config 기반 방식 유지
- **Stage 3-4**: manifest 생성 스크립트 + 기존 `src/data.py` Dataset 활용
- **코드 변경 최소화**: 새로운 데이터로더 구현 불필요
- **기존 컨벤션 준수**: `src/training/train_classification_stage.py` 활용

### **Stage 3-4 학습 명령어 (기존 컨벤션)**
```bash
# Stage 3 manifest 기반 학습 (기존 trainer 활용)
python -m src.training.train_classification_stage \
    --manifest artifacts/stage3/manifest_train.csv \
    --num-classes 1000 \
    --target-accuracy 0.85 \
    --epochs 50 \
    --batch-size 16

# Stage 4 manifest 기반 학습 (기존 trainer 활용)  
python -m src.training.train_classification_stage \
    --manifest artifacts/stage4/manifest_train.csv \
    --num-classes 4523 \
    --target-accuracy 0.92 \
    --epochs 100 \
    --batch-size 8
```
