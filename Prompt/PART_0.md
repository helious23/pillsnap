# Part 0 — 프로젝트 개요 및 데이터 준비 전략

## 🎯 프로젝트 개요

### 기본 정보
- **프로젝트명**: pillsnap-ml
- **목적**: 약품 이미지에서 약품 정보를 추출하여 약품을 식별하는 AI 서비스
- **데이터셋**: AIHub 166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터
- **데이터셋 Key**: 576
- **작업 환경**: WSL2 Ubuntu on Windows 11
- **작업자**: max16

### 하드웨어 스펙
```yaml
CPU: AMD Ryzen 7 7800X3D (8코어 16스레드, 최대 5.0GHz)
RAM: 128GB DDR5-5600 (삼성 32GB × 4)
GPU: NVIDIA GeForce RTX 5080 (16GB VRAM)
Storage:
  - OS/Code: 1TB NVMe PCIe 4.0 (키오시아 XG7)
  - Data: 
    - 8TB External HDD (삼성 T5 EVO, /mnt/data) - 원본 데이터셋 보관
    - 1TB Internal SSD (현재 사용, /home/max16/ssd_pillsnap/)
    - 4TB M.2 SSD (계획, Samsung 990 PRO, 7,450MB/s)
Network: Gigabit Ethernet
Cooling: 3RSYS RC1200 ARGB
Power: 850W 80+ Bronze ATX 3.1
```

### 경로 구조 (디스크 I/O 병목 해결 완룼)
```yaml
코드 경로: /home/max16/pillsnap
# 디스크 I/O 병목 해결 과정:
# - 기존 HDD: /mnt/data/pillsnap_dataset (100MB/s, 병목)
# - 현재 SSD: /home/max16/ssd_pillsnap/dataset (3,500MB/s, 35배 향상)
데이터 경로: /home/max16/ssd_pillsnap/dataset  # SSD 이전 완룼
가상환경: $HOME/pillsnap/.venv
실험 경로: /home/max16/ssd_pillsnap/exp/exp01  # SSD 이전 완룼
```

## 🚀 점진적 검증 전략 (Progressive Validation Strategy)

### 단계별 데이터 샘플링 전략 (디스크 I/O 최적화 기반)
**목표**: 디스크 I/O 병목을 해결하며 대용량 데이터셋을 단계적으로 확장

| Stage | 데이터 규모 | 클래스 수 | 목적 | 예상 소요시간 |
|-------|-----------|----------|------|------------|
| **Stage 1** | 5,000장 | 50개 | 파이프라인 검증 | 2시간 | ✅ **SSD 완룼** (7.0GB, 35배 향상) |
| **Stage 2** | 25,000장 | 250개 | 성능 기준선 확립 | 8시간 | 🔄 **SSD 준비완룼** |
| **Stage 3** | 100,000장 | 1,000개 | 확장성 테스트 | 32시간 | 🔄 **SSD 준비완룼** |
| **Stage 4** | 2,000,000장 | 4,523개 | 최종 프로덕션 학습 | 200시간 (8일) | 🔜 **M.2 SSD 4TB 계획** |

#### **계층적 균형 샘플링 (Stratified Balanced Sampling)**

**선정 이유:**
1. **클래스 불균형 완화**: 5000개 클래스의 분포 차이 대응
2. **대표성 보장**: 각 클래스 최소 2장 포함 (train 1장, val 1장)
3. **재현 가능성**: seed=42 기반 고정된 샘플링

```python
# 샘플링 규칙 (실제 데이터 기반 조정)
def progressive_sampling_strategy():
    """
    Stage 1 (5,000장): 상위 50개 클래스 선택 (데이터 많은 순)
    - 단일:조합 = 7:3 비율 유지 (원본 데이터 분포)
    - 클래스당 평균 100장 (train:val = 8:2)
    
    Stage 2 (25,000장): 상위 250개 클래스
    - 클래스당 평균 100장, 기준선 성능 확립
    
    Stage 3 (100,000장): 상위 1,000개 클래스
    - 클래스당 평균 100장, 확장성 테스트
    
    Stage 4 (2,000,000장): 전체 4,523개 클래스  
    - 클래스당 평균 442장, 최종 프로덕션 성능
    """
    pass
```

### 단계별 성능 벤치마크 지표

#### **Stage 1: 파이프라인 검증 (5,000장, 50클래스)**
```yaml
target_metrics:
  detection:
    mAP_0.5: ≥0.30      # 기본 검출 가능성 확인
    inference_time: ≤50ms  # RTX 5080에서 640px 실시간 처리 가능성
  classification: 
    accuracy: ≥0.40     # 50클래스 기준 (무작위 2% × 20배)
    inference_time: ≤10ms  # 384px 분류 실시간 가능
  system:
    memory_usage: ≤14GB  # VRAM 안정성 확인
    data_loading: ≤2s/batch # 128GB RAM 활용도 검증
```

#### **Stage 2: 성능 기준선 (25,000장, 250클래스)**  
```yaml
target_metrics:
  detection:
    mAP_0.5: ≥0.50      # 기본 검출 성능
    mAP_0.5_0.95: ≥0.35 # COCO 표준 지표
  classification:
    accuracy: ≥0.60     # 250클래스 대상 향상된 성능
    macro_f1: ≥0.55     # 클래스 불균형 환경에서의 실제 성능
  optimization:
    batch_size: ≥8      # RTX 5080 메모리 효율성 입증
    throughput: ≥100img/s # API 서빙을 위한 처리량 기준선
```

#### **Stage 3: 확장성 테스트 (100,000장, 1,000클래스)**
```yaml
target_metrics:
  detection:
    mAP_0.5: ≥0.70      # 높은 검출 성능
    mAP_0.5_0.95: ≥0.50 # 고급 검출 성능
  classification:
    accuracy: ≥0.75     # 1,000클래스 대상 높은 성능
    macro_f1: ≥0.70     # 불균형 환경에서 높은 성능
  system:
    memory_stable_hours: ≥12 # 장시간 안정성
```

#### **Stage 4: 최종 목표 (2,000,000장, 4,523클래스)** 
```yaml  
target_metrics:
  detection:
    mAP_0.5: ≥0.85      # PART_A에서 설정한 조합약품 최종 목표
    mAP_0.5_0.95: ≥0.65 # 최고 수준 검출 성능
  classification:
    accuracy: ≥0.92     # PART_A에서 설정한 단일약품 최종 목표
    macro_f1: ≥0.88     # 최고 수준 F1 스코어 (4,523 클래스)
    inference_latency: ≤30ms # 전체 파이프라인 실시간 성능
```

### OptimizationAdvisor 권장 시스템

#### **🎯 반자동화 평가 시스템**
각 Stage 완료 후 **OptimizationAdvisor**가 성능을 평가하고 구체적 권장사항을 제공합니다:

**✨ 핵심 설계 철학**: 완전 자동화 대신 **사용자 선택권과 투명성**을 제공하는 반자동화 접근

```yaml
# Stage 1 → Stage 2 권장 기준
stage1_evaluation:
  classification_accuracy: ≥0.40 (50클래스)
  detection_mAP_0.5: ≥0.30
  pipeline_complete: true
  memory_usage: ≤14GB
  
  권장사항_예시:
    - RECOMMEND_PROCEED: "모든 목표 달성, Stage 2 진행 권장"
    - SUGGEST_OPTIMIZE: "배치 크기 16→32 증가, 학습률 1e-4→1e-5 감소"
    - WARN_STOP: "과적합 징후, 드롭아웃 0.3→0.5 증가 권장"

# Stage 2 → Stage 3 권장 기준  
stage2_evaluation:
  classification_accuracy: ≥0.60 (250클래스)
  detection_mAP_0.5: ≥0.50
  macro_f1: ≥0.55
  memory_stable: true

# Stage 3 → Stage 4 권장 기준
stage3_evaluation:
  classification_accuracy: ≥0.75 (1,000클래스)
  detection_mAP_0.5: ≥0.70
  scalability_confirmed: true

# Stage 4 완료 조건
stage4_complete:
  classification_accuracy: ≥0.92 (4,523클래스)
  detection_mAP_0.5: ≥0.85
  production_ready: true
```

#### **🤖 사용자 선택권 제공**
시스템이 자동으로 결정하지 않고, 사용자에게 선택권을 제공:
- **[1] 권장사항 적용 후 재시도**
- **[2] 현재 성능으로 다음 Stage 진행** 
- **[3] 수동 디버깅 모드**

**반자동화 설계 이유**:
- 🔍 **투명성**: 모든 결정 과정을 사용자에게 공개
- 🎯 **제어권**: 최종 결정은 항상 사용자가 선택
- 📊 **학습**: 권장사항을 통해 최적화 지식 축적
- ⚡ **효율성**: 전문가 지식을 자동 제안하되 강제하지 않음
```


## 📊 데이터셋 분석

### 🚨 중요: 데이터 사용 정책
**Train/Val 데이터 분리는 최종 test 목적을 위함입니다:**
- **Train 데이터 (178개 ZIP)**: 학습 및 검증 분할용 (train:val = 85:15)
- **Val 데이터 (22개 ZIP)**: 최종 test 전용 데이터 (학습에 절대 사용 금지)
- **Progressive Validation Stage 1-4**: Train 데이터만 사용
- **최종 평가**: 모든 Stage 완료 후 Val 데이터로 1회 test만 수행

### 실제 데이터 현황 (2025-08-19 기준)
- **총 263만개 이미지** 
  - **Train**: 247만개 (학습/검증 분할용)
  - **Val**: 16만개 (최종 test 전용, 학습 금지)
- **실제 클래스 수**: 4,523개 (기존 목표 5,000개보다 적음)
- **Single:Combo 비율**: 99.3% : 0.7% (매우 불균형)
- **이미지 해상도**: 100% 동일 (976x1280)

### COCO 포맷 JSON 구조
```json
{
  "images": [{
    "file_name": "K-000059_0_0_0_0_60_000_200.png",
    "width": 976,
    "height": 1280,
    "di_edi_code": "653700060",  // 핵심: 약품 식별 코드
    "dl_name": "게루삼정 200mg/PTP",
    "drug_shape": "원형",
    "color_class1": "하양",
    "print_front": "G",
    "print_back": "G"
  }],
  "annotations": [{
    "bbox": [553, 184, 272, 279],  // [x, y, width, height]
    "category_id": 1,
    "image_id": 1
  }]
}
```

## 🚀 핵심 전략

### 1. Frontend 사용자 선택권 기반 설계

**핵심 철학**: 복잡한 자동 판단 대신 **사용자가 직접 모드를 선택**하는 단순한 구조

#### **🎯 모드 선택 방식**
- **Single 모드** (기본, 90% 케이스): 약품 하나만 촬영 → 직접 분류 (92% 정확도)
- **Combo 모드** (명시적 선택): 여러 약품 → 검출 후 분류 (85% 정확도)

#### **🔧 설계 이점**
- **모델 복잡도 최소화**: 자동 판단 로직 완전 제거
- **사용자 컨트롤**: 명확한 선택권으로 혼란 방지
- **높은 정확도**: Single 모드 권장으로 최적 성능
- **개발 단순성**: 디버깅과 최적화 용이

### 2. 단순화된 Two-Stage 파이프라인
```python
class PillSnapPipeline:
    def __init__(self):
        self.classifier = timm.create_model('efficientnetv2_l', num_classes=5000)
        self.detector = YOLO('yolov11x.pt')  # 조합 약품용, 지연 로딩
        
    def predict(self, image, mode="single"):
        # Frontend에서 사용자가 선택한 모드로 처리
        if mode == "single":
            # 단일 약품: 직접 분류 (90% 케이스, 권장)
            edi_code = self.classifier(image)
            return {"edi_code": edi_code, "mode": "single"}
        elif mode == "combo":
            # 조합 약품: 검출 → 개별 분류 (명시적 선택시)
            detections = self.detector(image)
            results = []
            for det in detections:
                cropped = crop(image, det.bbox)
                edi_code = self.classifier(cropped)
                results.append({"bbox": det.bbox, "edi_code": edi_code})
            return {"detections": results, "mode": "combo"}
        # 복잡한 자동 판단 로직 제거됨
```

### 3. 순차 압축 해제 전략
```bash
# 디스크 공간 효율적 관리
Available: 8TB (Samsung T5 EVO)
Required: ~5.1TB (해제 후)
Solution: 1개씩 처리 → 즉시 원본 삭제

# 처리 순서
1. 라벨링 데이터 전체 (3GB) → 메모리 캐시
2. Training 원천 1개씩 → 처리 → 삭제
3. Validation 마지막 일괄 처리
```

### 4. 단계별 학습 접근
| 단계 | 데이터 규모 | 목적 | RTX 5080 예상 시간 |
|------|------------|------|-------------------|
| **Stage 1** | 1,000개 | 파이프라인 검증 | 30분 |
| **Stage 2** | 50,000개 (10%) | 하이퍼파라미터 최적화 | 4시간 |
| **Stage 3** | 500,000개 (100%) | 최종 학습 | 40-50시간 |

## 📈 성능 목표

### 현실적 목표 (달성 가능)
- **단일 약품 분류**: 92% accuracy
- **조합 약품 검출**: mAP@0.5 = 0.85
- **전체 파이프라인**: 88% accuracy
- **추론 속도**: 100ms/image (RTX 5080)

### 도전 목표 (최적화 후)
- **단일 약품 분류**: 95% accuracy
- **조합 약품 검출**: mAP@0.5 = 0.90
- **전체 파이프라인**: 92% accuracy
- **추론 속도**: 50ms/image

[원본 데이터셋 목록 - AIHub 166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터]

**데이터셋 Key**: 576

## Training 데이터 (178개 파일)

### 라벨링데이터 (89개 ZIP)
**경구약제조합 5000종 (8개)**:
- TL_1_조합.zip (7.84MB, key: 66065)
- TL_2_조합.zip (7.31MB, key: 66066) 
- TL_3_조합.zip (7.38MB, key: 66067)
- TL_4_조합.zip (6.95MB, key: 66068)
- TL_5_조합.zip (7.76MB, key: 66069)
- TL_6_조합.zip (7.72MB, key: 66070)
- TL_7_조합.zip (7.26MB, key: 66071)
- TL_8_조합.zip (7.35MB, key: 66072)

**단일경구약제 5000종 (81개)**:
- TL_1_단일.zip (70.37MB, key: 66083)
- TL_2_단일.zip (70.36MB, key: 66094)
- TL_3_단일.zip (71.14MB, key: 66105)
- TL_4_단일.zip (70.35MB, key: 66116)
- TL_5_단일.zip (71.00MB, key: 66127)
- TL_6_단일.zip (71.99MB, key: 66138)
- TL_7_단일.zip (71.70MB, key: 66149)
- TL_8_단일.zip (71.70MB, key: 66152)
- TL_9_단일.zip (71.07MB, key: 66153)
- TL_10_단일.zip (71.91MB, key: 66073)
- TL_11_단일.zip (71.80MB, key: 66074)
- TL_12_단일.zip (71.61MB, key: 66075)
- TL_13_단일.zip (71.69MB, key: 66076)
- TL_14_단일.zip (71.65MB, key: 66077)
- TL_15_단일.zip (71.46MB, key: 66078)
- TL_16_단일.zip (72.84MB, key: 66079)
- TL_17_단일.zip (73.08MB, key: 66080)
- TL_18_단일.zip (71.29MB, key: 66081)
- TL_19_단일.zip (70.05MB, key: 66082)
- TL_20_단일.zip (68.73MB, key: 66084)
- TL_21_단일.zip (31.92MB, key: 66085)
- TL_22_단일.zip (27.52MB, key: 66086)
- TL_23_단일.zip (33.47MB, key: 66087)
- TL_24_단일.zip (22.22MB, key: 66088)
- TL_25_단일.zip (26.73MB, key: 66089)
- TL_26_단일.zip (23.90MB, key: 66090)
- TL_27_단일.zip (20.23MB, key: 66091)
- TL_28_단일.zip (19.13MB, key: 66092)
- TL_29_단일.zip (21.20MB, key: 66093)
- TL_30_단일.zip (20.30MB, key: 66095)
- TL_31_단일.zip (22.56MB, key: 66096)
- TL_32_단일.zip (24.93MB, key: 66097)
- TL_33_단일.zip (22.28MB, key: 66098)
- TL_34_단일.zip (30.72MB, key: 66099)
- TL_35_단일.zip (18.05MB, key: 66100)
- TL_36_단일.zip (24.20MB, key: 66101)
- TL_37_단일.zip (24.88MB, key: 66102)
- TL_38_단일.zip (14.25MB, key: 66103)
- TL_39_단일.zip (16.86MB, key: 66104)
- TL_40_단일.zip (23.85MB, key: 66106)
- TL_41_단일.zip (16.48MB, key: 66107)
- TL_42_단일.zip (19.39MB, key: 66108)
- TL_43_단일.zip (15.92MB, key: 66109)
- TL_44_단일.zip (28.33MB, key: 66110)
- TL_45_단일.zip (19.62MB, key: 66111)
- TL_46_단일.zip (14.39MB, key: 66112)
- TL_47_단일.zip (20.13MB, key: 66113)
- TL_48_단일.zip (17.63MB, key: 66114)
- TL_49_단일.zip (20.67MB, key: 66115)
- TL_50_단일.zip (29.49MB, key: 66117)
- TL_51_단일.zip (18.71MB, key: 66118)
- TL_52_단일.zip (21.05MB, key: 66119)
- TL_53_단일.zip (23.62MB, key: 66120)
- TL_54_단일.zip (18.33MB, key: 66121)
- TL_55_단일.zip (23.57MB, key: 66122)
- TL_56_단일.zip (21.34MB, key: 66123)
- TL_57_단일.zip (17.14MB, key: 66124)
- TL_58_단일.zip (17.64MB, key: 66125)
- TL_59_단일.zip (22.38MB, key: 66126)
- TL_60_단일.zip (26.92MB, key: 66128)
- TL_61_단일.zip (20.24MB, key: 66129)
- TL_62_단일.zip (25.85MB, key: 66130)
- TL_63_단일.zip (18.52MB, key: 66131)
- TL_64_단일.zip (19.61MB, key: 66132)
- TL_65_단일.zip (19.02MB, key: 66133)
- TL_66_단일.zip (17.80MB, key: 66134)
- TL_67_단일.zip (17.58MB, key: 66135)
- TL_68_단일.zip (22.31MB, key: 66136)
- TL_69_단일.zip (20.44MB, key: 66137)
- TL_70_단일.zip (16.14MB, key: 66139)
- TL_71_단일.zip (19.72MB, key: 66140)
- TL_72_단일.zip (19.11MB, key: 66141)
- TL_73_단일.zip (17.39MB, key: 66142)
- TL_74_단일.zip (18.35MB, key: 66143)
- TL_75_단일.zip (17.83MB, key: 66144)
- TL_76_단일.zip (13.73MB, key: 66145)
- TL_77_단일.zip (19.97MB, key: 66146)
- TL_78_단일.zip (21.02MB, key: 66147)
- TL_79_단일.zip (19.32MB, key: 66148)
- TL_80_단일.zip (18.65MB, key: 66150)
- TL_81_단일.zip (10.74MB, key: 66151)

### 원천데이터 (89개 ZIP)
**경구약제조합 5000종 (8개)**:
- TS_1_조합.zip (3.46GB, key: 66154)
- TS_2_조합.zip (3.38GB, key: 66155)
- TS_3_조합.zip (3.40GB, key: 66156)
- TS_4_조합.zip (3.33GB, key: 66157)
- TS_5_조합.zip (3.37GB, key: 66158)
- TS_6_조합.zip (3.42GB, key: 66159)
- TS_7_조합.zip (3.39GB, key: 66160)
- TS_8_조합.zip (3.44GB, key: 66161)

**단일경구약제 5000종 (81개)**:
- TS_1_단일.zip (92.22GB, key: 66172)
- TS_2_단일.zip (91.73GB, key: 66183)
- TS_3_단일.zip (93.49GB, key: 66194)
- TS_4_단일.zip (90.87GB, key: 66205)
- TS_5_단일.zip (92.61GB, key: 66216)
- TS_6_단일.zip (94.47GB, key: 66227)
- TS_7_단일.zip (91.13GB, key: 66238)
- TS_8_단일.zip (94.50GB, key: 66241)
- TS_9_단일.zip (90.68GB, key: 66242)
- TS_10_단일.zip (93.05GB, key: 66162)
- TS_11_단일.zip (94.48GB, key: 66163)
- TS_12_단일.zip (92.64GB, key: 66164)
- TS_13_단일.zip (95.04GB, key: 66165)
- TS_14_단일.zip (95.13GB, key: 66166)
- TS_15_단일.zip (94.71GB, key: 66167)
- TS_16_단일.zip (95.00GB, key: 66168)
- TS_17_단일.zip (94.36GB, key: 66169)
- TS_18_단일.zip (92.72GB, key: 66170)
- TS_19_단일.zip (93.06GB, key: 66171)
- TS_20_단일.zip (91.41GB, key: 66173)
- TS_21_단일.zip (42.07GB, key: 66174)
- TS_22_단일.zip (36.90GB, key: 66175)
- TS_23_단일.zip (44.28GB, key: 66176)
- TS_24_단일.zip (28.49GB, key: 66177)
- TS_25_단일.zip (33.93GB, key: 66178)
- TS_26_단일.zip (31.39GB, key: 66179)
- TS_27_단일.zip (26.02GB, key: 66180)
- TS_28_단일.zip (24.45GB, key: 66181)
- TS_29_단일.zip (28.90GB, key: 66182)
- TS_30_단일.zip (26.06GB, key: 66184)
- TS_31_단일.zip (27.50GB, key: 66185)
- TS_32_단일.zip (32.42GB, key: 66186)
- TS_33_단일.zip (27.70GB, key: 66187)
- TS_34_단일.zip (40.03GB, key: 66188)
- TS_35_단일.zip (23.03GB, key: 66189)
- TS_36_단일.zip (31.10GB, key: 66190)
- TS_37_단일.zip (32.25GB, key: 66191)
- TS_38_단일.zip (18.15GB, key: 66192)
- TS_39_단일.zip (22.01GB, key: 66193)
- TS_40_단일.zip (29.87GB, key: 66195)
- TS_41_단일.zip (21.38GB, key: 66196)
- TS_42_단일.zip (23.37GB, key: 66197)
- TS_43_단일.zip (20.03GB, key: 66198)
- TS_44_단일.zip (35.45GB, key: 66199)
- TS_45_단일.zip (25.11GB, key: 66200)
- TS_46_단일.zip (17.31GB, key: 66201)
- TS_47_단일.zip (24.78GB, key: 66202)
- TS_48_단일.zip (19.32GB, key: 66203)
- TS_49_단일.zip (24.67GB, key: 66204)
- TS_50_단일.zip (36.68GB, key: 66206)
- TS_51_단일.zip (23.44GB, key: 66207)
- TS_52_단일.zip (25.41GB, key: 66208)
- TS_53_단일.zip (29.86GB, key: 66209)
- TS_54_단일.zip (21.29GB, key: 66210)
- TS_55_단일.zip (29.97GB, key: 66211)
- TS_56_단일.zip (27.81GB, key: 66212)
- TS_57_단일.zip (21.82GB, key: 66213)
- TS_58_단일.zip (24.27GB, key: 66214)
- TS_59_단일.zip (28.60GB, key: 66215)
- TS_60_단일.zip (34.08GB, key: 66217)
- TS_61_단일.zip (24.21GB, key: 66218)
- TS_62_단일.zip (33.06GB, key: 66219)
- TS_63_단일.zip (24.50GB, key: 66220)
- TS_64_단일.zip (27.29GB, key: 66221)
- TS_65_단일.zip (23.92GB, key: 66222)
- TS_66_단일.zip (23.21GB, key: 66223)
- TS_67_단일.zip (23.19GB, key: 66224)
- TS_68_단일.zip (27.31GB, key: 66225)
- TS_69_단일.zip (26.52GB, key: 66226)
- TS_70_단일.zip (20.52GB, key: 66228)
- TS_71_단일.zip (23.87GB, key: 66229)
- TS_72_단일.zip (21.91GB, key: 66230)
- TS_73_단일.zip (20.61GB, key: 66231)
- TS_74_단일.zip (23.44GB, key: 66232)
- TS_75_단일.zip (21.74GB, key: 66233)
- TS_76_단일.zip (16.83GB, key: 66234)
- TS_77_단일.zip (24.23GB, key: 66235)
- TS_78_단일.zip (26.32GB, key: 66236)
- TS_79_단일.zip (24.63GB, key: 66237)
- TS_80_단일.zip (23.97GB, key: 66239)
- TS_81_단일.zip (13.46GB, key: 66240)

## Validation 데이터 (22개 파일)

### 라벨링데이터 (11개 ZIP)
**경구약제조합 5000종 (1개)**:
- VL_1_조합.zip (6.98MB, key: 66243)

**단일경구약제 5000종 (10개)**:
- VL_1_단일.zip (16.10MB, key: 66245)
- VL_2_단일.zip (17.94MB, key: 66246)
- VL_3_단일.zip (15.85MB, key: 66247)
- VL_4_단일.zip (20.53MB, key: 66248)
- VL_5_단일.zip (16.72MB, key: 66249)
- VL_6_단일.zip (16.91MB, key: 66250)
- VL_7_단일.zip (15.04MB, key: 66251)
- VL_8_단일.zip (17.00MB, key: 66252)
- VL_9_단일.zip (24.13MB, key: 66253)
- VL_10_단일.zip (17.64MB, key: 66244)

### 원천데이터 (11개 ZIP)
**경구약제조합 5000종 (1개)**:
- VS_1_조합.zip (3.35GB, key: 66254)

**단일경구약제 5000종 (10개)**:
- VS_1_단일.zip (19.11GB, key: 66256)
- VS_2_단일.zip (23.30GB, key: 66257)
- VS_3_단일.zip (20.14GB, key: 66258)
- VS_4_단일.zip (27.28GB, key: 66259)
- VS_5_단일.zip (21.07GB, key: 66260)
- VS_6_단일.zip (21.64GB, key: 66261)
- VS_7_단일.zip (18.99GB, key: 66262)
- VS_8_단일.zip (22.29GB, key: 66263)
- VS_9_단일.zip (30.65GB, key: 66264)
- VS_10_단일.zip (24.34GB, key: 66255)

**총 파일 수**: 200개 ZIP 파일
**총 예상 용량**: 약 4.2TB

[aihub_downloader.py 다운로드 스크립트]

현재 WSL에서 `aihub_downloader.py`가 실행 중입니다. 이 스크립트는:
- API Key: AE22E788-17B4-493D-B8C8-3BF3516590D8 사용
- 순차적 그룹 다운로드로 디스크 공간 관리
- 실시간 진행상황 모니터링
- 중단/재개 기능 지원
- 다운로드 상태를 `download_status.json`에 저장

**재다운로드가 필요한 경우**:
누락되거나 손상된 파일이 발견되면 해당 filekey로 재다운로드:
```bash
# 개별 파일 다운로드 예시
aihubshell -mode d -datasetkey 576 -filekey 66065 -aihubapikey AE22E788-17B4-493D-B8C8-3BF3516590D8
```

[데이터셋 검증 필수 절차]

## Step 1: 다운로드 완료 검증
```bash
# 필수 검증 스크립트
#!/bin/bash
echo "=== AIHub 166 데이터셋 검증 ==="

# 예상 파일 개수 확인
EXPECTED_TRAINING_LABELS=89  # 81 + 8 (단일 + 조합)
EXPECTED_TRAINING_SOURCE=89  # 81 + 8 (단일 + 조합)
EXPECTED_VALIDATION_LABELS=11 # 10 + 1 (단일 + 조합)
EXPECTED_VALIDATION_SOURCE=11 # 10 + 1 (단일 + 조합)
EXPECTED_TOTAL=200

BASE_DIR="/mnt/data/AIHub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터"

# 실제 파일 개수 확인
ACTUAL_TRAINING_LABELS=$(find "$BASE_DIR/01.데이터/1.Training/라벨링데이터/" -name "*.zip" 2>/dev/null | wc -l)
ACTUAL_TRAINING_SOURCE=$(find "$BASE_DIR/01.데이터/1.Training/원천데이터/" -name "*.zip" 2>/dev/null | wc -l)
ACTUAL_VALIDATION_LABELS=$(find "$BASE_DIR/01.데이터/2.Validation/라벨링데이터/" -name "*.zip" 2>/dev/null | wc -l)
ACTUAL_VALIDATION_SOURCE=$(find "$BASE_DIR/01.데이터/2.Validation/원천데이터/" -name "*.zip" 2>/dev/null | wc -l)

echo "Training 라벨링데이터: ${ACTUAL_TRAINING_LABELS}/${EXPECTED_TRAINING_LABELS}"
echo "Training 원천데이터: ${ACTUAL_TRAINING_SOURCE}/${EXPECTED_TRAINING_SOURCE}"
echo "Validation 라벨링데이터: ${ACTUAL_VALIDATION_LABELS}/${EXPECTED_VALIDATION_LABELS}"
echo "Validation 원천데이터: ${ACTUAL_VALIDATION_SOURCE}/${EXPECTED_VALIDATION_SOURCE}"

TOTAL_ACTUAL=$((ACTUAL_TRAINING_LABELS + ACTUAL_TRAINING_SOURCE + ACTUAL_VALIDATION_LABELS + ACTUAL_VALIDATION_SOURCE))
echo "총 파일: ${TOTAL_ACTUAL}/${EXPECTED_TOTAL}"

if [ $TOTAL_ACTUAL -eq $EXPECTED_TOTAL ]; then
    echo "✅ 모든 파일 다운로드 완료"
else
    echo "❌ 다운로드 미완료: $((EXPECTED_TOTAL - TOTAL_ACTUAL))개 파일 누락"
    echo ""
    echo "=== 누락 파일 상세 분석 ==="
    
    # 누락된 파일을 찾아 filekey와 함께 출력
    python3 - <<'PYTHON'
import os
import glob

# 예상 파일 목록 (원본 데이터셋 목록에서)
expected_files = {
    # Training 라벨링데이터 - 조합
    "TL_1_조합.zip": 66065, "TL_2_조합.zip": 66066, "TL_3_조합.zip": 66067, "TL_4_조합.zip": 66068,
    "TL_5_조합.zip": 66069, "TL_6_조합.zip": 66070, "TL_7_조합.zip": 66071, "TL_8_조합.zip": 66072,
    
    # Training 라벨링데이터 - 단일
    "TL_1_단일.zip": 66083, "TL_2_단일.zip": 66094, "TL_3_단일.zip": 66105, "TL_4_단일.zip": 66116,
    "TL_5_단일.zip": 66127, "TL_6_단일.zip": 66138, "TL_7_단일.zip": 66149, "TL_8_단일.zip": 66152,
    "TL_9_단일.zip": 66153, "TL_10_단일.zip": 66073, "TL_11_단일.zip": 66074, "TL_12_단일.zip": 66075,
    "TL_13_단일.zip": 66076, "TL_14_단일.zip": 66077, "TL_15_단일.zip": 66078, "TL_16_단일.zip": 66079,
    "TL_17_단일.zip": 66080, "TL_18_단일.zip": 66081, "TL_19_단일.zip": 66082, "TL_20_단일.zip": 66084,
    "TL_21_단일.zip": 66085, "TL_22_단일.zip": 66086, "TL_23_단일.zip": 66087, "TL_24_단일.zip": 66088,
    "TL_25_단일.zip": 66089, "TL_26_단일.zip": 66090, "TL_27_단일.zip": 66091, "TL_28_단일.zip": 66092,
    "TL_29_단일.zip": 66093, "TL_30_단일.zip": 66095, "TL_31_단일.zip": 66096, "TL_32_단일.zip": 66097,
    "TL_33_단일.zip": 66098, "TL_34_단일.zip": 66099, "TL_35_단일.zip": 66100, "TL_36_단일.zip": 66101,
    "TL_37_단일.zip": 66102, "TL_38_단일.zip": 66103, "TL_39_단일.zip": 66104, "TL_40_단일.zip": 66106,
    "TL_41_단일.zip": 66107, "TL_42_단일.zip": 66108, "TL_43_단일.zip": 66109, "TL_44_단일.zip": 66110,
    "TL_45_단일.zip": 66111, "TL_46_단일.zip": 66112, "TL_47_단일.zip": 66113, "TL_48_단일.zip": 66114,
    "TL_49_단일.zip": 66115, "TL_50_단일.zip": 66117, "TL_51_단일.zip": 66118, "TL_52_단일.zip": 66119,
    "TL_53_단일.zip": 66120, "TL_54_단일.zip": 66121, "TL_55_단일.zip": 66122, "TL_56_단일.zip": 66123,
    "TL_57_단일.zip": 66124, "TL_58_단일.zip": 66125, "TL_59_단일.zip": 66126, "TL_60_단일.zip": 66128,
    "TL_61_단일.zip": 66129, "TL_62_단일.zip": 66130, "TL_63_단일.zip": 66131, "TL_64_단일.zip": 66132,
    "TL_65_단일.zip": 66133, "TL_66_단일.zip": 66134, "TL_67_단일.zip": 66135, "TL_68_단일.zip": 66136,
    "TL_69_단일.zip": 66137, "TL_70_단일.zip": 66139, "TL_71_단일.zip": 66140, "TL_72_단일.zip": 66141,
    "TL_73_단일.zip": 66142, "TL_74_단일.zip": 66143, "TL_75_단일.zip": 66144, "TL_76_단일.zip": 66145,
    "TL_77_단일.zip": 66146, "TL_78_단일.zip": 66147, "TL_79_단일.zip": 66148, "TL_80_단일.zip": 66150,
    "TL_81_단일.zip": 66151,
    
    # Training 원천데이터 - 조합
    "TS_1_조합.zip": 66154, "TS_2_조합.zip": 66155, "TS_3_조합.zip": 66156, "TS_4_조합.zip": 66157,
    "TS_5_조합.zip": 66158, "TS_6_조합.zip": 66159, "TS_7_조합.zip": 66160, "TS_8_조합.zip": 66161,
    
    # Training 원천데이터 - 단일 (대용량 파일들)
    "TS_1_단일.zip": 66172, "TS_2_단일.zip": 66183, "TS_3_단일.zip": 66194, "TS_4_단일.zip": 66205,
    "TS_5_단일.zip": 66216, "TS_6_단일.zip": 66227, "TS_7_단일.zip": 66238, "TS_8_단일.zip": 66241,
    "TS_9_단일.zip": 66242, "TS_10_단일.zip": 66162, "TS_11_단일.zip": 66163, "TS_12_단일.zip": 66164,
    "TS_13_단일.zip": 66165, "TS_14_단일.zip": 66166, "TS_15_단일.zip": 66167, "TS_16_단일.zip": 66168,
    "TS_17_단일.zip": 66169, "TS_18_단일.zip": 66170, "TS_19_단일.zip": 66171, "TS_20_단일.zip": 66173,
    "TS_21_단일.zip": 66174, "TS_22_단일.zip": 66175, "TS_23_단일.zip": 66176, "TS_24_단일.zip": 66177,
    "TS_25_단일.zip": 66178, "TS_26_단일.zip": 66179, "TS_27_단일.zip": 66180, "TS_28_단일.zip": 66181,
    "TS_29_단일.zip": 66182, "TS_30_단일.zip": 66184, "TS_31_단일.zip": 66185, "TS_32_단일.zip": 66186,
    "TS_33_단일.zip": 66187, "TS_34_단일.zip": 66188, "TS_35_단일.zip": 66189, "TS_36_단일.zip": 66190,
    "TS_37_단일.zip": 66191, "TS_38_단일.zip": 66192, "TS_39_단일.zip": 66193, "TS_40_단일.zip": 66195,
    "TS_41_단일.zip": 66196, "TS_42_단일.zip": 66197, "TS_43_단일.zip": 66198, "TS_44_단일.zip": 66199,
    "TS_45_단일.zip": 66200, "TS_46_단일.zip": 66201, "TS_47_단일.zip": 66202, "TS_48_단일.zip": 66203,
    "TS_49_단일.zip": 66204, "TS_50_단일.zip": 66206, "TS_51_단일.zip": 66207, "TS_52_단일.zip": 66208,
    "TS_53_단일.zip": 66209, "TS_54_단일.zip": 66210, "TS_55_단일.zip": 66211, "TS_56_단일.zip": 66212,
    "TS_57_단일.zip": 66213, "TS_58_단일.zip": 66214, "TS_59_단일.zip": 66215, "TS_60_단일.zip": 66217,
    "TS_61_단일.zip": 66218, "TS_62_단일.zip": 66219, "TS_63_단일.zip": 66220, "TS_64_단일.zip": 66221,
    "TS_65_단일.zip": 66222, "TS_66_단일.zip": 66223, "TS_67_단일.zip": 66224, "TS_68_단일.zip": 66225,
    "TS_69_단일.zip": 66226, "TS_70_단일.zip": 66228, "TS_71_단일.zip": 66229, "TS_72_단일.zip": 66230,
    "TS_73_단일.zip": 66231, "TS_74_단일.zip": 66232, "TS_75_단일.zip": 66233, "TS_76_단일.zip": 66234,
    "TS_77_단일.zip": 66235, "TS_78_단일.zip": 66236, "TS_79_단일.zip": 66237, "TS_80_단일.zip": 66239,
    "TS_81_단일.zip": 66240,
    
    # Validation 라벨링데이터
    "VL_1_조합.zip": 66243,
    "VL_1_단일.zip": 66245, "VL_2_단일.zip": 66246, "VL_3_단일.zip": 66247, "VL_4_단일.zip": 66248,
    "VL_5_단일.zip": 66249, "VL_6_단일.zip": 66250, "VL_7_단일.zip": 66251, "VL_8_단일.zip": 66252,
    "VL_9_단일.zip": 66253, "VL_10_단일.zip": 66244,
    
    # Validation 원천데이터
    "VS_1_조합.zip": 66254,
    "VS_1_단일.zip": 66256, "VS_2_단일.zip": 66257, "VS_3_단일.zip": 66258, "VS_4_단일.zip": 66259,
    "VS_5_단일.zip": 66260, "VS_6_단일.zip": 66261, "VS_7_단일.zip": 66262, "VS_8_단일.zip": 66263,
    "VS_9_단일.zip": 66264, "VS_10_단일.zip": 66255
}

base_dir = "/mnt/data/AIHub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터"
search_paths = [
    f"{base_dir}/01.데이터/1.Training/라벨링데이터/**/*.zip",
    f"{base_dir}/01.데이터/1.Training/원천데이터/**/*.zip", 
    f"{base_dir}/01.데이터/2.Validation/라벨링데이터/**/*.zip",
    f"{base_dir}/01.데이터/2.Validation/원천데이터/**/*.zip"
]

# 실제 존재하는 파일들 수집
existing_files = set()
for pattern in search_paths:
    for filepath in glob.glob(pattern, recursive=True):
        filename = os.path.basename(filepath)
        existing_files.add(filename)

# 누락된 파일들 찾기
missing_files = []
for expected_file, filekey in expected_files.items():
    if expected_file not in existing_files:
        missing_files.append((expected_file, filekey))

if missing_files:
    print(f"누락된 파일 {len(missing_files)}개:")
    print("파일명 | filekey")
    print("=" * 40)
    for filename, filekey in missing_files:
        print(f"{filename} | {filekey}")
    
    print(f"\n재다운로드 명령어:")
    filekeys = ",".join(str(fk) for _, fk in missing_files)
    print(f"aihubshell -mode d -datasetkey 576 -filekey {filekeys} -aihubapikey AE22E788-17B4-493D-B8C8-3BF3516590D8")
else:
    print("모든 파일이 존재합니다!")
PYTHON
    
    exit 1
fi
```

## Step 2: 파일 무결성 검증
```bash
# ZIP 파일 무결성 테스트
echo "=== ZIP 파일 무결성 검증 ==="
CORRUPTED=0

find /mnt/data/AIHub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터/ -name "*.zip" | while read zipfile; do
    echo "검증 중: $(basename "$zipfile")"
    if ! unzip -t "$zipfile" >/dev/null 2>&1; then
        echo "❌ 손상된 파일: $zipfile"
        CORRUPTED=$((CORRUPTED + 1))
    fi
done

if [ $CORRUPTED -eq 0 ]; then
    echo "✅ 모든 ZIP 파일 무결성 확인"
else
    echo "❌ $CORRUPTED 개 파일 손상, 재다운로드 필요"
    exit 1
fi
```

## Step 3: 예상 용량 대비 검증
```bash
# 용량 검증 (±5% 허용 오차)
echo "=== 용량 검증 ==="

EXPECTED_SIZE_KB=4398046511  # 4.2TB in KB
ACTUAL_SIZE_KB=$(du -sk /mnt/data/AIHub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터/ | cut -f1)
DIFF_PERCENT=$(echo "scale=2; ($ACTUAL_SIZE_KB - $EXPECTED_SIZE_KB) * 100 / $EXPECTED_SIZE_KB" | bc)

echo "예상 용량: 4.2TB"
echo "실제 용량: $(echo "scale=2; $ACTUAL_SIZE_KB / 1024 / 1024 / 1024" | bc)GB"
echo "차이: ${DIFF_PERCENT}%"

if [ $(echo "$DIFF_PERCENT > 5 || $DIFF_PERCENT < -5" | bc) -eq 1 ]; then
    echo "❌ 용량 차이가 5% 이상, 데이터 재검토 필요"
    exit 1
else
    echo "✅ 용량 검증 통과"
fi
```

## 📁 영문 변환 디렉토리 구조

```bash
# 실제 생성되는 ZIP별 폴더 구조 (2025년 8월 18일 현재 - 검증완료)
/mnt/data/pillsnap_dataset/
├── data/
│   ├── train/
│   │   ├── images/
│   │   │   ├── single/
│   │   │   │   ├── TS_1_single/         # TS_1_단일.zip → TS_1_single
│   │   │   │   │   ├── K-000001/        # K-코드별 폴더
│   │   │   │   │   │   └── K-000001_*.png   # 약품 이미지들
│   │   │   │   │   ├── K-000002/
│   │   │   │   │   │   └── K-000002_*.png
│   │   │   │   │   └── ... (50개 K-코드 폴더)
│   │   │   │   ├── TS_2_single/         # TS_2_단일.zip → TS_2_single
│   │   │   │   │   └── ... (50개 K-코드 폴더)
│   │   │   │   └── ... (TS_1_single~TS_81_single)   # 81개 단일 약품 ZIP
│   │   │   └── combination/
│   │   │       ├── TS_1_combo/          # TS_1_조합.zip → TS_1_combo
│   │   │       │   ├── K-000250-000573-002483-006192/  # 약품 조합별 폴더
│   │   │       │   │   └── K-000250-000573-002483-006192_*.png
│   │   │       │   ├── K-000250-000573-002483-012778/
│   │   │       │   └── ... (547개 조합 폴더)
│   │   │       ├── TS_2_combo/          # TS_2_조합.zip → TS_2_combo
│   │   │       │   └── ... (500개 조합 폴더)
│   │   │       └── ... (TS_1_combo~TS_8_combo)  # 8개 조합 약품 ZIP
│   │   └── labels/
│   │       ├── single/
│   │       │   ├── TL_1_single/         # TL_1_단일.zip → TL_1_single
│   │       │   │   ├── K-000001_json/   # K-코드별 json 폴더
│   │       │   │   │   └── K-000001_*.json  # COCO JSON 라벨 파일들
│   │       │   │   ├── K-000002_json/
│   │       │   │   │   └── K-000002_*.json
│   │       │   │   └── ... (50개 K-코드_json 폴더)
│   │       │   ├── TL_2_single/
│   │       │   │   └── ... (50개 K-코드_json 폴더)
│   │       │   └── ... (TL_1_single~TL_81_single)   # 81개 단일 라벨 ZIP
│   │       └── combination/
│   │           ├── TL_1_combo/          # TL_1_조합.zip → TL_1_combo
│   │           │   ├── K-000250-000573-002483-006192_json/  # 조합 K-코드_json 폴더
│   │           │   │   └── K-000250-000573-002483-006192_*.json  # JSON 라벨 파일들
│   │           │   └── ... (547개 조합 K-코드_json 폴더)
│   │           └── ... (TL_1_combo~TL_8_combo)  # 8개 조합 라벨 ZIP
│   └── val/
│       ├── images/
│       │   ├── single/
│       │   │   ├── VS_1_single/         # VS_1_단일.zip → VS_1_single
│       │   │   │   ├── K-039148/        # K-코드별 폴더
│       │   │   │   │   └── K-039148_*.png
│       │   │   │   ├── K-039167/
│       │   │   │   └── ... (50개 K-코드 폴더)
│       │   │   ├── VS_2_single/
│       │   │   └── ... (VS_1_single~VS_10_single)   # 10개 단일 검증 ZIP
│       │   └── combination/
│       │       └── VS_1_combo/          # VS_1_조합.zip → VS_1_combo
│       │           ├── K-016235-027733-029667-031885/  # 조합별 폴더
│       │           │   └── K-016235-027733-029667-031885_*.png
│       │           └── ... (500개 조합 폴더)
│       └── labels/
│           ├── single/
│           │   ├── VL_1_single/         # VL_1_단일.zip → VL_1_single
│           │   │   ├── K-039148_json/   # K-코드별 json 폴더
│           │   │   │   └── K-039148_*.json  # JSON 라벨 파일들
│           │   │   └── ... (50개 K-코드_json 폴더)
│           │   └── ... (VL_1_single~VL_10_single)   # 10개 단일 검증 라벨
│           └── combination/
│               └── VL_1_combo/          # VL_1_조합.zip → VL_1_combo
│                   ├── K-016235-027733-029667-031885_json/  # 조합 K-코드_json 폴더
│                   │   └── K-016235-027733-029667-031885_*.json  # JSON 라벨 파일들
│                   └── ... (500개 조합 K-코드_json 폴더)
├── processed/                            # 전처리된 데이터
│   ├── yolo_format/                     # YOLO 변환 결과
│   ├── metadata/
│   │   ├── edi_mapping.json             # edi_code 매핑
│   │   ├── class_statistics.json        # 클래스별 통계
│   │   └── file_index.json              # 파일 인덱스
│   └── splits/
│       ├── train_split.json             # 학습 분할 정보
│       ├── val_split.json           
│       └── test_split.json
└── cache/                                # 최적화된 캐시
    ├── lmdb/                            # LMDB 캐시
    └── thumbnails/                      # 썸네일 캐시

# 실제 ZIP 파일명 → 폴더명 변환 규칙:
# TS_1_단일.zip → TS_1_single/
# TS_1_조합.zip → TS_1_combo/
# TL_1_단일.zip → TL_1_single/
# TL_1_조합.zip → TL_1_combo/
# VS_1_단일.zip → VS_1_single/
# VS_1_조합.zip → VS_1_combo/
# VL_1_단일.zip → VL_1_single/
# VL_1_조합.zip → VL_1_combo/
```

## 🔄 순차 압축 해제 스크립트

```python
#!/usr/bin/env python3
# scripts/sequential_extract.py
"""
한글 폴더명을 영문으로 변환하며 순차적으로 압축 해제
"""
import os
import shutil
import zipfile
import json
from pathlib import Path
from typing import Dict, List
import logging

class SequentialExtractor:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.name_mapping = {
            # 디렉토리 매핑
            "166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터": "pillsnap_dataset",
            "01.데이터": "data",
            "1.Training": "train",
            "2.Validation": "val",
            "라벨링데이터": "labels",
            "원천데이터": "images",
            "단일경구약제_5000종": "single",
            "경구약제조합_5000종": "combination",
            # 파일명 매핑
            "단일": "single",
            "조합": "combination"
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def convert_path(self, korean_path: str) -> str:
        """한글 경로를 영문으로 변환"""
        english_path = korean_path
        for kor, eng in self.name_mapping.items():
            english_path = english_path.replace(kor, eng)
        return english_path
    
    def check_disk_space(self, required_gb: float) -> bool:
        """디스크 공간 확인"""
        stat = shutil.disk_usage("/mnt/data")
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb + 100:  # 100GB 여유 확보
            self.logger.error(f"Insufficient space: {free_gb:.1f}GB < {required_gb+100:.1f}GB")
            return False
        return True
    
    def extract_batch(self, zip_files: List[str], output_base: str):
        """배치 단위로 압축 해제"""
        for zip_path in zip_files:
            try:
                # 파일 크기 확인
                size_gb = os.path.getsize(zip_path) / (1024**3)
                
                # 디스크 공간 체크
                if not self.check_disk_space(size_gb * 1.2):  # 압축 해제 시 1.2배 예상
                    raise Exception("Insufficient disk space")
                
                self.logger.info(f"Extracting {os.path.basename(zip_path)} ({size_gb:.1f}GB)")
                
                # 임시 디렉토리에 압축 해제
                temp_dir = f"/mnt/data/temp_{os.getpid()}"
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(temp_dir)
                
                # 영문으로 이름 변경하며 이동
                self.reorganize_extracted(temp_dir, output_base)
                
                # 임시 디렉토리 삭제
                shutil.rmtree(temp_dir)
                
                # 원본 ZIP 삭제
                os.remove(zip_path)
                self.logger.info(f"Deleted {zip_path}, freed {size_gb:.1f}GB")
                
            except Exception as e:
                self.logger.error(f"Failed to process {zip_path}: {e}")
                raise
    
    def reorganize_extracted(self, temp_dir: str, output_base: str):
        """추출된 파일을 영문 구조로 재구성"""
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                old_path = os.path.join(root, file)
                
                # 상대 경로를 영문으로 변환
                rel_path = os.path.relpath(old_path, temp_dir)
                new_rel_path = self.convert_path(rel_path)
                new_path = os.path.join(output_base, new_rel_path)
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                # 파일 이동
                shutil.move(old_path, new_path)
    
    def process_dataset(self, source_dir: str, output_dir: str):
        """전체 데이터셋 처리"""
        # 1. 라벨링 데이터 먼저 처리 (작은 용량)
        label_zips = sorted(Path(source_dir).glob("**/라벨링데이터/**/*.zip"))
        self.logger.info(f"Processing {len(label_zips)} label files...")
        self.extract_batch(label_zips, output_dir)
        
        # 2. 원천 데이터 배치 처리 (대용량)
        image_zips = sorted(Path(source_dir).glob("**/원천데이터/**/*.zip"))
        self.logger.info(f"Processing {len(image_zips)} image files...")
        
        for i in range(0, len(image_zips), self.batch_size):
            batch = image_zips[i:i+self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1}/{len(image_zips)//self.batch_size + 1}")
            self.extract_batch(batch, output_dir)
            
            # 진행 상황 저장
            progress = {
                "processed": i + len(batch),
                "total": len(image_zips),
                "percentage": (i + len(batch)) / len(image_zips) * 100
            }
            with open("extraction_progress.json", "w") as f:
                json.dump(progress, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/data/AIHub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터")  # 원본 데이터 경로 (한글)
    parser.add_argument("--output", default="/mnt/data/pillsnap_dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    
    extractor = SequentialExtractor(batch_size=args.batch_size)
    extractor.process_dataset(args.source, args.output)
```

### 실행 방법
```bash
# 다운로드 완료 후 실행
cd /home/max16/pillsnap
python scripts/sequential_extract.py --batch-size 1
```

## 📁 **프로젝트 구조 (PART_B 기준)**

### **코드 프로젝트 구조**
```
/home/max16/pillsnap/
├─ .env                         # 환경 변수 (API 키, 경로 등)
├─ .env.example                 # 환경 변수 예시
├─ .gitignore                   # Git 무시 파일
├─ .gitattributes              # Git 속성 설정
├─ .editorconfig               # 에디터 설정
├─ requirements.txt             # Python 의존성
├─ config.yaml                 # 메인 설정 파일
├─ VERSION.txt                 # 프로젝트 버전
├─ README.md                   # 프로젝트 문서
├─ scripts/                    # 실행 스크립트
│  ├─ bootstrap_venv.sh        # 가상환경 설정
│  ├─ train.sh                 # 학습 실행
│  ├─ export_onnx.sh          # ONNX 변환 스크립트
│  ├─ run_api.sh              # API 서버 실행
│  ├─ evaluate_stage.sh       # 단계별 자동 평가
│  ├─ maintenance.sh          # 유지보수 스크립트
│  ├─ backup_release.sh       # 백업 릴리스
│  ├─ reload_model.sh         # 모델 무중단 교체
│  ├─ perf_bench_infer.py     # 성능 벤치마크
│  ├─ ort_optimize.py         # ONNX Runtime 최적화
│  ├─ quantize_dynamic.py     # 동적 양자화
│  ├─ cf_start.ps1            # Cloudflare 시작 (Windows)
│  ├─ cf_stop.ps1             # Cloudflare 중지 (Windows)
│  └─ cf_status.ps1           # Cloudflare 상태 (Windows)
├─ src/                       # 소스 코드
│  ├─ __init__.py
│  ├─ data.py                 # 단일/조합 구분 데이터로더
│  ├─ train.py                # 조건부 Two-Stage 학습
│  ├─ evaluate.py             # 성능 평가 (accuracy, mAP)
│  ├─ infer.py                # 추론 파이프라인
│  ├─ utils.py                # 128GB RAM 최적화 유틸
│  ├─ models/                 # 모델 정의
│  │  ├─ __init__.py
│  │  ├─ detector.py          # YOLOv11x for 조합약품
│  │  ├─ classifier.py        # EfficientNetV2-L for 5000 classes
│  │  └─ pipeline.py          # Two-Stage 조건부 파이프라인
│  └─ api/                    # FastAPI 서빙
│     ├─ __init__.py
│     ├─ main.py              # FastAPI 앱
│     ├─ schemas.py           # edi_code 스키마
│     ├─ service.py           # Two-Stage 서비스
│     └─ security.py          # API 키 인증
└─ tests/                     # 테스트 코드
   ├─ test_smoke_detection.py    # YOLO 검출 테스트
   ├─ test_smoke_classification.py # 분류 테스트
   ├─ test_pipeline.py           # Two-Stage 파이프라인 테스트
   ├─ test_export_compare.py     # ONNX 변환 테스트
   ├─ test_api_min.py            # API 테스트
   ├─ stage_evaluator.py         # 점진적 검증 자동 평가
   ├─ stage_1_evaluator.py       # Stage 1 전용 평가
   ├─ stage_2_evaluator.py       # Stage 2 전용 평가
   ├─ stage_3_evaluator.py       # Stage 3 전용 평가
   ├─ stage_4_evaluator.py       # Stage 4 전용 평가
   └─ stage_progress_tracker.py  # 전체 Stage 진행 상황 추적
```

### **데이터 구조 (영문 변환 후)**
```
/mnt/data/pillsnap_dataset/              # 영문 변환된 데이터셋
├─ data/
│  ├─ train/
│  │  ├─ labels/
│  │  │  ├─ single/                      # 단일경구약제_5000종 → single
│  │  │  │  └─ TL_*_single/              # TL_*_단일.zip → TL_*_single/
│  │  │  │     └─ *.json                # COCO JSON 파일들
│  │  │  └─ combination/                 # 경구약제조합_5000종 → combination
│  │  │     └─ TL_*_combo/              # TL_*_조합.zip → TL_*_combo/
│  │  │        └─ *.json                # COCO JSON 파일들
│  │  └─ images/
│  │     ├─ single/                      # TS_*_단일.zip → TS_*_single/
│  │     │  └─ TS_*_single/              # 개별 폴더들
│  │     └─ combination/                 # TS_*_조합.zip → TS_*_combo/
│  │        └─ TS_*_combo/               # 개별 폴더들
│  └─ val/
│     ├─ labels/
│     │  ├─ single/                      # VL_*_단일.zip → VL_*_single/
│     │  │  └─ VL_*_single/              # 개별 폴더들
│     │  └─ combination/                 # VL_*_조합.zip → VL_*_combo/
│     │     └─ VL_*_combo/               # 개별 폴더들
│     └─ images/
│        ├─ single/                      # VS_*_단일.zip → VS_*_single/
│        │  └─ VS_*_single/              # 개별 폴더들
│        └─ combination/                 # VS_*_조합.zip → VS_*_combo/
│           └─ VS_*_combo/               # 개별 폴더들
├─ processed/                            # 전처리된 데이터
│  ├─ yolo_format/                       # YOLO 변환 데이터
│  ├─ lmdb_cache/                        # LMDB 캐시 (128GB RAM 활용)
│  ├─ edi_mapping.json                   # edi_code → class_id 매핑
│  ├─ class_names.json                   # class_id → 약품명 매핑
│  └─ splits.json                        # train/val 분할 정보
└─ cache/                                # 임시 캐시 파일
```

### **실험 디렉토리 구조**
```
/mnt/data/exp/exp01/                     # 실험별 결과 관리
├─ logs/                                 # 로그 파일
│  ├─ train.out, train.err              # 학습 로그
│  ├─ api.out, api.err                  # API 서빙 로그
│  ├─ export.out, export.err            # ONNX 변환 로그
│  └─ archive/                          # 압축된 과거 로그
├─ tb/                                  # TensorBoard 이벤트
│  ├─ stage_evaluation/                 # Stage별 평가 결과
│  ├─ train/                           # 학습 메트릭
│  └─ perf/                            # 성능 지표
├─ checkpoints/                         # 모델 체크포인트
│  ├─ detection_best.pt                 # YOLOv11x 최고 성능
│  ├─ detection_last.pt                 # YOLOv11x 최신
│  ├─ classification_best.pt            # EfficientNetV2-L 최고 성능
│  └─ classification_last.pt            # EfficientNetV2-L 최신
├─ export/                              # ONNX 모델
│  ├─ detection-YYYYMMDD-HHMMSS-SHA.onnx
│  ├─ classification-YYYYMMDD-HHMMSS-SHA.onnx
│  ├─ latest_detection.onnx             # 심볼릭 링크
│  ├─ latest_classification.onnx        # 심볼릭 링크
│  └─ export_report.json                # 변환 상세 리포트
├─ reports/                             # 평가 리포트
│  ├─ stage_1_evaluation.json           # Stage 1 평가 결과
│  ├─ stage_2_evaluation.json           # Stage 2 평가 결과
│  ├─ stage_3_evaluation.json           # Stage 3 평가 결과
│  ├─ stage_4_evaluation.json           # Stage 4 평가 결과
│  ├─ metrics.json                      # 최종 성능 지표
│  └─ confusion_matrix.png              # 혼동 행렬
└─ releases/                            # 배포 아카이브
   ├─ release-YYYYMMDD-HHMMSS-SHA.tar.gz
   ├─ release-YYYYMMDD-HHMMSS-SHA.tar.gz.sha256
   └─ MANIFEST.json                     # 릴리스 메타데이터
```

## 🚨 **중요한 주의사항**

### **데이터 무결성이 최우선**
1. **압축 해제 전 반드시 검증 스크립트 실행**
2. **손상된 파일 발견 시 즉시 재다운로드**  
3. **디스크 공간 모니터링 (85% 이상 시 중단)**
4. **원본 삭제는 해제 성공 후에만**

### **재다운로드가 필요한 경우**
- ZIP 파일 개수가 200개 미만
- 총 용량이 4.2TB ±5% 범위 벗어남
- ZIP 무결성 테스트 실패
- 분할 파일(*.part*) 미완성 상태

## 📋 **다음 단계 (압축 해제 완료 후)**

1. **프로젝트 초기화**: `bash scripts/core/setup_venv.sh`
2. **COCO→YOLO 변환**: Part C 데이터 파이프라인
3. **edi_code 매핑 생성**: Part C 매핑 테이블
4. **조건부 Two-Stage 학습**: Part D 학습 파이프라인
5. **ONNX Export & 검증**: Part E 모델 변환
6. **FastAPI 서빙**: Part F API 서비스
7. **Cloudflare Tunnel**: Part G 외부 접속
8. **운영 자동화**: Part H 모니터링

**⚠️ 이 Part 0을 완료하지 않고는 절대 다음 파트로 진행하지 마세요.**