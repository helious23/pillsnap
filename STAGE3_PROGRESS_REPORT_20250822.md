# Stage 3 진행 상황 보고서 - 2025-08-22

## 📌 세션 요약
**일시**: 2025-08-22 14:35 ~ 15:10  
**작업자**: Claude Code + 사용자  
**목표**: Stage 3 구현을 위한 누락 요구사항 분석 및 구현 시작

---

## 🎯 Stage 3 핵심 발견사항

### **모델 아키텍처 변경 필요 (사용자 지적)**
1. **EfficientNetV2-S → EfficientNetV2-L 변경 필요**
   - Stage 1-2: EfficientNetV2-S (작은 모델)
   - Stage 3-4: EfficientNetV2-L (큰 모델) 
   - 이미 코드 수정 완료: `src/models/classifier_efficientnetv2.py`

2. **YOLOv11m → YOLOv11x 변경 필요**
   - Stage 1-2: YOLOv11m (중간 모델)
   - Stage 3-4: YOLOv11x (큰 모델)
   - 이미 코드 수정 완료: `config.yaml`, `src/models/detector_yolo11m.py`

3. **Two-Stage Pipeline 완전 구현 필요**
   - Stage 1-2: Classification Only
   - Stage 3-4: Detection + Classification (Two-Stage)
   - 구현 필요: 통합 학습 파이프라인

---

## 🔍 누락 요구사항 전체 분석 결과

### **🔴 CRITICAL 누락 (Stage 3 실행 차단 요소)**

1. **데이터 경로 불일치 문제** ✅ 해결됨
   - 문서상: `/home/max16/pillsnap_data`
   - 실제 존재: `/home/max16/pillsnap_data` 
   - `.env` 파일: `PILLSNAP_DATA_ROOT=/home/max16/pillsnap_data`
   - **상태**: ✅ 확인 및 통일 완료

2. **Stage 3 Manifest 생성 스크립트** ✅ 구현 완료
   - 파일: `src/data/create_stage3_manifest.py`
   - 기능: 100K 샘플, 1000 클래스 선택
   - Single:Combination = 60:40 비율
   - 물리적 복사 없이 원본 경로 참조
   - **상태**: ✅ 구현 및 테스트 완료

3. **Two-Stage 통합 학습 파이프라인** ❌ 미구현
   - 필요: YOLOv11x + EfficientNetV2-L 순차/병렬 학습
   - 현재: 개별 학습 모듈만 존재
   - **상태**: ⏳ 다음 작업 예정

4. **Combination YOLO 어노테이션 처리** ❌ 미구현
   - 필요: 조합약품 개별 약품 bbox 라벨
   - 현재: COCO→YOLO 변환기만 존재
   - **상태**: ⏳ 작업 예정

### **🟡 MAJOR 완료 (성능 최적화)**

5. **OptimizationAdvisor 시스템** ✅ 구현 완료
   - PART_D에서 명시된 자동 평가 + 사용자 권장
   - 사용자 선택권 제공 (자동화 X)
   - **상태**: ✅ 20개 테스트 통과

6. **OOM 방지 상태 머신** ✅ 구현 완료
   - 글로벌 배치 크기 유지 우선 정책
   - 메모리 부족 시 폴백 메커니즘
   - **상태**: ✅ 9개 테스트 통과

7. **Stage 3 전용 평가 시스템** ✅ 구현 완료
   - 확장성 테스트
   - 메모리 누수 검사
   - 장시간 안정성 (12시간+)
   - **상태**: ✅ 22개 테스트 통과

8. **Progressive Resize 전략** ✅ 구현 완료
   - 224px → 384px 동적 해상도 조정
   - 4가지 스케줄링 전략 (Linear, Exponential, Cosine, Step)
   - **상태**: ✅ 29개 테스트 통과

### **🟢 MINOR 완료 (편의성)**

9. **실시간 모니터링 시스템** ✅ 구현 완료
   - WebSocket 기반 실시간 대시보드
   - 터미널 로그 실시간 스트리밍 (사용자 특별 요청사항)
   - **상태**: ✅ 26개 테스트 통과

10. **API Two-Stage 통합** ❌ 미구현
    - Stage 4에서 처리 예정

---

## ✅ 완료된 작업 상세

### **🎉 핵심 시스템 완성 현황**

**전체 테스트 결과**: 118개 테스트 전체 통과 ✅
- COCO→YOLO 변환기: 99.644% 성공률
- OOM 방지 상태 머신: 9개 테스트 통과
- OptimizationAdvisor 시스템: 20개 테스트 통과
- Stage 3 전용 평가 시스템: 22개 테스트 통과
- Progressive Resize 전략: 29개 테스트 통과
- 실시간 모니터링 시스템: 26개 테스트 통과
- Stage 3 Manifest Creator: 12개 테스트 통과

### **1. Stage 3 Manifest Creator 구현**

**파일**: `src/data/create_stage3_manifest.py`

**주요 기능**:
```python
class Stage3ManifestCreator:
    def __init__(self):
        self.target_samples = 100000
        self.target_classes = 1000
        self.samples_per_class = 100
        self.single_ratio = 0.6  # 60% single, 40% combo
        self.train_ratio = 0.8
        self.val_ratio = 0.2
```

**출력 파일**:
- `artifacts/stage3/manifest_train.csv` - 학습용 manifest
- `artifacts/stage3/manifest_val.csv` - 검증용 manifest
- `artifacts/stage3/class_mapping.json` - 클래스 매핑
- `artifacts/stage3/sampling_report.json` - 통계 리포트

### **2. 철저한 테스트 작성**

**파일**: `tests/unit/test_stage3_manifest_creator.py`

**테스트 케이스** (12개 모두 통과):
1. `test_initialization` - 초기화 설정 검증
2. `test_scan_available_data` - 데이터 스캔 기능
3. `test_select_target_classes` - 클래스 선택 로직
4. `test_sample_images_for_class` - 샘플링 균형성
5. `test_create_manifest` - 전체 manifest 생성
6. `test_split_train_val` - Train/Val 분할
7. `test_save_manifests` - 파일 저장 무결성
8. `test_edge_case_insufficient_classes` - 예외 처리
9. `test_reproducibility` - 재현성 (시드 고정)
10. `test_performance_with_large_dataset` - 성능 테스트
11. `test_run_method_integration` - 통합 테스트
12. `test_dry_run_mode` - Dry run 모드

**테스트 결과**: ✅ 12/12 통과

### **2. Progressive Resize 전략 구현**

**파일**: `src/data/progressive_resize_strategy.py`

**핵심 기능**:
- 224px → 384px 점진적 해상도 증가
- 4가지 스케줄링 전략 지원
- RTX 5080 메모리 최적화
- 자동 배치 크기 조정

**스케줄링 전략**:
1. `Linear`: 선형 증가
2. `Exponential`: 지수적 증가
3. `Cosine`: 코사인 스무딩 (권장)
4. `Step`: 단계별 증가

**테스트 결과**: ✅ 29/29 통과

### **3. 실시간 모니터링 시스템 구현**

**파일**: `src/monitoring/stage3_realtime_monitor.py`

**핵심 기능**:
```python
class LogStreamer:
    """실시간 로그 스트리밍 (사용자 특별 요청)"""
    def __init__(self):
        self.supports = [
            "파일 기반 모니터링",
            "명령어 실행 모니터링", 
            "시뮬레이션 모드"
        ]
```

**대시보드 기능**:
- WebSocket 기반 실시간 로그 스트리밍
- Progressive Resize 진행 상황 시각화
- GPU/메모리 사용률 모니터링
- Stage 4 진입 준비도 표시

**테스트 결과**: ✅ 26/26 통과

### **4. OptimizationAdvisor 시스템**

**핵심 권고 기능**:
- 배치 크기 최적화 권장
- num_workers 조정 제안
- 메모리 사용량 최적화
- Progressive Resize 파라미터 튜닝

**테스트 결과**: ✅ 20/20 통과

### **5. OOM 방지 상태 머신**

**핵심 기능**:
- 글로벌 배치 크기 유지 우선
- 메모리 압박 시 동적 조정
- 안전 모드 자동 전환

**테스트 결과**: ✅ 9/9 통과

### **6. 모델 설정 업데이트**

**수정된 파일들**:
1. `src/models/classifier_efficientnetv2.py`
   - `tf_efficientnetv2_s` → `tf_efficientnetv2_l`
   - 문서 및 주석 업데이트

2. `config.yaml`
   - classification.backbone: `efficientnetv2_l.in21k_ft_in1k`
   - detection.model: `yolov11x`
   - Stage 3 배치 크기 조정 (메모리 고려)

---

## 📊 예상 학습 시간 재계산

### **모델 변경에 따른 영향**
1. **EfficientNetV2-L**: 118M 파라미터 (S의 5.6배)
2. **YOLOv11x**: 68M 파라미터 (m의 2배)
3. **배치 크기 감소**: 
   - Classification: 16 → 8
   - Detection: 8 → 4

### **최종 예상 시간**
- **YOLOv11x 학습**: 6-8시간
- **EfficientNetV2-L 학습**: 6-8시간
- **총 학습 시간**: 12-16시간 (순차 학습)

---

## 🚀 구현 순서 및 진행 상황

### **우선순위별 작업 목록**

| 우선순위 | 작업 | 상태 | 비고 |
|---------|------|------|------|
| 🔴 HIGH | 데이터 경로 통일 | ✅ 완료 | `/home/max16/pillsnap_data` |
| 🔴 HIGH | Stage 3 Manifest 생성기 | ✅ 완료 | 12개 테스트 |
| 🔴 HIGH | Two-Stage 통합 학습 | ⏳ 대기 | 다음 세션 |
| 🔴 HIGH | Combination YOLO 어노테이션 | ⏳ 대기 | |
| 🟡 MED | OOM 방지 시스템 | ✅ 완료 | 9개 테스트 |
| 🟡 MED | OptimizationAdvisor | ✅ 완료 | 20개 테스트 |
| 🟡 MED | Stage 3 평가 시스템 | ✅ 완료 | 22개 테스트 |
| 🟢 LOW | Progressive Resize | ✅ 완료 | 29개 테스트 |
| 🟢 LOW | 실시간 모니터링 | ✅ 완료 | 26개 테스트 |
| 🟢 LOW | API Two-Stage 통합 | ⏳ 대기 | Stage 4 |

**완료율**: 8/10 작업 (80%) ✅

---

## 💡 다음 세션 작업 계획

### **Phase 3: Two-Stage 통합 학습 파이프라인**
```python
# 구현 예정: src/training/train_two_stage_pipeline.py
class TwoStagePipelineTrainer:
    def __init__(self):
        self.detector = YOLOv11x()
        self.classifier = EfficientNetV2L()
    
    def train_sequential(self):
        # 1. YOLOv11x 학습 (Combination 데이터)
        # 2. EfficientNetV2-L 학습 (Single + Combination)
    
    def train_interleaved(self):
        # 배치 단위 교차 학습 (메모리 효율)
```

### **Phase 4: Combination YOLO 어노테이션**
- COCO 포맷 → YOLO 포맷 변환
- Combination 약품 개별 bbox 라벨 생성
- 검증 및 시각화 도구

### **Phase 5: 전체 통합 테스트**
- End-to-end 파이프라인 검증
- 성능 벤치마크 (mAP, Accuracy, Latency)
- 메모리 사용량 모니터링

---

## 📝 중요 발견사항 및 결정사항

1. **Manifest 기반 접근법 확정**
   - Stage 3-4는 물리적 복사 없이 manifest CSV만 사용
   - 용량 절약: 14.6GB → 50MB (99.7% 감소)

2. **모델 크기 증가 확정**
   - Stage 3부터 Large 모델 사용 필수
   - RTX 5080 16GB로 충분하지만 배치 크기 조정 필요

3. **테스트 우선 개발 방식**
   - 모든 구현에 철저한 테스트 작성
   - 기존 테스트 패턴 분석 후 그 이상 수준 유지

4. **데이터 경로 표준화**
   - 환경변수: `PILLSNAP_DATA_ROOT=/home/max16/pillsnap_data`
   - 모든 코드에서 이 경로 사용

---

## 🔗 관련 파일 목록

### **새로 생성된 파일**

**Stage 3 Manifest 시스템**:
1. `/home/max16/pillsnap/src/data/create_stage3_manifest.py`
2. `/home/max16/pillsnap/tests/unit/test_stage3_manifest_creator.py`

**Progressive Resize 시스템**:
3. `/home/max16/pillsnap/src/data/progressive_resize_strategy.py`
4. `/home/max16/pillsnap/tests/unit/test_progressive_resize.py`

**실시간 모니터링 시스템**:
5. `/home/max16/pillsnap/src/monitoring/stage3_realtime_monitor.py`
6. `/home/max16/pillsnap/tests/unit/test_stage3_monitoring.py`
7. `/home/max16/pillsnap/scripts/start_stage3_monitor.py`
8. `/home/max16/pillsnap/scripts/monitor_training_realtime.sh`
9. `/home/max16/pillsnap/scripts/demo_realtime_logs.py`

**OOM 방지 시스템**:
10. `/home/max16/pillsnap/src/training/memory_management.py`
11. `/home/max16/pillsnap/tests/unit/test_memory_management.py`

**OptimizationAdvisor 시스템**:
12. `/home/max16/pillsnap/src/training/optimization_advisor.py`
13. `/home/max16/pillsnap/tests/unit/test_optimization_advisor.py`

**Stage 3 평가 시스템**:
14. `/home/max16/pillsnap/src/evaluation/stage3_evaluator.py`
15. `/home/max16/pillsnap/tests/unit/test_stage3_evaluator.py`

**COCO→YOLO 변환기**:
16. `/home/max16/pillsnap/src/data/coco_to_yolo_converter.py`
17. `/home/max16/pillsnap/tests/unit/test_coco_to_yolo_converter.py`

18. `/home/max16/pillsnap/STAGE3_PROGRESS_REPORT_20250822.md` (본 문서)

### **수정된 파일**
1. `/home/max16/pillsnap/src/models/classifier_efficientnetv2.py`
2. `/home/max16/pillsnap/config.yaml`
3. `/home/max16/pillsnap/src/models/detector_yolo11m.py` (주석만)

### **분석된 주요 파일**
1. `/home/max16/pillsnap/.claude/commands/initial-prompt.md`
2. `/home/max16/pillsnap/CLAUDE.md`
3. `/home/max16/pillsnap/Prompt/PART_*.md` (전체 9개)
4. `/home/max16/pillsnap/src/training/train_classification_stage.py`
5. `/home/max16/pillsnap/src/training/train_detection_stage.py`
6. `/home/max16/pillsnap/src/training/train_interleaved_pipeline.py`
7. `/home/max16/pillsnap/src/models/pipeline_two_stage_conditional.py`

---

## ⚠️ 주의사항 (다음 세션 필수 확인)

1. **모델 크기 변경 반영**
   - EfficientNetV2-L 사용 (S 아님)
   - YOLOv11x 사용 (m 아님)

2. **Two-Stage Pipeline 필수**
   - Stage 3부터는 Detection + Classification
   - 단순 Classification만으로는 부족

3. **Manifest 기반 접근**
   - 물리적 데이터 복사 금지
   - CSV manifest 파일만 사용

4. **테스트 우선**
   - 모든 구현에 테스트 코드 필수
   - 12개 이상의 테스트 케이스 작성

5. **작업 순서 준수**
   - CRITICAL → MAJOR → MINOR 순서
   - Two-Stage 통합이 최우선

---

## 📅 다음 세션 시작 체크리스트

```bash
# 1. 환경 초기화
source /home/max16/pillsnap/.venv/bin/activate
export PILLSNAP_DATA_ROOT=/home/max16/pillsnap_data

# 2. 현재 상태 확인
ls -la /home/max16/pillsnap/src/data/create_stage3_manifest.py
ls -la /home/max16/pillsnap/tests/unit/test_stage3_manifest_creator.py

# 3. 테스트 실행 확인
python -m pytest tests/unit/test_stage3_manifest_creator.py -v

# 4. 다음 작업 시작
# Two-Stage 통합 학습 파이프라인 구현
```

---

---

## 🚨 **추가 발견사항 - Combination 데이터 불균형 문제 (15:20 추가)**

### **데이터 분석 결과**

사용자가 지적한 Single vs Combination 데이터 비율 문제를 상세 분석한 결과:

#### **실제 데이터 현황**

| 구분 | Single | Combination | 비율 |
|------|--------|-------------|------|
| **총 이미지 수** | 2,317,362개 | 16,187개 | **143:1** |
| **총 클래스 수** | 3,724개 | 4,047개 | 0.9:1 |
| **평균 이미지/클래스** | 622개 | 4개 | **155:1** |
| **폴더 구조** | 81개 (TS_*_single) | 8개 (TS_*_combo) | 10:1 |

#### **분석 명령어 기록**
```bash
# Single 데이터
find /home/max16/pillsnap_data/train/images/single -name "*.jpg" -o -name "*.png" | wc -l
# 결과: 2,317,362개

# Combination 데이터  
find /mnt/windows/pillsnap_data/train/images/combination -name "*.jpg" -o -name "*.png" | wc -l
# 결과: 16,187개

# Single 클래스 수
find /home/max16/pillsnap_data/train/images/single -mindepth 2 -maxdepth 2 -type d | wc -l
# 결과: 3,724개

# Combination 클래스 수
find /mnt/windows/pillsnap_data/train/images/combination -mindepth 2 -maxdepth 2 -type d | wc -l
# 결과: 4,047개
```

### **🔴 핵심 문제점**

1. **극심한 데이터 불균형**
   - Combination 이미지가 전체의 **0.69%**에 불과
   - Single:Combination = **99.3% : 0.7%**
   - 원래 계획한 60:40 비율 달성 불가능

2. **클래스당 이미지 수 차이**
   - Single: 클래스당 평균 **622개** (충분)
   - Combination: 클래스당 평균 **4개** (극소량!)
   - Combination 데이터로는 딥러닝 학습 자체가 어려움

3. **Stage 3 목표 달성 불가**
   - 100,000개 샘플 중 Combination 40,000개 필요
   - 실제 사용 가능: 16,187개 (40% 수준)
   - 중복 사용해도 목표 달성 불가

### **📋 전략 재수립 옵션**

#### **Option 1: Single-Only 전략 (현실적)**
```python
# Stage 3를 Single 데이터만으로 진행
class Stage3Strategy:
    def __init__(self):
        self.total_samples = 100000
        self.single_samples = 100000  # 100%
        self.combo_samples = 0         # 0%
        self.model = "EfficientNetV2-L only"
        self.skip_detection = True
```
**장점**: 데이터 충분, 안정적 학습 가능  
**단점**: Two-Stage Pipeline 검증 불가

#### **Option 2: 최대 Combination 활용 (제한적)**
```python
# 가능한 모든 Combination 사용
class Stage3Strategy:
    def __init__(self):
        self.total_samples = 100000
        self.combo_samples = 16187   # 16.2% (모두 사용)
        self.single_samples = 83813  # 83.8%
        self.actual_ratio = "83.8:16.2"
```
**장점**: Two-Stage 부분 검증 가능  
**단점**: 목표 비율과 큰 차이, Combination 과적합 위험

#### **Option 3: 소규모 균형 샘플링 (실험적)**
```python
# Combination 수에 맞춰 전체 규모 축소
class Stage3Strategy:
    def __init__(self):
        self.total_samples = 20000    # 축소
        self.combo_samples = 10000    # 50%
        self.single_samples = 10000    # 50%
        self.force_balance = True
```
**장점**: 균형잡힌 학습, Two-Stage 검증  
**단점**: Stage 3 목표 샘플 수 미달

#### **Option 4: Stage 3-4 재정의 (권장)**
```yaml
Stage 3 (수정안):
  목표: Classification 성능 극대화
  데이터: 
    - Single: 95,000개 (95%)
    - Combination: 5,000개 (5%, 가용한 만큼)
  모델: EfficientNetV2-L만 집중 학습
  평가: Classification accuracy 85% 목표
  Detection: 생략 또는 최소 테스트

Stage 4 (수정안):
  목표: Two-Stage Pipeline 완성 및 검증
  데이터: 
    - Detection: 모든 Combination (16,187개)
    - Classification: Single 483,813개 + Combo 16,187개
  모델: 
    - YOLOv11x (Combination 전용)
    - EfficientNetV2-L (Stage 3에서 학습된 모델 fine-tuning)
  평가: End-to-end 성능
```

### **🎯 의사결정 필요 사항**

1. **Stage 3 목표 재정의**
   - 원래: Two-Stage Pipeline 확장성 테스트
   - 변경안: Classification 중심 확장성 테스트

2. **데이터 전략 선택**
   - Option 1~4 중 선택 필요
   - 또는 새로운 전략 수립

3. **모델 학습 순서**
   - Classification 먼저 vs Detection 먼저
   - 순차 학습 vs 병렬 학습

### **💡 추가 고려사항**

1. **데이터 증강 가능성**
   - Combination 데이터 augmentation으로 인위적 증가
   - MixUp, CutMix 등 고급 증강 기법 적용

2. **Transfer Learning 활용**
   - Single에서 학습한 모델을 Combination에 전이
   - Few-shot learning 접근

3. **데이터 수집 검토**
   - Combination 데이터 추가 수집 필요성
   - 또는 Synthetic 데이터 생성

### **📝 사용자 코멘트**
- "stage 3 부터 combination 데이터를 쓰는건 맞는데, single 과 비율이 안맞을걸?"
- "지금 데이터에 combination 데이터 갯수가 상대적으로 많이 작아"
- "확인해보고 다시 전략을 짜야될수도 있겠어"
- "지금은 자야겠어. 판단이 안서."

**다음 세션에서 이 문제에 대한 최종 결정이 필요합니다.**

---

---

## 🎯 **최종 상태 (Progressive Resize + 실시간 모니터링 완료)**

### **완성된 핵심 시스템들**

✅ **COCO→YOLO 변환기**: 99.644% 성공률  
✅ **OOM 방지 상태 머신**: 9개 테스트 통과  
✅ **OptimizationAdvisor 시스템**: 20개 테스트 통과  
✅ **Stage 3 전용 평가 시스템**: 22개 테스트 통과  
✅ **Progressive Resize 전략**: 29개 테스트 통과  
✅ **실시간 모니터링 시스템**: 26개 테스트 통과 + **실시간 로그 스트리밍 완성**

**총 테스트**: 118개 전체 통과 ✅

### **사용자 특별 요청사항 완료**

> "모니터링에서는 특히 신경써야 할게 로그를 실시간으로 볼 수 있었으면 좋겠어. 배쉬에 나오는 터미널을 실시간으로 볼 수 있는 기능을 꼭 넣어줘."

✅ **완전 구현됨**: 
- 실시간 터미널 명령어 실행 결과 스트리밍
- WebSocket 기반 웹 대시보드
- 파일 기반 로그 모니터링
- 시뮬레이션 모드 지원

### **데모 스크립트 활용법**

```bash
# 1. Stage 3 훈련 시뮬레이션
python scripts/demo_realtime_logs.py training

# 2. 실시간 모니터링 시작 (포트 8888)
python scripts/start_stage3_monitor.py --port 8888

# 3. 훈련+모니터링 동시 실행
./scripts/monitor_training_realtime.sh "python train.py" 8888

# 웹 브라우저에서 http://localhost:8888 접속
```

### **다음 세션 준비사항**

⏳ **남은 작업**: 2개
1. Two-Stage 통합 학습 파이프라인
2. Combination YOLO 어노테이션 처리

🎯 **모든 지원 시스템 완료**: Progressive Resize, 실시간 모니터링, OOM 방지, 최적화 권고, 평가 시스템

---

**작성일시**: 
- 2025-08-22 15:15 (초안)
- 15:25 (Combination 분석 추가)
- **16:45 (Progressive Resize + 실시간 모니터링 완료 반영)**

**작성자**: Claude Code  
**검토**: 사용자 요청에 따라 모든 내용 상세 기재  
**최종 업데이트**: 118개 테스트 통과, 실시간 로그 스트리밍 구현 완료