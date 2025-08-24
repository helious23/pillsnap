# 📊 Data Quality Utilities

PillSnap ML 데이터 품질 점검 및 수정 유틸리티 모음

## 🚀 Quick Start

### 1. 전체 파이프라인 실행 (권장)

```bash
cd /home/max16/pillsnap/scripts/data_quality

# Dry run (변경사항 미리보기)
./run_all_fixes.sh --dry-run

# 실제 실행 (변경사항 적용)
./run_all_fixes.sh --execute
```

### 2. 개별 유틸리티 실행

```bash
# Python 환경 활성화
source /home/max16/pillsnap/.venv/bin/activate

# 각 유틸리티 실행 (dry-run이 기본)
python clean_corrupted_files.py
python fix_val_only_classes.py
python balance_combination_ratio.py
python calculate_class_weights.py
python final_quality_check.py
```

## 📋 유틸리티 목록

### 1️⃣ **clean_corrupted_files.py** (최우선)
손상된 이미지 파일 검출 및 제거

```bash
# Dry run
python clean_corrupted_files.py

# 실제 적용
python clean_corrupted_files.py --no-dry-run

# 병렬 처리 workers 조정
python clean_corrupted_files.py --max-workers 16
```

**효과:**
- 학습 안정성 즉시 향상
- K-001900 등 알려진 손상 파일 제거
- 블랙리스트 자동 관리

### 2️⃣ **fix_val_only_classes.py**
Val에만 있는 클래스 처리

```bash
# Val-only 클래스 제거 (기본)
python fix_val_only_classes.py --mode remove

# Train에 추가하는 방식
python fix_val_only_classes.py --mode add-to-train --max-per-class 5

# 실제 적용
python fix_val_only_classes.py --mode remove --no-dry-run
```

**효과:**
- Val accuracy +0.5~1% 개선
- Train/Val 클래스 일치성 확보

### 3️⃣ **balance_combination_ratio.py**
Single/Combination 비율 조정

```bash
# Combination 25%로 오버샘플링 (기본)
python balance_combination_ratio.py --target-ratio 0.25

# 언더샘플링 방식
python balance_combination_ratio.py --target-ratio 0.2 --strategy undersample

# 혼합 전략
python balance_combination_ratio.py --strategy mixed --target-ratio 0.25
```

**효과:**
- Detection mAP +10~15% 개선
- Two-Stage Pipeline 성능 향상

### 4️⃣ **calculate_class_weights.py**
클래스 불균형 가중치 계산

```bash
# Balanced weights (기본)
python calculate_class_weights.py --method balanced

# Effective number 방식
python calculate_class_weights.py --method effective --beta 0.999

# Square root (moderate)
python calculate_class_weights.py --method sqrt --clip-max 5.0
```

**효과:**
- 과적합 5~10%p 감소
- 희귀 클래스 성능 개선

### 5️⃣ **final_quality_check.py**
최종 품질 종합 검증

```bash
# 현재 manifest 검사
python final_quality_check.py

# 수정된 manifest 검사
python final_quality_check.py \
  --train-manifest artifacts/stage3/manifest_train.cleaned.csv \
  --val-manifest artifacts/stage3/manifest_val.cleaned.csv
```

## 📊 공통 옵션

모든 유틸리티가 지원하는 공통 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dry-run` | True | 실제 변경 없이 리포트만 생성 |
| `--no-dry-run` | - | 실제 변경 적용 |
| `--backup` | True | 원본 파일 백업 생성 |
| `--no-backup` | - | 백업 생성 안 함 |
| `--out-suffix` | .cleaned | 출력 파일 접미사 |
| `--train-manifest` | auto | Train manifest 경로 |
| `--val-manifest` | auto | Val manifest 경로 |
| `--verbose` | False | 상세 로그 출력 |

## 🔄 권장 실행 순서

1. **초기 검사**: `final_quality_check.py`
2. **손상 파일 정리**: `clean_corrupted_files.py`
3. **Val-only 처리**: `fix_val_only_classes.py`
4. **비율 조정**: `balance_combination_ratio.py`
5. **가중치 계산**: `calculate_class_weights.py`
6. **최종 검증**: `final_quality_check.py`

또는 통합 스크립트 사용:
```bash
./run_all_fixes.sh --execute
```

## 📁 출력 파일 위치

- **수정된 Manifest**: `artifacts/stage3/manifest_*.{cleaned,remove,balanced}.csv`
- **리포트**: `artifacts/data_quality_reports/`
- **가중치 파일**: `artifacts/data_quality_reports/class_weights_*.{json,npy}`
- **블랙리스트**: `artifacts/data_quality_reports/blacklist.txt`
- **로그**: `artifacts/data_quality_reports/*_*.log`

## 💡 학습 스크립트 통합

### 1. 수정된 Manifest 사용

```python
# train_stage3_two_stage.py에서
train_manifest = "artifacts/stage3/manifest_train.balanced_oversample.csv"
val_manifest = "artifacts/stage3/manifest_val.balanced_oversample.csv"
```

### 2. 클래스 가중치 적용

```python
import json
import numpy as np
import torch

# 가중치 로드
weights = json.load(open("artifacts/data_quality_reports/class_weights_balanced_*.json"))
# 또는
weight_array = np.load("artifacts/data_quality_reports/class_weights_balanced_*.npy")

# Loss function에 적용
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight_array))
```

### 3. 샘플러 설정 적용

```python
# DataLoader에서
sampler_config = json.load(open("artifacts/data_quality_reports/sampler_config_*.json"))

if sampler_config['type'] == 'WeightedRandomSampler':
    weights = sampler_config['weights']
    # WeightedRandomSampler 구현
```

## 🎯 예상 성능 개선

현재 Stage 3 기준 (Val Top-1: 39.1%):

| 개선 사항 | 예상 효과 |
|-----------|-----------|
| 손상 파일 제거 | 학습 안정성 |
| Val-only 클래스 제거 | +0.5~1% |
| Combination 25% | Detection +10~15% |
| 클래스 가중치 | 과적합 -5~10%p |
| **종합** | **Val Top-1: 45~50%** |

## ⚠️ 주의사항

1. **항상 dry-run 먼저 실행**하여 변경사항 확인
2. **백업 확인**: 원본 manifest는 자동 백업되지만 확인 필요
3. **순차 실행**: 각 단계의 출력이 다음 단계 입력이 됨
4. **디스크 공간**: 오버샘플링 시 manifest 크기 증가 고려

## 🐛 문제 해결

### "Module not found" 에러
```bash
# Python 경로 추가
export PYTHONPATH=/home/max16/pillsnap/scripts/data_quality:$PYTHONPATH
```

### 메모리 부족
```bash
# Worker 수 줄이기
python clean_corrupted_files.py --max-workers 4
```

### 권한 에러
```bash
chmod +x *.py *.sh
```

## 📞 Support

문제 발생 시 다음 정보와 함께 리포트:
- 실행한 명령어
- 에러 메시지
- 로그 파일 (`artifacts/data_quality_reports/*.log`)

---

*Generated by PillSnap ML Data Quality Pipeline v1.0*