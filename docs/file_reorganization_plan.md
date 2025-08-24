# 📁 파일 재구성 계획

## 🎯 현재 상황

최근 생성된 파일들이 scripts 루트에 직접 위치하고 있어 정리가 필요합니다.

## 📋 재구성 대상 파일들

### 1. 평가 및 검증 도구
- `scripts/sanity_check.py` → `scripts/evaluation/sanity_check.py`
- `scripts/sanity_check_fixed.py` → `scripts/evaluation/sanity_check_fixed.py`

### 2. 백업 및 아카이빙 도구
- `scripts/freeze_stage3_results.py` → `scripts/backup/freeze_stage_results.py`

### 3. Detection 관련 도구
- `scripts/test_detection_state.py` → `scripts/testing/test_detection_state.py`
- `scripts/tune_detection_precision.py` → `scripts/optimization/tune_detection_precision.py`

### 4. 데이터 준비 도구
- `scripts/create_yolo_label_symlinks.py` → `scripts/data_prep/create_yolo_label_symlinks.py`

### 5. 유틸리티 클래스들 (src로 이동)
- `src/utils/detection_state_manager.py` ✅ (이미 올바른 위치)
- `src/utils/robust_csv_parser.py` ✅ (이미 올바른 위치)

## 🗂️ 제안하는 디렉토리 구조

```
scripts/
├── backup/                          # 백업 및 아카이빙
│   ├── freeze_stage_results.py      # Stage별 결과 동결
│   └── create_experiment_card.py    # 실험 카드 생성
│
├── evaluation/                      # 평가 및 검증
│   ├── sanity_check.py             # 기본 평가 스크립트
│   ├── sanity_check_fixed.py       # 개선된 평가 스크립트
│   └── evaluate_model.py           # 통합 평가 도구
│
├── optimization/                    # 최적화 및 튜닝
│   ├── tune_detection_precision.py # Detection 파라미터 튜닝
│   ├── tune_classification.py      # Classification 튜닝
│   └── hyperparameter_search.py    # 하이퍼파라미터 탐색
│
├── data_prep/                      # 데이터 준비
│   ├── create_yolo_label_symlinks.py
│   ├── prepare_manifests.py
│   └── validate_dataset.py
│
├── testing/                        # 테스트 실행
│   ├── test_detection_state.py    # Detection State 테스트
│   ├── run_stage*_test_suite.py   # Stage별 테스트
│   └── integration_tests.py       # 통합 테스트
│
├── monitoring/                     # 모니터링 (기존 유지)
├── stage1/                        # Stage 1 (기존 유지)
├── stage2/                        # Stage 2 (기존 유지)
├── stage3/                        # Stage 3 (기존 유지)
├── stage4/                        # Stage 4 (신규 생성)
│   ├── prepare_stage4_data.py
│   └── train_stage4_production.sh
│
├── core/                          # 핵심 유틸리티 (기존 유지)
└── utils/                         # 기타 유틸리티
    └── reorganize_scripts.py     # 스크립트 정리 도구
```

## 🔧 재구성 명령어

```bash
# 1. 새 디렉토리 생성
mkdir -p scripts/{backup,evaluation,optimization,data_prep,stage4}

# 2. 파일 이동
# 평가 도구
mv scripts/sanity_check.py scripts/evaluation/
mv scripts/sanity_check_fixed.py scripts/evaluation/

# 백업 도구
mv scripts/freeze_stage3_results.py scripts/backup/freeze_stage_results.py

# 최적화 도구
mv scripts/tune_detection_precision.py scripts/optimization/

# 데이터 준비
mv scripts/create_yolo_label_symlinks.py scripts/data_prep/

# 테스트
mv scripts/test_detection_state.py scripts/testing/
```

## 📝 코드 컨벤션 준수 사항

### 1. 파일 명명 규칙
- **스네이크 케이스**: `file_name.py`
- **동사로 시작**: `run_`, `test_`, `prepare_`, `train_`
- **명확한 목적 표현**: `freeze_stage3_results.py` → `freeze_stage_results.py` (재사용 가능)

### 2. 클래스 명명 규칙
- **파스칼 케이스**: `DetectionStateManager`
- **명사형**: Manager, Parser, Evaluator, Trainer

### 3. 함수 명명 규칙
- **스네이크 케이스**: `load_state()`, `parse_csv()`
- **동사로 시작**: `get_`, `set_`, `update_`, `validate_`

### 4. 모듈 위치
- **유틸리티 클래스**: `src/utils/`
- **실행 스크립트**: `scripts/`
- **테스트 코드**: `tests/`
- **문서**: `docs/`

## 🎯 재구성 후 장점

1. **명확한 구조**: 기능별로 분류되어 찾기 쉬움
2. **재사용성**: Stage별 도구를 범용화
3. **유지보수**: 관련 파일들이 모여있어 관리 용이
4. **확장성**: 새로운 Stage나 기능 추가 시 구조 확장 쉬움

## ⚠️ 주의사항

### Import 경로 수정 필요
재구성 후 다음 파일들의 import 경로 확인 및 수정:
- Stage 3 학습 스크립트
- 테스트 파일들
- 문서의 명령어 예시

### Git 이력 보존
```bash
# git mv 사용으로 이력 보존
git mv scripts/sanity_check.py scripts/evaluation/
```

### 심볼릭 링크 생성 (선택)
호환성을 위해 임시로 심볼릭 링크 생성:
```bash
ln -s evaluation/sanity_check_fixed.py scripts/sanity_check_fixed.py
```

---

*작성일: 2025-08-24*
*작성자: Claude Code*