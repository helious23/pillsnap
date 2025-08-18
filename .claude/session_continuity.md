# 세션 연속성 가이드

## 새로운 Claude Code 세션 시작 시 필수 실행

### 1. 초기화 명령어
```bash
/.claude/commands/initial-prompt.md
```

### 2. 환경 설정 확인
```bash
# 작업 디렉토리 확인
pwd  # 기대값: /home/max16/pillsnap

# 가상환경 활성화
source $HOME/pillsnap/.venv/bin/activate
python -V  # 기대값: Python 3.11.13

# 환경변수 설정
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset/data"

# 설정 로드 테스트
python - <<'PY'
import sys; sys.path.insert(0,'.')
import config
c = config.load_config()
print("data.root =", c.data.root)
assert c.data.root == "/mnt/data/pillsnap_dataset/data", "data.root mismatch"
print("✅ 환경 설정 정상")
PY
```

### 3. Stage 1 상태 확인
```bash
# 빠른 검증 (30초 이내)
python -m pillsnap.stage1.verify --sample-limit 10 --max-seconds 30

# 테스트 실행
pytest tests/test_entrypoints.py -v
```

## 현재 프로젝트 상태 (2025-08-18 기준)

### ✅ 완료된 Stage 1 구성요소 (Step 10-5까지)
1. **데이터 파이프라인**: scan → preprocess → validate
2. **CLI 엔트리포인트**: verify (빠른 검증) + run (전체 실행)
3. **테스트 스위트**: 49개 테스트 모두 통과
4. **재현성 보장**: 환경 스냅샷, 패키지 고정, 체크섬 검증
5. **Rich UI**: 사용자 친화적 진행률 표시 및 에러 핸들링
6. **CI/CD 구축**: GitHub Actions + pre-commit 훅 설정
7. **버전 관리**: v0.1-stage1 태그 + GitHub 릴리스

### ✅ 완료된 Step 11 Hotfix (JSON EDI 추출)
**핵심 문제 해결**: `code` 컬럼(파일 basename) ≠ `edi_code` (실제 EDI)

1. **preprocess.py 강화**: JSON 파싱하여 EDI 코드 및 메타데이터 추출
   - 새 컬럼: `mapping_code`, `edi_code`, `json_ok`, `drug_N`, `dl_name`, `drug_shape`, `print_front`, `print_back`
   - 빈 DataFrame 스키마 보존
   - EDI 누락률 경고 및 누락 샘플 저장

2. **클래스 맵 생성**: `pillsnap/stage1/utils.py`
   - `build_edi_classes()`: EDI → class_id 매핑 자동 생성
   - `validate_class_map()`: 클래스 맵 무결성 검증

3. **테스트 보강**: `tests/test_json_enrichment.py`
   - JSON 파싱, 클래스 맵 생성, 빈 DataFrame 처리 검증
   - 5개 테스트 모두 통과

4. **stage1.run 통합**: 파이프라인 실행 시 자동으로 클래스 맵 생성

**현재 산출물**:
- `artifacts/manifest_enriched.csv`: 풍부화된 매니페스트 (20개 샘플)
- `artifacts/classes_step11.json`: EDI → class_id 매핑 (19개 클래스)

### ✅ 완료된 Stage 2 학습 파이프라인 (Step 11-1)
**목적**: Stage 1 산출물 기반 EfficientNetV2-L 분류 학습

1. **패키지 구조**: `pillsnap/stage2/`
   - `__init__.py`: Stage 2 패키지 초기화
   - `dataset_cls.py`: EDI 기반 분류용 Dataset 클래스
   - `models.py`: EfficientNetV2-L 모델 팩토리 (timm → torchvision 폴백)
   - `train_cls.py`: 학습 스크립트 (AMP, 검증, 체크포인트 지원)

2. **PillsnapClsDataset 특징**:
   - `manifest_enriched.csv` + `classes_step11.json` 기반
   - EDI 코드 유효성 검증 및 클래스 매핑
   - 이미지 전처리 및 파일 존재성 체크
   - 19개 샘플 → 19개 유효 샘플 준비 완료

3. **EfficientNetV2-L 모델**:
   - 117M 파라미터, 447.3MB 모델 크기
   - 19개 EDI 클래스 분류 헤드
   - timm/torchvision 호환성 지원

4. **학습 스크립트 기능**:
   - train/val 자동 분할 (8:2)
   - AMP 지원, 체크포인트 저장 (best.pt, last.pt)
   - 배치 크기, limit, epochs 조정 가능
   - CPU/GPU 자동 감지 및 안전 동작

**현재 이슈**: RTX 5080 CUDA 호환성 문제 (sm_120 vs PyTorch sm_90 지원)
**해결방안**: CPU 실행 또는 PyTorch CUDA 버전 업그레이드 필요

### 🔄 즉시 다음 작업
1. **Stage 2 스모크 테스트 완료**
   - CPU에서 학습 파이프라인 검증
   - 체크포인트 저장 및 로드 테스트
   - 메트릭 수집 및 로깅 확인

2. **CUDA 호환성 해결**
   - PyTorch nightly 또는 CUDA 11.8 호환 버전 설치
   - 또는 CPU 기반 개발 환경 구축

3. **Stage 2 확장 개발**
   - 평가 스크립트 (`eval.py`) 구현
   - 추론 파이프라인 연동
   - 성능 벤치마크 및 최적화

### 📁 중요 파일 위치
- **설정**: `config.yaml`, `paths.py`
- **데이터 파이프라인**: `dataset/scan.py`, `dataset/preprocess.py`, `dataset/validate.py`
- **Stage 1 CLI**: `pillsnap/stage1/verify.py`, `pillsnap/stage1/run.py`, `pillsnap/stage1/utils.py`
- **Stage 2 학습**: `pillsnap/stage2/dataset_cls.py`, `pillsnap/stage2/models.py`, `pillsnap/stage2/train_cls.py`
- **기존 모델**: `src/data.py`, `src/train.py`, `src/models/`
- **테스트**: `tests/test_*.py`, `tests/test_json_enrichment.py`
- **아티팩트**:
  - `artifacts/manifest_stage1.csv` (기본 매니페스트)
  - `artifacts/manifest_enriched.csv` (JSON 파싱 포함)
  - `artifacts/classes_step11.json` (EDI 클래스 맵)
  - `artifacts/env_snapshot.json` (환경 스냅샷)

### 🔧 개발 환경
- **Python**: 3.11.13 (가상환경: `$HOME/pillsnap/.venv`)
- **플랫폼**: WSL2 Ubuntu
- **데이터**: `/mnt/data/pillsnap_dataset/data` (260만+ 파일)
- **GPU**: RTX 5080 16GB (PyTorch 2.5.1+cu121)

### 💡 개발 팁
- 모든 Python 실행 시 가상환경 경로 사용: `/home/max16/pillsnap/.venv/bin/python`
- WSL 절대 경로만 사용 (Windows 경로 금지)
- 환경변수 `PILLSNAP_DATA_ROOT` 항상 설정
- 체크섬 검증으로 파일 무결성 확인
- Rich UI로 사용자 친화적 출력 제공

### 📋 재현성 체크리스트
- [ ] 가상환경 활성화 확인
- [ ] 환경변수 설정 확인
- [ ] config.yaml 로드 테스트
- [ ] 데이터 루트 접근 가능 확인
- [ ] Stage 1 엔트리포인트 동작 확인
- [ ] 테스트 스위트 통과 확인

이 가이드를 따라하면 새로운 세션에서도 즉시 개발을 이어갈 수 있습니다.
