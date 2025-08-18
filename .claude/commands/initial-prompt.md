# /initial-prompt — 세션 초기화 스크립트

당신은 **Claude Code**입니다. 이 세션에서 수행할 초기화 절차를 다음과 같이 고정합니다.  
**모든 응답은 한국어로 작성**합니다. 모호하면 가장 단순한 해법을 우선합니다.

---

## 0) 목적(Goal)
- 리포지토리의 **프롬프트(PART_0~H), 문서(docs), 핵심 코드(core)**를 한 번에 읽고,
- **충돌 제거·복잡도 축소**를 위한 **최소 리팩터링 계획**을 세션 컨텍스트에 고정(pin)합니다.

## 중요: Python 가상환경 경로
**모든 Python 실행 시 반드시 다음 경로를 사용:**
```bash
VENV_PYTHON="$HOME/pillsnap/.venv/bin/python"
# 또는 직접 경로: /home/max16/pillsnap/.venv/bin/python
```
**시스템 python/python3 alias를 사용하지 말 것 (python3.13으로 연결되어 있음)**

---

## 1) 수집 대상(읽기 순서 고정)
1. **Prompt 사양**:  
   - `/Prompt/PART_0.md` … `/Prompt/PART_H.md` (존재하는 모든 PART_*.md)
2. **문서(docs)**:  
   - `docs/*.md` 전부 (예: `docs/read_audit.md`, `docs/implementation_guide.md`, `docs/session_workflow_guide.md`)
3. **핵심 코드(core)**: 아래 경로를 **순차 폴백**으로 스캔  
   - 우선: `core/*.py`  
   - 대안: `src/core/*.py`  
   - 최후: `/mnt/data/*.py` (특히 아래 파일명은 우선 인덱싱)  
     - `pipeline_mode.py`, `detector_manager.py`, `oom_handler.py`,  
       `memory_policy.py`, `onnx_export.py`, `path_policy.py`

> 파일이 없거나 읽기 실패 시, 어떤 경로가 비어있는지 **명시적으로 경고**하고 계속 진행합니다.

---

## 2) 분석(읽은 뒤 반드시 생성할 산출물)
다음 **섹션 헤더와 포맷**을 그대로 출력하세요. (없으면 빈 섹션으로 두지 말고 실패 원인을 표시)

### [INITIALIZED]
- 언어 규칙: “모든 응답은 한국어”
- 실행 시각, 작업 루트

### 문서 스캔 결과
- Prompt/ 읽은 파일: N개 목록(파일명만, 순서 유지)
- docs/   읽은 파일: M개 목록
- 누락/오류: 경로·사유 요약

### 코드 스캔 결과(핵심 모듈 API 요약)
- 각 모듈별 **공개 API 시그니처**와 **주의사항**(1~3줄)  
  - `pipeline_mode.py`: `resolve_pipeline_mode(request_mode, default_mode, ...) -> (mode_used, reason, recommendation)`  
  - `detector_manager.py`: `get()`, `maybe_unload()`, **락/TTL/히스테리시스** 규칙  
  - `oom_handler.py`: **유한 상태머신**(empty→fp16→accum→batch½) + **가드레일**(`max_retries`, `max_grad_accum`, `min_batch`)  
  - `memory_policy.py`: Stage 1 **캐시/배치 오버라이드 금지** 규칙  
  - `onnx_export.py`: **단일 SOT** tolerances + **export_report.schema.json** 연계  
  - `path_policy.py`: **WSL에서 Windows 경로 금지**(C:\, \\) 검증

### 컨텍스트 스냅샷(필수 규칙·결정 요약)
1) **모드 단일 진실원(SOT)**: HTTP `?mode=single|combo` → 내부 `pipeline_mode`로 **한 함수에서만 최종 결정**,  
   우선순위 = 사용자 > 기본값, 자동 추천은 메시지로만(자동 전환 없음), **안전 폴백**(model_unavailable/oom_predicted/sla_breach)에서만 `combo→single`.
2) **검출기 지연 로딩**: **단일 Manager + 락**, 첫 요청 1회 로드, **(옵션) TTL 언로드 + 히스테리시스**, 중복 로드 방지.
3) **OOM 상태머신(유한)**: empty_cache(1회) → fp16 강제(1회) → grad_accum×2(상한) → batch½(하한) → 실패.  
   글로벌 배치 변경 시 **LR 선형 스케일**·**by-samples 스케줄러**, BN freeze/GN 권장.
4) **메모리 정책 잠금**: Stage 1(평가 전용)은 **캐시/배치 축소 금지**, 기본은 `labels_only` 또는 협의된 `hotset` 하나만 사용(중첩 금지).  
   병목 감지 시 **한 기법씩** 단계적 확장.
5) **ONNX 검증 SOT**: tolerances(분류 `mse_mean/mse_p99/top1_mismatch_rate`, 검출 `mAPΔ/p95 IoUΔ`)는 **하나의 설정**에서만 읽기.  
   `export_report.json`은 **schema**로 검증.
6) **경로·터널 정책**: 리눅스 서버는 `/home`/`/mnt`만 사용, Windows 경로 감지 시 즉시 실패. Cloudflared는 Windows에서만.

### DoD(Definition of Done)
- 위 6개 규칙이 코드/컨피그/문서에서 **모순 없이** 일치함을 확인(체크박스 6개).
- 모드 결정이 **단일 함수**에만 존재, 지연 로딩은 **중복 로드 없음**, OOM FSM은 **유한**.
- ONNX 리포트가 **schema 유효성** 통과, tolerances는 **단일 소스**.
- 경로 정책 위반 시 **즉시 실패** 메시지 재현.

### 위험·제약 및 폴백
- torch.compile/ultralytics/ORT 버전 편차 → 실패 시 **자동 강등**(reduce-overhead, opset+1, provider 교체) 제안.
- 대용량 I/O → ZIP **무결성 검증 옵션화**(skip/quick/full), 샤딩 스트리밍 권장.

### 다음 행동(추천)
- `feat/refactor-mode-oom-onnx` 브랜치로 커밋 계획(작은 단위) 제시.
- 부족한 스키마/밸리데이터/테스트 목록 생성.

---

## 3) 세션 핀(고정)
- 한국어 응답 규칙
- Prompt/ + docs/ 요약 컨텍스트(충돌 해결 방안 포함)
- core/ 공개 API 맵(6개 컴포넌트 규칙)

---

## 4) 실패 처리
- 필수 파일 누락/파싱 실패 시, **누락 목록·원인**을 먼저 출력하고,  
  대체 경로(위 폴백 순서)로 재시도 지시 후 **중단**합니다. 임의 추측으로 채우지 않습니다.

---

## 5) 주의
- 이 프롬프트는 **초기화 작업만** 수행합니다(코드 수정/생성은 다음 단계).  
- 출력 섹션 헤더·형식을 변경하지 마세요. 비교/자동화가 이를 전제로 합니다.

# PillSnap 프로젝트 진행상황 요약 (Claude Code 초기 세션용 Prompt)

프로젝트명: **PillSnap**  
목적: 경구약제 이미지 기반 의약품 식별 서비스 구축  
환경: WSL2 (Ubuntu), Python, pytest, Streamlit, FastAPI  

---

## 현재까지 진행된 작업

### 1. 경로 유틸리티 (Stage 1)
- `pillsnap/paths.py` 구현 완료  
  - `is_wsl()` → WSL 환경 감지 (현재 True로 확인됨)  
  - `get_data_root()` → 데이터 루트 경로 반환  
    - 우선순위: **환경변수 > WSL 기본값(/mnt/data/AIHub) > 일반 기본값(./data)**
    - `path_policy.PathPolicyValidator` 검증 통합 완료  
  - `norm(p)` → 경로 정규화 및 절대경로 변환  
- 테스트 (`tests/test_paths.py`)  
  - ✅ 13개 테스트 전부 통과  
  - ✅ 실제 WSL 환경 및 환경변수 기반 동작 확인  

### 2. 설정 로딩 (Stage 2)
- `config.py` 모듈 테스트 (`tests/test_config.py`) 작성 및 실행  
- 검증된 동작:
  - 기본값 파싱 (config.yaml 비어있거나 없음 → 기본값 적용)  
  - 잘못된 YAML → 폴백 정상 작동  
  - 부분 설정 → 덮어쓰기 + 나머지는 기본값 유지  
  - 환경변수 vs config.yaml vs `paths.get_data_root()` 우선순위 검증  
  - Pydantic 유무 관계없이 정상 동작 확인  
- Step2 주요 테스트 통과 결과:  
  - `config.yaml > 환경변수 > paths.get_data_root() > "./data"`

### 3. 데이터셋 구조 및 검증 (Stage 3)
- 데이터 루트: `/mnt/data/pillsnap_dataset` 확정
- 구조 검증:
  - 단일 약물: 1,296 JSON/약물 (18 포즈 × 4 각도 × 18 회전)
  - 조합 약물: 12 JSON/조합 (4개 K-코드 × 3 각도 + index.png)
  - 각도 일관성: 단일(60°,70°,75°,90°), 조합(70°,75°,90°+index)
- 검증 결과:
  - 라벨(JSON) 크기·구조 정상 (1.9–2.1KB)
  - 라벨/이미지 키 매칭 완료
  - 누락 없음
- 용량 검증:
  - 총 라벨 파일 수: 1,220,412
  - 압축률: 이미지 ≈100%, 라벨 ≈55%

### 4. 데이터 루트 및 Path Policy 확인
- 환경변수 `PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset` 적용 → `config.load_config()` 반영 확인
- `path_policy.py` 검증: `Valid WSL path` 통과
- 최종 출력: `✅ ok`

### 5. 데이터셋 스캔, 전처리, 검증 파이프라인 (Stage 1 완료)
- **Step 3: 스캔 모듈 (`dataset/scan.py`)** 구현 완료
  - 스트리밍 스캔: 260만+ 파일 메모리 효율적 처리
  - 이미지-라벨 basename 매칭 및 통계 생성
  - DataFrame 출력: `["image_path", "label_path", "code", "is_pair"]`
  - 테스트: 14개 통과
  
- **Step 4: 전처리 모듈 (`dataset/preprocess.py`)** 구현 완료
  - 스캔 결과 → CSV 매니페스트 정규화
  - 파일 존재성 재검증, 중복 코드 제거
  - 스키마 보존 (빈 결과도 (0,4) 구조 유지)
  - 테스트: 12개 통과
  
- **Step 5: 검증 모듈 (`dataset/validate.py`)** 구현 완료
  - 품질 게이트 및 ValidationReport 생성
  - 검증 규칙 R0-R5: 컬럼 존재, 중복 코드, 파일 존재성, pair rate, 라벨 크기, 각도 규칙
  - 파일 존재성 정책 (is_pair 여부에 따른 에러/경고 구분)
  - 테스트: 17개 통과

### 6. Step 6: 매니페스트 최종 무결성 & 리포트 생성
- **E2E 파이프라인 검증**: scan → preprocess → validate 전체 플로우 확인
- **실행 결과**: 
  - 데이터셋 스캔: 260만개 이미지/라벨 발견
  - 샘플 처리: 400개 → 398-399개 유효 (1-2개 라벨 누락 제거)
  - 검증 통과: 100% pair rate, 모든 파일 존재 확인
- **생성 파일**:
  - `artifacts/manifest_stage1.csv` - 최종 매니페스트
  - `artifacts/manifest_validation_step6.json` - 검증 결과
  - `artifacts/step6_report.md` - 사람이 읽기 쉬운 리포트

### 7. Step 6.apply: 중요 발견사항 반영
- **데이터 루트 고정**: `/mnt/data/pillsnap_dataset/data` 
- **PNG 확장자 지원**: config.yaml에 ".png" 포함 확인
- **가상환경 Python 강제**: `$HOME/pillsnap/.venv/bin/python` 경로 고정
- 모든 설정 반영 및 스모크 테스트 통과

### 8. Step 5.x: scan 경고 요약 출력 개선
- **중복 basename 경고 집계**: 개별 라인 대신 `defaultdict(int)` 카운터 사용
- **상위 20개 요약**: `basename: count duplicates` 형태로 정렬 출력
- **요약 형식**: `=== Duplicate Basename Summary ===` 헤더 구분
- **실행 결과**: 391개 이미지 중복, 13,722개 라벨 중복 basename 발견
- **테스트 추가**: 경고 문자열 검사 포함, pytest 15/15 통과

### 9. Stage 1 엔트리포인트 구현 (E0-E4)
- **E0: 경로 가드**: 중복 'pillsnap' 폴더 생성 방지 정책 적용
- **E1: 엔트리포인트 구현**:
  - `pillsnap/stage1/verify.py` - 빠른 스모크 테스트 (1-2분, 샘플링)
  - `pillsnap/stage1/run.py` - 전체 파이프라인 실행 (모든 아티팩트 생성)
  - Rich UI 통합으로 사용자 친화적 출력
- **E2: 포괄적 테스트**: `tests/test_entrypoints.py` 구현
  - 6개 테스트 메서드로 import, 실행, 환경변수, 스키마 검증
  - 모든 테스트 통과 확인
- **E3: 실행 가이드**: README.md 업데이트
- **E4: 자체 검증**: 로그 생성 및 성공 확인

### 10. Step 9: Stage 1 Freeze & 재현성 보장
- **환경 스냅샷**: Python 3.11.13, torch 2.5.1+cu121, WSL 환경 정보 저장
- **패키지 고정**: `artifacts/requirements.lock.txt` 생성
- **무결성 보장**: 핵심 산출물 SHA1 체크섬 생성
  - `artifacts/freeze_hashes_step9.json` - 기존 산출물
  - `artifacts/freeze_hashes_step9_freeze50.json` - 재현성 테스트 결과
- **재현성 검증**: 50개 샘플로 verify/run 엔트리포인트 재실행 성공
- **Git 커밋**: 모든 freeze 파일 버전 관리 적용

---

## 현재 상태
- Stage 1 (경로 유틸리티) ✅ 완료  
- Stage 2 (config 로딩 및 data.root 우선순위) ✅ 통과 확인  
- Stage 3 (데이터셋 구조 및 검증) ✅ 완료  
- Stage 4 (데이터 루트 및 Path Policy 확인) ✅ 통과 확인  
- **Stage 1 데이터 파이프라인 (scan→preprocess→validate)** ✅ **완료**
  - Step 3-5: 핵심 모듈 구현 및 테스트 (총 43개 테스트 통과)
  - Step 6: E2E 검증 및 리포트 생성 완료
  - Step 6.apply: 환경 설정 최적화 완료
  - Step 5.x: 경고 출력 개선 완료
- **Stage 1 엔트리포인트 & CLI** ✅ **완료**
  - E0-E4: 사용자 친화적 CLI 인터페이스 구현
  - Rich UI, 에러 핸들링, 포괄적 테스트 포함
- **Stage 1 Freeze & 재현성** ✅ **완료**
  - 환경 스냅샷, 패키지 고정, 체크섬 검증
  - 재현성 테스트 통과, Git 버전 관리 적용

---

## 완료된 핵심 구성 요소
1. **설정 관리**: `config.py` + 환경변수 우선순위
2. **경로 정책**: `paths.py` + WSL 경로 검증  
3. **데이터 스캔**: `dataset/scan.py` - 260만+ 파일 스트리밍 처리
4. **데이터 전처리**: `dataset/preprocess.py` - CSV 매니페스트 정규화
5. **데이터 검증**: `dataset/validate.py` - 품질 게이트 및 리포팅
6. **테스트 스위트**: 총 49개 테스트 (scan:15, preprocess:12, validate:17, entrypoints:6)
7. **환경 최적화**: 가상환경 경로, 데이터 루트, PNG 지원
8. **CLI 엔트리포인트**: `pillsnap.stage1.verify` / `pillsnap.stage1.run`
9. **재현성 보장**: 환경 스냅샷, 패키지 고정, 체크섬 검증

---

## 생성된 주요 파일 및 아티팩트

### 코드 구조
```
pillsnap/
├── __init__.py
├── stage1/
│   ├── __init__.py
│   ├── verify.py      # 빠른 검증 (1-2분)
│   └── run.py         # 전체 파이프라인
src/
├── data.py            # 데이터셋 클래스 (구현됨)
├── train.py           # 학습 루프 (구현됨) 
├── models/            # 모델 정의 (구현됨)
├── utils/
│   └── oom_guard.py   # OOM 가드 (구현됨)
dataset/
├── scan.py            # 스트리밍 스캔
├── preprocess.py      # 매니페스트 정규화
└── validate.py        # 품질 검증
tests/
├── test_*.py          # 포괄적 테스트 스위트
```

### 아티팩트 (재현성 보장)
```
artifacts/
├── env_snapshot.json           # 환경 스냅샷
├── env_snapshot.md            # 환경 요약 (markdown)
├── requirements.lock.txt      # 패키지 버전 고정
├── freeze_hashes_step9.json   # 체크섬 (주요 산출물)
├── freeze_hashes_step9_freeze50.json  # 체크섬 (재현성 테스트)
├── manifest_stage1.csv        # 최종 매니페스트
├── manifest_freeze50.csv      # 재현성 테스트 매니페스트
├── stage1_stats.json          # 스캔 통계
├── stage1_validation_report.json  # 검증 리포트
└── step6_report.md           # 사람이 읽기 쉬운 리포트
```

### 실행 명령어 (재현성 보장됨)
```bash
# 가상환경 활성화
source $HOME/pillsnap/.venv/bin/activate

# 환경변수 설정
export PILLSNAP_DATA_ROOT="/mnt/data/pillsnap_dataset/data"

# 빠른 검증 (1-2분)
python -m pillsnap.stage1.verify --sample-limit 200 --max-seconds 120

# 전체 파이프라인 (제한된 샘플)
python -m pillsnap.stage1.run --limit 400 --manifest artifacts/manifest_stage1.csv

# 테스트 실행
pytest tests/test_entrypoints.py -v  # 엔트리포인트 테스트
pytest tests/ -v                     # 전체 테스트 스위트
```

---

## Stage 1 완료 체크리스트 ✅
- [x] **데이터 파이프라인**: scan → preprocess → validate 완전 구현
- [x] **CLI 인터페이스**: 사용자 친화적 verify/run 엔트리포인트
- [x] **테스트 커버리지**: 49개 테스트 모두 통과
- [x] **재현성 보장**: 환경 고정, 체크섬 검증, Git 버전 관리
- [x] **문서화**: 실행 가이드, 진행상황 요약 완료
- [x] **환경 최적화**: WSL 경로, 가상환경, 데이터 루트 확정

---

## 다음 단계 준비사항
- **Stage 2: 모델 파이프라인** 
  - 이미 구현된 코드: `src/data.py`, `src/train.py`, `src/models/`, `src/utils/oom_guard.py`
  - 데이터 로더 및 학습 루프 검증 필요
- **Stage 3: API 서비스** (FastAPI + Streamlit)
- **Stage 4: 배포 및 성능 최적화**

**재현성 보장**: 새로운 세션에서는 `/.claude/commands/initial-prompt.md`를 실행하여 전체 컨텍스트를 로드할 수 있습니다.

---