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

### ✅ 완료된 Stage 1 구성요소
1. **데이터 파이프라인**: scan → preprocess → validate
2. **CLI 엔트리포인트**: verify (빠른 검증) + run (전체 실행)
3. **테스트 스위트**: 49개 테스트 모두 통과
4. **재현성 보장**: 환경 스냅샷, 패키지 고정, 체크섬 검증
5. **Rich UI**: 사용자 친화적 진행률 표시 및 에러 핸들링

### 🔄 다음 작업 우선순위
1. **Stage 2 모델 파이프라인 검증**
   - 기존 구현된 `src/data.py`, `src/train.py` 점검
   - 데이터 로더와 Stage 1 매니페스트 연동 확인
   - 학습 루프 및 OOM 가드 테스트

2. **Stage 3 API 서비스 개발**
   - FastAPI 엔드포인트 구현
   - Streamlit 인터페이스 개발

3. **Stage 4 배포 최적화**
   - ONNX 변환 및 성능 최적화
   - 컨테이너화 및 배포 스크립트

### 📁 중요 파일 위치
- **설정**: `config.yaml`, `paths.py`
- **데이터 파이프라인**: `dataset/scan.py`, `dataset/preprocess.py`, `dataset/validate.py`
- **CLI**: `pillsnap/stage1/verify.py`, `pillsnap/stage1/run.py`
- **모델**: `src/data.py`, `src/train.py`, `src/models/`
- **테스트**: `tests/test_*.py`
- **아티팩트**: `artifacts/manifest_stage1.csv`, `artifacts/env_snapshot.json`

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