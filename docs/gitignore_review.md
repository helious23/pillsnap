# 📋 .gitignore 검토 보고서

## ✅ 검토 완료 (2025-08-24)

`.gitignore` 파일이 GitHub 저장소에 불필요한 대용량 파일들이 올라가지 않도록 적절히 설정되었습니다.

## 🚫 무시되는 주요 파일/디렉토리

### 1. 대용량 모델 파일 (3.4GB+)
- `*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.pkl`
- `artifacts/stage*/checkpoints/` (3.4GB)
- `artifacts/yolo/` (545MB)
- `frozen_experiments/` (백업된 체크포인트)
- `backup/`, `archive/` (수동 백업)

### 2. 데이터셋
- `/home/max16/pillsnap_data/` (외부 데이터)
- `dataset/` (로컬 데이터)
- `data/raw/`, `data/processed/`

### 3. 실험 추적 파일
- `wandb/`, `mlruns/`, `tensorboard/`
- `runs/` (YOLO 학습 결과)
- `events.out.tfevents.*`

### 4. 임시 파일
- `*.log`, `logs/`
- `tmp/`, `temp/`, `*.tmp`
- `__pycache__/`, `*.pyc`
- `.pytest_cache/`, `.coverage`

### 5. 환경 설정
- `.venv/`, `venv/`, `env/`
- `.env`, `.env.local`
- `config.local.yaml`

### 6. IDE 설정
- `.vscode/`, `.idea/`
- `*.swp`, `*.swo` (Vim)

## ✅ 유지되는 중요 파일

### GitHub에 포함되는 파일들:
```
artifacts/
├── manifests/*.csv         ✅ (Manifest 파일)
├── stage*/*.json           ✅ (설정 파일)
├── stage*/*.csv            ✅ (결과 요약)
├── logs/*.sh              ✅ (스크립트)
└── *.md                   ✅ (문서)

docs/                      ✅ (모든 문서)
scripts/                   ✅ (모든 스크립트)
src/                       ✅ (소스 코드)
tests/                     ✅ (테스트 코드)
*.md                       ✅ (README, CHANGELOG 등)
*.yaml, *.json             ✅ (설정 파일)
requirements.txt           ✅ (패키지 목록)
```

## 📊 용량 절약 효과

| 카테고리 | 크기 | 상태 |
|---------|------|------|
| 체크포인트 | 3.4GB | 🚫 무시됨 |
| YOLO 아티팩트 | 545MB | 🚫 무시됨 |
| 백업/아카이브 | ~2GB | 🚫 무시됨 |
| 데이터셋 | ~50GB | 🚫 무시됨 |
| **총 절약** | **~56GB** | ✅ |

## 🎯 권장사항

### 개발자가 해야 할 일:
1. **체크포인트 공유 필요시**: Google Drive나 Hugging Face Hub 사용
2. **데이터셋 공유**: 별도 스토리지 또는 DVC 사용
3. **실험 결과 공유**: WandB나 MLflow 사용

### 저장소 사용자가 해야 할 일:
1. 데이터셋은 별도 다운로드
2. Pretrained 모델은 스크립트로 자동 다운로드
3. 실험 재현시 manifest 파일 활용

## 📝 추가 개선사항

현재 `.gitignore`는 다음을 잘 처리합니다:
- ✅ Python 프로젝트 표준 파일
- ✅ ML/DL 대용량 파일
- ✅ 프로젝트별 아티팩트
- ✅ OS 및 IDE 파일
- ✅ 임시 및 캐시 파일

## 🔍 검증 명령어

```bash
# 무시되는 파일 확인
git status --ignored

# 특정 파일이 무시되는지 확인
git check-ignore -v [파일경로]

# 저장소 크기 확인 (무시 파일 제외)
git ls-files | xargs du -ch | tail -1

# 실제 디스크 사용량
du -sh .
```

---

*검토일: 2025-08-24*
*검토자: Claude Code*