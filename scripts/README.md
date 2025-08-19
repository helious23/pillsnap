# PillSnap ML Scripts 사용법

## 가상환경 Python 실행

### 기본 사용법
```bash
# Python 명령어 실행
./scripts/python_safe.sh [Python 명령어와 인수들]

# 예시
./scripts/python_safe.sh --version
./scripts/python_safe.sh -c "print('Hello PillSnap!')"
./scripts/python_safe.sh -m pytest tests/ -v
./scripts/python_safe.sh -m pip install numpy
```

### 자주 사용하는 명령어들

#### 테스트 실행
```bash
# 전체 테스트
./scripts/python_safe.sh -m pytest

# 특정 테스트 파일
./scripts/python_safe.sh -m pytest tests/unit/test_dataloaders_strict_validation.py -v

# 특정 테스트 함수
./scripts/python_safe.sh -m pytest tests/unit/test_dataloaders_strict_validation.py::TestSinglePillDatasetHandlerStrictValidation::test_getitem_error_handling_robustness -v
```

#### 패키지 관리
```bash
# 패키지 설치
./scripts/python_safe.sh -m pip install [패키지명]

# 설치된 패키지 목록
./scripts/python_safe.sh -m pip list

# requirements.txt 설치
./scripts/python_safe.sh -m pip install -r requirements.txt
```

#### 스크립트 실행
```bash
# 프로젝트 스크립트 실행
./scripts/python_safe.sh -m src.train
./scripts/python_safe.sh -m src.evaluate
./scripts/python_safe.sh -m src.infer
```

## 별칭 설정 (선택사항)

더 짧은 명령어를 원한다면:
```bash
# 별칭 설정 로드
source scripts/setup_aliases.sh

# 사용법
pp --version                    # Python 실행
ptest tests/ -v                 # pytest 실행  
ppip install numpy              # pip 실행
```

## 환경 활성화 (전체 환경)

전체 환경을 활성화하려면:
```bash
source scripts/env/activate_environment.sh
```

이후 일반적인 `python`, `pytest` 명령어 사용 가능

## 문제 해결

### 가상환경을 찾을 수 없다는 오류
```bash
# 가상환경 경로 확인
ls -la /home/max16/pillsnap/.venv/bin/python

# 가상환경 재생성이 필요한 경우
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 권한 오류
```bash
# 스크립트 실행 권한 부여
chmod +x scripts/python_safe.sh
chmod +x scripts/setup_aliases.sh
```