#!/bin/bash
# PillSnap ML 프로젝트 별칭 설정
# 사용법: source scripts/core/setup_aliases.sh

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 PillSnap ML 별칭 설정 중...${NC}"

# Python 실행 별칭들
alias pillsnap-python='./scripts/core/python_safe.sh'
alias pp='./scripts/core/python_safe.sh'  # 짧은 별칭
alias ptest='./scripts/core/python_safe.sh -m pytest'
alias ppip='./scripts/core/python_safe.sh -m pip'

# 자주 사용하는 명령어들
alias pillsnap-env='source scripts/env/activate_environment.sh'
alias pillsnap-test='./scripts/core/python_safe.sh -m pytest'
alias pillsnap-lint='./scripts/core/python_safe.sh -m ruff check'
alias pillsnap-format='./scripts/core/python_safe.sh -m black'

echo -e "${GREEN}✅ 별칭 설정 완료!${NC}"
echo -e "${GREEN}사용 가능한 별칭:${NC}"
echo -e "${GREEN}  pp [명령어]         # Python 실행${NC}"
echo -e "${GREEN}  ptest [옵션]        # pytest 실행${NC}"
echo -e "${GREEN}  ppip [명령어]       # pip 실행${NC}"
echo -e "${GREEN}  pillsnap-python     # 전체 이름${NC}"
echo -e "${GREEN}  pillsnap-test       # 테스트 실행${NC}"
echo ""
echo -e "${BLUE}예시:${NC}"
echo -e "${BLUE}  pp --version${NC}"
echo -e "${BLUE}  ptest tests/unit/test_dataloaders_strict_validation.py -v${NC}"
echo -e "${BLUE}  ppip install numpy${NC}"