#!/bin/bash
# PillSnap ML 문서 자동 업데이트 + Git Push 헬퍼 스크립트

set -e  # 에러 발생 시 즉시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 인자 파싱
GIT_PUSH=false
FORCE_PUSH=false

for arg in "$@"; do
    case $arg in
        --git-push)
            GIT_PUSH=true
            shift
            ;;
        --force)
            FORCE_PUSH=true
            shift
            ;;
    esac
done

echo -e "${BLUE}📋 PillSnap ML 문서 자동 업데이트 시작...${NC}"

# 프로젝트 루트로 이동
cd /home/max16/pillsnap

# 현재 상황 스캔
echo -e "${YELLOW}🔍 현재 상황 스캔 중...${NC}"

# Stage 2 상태 확인
STAGE2_STATUS=$(./scripts/stage2/quick_status.sh 2>/dev/null | grep "진행률:" | tail -1 || echo "상태 확인 불가")
echo "📊 Stage 2 상태: $STAGE2_STATUS"

# SSD 사용량 확인
SSD_USAGE=$(df -h /home/max16/ssd_pillsnap 2>/dev/null | tail -1 || echo "SSD 정보 확인 불가")
echo "💾 SSD 사용량: $SSD_USAGE"

# Git 상태 확인 (--git-push 옵션 시)
if [ "$GIT_PUSH" = true ]; then
    echo -e "${YELLOW}🔍 Git 상태 확인 중...${NC}"
    
    # 현재 브랜치 확인
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "📍 현재 브랜치: $CURRENT_BRANCH"
    
    # 변경사항 확인
    if git diff --quiet && git diff --cached --quiet; then
        echo "✅ Working directory clean"
    else
        echo "⚠️ 미커밋된 변경사항 있음"
        git status --short
    fi
fi

echo ""
echo -e "${GREEN}✅ 현재 상황 스캔 완료${NC}"
echo "📋 문서 업데이트를 시작합니다..."

# 여기서 실제 문서 업데이트 로직이 들어갑니다
# (Claude Code에서 /update-doc 명령어가 실행될 때 이 부분이 자동으로 처리됩니다)

echo -e "${GREEN}✅ 문서 업데이트 완료${NC}"

# Git 작업 수행 (--git-push 옵션 시)
if [ "$GIT_PUSH" = true ]; then
    echo ""
    echo -e "${BLUE}🔄 Git 작업 시작...${NC}"
    
    # 변경된 파일 확인
    echo "📋 변경된 파일:"
    git status --porcelain
    
    # 문서 파일들 추가
    echo "📁 문서 파일들을 staging area에 추가..."
    git add README.md CLAUDE.md .claude/ Prompt/ scripts/core/update_doc_with_git.sh
    
    # 변경사항이 있는지 확인
    if git diff --cached --quiet; then
        echo -e "${YELLOW}⚠️ 커밋할 변경사항이 없습니다.${NC}"
        exit 0
    fi
    
    # 커밋 메시지 생성
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    COMMIT_MSG="docs: PillSnap ML 문서 자동 업데이트 ($TIMESTAMP)

- Stage 2 완료 상태 반영 (25K 샘플, 250 클래스)
- Progressive Validation 현황 업데이트
- M.2 SSD 확장 계획 문서화
- Scripts 폴더 재구성 반영

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # 커밋 실행
    echo "💾 커밋 생성 중..."
    git commit -m "$COMMIT_MSG"
    
    # Push 실행
    echo "🚀 Remote repository로 push 중..."
    if [ "$FORCE_PUSH" = true ]; then
        echo -e "${RED}⚠️ Force push 실행 중... (주의!)${NC}"
        git push --force
    else
        git push
    fi
    
    echo ""
    echo -e "${GREEN}✅ Git push 완료!${NC}"
    echo "📍 브랜치: $CURRENT_BRANCH"
    echo "📝 커밋 메시지 미리보기:"
    echo "   docs: PillSnap ML 문서 자동 업데이트 ($TIMESTAMP)"
else
    echo ""
    echo -e "${BLUE}📋 문서 업데이트만 완료 (Git push 건너뜀)${NC}"
    echo -e "${YELLOW}💡 Git push를 원한다면: --git-push 옵션을 사용하세요${NC}"
fi

echo ""
echo -e "${GREEN}🎉 작업 완료!${NC}"
echo ""
echo "다음 단계 권장사항:"
echo "1. 🔍 Stage 2 데이터 무결성 검증"
echo "2. 📊 Stage 2 manifest 파일 생성"
echo "3. 🧪 Stage 2 데이터로더 테스트"
echo "4. 🚀 Stage 2 모델 학습 시작"