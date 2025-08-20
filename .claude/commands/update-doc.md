# /update-doc — PillSnap ML 문서 자동 업데이트 명령어

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 모든 관련 문서를 현재 상황에 맞게 자동으로 업데이트합니다.
**모든 응답은 한국어로 작성**합니다.

---

## 📋 옵션 및 사용법

### 기본 사용법
```bash
/.claude/commands/update-doc.md                    # 문서만 업데이트
/.claude/commands/update-doc.md --git-push         # 문서 업데이트 + Git Push
/.claude/commands/update-doc.md --git-push --force # 문서 업데이트 + Force Git Push
```

### 🎯 옵션 설명
- **기본 모드**: 문서 업데이트만 수행 (안전 모드)
- **--git-push**: 문서 업데이트 후 자동으로 git add, commit, push 수행
- **--force**: git push에 --force 옵션 추가 (신중히 사용)

---

## 📋 업데이트 대상 문서

### 1. 세션 초기화 문서
- `.claude/commands/initial-prompt.md` - 세션 초기화 스크립트
- `CLAUDE.md` - 프로젝트 가이드

### 2. PART 설계 문서 (9개)
- `Prompt/PART_0.md` - Progressive Validation Strategy + OptimizationAdvisor
- `Prompt/PART_A.md` - Two-Stage Conditional Pipeline 아키텍처 + 디스크 I/O 병목 해결
- `Prompt/PART_B.md` - 프로젝트 구조 + RTX 5080 최적화 + SSD 이전
- `Prompt/PART_C.md` - Two-Stage 데이터 파이프라인 + SSD 최적화
- `Prompt/PART_D.md` - YOLOv11m 검출 모델
- `Prompt/PART_E.md` - EfficientNetV2-S 분류 모델
- `Prompt/PART_F.md` - API 서빙 + FastAPI
- `Prompt/PART_G.md` - 최적화 + 컴파일러
- `Prompt/PART_H.md` - 배포 + ONNX 내보내기

### 3. 프로젝트 메인 문서
- `README.md` - 프로젝트 전체 개요
- `config.yaml` - 설정 파일 검토 및 업데이트

---

## 🔍 현재 상황 분석 (2025-08-20 기준)

### ✅ 완료된 주요 변경사항
1. **Stage 2 완료**: SSD 이전 완료
   - Stage 1 + Stage 2 데이터 완전 SSD 이전 완료
   - 데이터 루트: `/home/max16/ssd_pillsnap/dataset`
   - 실험 디렉토리: `/home/max16/ssd_pillsnap/exp/exp01`
   - 성능 향상: 35배 (100MB/s → 3,500MB/s)

2. **M.2 SSD 확장 계획**: Samsung 990 PRO 4TB 
   - 성능: 7,450MB/s (75배 향상)
   - Stage 3-4까지 전체 데이터셋 수용 가능

3. **Progressive Validation 현황**:
   - Stage 1: ✅ 완료 (5K 샘플, 50 클래스)
   - Stage 2: ✅ 완료 (25K 샘플, 250 클래스, 307,152개 이미지)
   - Stage 3: ⚠️ M.2 SSD 필요 (현재 SSD 용량 부족)
   - Stage 4: ⏳ M.2 SSD 추가 후 진행

4. **Commercial-Grade 아키텍처 완성**:
   - 6단계 상업용 시스템 구현 완료
   - 22개 통합 테스트 (기본 + 엄격한 검증)
   - Training/Evaluation Components 완성

5. **Scripts 폴더 재구성**:
   - 기능별/Stage별 직관적 구조 완성
   - 모든 경로 참조 20개 파일 업데이트 완료

### 📍 업데이트 필요 사항
1. **최신 Progress**: Stage 2 완료 상태 반영
2. **Git Push 옵션**: 문서 업데이트 후 자동 커밋/푸시 기능
3. **하드웨어 확장**: M.2 SSD 추가 계획 문서화
4. **다음 단계**: Stage 2 학습 준비 상태 강조

---

## 🔄 업데이트 프로세스

### 1단계: 옵션 파싱 및 현재 상황 스캔
```bash
# 인자 분석
ARGS="$@"
GIT_PUSH=false
FORCE_PUSH=false

if [[ "$ARGS" == *"--git-push"* ]]; then
    GIT_PUSH=true
fi

if [[ "$ARGS" == *"--force"* ]]; then
    FORCE_PUSH=true
fi
```

- 프로젝트 구조 및 파일 상태 확인
- 디스크 사용량 및 SSD 이전 상태 검증
- config.yaml 설정 검토
- Git 상태 확인 (--git-push 옵션 시)

### 2단계: 문서별 업데이트 실행
각 문서를 현재 상황에 맞게 업데이트:

#### initial-prompt.md 업데이트 내용:
- Stage 2 완료 상태 반영
- SSD 데이터 경로 최신화
- M.2 SSD 확장 계획 업데이트
- 현재 Stage 준비 상태

#### PART 문서 업데이트 내용:
- **PART_A**: 경로 정책 + 디스크 I/O 최적화 완료 상황
- **PART_B**: SSD 경로 + Stage별 최적화 설정
- **PART_C**: 데이터 파이프라인 SSD 최적화 구조
- **PART_D~H**: 현재 구현 상태 반영

#### README.md 업데이트 내용:
- Stage 2 완료 상태 강조
- M.2 SSD 확장 계획 추가
- Progressive Validation 현황 업데이트
- Commercial-Grade 아키텍처 강조

### 3단계: Git 작업 수행 (--git-push 옵션 시)
```bash
if [ "$GIT_PUSH" = true ]; then
    echo "🔄 Git 작업 시작..."
    
    # 변경된 파일 확인
    git status --porcelain
    
    # 문서 파일들 추가
    git add README.md CLAUDE.md .claude/ Prompt/
    
    # 커밋 메시지 생성
    COMMIT_MSG="docs: PillSnap ML 문서 자동 업데이트 ($(date '+%Y-%m-%d %H:%M'))

- Stage 2 완료 상태 반영 (25K 샘플, 250 클래스)
- Progressive Validation 현황 업데이트
- M.2 SSD 확장 계획 문서화
- Scripts 폴더 재구성 반영

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # 커밋 실행
    git commit -m "$COMMIT_MSG"
    
    # Push 실행
    if [ "$FORCE_PUSH" = true ]; then
        git push --force
    else
        git push
    fi
    
    echo "✅ Git push 완료!"
fi
```

### 4단계: 완료 보고
- 업데이트된 파일 목록
- 주요 변경사항 요약
- Git 작업 결과 (해당 시)
- 다음 단계 권장사항

---

## 🎯 실행 예시

### 기본 문서 업데이트만
```bash
/.claude/commands/update-doc.md
```

### 문서 업데이트 + Git Push
```bash
/.claude/commands/update-doc.md --git-push
```

### 문서 업데이트 + Force Git Push (주의!)
```bash
/.claude/commands/update-doc.md --git-push --force
```

**실행 후 기대 결과**:
- 모든 관련 문서가 현재 상황에 맞게 업데이트
- 새로운 세션에서도 100% 현재 컨텍스트 이해 가능
- Stage 2 완료 상태 완전 반영
- M.2 SSD 확장 계획 문서화
- (옵션) Git 커밋 및 푸시 자동 완료

---

## ⚠️ 주의사항

### 일반 사항
1. **백업**: 업데이트 전 중요 문서 백업 권장
2. **검증**: 업데이트 후 주요 경로 및 설정 검증 필요
3. **일관성**: 모든 문서 간 정보 일관성 유지
4. **완전성**: 한 줄도 누락 없이 모든 관련 정보 반영

### Git Push 관련 주의사항
1. **--git-push 사용 전 확인**:
   - 현재 브랜치가 올바른지 확인
   - 충돌할 수 있는 로컬 변경사항 없는지 확인
   - 원격 저장소 상태 확인

2. **--force 옵션 사용 금지 상황**:
   - 공유 브랜치 (main, develop 등)에서 사용 금지
   - 다른 개발자와 협업 중인 브랜치에서 사용 금지
   - 확실하지 않은 상황에서 사용 금지

3. **안전한 사용법**:
   ```bash
   # 1. 먼저 문서만 업데이트해서 확인
   /.claude/commands/update-doc.md
   
   # 2. 변경사항 확인 후 Git Push
   /.claude/commands/update-doc.md --git-push
   ```

---

## 🔧 고급 기능

### 선택적 문서 업데이트 (향후 확장 예정)
```bash
/.claude/commands/update-doc.md --only-readme        # README.md만 업데이트
/.claude/commands/update-doc.md --only-claude        # CLAUDE.md만 업데이트
/.claude/commands/update-doc.md --only-parts         # PART 문서들만 업데이트
```

### 브랜치 관리 (향후 확장 예정)
```bash
/.claude/commands/update-doc.md --git-push --branch docs-update  # 새 브랜치 생성 후 push
```

---

이 명령어는 **현재 상황을 100% 정확하게 반영**하여 새로운 세션에서도 완전한 컨텍스트 이해가 가능하도록 모든 문서를 업데이트하며, 선택적으로 Git 작업까지 자동화합니다.