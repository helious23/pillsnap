# /update-doc — PillSnap ML 문서 자동 업데이트 명령어

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 모든 관련 문서를 현재 상황에 맞게 자동으로 업데이트합니다.
**모든 응답은 한국어로 작성**합니다.

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

## 🔍 현재 상황 분석 (2025-08-19 기준)

### ✅ 완료된 주요 변경사항
1. **디스크 I/O 병목 해결**: 외장 HDD → SSD 이전 완료
   - Stage 1 데이터 5,000장 SSD 이전 (7.0GB)
   - 데이터 루트: `/home/max16/ssd_pillsnap/dataset`
   - 실험 디렉토리: `/home/max16/ssd_pillsnap/exp/exp01`
   - 성능 향상: 35배 (100MB/s → 3,500MB/s)

2. **M.2 SSD 확장 계획**: Samsung 990 PRO 4TB 
   - 성능: 7,450MB/s (75배 향상)
   - Stage 4까지 전체 데이터셋 수용 가능

3. **Progressive Validation 현황**:
   - Stage 1: ✅ 완료 (5K 샘플, 50 클래스, SSD 이전 완료)
   - Stage 2-3: 🔄 SSD 이전 예정
   - Stage 4: ⏳ M.2 SSD 추가 후 진행

4. **Commercial-Grade 아키텍처 완성**:
   - 6단계 상업용 시스템 구현 완료
   - 22개 통합 테스트 (기본 + 엄격한 검증)
   - Training/Evaluation Components 완성

### 📍 업데이트 필요 사항
1. **경로 정보**: HDD 경로를 SSD 경로로 일괄 변경
2. **데이터 처리 정책**: Stage별 SSD/M.2 SSD 사용 계획 반영
3. **성능 개선**: 디스크 I/O 병목 해결 과정 및 결과 반영
4. **하드웨어 확장**: M.2 SSD 추가 계획 문서화
5. **현재 준비 상태**: Stage 2-4 진행 가능한 상태 강조

---

## 🔄 업데이트 프로세스

### 1단계: 현재 상황 스캔
- 프로젝트 구조 및 파일 상태 확인
- 디스크 사용량 및 SSD 이전 상태 검증
- config.yaml 설정 검토

### 2단계: 문서별 업데이트 실행
각 문서를 현재 상황에 맞게 업데이트:

#### initial-prompt.md 업데이트 내용:
- 디스크 I/O 병목 해결 완료 상태
- SSD 데이터 경로 변경
- M.2 SSD 확장 계획
- 현재 Stage 준비 상태

#### PART 문서 업데이트 내용:
- **PART_A**: 경로 정책 + 디스크 I/O 최적화 상황
- **PART_B**: SSD 경로 + Stage별 최적화 설정
- **PART_C**: 데이터 파이프라인 SSD 최적화 구조
- **PART_D~H**: 현재 구현 상태 반영

#### README.md 업데이트 내용:
- 디스크 I/O 병목 해결 강조
- M.2 SSD 확장 계획 추가
- Progressive Validation 현황 업데이트
- Commercial-Grade 아키텍처 강조

### 3단계: 일관성 검증
- 모든 문서의 경로 정보 일관성 확인
- Stage 진행 상황 통일성 검증
- 하드웨어 정보 정확성 검토

### 4단계: 완료 보고
- 업데이트된 파일 목록
- 주요 변경사항 요약
- 다음 단계 권장사항

---

## 🎯 실행 명령어

```bash
# PillSnap ML 문서 자동 업데이트 실행
/.claude/commands/update-doc.md
```

**실행 후 기대 결과**:
- 모든 관련 문서가 현재 상황에 맞게 업데이트
- 새로운 세션에서도 100% 현재 컨텍스트 이해 가능
- 디스크 I/O 병목 해결 과정 완전 반영
- M.2 SSD 확장 계획 문서화

---

## ⚠️ 주의사항

1. **백업**: 업데이트 전 중요 문서 백업 권장
2. **검증**: 업데이트 후 주요 경로 및 설정 검증 필요
3. **일관성**: 모든 문서 간 정보 일관성 유지
4. **완전성**: 한 줄도 누락 없이 모든 관련 정보 반영

---

이 명령어는 **현재 상황을 100% 정확하게 반영**하여 새로운 세션에서도 완전한 컨텍스트 이해가 가능하도록 모든 문서를 업데이트합니다.