# 📜 PillSnap ML Changelog

모든 주요 변경사항과 개선사항을 기록합니다.

## [2025-08-24] - Detection 누적 학습 시스템 구현

### 🎯 Added
- **DetectionStateManager** (`src/utils/detection_state_manager.py`)
  - state.json 기반 에폭 추적 시스템
  - 원자적 파일 작업으로 동시성 안전 보장
  - 학습 정체 자동 감지 (3사이클 불변시 경고)
  - 메트릭 히스토리 관리 (최근 10사이클)

- **RobustCSVParser** (`src/utils/robust_csv_parser.py`)
  - 지수 백오프 재시도 로직 (최대 3회)
  - YOLO 버전별 컬럼명 자동 매핑
  - 메트릭 유효성 검증 시스템

- **Precision 튜닝 도구** (`scripts/tune_detection_precision.py`)
  - Confidence threshold 자동 스윕 (0.1~0.9)
  - IoU threshold 자동 스윕 (0.3~0.7)
  - 최적 파라미터 자동 탐색 및 저장

- **테스트 스크립트** (`scripts/test_detection_state.py`)
  - State Manager 동작 검증
  - CSV Parser 테스트
  - YOLO 아티팩트 확인

### 🔧 Changed
- **train_detection_epoch 함수** 개선
  - `epochs=1` → `epochs=target_epochs` (누적 학습)
  - State 기반 초기화 로직
  - 동적 검증 주기 (초반 5에폭 매번, 이후 3에폭마다)
  - yolo11x → yolo11m (메모리 최적화)
  - DET_CHECK 한줄 요약 로깅

### 📚 Documentation
- `docs/detection_cumulative_training_fix.md` - 구현 상세 문서
- `README.md` - Detection 누적 학습 시스템 추가
- `CLAUDE.md` - Detection 버그 해결 업데이트
- `docs/TRAINING_RESULTS.md` - 해결된 문제 및 개선사항 기록
- `docs/STAGE4_PLAN.md` - Detection 버그 수정 완료 표시

### 🐛 Fixed
- YOLO가 매 사이클마다 1 에폭만 학습하는 문제 해결
- results.csv 덮어쓰기 문제 해결
- Detection 학습 진행상황이 추적되지 않는 문제 해결

---

## [2025-08-24 오전] - Stage 3 학습 완료 및 백업

### 🎯 Achievements
- **Stage 3 학습 성공 완료**
  - Classification: 85.01% Top-1, 97.68% Top-5 (목표 85% 달성)
  - Detection: 32.73% mAP@0.5 (목표 30% 초과 달성)
  - 학습 시간: 276.2분 (22 에포크에서 조기 종료)

### 🎯 Added
- **Freeze Utility** (`scripts/freeze_stage3_results.py`)
  - 체크포인트 백업 및 MD5 검증
  - Manifest 스냅샷 저장
  - 실험 카드 생성
  - TensorBoard 로그 백업
  - 재현성 스모크 테스트

- **Sanity Check 수정** (`scripts/sanity_check_fixed.py`)
  - 클래스 매핑 문제 해결 (0% → 84.7% 정확도)
  - sorted(unique_codes) 사용으로 일관성 보장
  - SimpleEvalDataset으로 명시적 label_idx 전달

### 📦 Backup
- `frozen_experiments/stage3_frozen_20250824_225921/`
  - 모든 체크포인트 백업 완료
  - Manifest 파일 스냅샷
  - 실험 설정 및 결과 기록

---

## [2025-08-23] - Progressive Resize & 실시간 모니터링

### 🎯 Added
- **Progressive Resize 시스템**
  - 128px → 384px 점진적 해상도 증가
  - GPU 메모리 최적화
  - 자동 해상도 조정

- **실시간 모니터링 시스템**
  - WebSocket 기반 대시보드 (http://localhost:8888)
  - KST 표준시 지원
  - Stage 자동 감지

- **OOM 방지 시스템**
  - 동적 배치 크기 조정
  - 가비지 컬렉션
  - torch.compile 최적화

### 🔧 Changed
- TensorBoard 통합 강화
- CosineAnnealingWarmRestarts 스케줄러 적용
- 손상 파일 자동 스킵 (manifest.remove.csv)

---

## [2025-08-22] - Native Linux 마이그레이션

### 🚀 Infrastructure
- WSL → Native Ubuntu 이전 완료
- 128GB RAM + RTX 5080 16GB 완전 활용
- num_workers=8-12 멀티프로세싱 활성화

### 📊 Performance
- Stage 1: 74.9% 정확도 (1분 소요)
- Stage 2: 83.1% 정확도 달성
- 데이터 로딩 속도 8-12배 향상

---

## [2025-08-21] - Manifest 기반 데이터 로딩

### 🎯 Added
- **ManifestDataset** 클래스
  - 물리적 복사 없이 원본 직접 로딩
  - 14.6GB → 50MB 용량 절약 (99.7%)
  - 하이브리드 스토리지 지원

### 🔧 Changed
- Stage 3-4: Config 기반 → Manifest 기반 전환
- 데이터 로딩 파이프라인 최적화

---

## 📈 주요 성과 요약

| Stage | 목표 | 달성 | 상태 |
|-------|------|------|------|
| Stage 1 | 78% | 74.9% | ✅ 완료 |
| Stage 2 | 82% | 83.1% | ✅ 완료 |
| Stage 3 | 85% | 85.01% | ✅ 완료 |
| Stage 4 | 92% | - | 🎯 준비 완료 |

---

*생성일: 2025-08-24*
*유지보수: PillSnap ML Team*