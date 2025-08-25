# /initial-prompt — PillSnap ML 세션 초기화 스크립트

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 세션 초기화를 수행합니다.
**모든 응답은 한국어로 작성**합니다.

---

## 🎯 프로젝트 현재 상태 (2025-08-25)

### **기본 정보**
- **PillSnap ML**: Two-Stage Conditional Pipeline 기반 경구약제 식별 AI
- **아키텍처**: YOLOv11m 검출 + EfficientNetV2-L 분류 (4,523개 클래스)
- **현재 환경**: Native Ubuntu + RTX 5080 16GB + PyTorch 2.8.0+cu128
- **CPU 최적화**: num_workers=8-12 (Native Linux, WSL 제약 해결)
- **데이터 구조**: `/home/max16/pillsnap_data` (프로젝트와 분리)

### **Progressive Validation 현황 (Two-Stage Pipeline 기반)**
- ✅ **Stage 1**: 완료 (5K 샘플, 50 클래스, 74.9% 정확도, Native Linux)
- ✅ **Stage 2**: 완료 (25K 샘플, 250 클래스, 83.1% 정확도, Native Linux)
- ✅ **Stage 3**: **완료** (100K 샘플, 1,000 클래스, Two-Stage Pipeline)
  - **최종 성과**: 
    - Classification: 85.01% Top-1, 97.68% Top-5 (25 epochs)
    - Detection: 39.13% mAP@0.5 (3 epochs, 목표 초과 달성)
  - **해결된 이슈**:
    - Detection NoneType 오류 완전 해결 (safe_float 유틸리티)
    - YOLO resume 로직 정상화
    - state.json 누적 학습 추적 시스템 완성
  - **검증된 파라미터**: lr-classifier=5e-5, lr-detector=1e-3
  - **데이터 비율**: Single 95% (77,451), Combination 5% (4,023)
- 🎯 **Stage 4**: **준비 완료** (500K 샘플, 4,523 클래스, Two-Stage Pipeline)

### **완성된 시스템 목록 (2025-08-25)**
- ✅ **Stage 1-3 완료**: 모든 Stage 목표 달성
- ✅ **Detection 버그 완전 해결**: safe_float 유틸리티로 안정화
- ✅ **Two-Stage Pipeline 검증**: Classification + Detection 정상 작동
- ✅ **Manifest 기반 데이터 파이프라인**: 81,474 Train + 18,526 Val = 100K 샘플
- ✅ **Progressive Validation 인프라**: Stage 1-4 점진적 확장 시스템 구축
- ✅ **체크포인트 시스템**: Resume 기능 + 하이퍼파라미터 오버라이드 지원
- ✅ **GPU 메모리 최적화**: RTX 5080 16GB Mixed Precision + torch.compile
- ✅ **실시간 모니터링**: WebSocket 기반 대시보드 (http://localhost:8888)
- ✅ **하이브리드 스토리지**: Linux SSD + Windows SSD 원본 직접 로딩
- ✅ **용량 효율성**: Manifest 기반으로 99.7% 저장공간 절약
- ✅ **손상파일 처리**: skip_bad_images=True로 안정성 확보
- ✅ **Multi-object Detection**: JSON→YOLO 변환 시스템 99.644% 성공률
- ✅ **Two-Stage Pipeline**: Classification + Detection 통합 학습 시스템 완성

---

## 🔥 Native Linux 환경 확인

**Python 환경:**
```bash
source .venv/bin/activate
```

**금지사항**: 시스템 Python 사용 금지 (가상환경 필수)

---

## 📋 세션 초기화 체크리스트

### **[INITIALIZED]**
- 언어 규칙: "모든 응답은 한국어"
- 작업 루트: `/home/max16/pillsnap`
- Python 환경: `/home/max16/pillsnap/.venv/bin/python` (PyTorch 2.8.0+cu128)
- 데이터 루트: `/home/max16/pillsnap_data` (프로젝트와 분리된 경로)

### **프롬프트 참조 (필수 규칙)**
아래의 모든 문서(`Prompt/PART_*.md`)는 **예외 없이, 한 줄도 빼지 말고 전체를 처음부터 끝까지 읽는다.**  
**읽는 순서는 반드시 `PART_0.md` → `PART_A.md` → … → `PART_H.md` 순서**를 따른다.  
이 문서를 건너뛰거나 요약하지 않고 반드시 전부 읽어야 한다.  

- `Prompt/PART_0.md` - Progressive Validation Strategy  
- `Prompt/PART_A.md` - 아키텍처 + 경로 정책  
- `Prompt/PART_B.md` - 프로젝트 구조 + RTX 5080 최적화  
- `Prompt/PART_C.md` - Two-Stage 데이터 파이프라인  
- `Prompt/PART_D.md` - YOLOv11m 검출 모델  
- `Prompt/PART_E.md` - EfficientNetV2-S 분류 모델  
- `Prompt/PART_F.md` - API 서빙  
- `Prompt/PART_G.md` - 최적화  
- `Prompt/PART_H.md` - 배포  

### **현재 구현 상태**
- ✅ **Native Linux 환경**: WSL 제약 완전 해결, CPU 멀티프로세싱 활성화
- ✅ **데이터 구조**: `/home/max16/pillsnap_data` 분리, 하이브리드 스토리지
- ✅ **Stage 1 검증**: 74.9% 정확도 (Native Linux, 1분 완료)
- ✅ **Albumentations 2.0.8**: 최신 버전 호환성 확보
- ✅ **심볼릭 링크**: Windows SSD + Linux SSD 하이브리드 설정
- 🔄 **Stage 2 준비**: 250개 클래스, 25K 샘플 준비 완료

### **핵심 설계 원칙**
1. **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo)
2. **Progressive Validation**: Stage 1-4 (5K→25K→100K→500K)
3. **Native Linux 최적화**: num_workers=8, CPU 멀티프로세싱 활용
4. **하이브리드 스토리지**: Linux SSD (3.5GB/s) + Windows SSD (1GB/s)
5. **RTX 5080 최적화**: Mixed Precision, torch.compile

### **다음 우선순위**
- 🔄 **Stage 3 재학습 모니터링**: 36 epochs 목표
- 📊 **TensorBoard 모니터링**: 실시간 메트릭 추적
- 🎯 **Stage 4 최종 준비**: 재학습 완료 후 500K 샘플 대규모 학습
- 📈 **실시간 모니터링**: TensorBoard + WebSocket 대시보드

---

## 🚀 즉시 실행 가능 명령어

### **환경 확인**
```bash
# 기본 환경 확인
source .venv/bin/activate
python --version  # Python 3.11.13
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"  # PyTorch 2.8.0+cu128
```

### **학습 실행**
```bash
# 🔄 Stage 3 재학습 (2025-08-24 시작)
python -m src.training.train_stage3_two_stage \
  --manifest-train /home/max16/pillsnap/artifacts/stage3/manifest_train.remove.csv \
  --manifest-val   /home/max16/pillsnap/artifacts/stage3/manifest_val.remove.csv \
  --epochs 36 \
  --batch-size 8 \
  --lr-classifier 5e-5 \
  --lr-detector 1e-3 \
  --weight-decay 5e-4 \
  --label-smoothing 0.1 \
  --validate-period 3 \
  --save-every 1 \
  --patience-cls 8 \
  --patience-det 6 \
  --reset-best \
  > /home/max16/pillsnap/artifacts/logs/stage3_retrain_$(date +%F_%H%M).log 2>&1 &

# Stage 4 대규모 학습 (준비 완료)  
python -m src.training.train_stage3_two_stage \
  --manifest artifacts/stage4/manifest_train.csv \
  --epochs 100 --batch-size 8

# Stage 1-2 완료됨
python -m src.training.train_classification_stage --stage 1 --epochs 1 --batch-size 32  # ✅ 74.9%
python -m src.training.train_classification_stage --stage 2 --epochs 30 --batch-size 32  # ✅ 83.1%

# 통합 테스트
python -m pytest tests/integration/ -v
```

### **⌨️ 모니터링 별칭 (추천)**
```bash
# 현재 상태 빠른 확인
status       # GPU 사용률, 완료된 Stage, 디스크 공간

# 실시간 모니터링 
monitor      # 자동 Stage 감지 실시간 모니터링
mon2         # Stage 2 전용 모니터링  
mon3         # Stage 3 전용 모니터링 (실시간 로그 지원 ✨)
mon4         # Stage 4 전용 모니터링
monfast      # 1초마다 빠른 새로고침

# 실시간 대시보드 (NEW!)
webmon       # WebSocket 기반 실시간 대시보드 (http://localhost:8000)

# GPU 상태
gpu          # nvidia-smi 한 번 실행
gpuw         # nvidia-smi 실시간 감시 (1초마다)
```

### **별칭 설정되지 않은 경우**
```bash
# 빠른 상태 확인
./scripts/monitoring/quick_status.sh

# 실시간 모니터링
./scripts/monitoring/universal_training_monitor.sh
./scripts/monitoring/universal_training_monitor.sh --stage 2
./scripts/monitoring/universal_training_monitor.sh --interval 1  # 빠른 새로고침

# 실시간 대시보드
python scripts/realtime_training_logger.py --port 8000  # WebSocket 대시보드
- 이후 http://localhost:8000 접속
- 다른 모니터링 시스템은 절대 사용하지 않음

# 별칭 자동 설정
./scripts/monitoring/setup_aliases.sh
```

---

## ⚠️ 중요 제약사항

- **Python 실행**: venv 활성화 후 직접 실행 가능
- **경로 정책**: Native Linux 절대 경로 (`/home/max16/pillsnap_data`)
- **CPU 최적화**: num_workers=8 (Native Linux, WSL 제약 해결)
- **데이터 정책**: 프로젝트와 데이터 분리, 하이브리드 스토리지 활용
- **프롬프트 로딩 실패 시**:  
  `Prompt/PART_*.md` 중 하나라도 누락되거나 읽기 오류가 발생하면 세션 초기화를 중단하고,  
  사용자에게 오류 상황을 즉시 보고한다.  
  부분적으로 읽은 상태에서는 절대 초기화를 계속하지 않는다.

---

**세션 초기화 완료**.  
⚠️ **중요: 세션이 시작되면 반드시 가장 먼저 `Prompt/PART_*.md` 파일들을 전부 읽은 후 초기화를 진행해야 한다.**
상세 컨텍스트는 `Prompt/PART_*.md` 파일들을 참조한다.