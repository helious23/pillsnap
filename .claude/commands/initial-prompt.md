# /initial-prompt — PillSnap ML 세션 초기화 스크립트

당신은 **Claude Code**입니다. **PillSnap ML** 프로젝트의 세션 초기화를 수행합니다.
**모든 응답은 한국어로 작성**합니다.

---

## 🎯 프로젝트 현재 상태 (2025-08-20)

### **기본 정보**
- **PillSnap ML**: Two-Stage Conditional Pipeline 기반 경구약제 식별 AI
- **아키텍처**: YOLOv11m 검출 + EfficientNetV2-S 분류 (4,523개 EDI 코드)
- **현재 환경**: WSL2 + RTX 5080 16GB + PyTorch 2.7.0+cu128
- **주요 제약**: num_workers=0 (DataLoader 멀티프로세싱 비활성화)
- **Migration 계획**: Native Ubuntu + M.2 SSD 4TB (CPU 멀티프로세싱 최적화)

### **Progressive Validation 현황**
- ✅ **Stage 1**: 완료 (5K 샘플, 50 클래스, 83.2% 정확도 달성)
- 🔄 **Stage 2**: 준비됨 (25K 샘플, 250 클래스)
- ⏳ **Stage 3-4**: Native Ubuntu 이전 후 진행 예정

### **최근 완료 작업 (2025-08-20)**
- ✅ WSL DataLoader 최적화 (num_workers=0, 안정성 확보)
- ✅ Albumentations 2.0.8 업그레이드 (API 호환성 완료)
- ✅ 데드락 모니터링 시스템 구현
- ✅ Native Ubuntu 이전 계획 문서화

---

## 🔥 필수 Python 실행 규칙

**모든 Python 실행 시 반드시 사용:**
```bash
./scripts/python_safe.sh [명령어]
```

**금지사항**: `python`, `python3` 시스템 명령어 사용 금지 (Python 3.13 충돌)

---

## 📋 세션 초기화 체크리스트

### **[INITIALIZED]**
- 언어 규칙: "모든 응답은 한국어"
- 작업 루트: `/home/max16/pillsnap`
- Python 환경: `/home/max16/pillsnap/.venv/bin/python` (PyTorch 2.7.0+cu128)
- 데이터 루트: `/home/max16/ssd_pillsnap/dataset` (SSD 최적화)

### **프롬프트 참조**
상세 설계는 다음 문서 참조:
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
- ✅ **모델 아키텍처**: YOLOv11m + EfficientNetV2-S + Two-Stage Pipeline 완료
- ✅ **데이터 파이프라인**: Progressive Validation 샘플링 완료
- ✅ **Training/Evaluation 시스템**: 상업용 컴포넌트 완료
- ✅ **통합 테스트**: 22개 테스트 (기본 + 엄격한 검증)
- ✅ **Stage 1 검증**: 5K 샘플, 50 클래스, 83.2% 정확도

### **핵심 설계 원칙**
1. **Two-Stage Conditional Pipeline**: 사용자 제어 모드 (single/combo)
2. **Progressive Validation**: Stage 1-4 (5K→25K→100K→500K)
3. **WSL 제약 인식**: num_workers=0, Native Ubuntu 이전 계획
4. **SSD 최적화**: 35배 성능 향상 (HDD→SSD)
5. **RTX 5080 최적화**: Mixed Precision, torch.compile

### **다음 우선순위**
- **Stage 2 준비**: 25K 샘플, 250 클래스
- **Native Ubuntu 이전**: CPU 멀티프로세싱 활용
- **Production API**: Cloud tunnel 배포

---

## 🚀 즉시 실행 가능 명령어

```bash
# 환경 확인
./scripts/python_safe.sh --version
./scripts/python_safe.sh -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"

# Stage 1 테스트
./scripts/python_safe.sh tests/test_stage1_real_image.py

# 통합 테스트
./scripts/python_safe.sh -m pytest tests/integration/ -v

# Stage 2 시작 (예정)
./scripts/python_safe.sh -m src.training.train_classification_stage --stage 2
```

---

## ⚠️ 중요 제약사항

- **Python 실행**: 반드시 `./scripts/python_safe.sh` 사용
- **경로 정책**: SSD 기반 절대 경로 (`/home/max16/ssd_pillsnap/`)
- **WSL 제약**: num_workers=0 (Native Ubuntu 이전으로 해결 예정)
- **데이터 정책**: Train 데이터만 학습/검증 분할, Val은 최종 테스트 전용

---

**세션 초기화 완료**. 상세 컨텍스트는 `Prompt/PART_*.md` 파일들을 참조하세요.