# CLAUDE.md

PillSnap ML 프로젝트의 Claude Code 종합 가이드입니다. 프로젝트 개요, 기술적 세부사항, 세션 초기화 지침을 통합하여 일관되고 최적화된 상호작용을 보장합니다.

---

## 세션 초기화

**모든 세션 시작 시 반드시 다음 명령어로 Claude Code 환경을 초기화하세요:**

```
/.claude/commands/initial-prompt.md
```

이 명령어는 정확하고 효율적인 지원을 위해 컨텍스트, 환경변수, 프로젝트별 설정을 구성합니다.

**목적:**  
- PillSnap ML 프로젝트의 기본 지식 구축
- SSD 기반 절대 경로 사용 강제
- 모든 응답의 기본 언어를 한국어로 설정
- Two-Stage Conditional Pipeline 로직 등 핵심 제약사항 로드
- **Stage 3 Two-Stage 학습 준비 완료** 상태 및 올바른 Manifest 반영

초기화를 실행하지 않으면 일관성 없는 출력이나 프로젝트 규칙 위반이 발생할 수 있습니다.

---

## 핵심 규칙

- **언어:** 별도 지시가 없는 한 모든 응답은 **한국어**로 작성
- **경로 사용:** **Native Linux 절대 경로만 사용** (`/home/max16/pillsnap_data/`). Windows 스타일 경로(예: `C:\`) 금지  
- **데이터 위치:** 모든 데이터셋은 프로젝트와 분리된 전용 경로 사용 (`/home/max16/pillsnap_data`)
- **Two-Stage Pipeline 강제:** 조건부 파이프라인 로직 준수:
  - Single pills → EfficientNetV2-S 직접 분류 (384px)
  - Combination pills → YOLOv11m 검출 → crop → 분류 (640px→384px)  
- **API Security:** Always assume API key authentication and rate limiting are in place (100 requests/minute).  
- **Performance Targets:**  
  - Single pill accuracy: 92%  
  - Combination pill mAP@0.5: 0.85  
- **Hardware Optimization:**  
  - Use mixed precision (TF32) and channels_last memory format on RTX 5080 (16GB) GPUs.  
  - Enable `torch.compile(model, mode='max-autotune')` for training speedups.  
  - **Native Linux 환경**: num_workers=8-12 (16 CPU 코어 활용)
  - **2025-08-22 업데이트**: Native Ubuntu 이전 완료, CPU 멀티프로세싱 최적화 활성화
  - Monitor VRAM usage to stay under 14GB.  

---

## Recommended Workflow

1. **Initialize session:** Run `/ .claude/commands/initial-prompts.md` first.  
2. **Environment setup:**  
   ```bash
   source /home/max16/pillsnap/.venv/bin/activate
   # Python 3.11.13, PyTorch 2.8.0+cu128, CUDA 12.8
   ```  
3. **Training (Manifest 기반 - Stage 3-4 표준):**  
   ```bash
   # ⭐ IMPORTANT: Stage 3-4는 반드시 manifest 기반으로만 진행
   # 물리적 데이터 복사 없이 원본에서 직접 로딩 (용량 절약)
   
   # Stage 3 완료됨 (44.1% Classification + 25.0% Detection) - Resume 기능으로 개선 가능
   python -m src.training.train_stage3_two_stage \
     --resume /home/max16/pillsnap_data/exp/exp01/checkpoints/stage3_classification_best.pt \
     --epochs 50 --lr-classifier 1e-4 --lr-detector 5e-3 --batch-size 12
   
   # Stage 4 준비 중 (500K 샘플, 4523 클래스) 
   python -m src.training.train_classification_stage --manifest artifacts/stage4/manifest_train.csv --epochs 100 --batch-size 8
   
   # Stage 1-2 (완료됨)
   python -m src.train --cfg config.yaml
   python -m src.train --cfg config.yaml train.resume=last
   ```  
4. **Monitoring & Status Check:**  
   ```bash
   # 빠른 상태 확인 (별칭 사용 - 추천)
   status       # GPU, Stage 완료 현황, 디스크 공간
   
   # 실시간 모니터링 (별칭 사용 - 추천) 
   monitor      # 자동 Stage 감지 실시간 모니터링
   mon2         # Stage 2 전용 모니터링
   monfast      # 1초마다 빠른 새로고침
   gpu          # nvidia-smi
   
   # 전체 경로 (별칭 미설정시)
   ./scripts/monitoring/quick_status.sh
   ./scripts/monitoring/universal_training_monitor.sh --stage 2
   ```
5. **Testing & Evaluation:**  
   ```bash
   pytest tests/
   bash tests/evaluate_stage.sh 1  # Replace with appropriate stage number (1-4)
   python -m tests.stage_1_evaluator  # Replace with stage 1-4 as needed
   python -m tests.stage_progress_tracker
   ```  
6. **Inference:**  
   ```bash
   python -m src.infer --engine torch --model /mnt/data/exp/exp01/checkpoints/best.pt --inputs "/path/to/images/*.jpg" --batch 16
   python -m src.infer --engine onnx --model /mnt/data/exp/exp01/export/model.onnx --inputs "/path/to/images/*.jpg" --batch 16
   ```  
7. **API & Deployment:**  
   ```bash
   bash scripts/deployment/run_api.sh
   bash scripts/deployment/export_onnx.sh
   bash scripts/deployment/maintenance.sh
   ```  

---

## Project Overview

**PillSnap ML** is an AI-powered pharmaceutical pill identification system using a **Two-Stage Conditional Pipeline** designed to extract `edi_code` from pill images efficiently and accurately.

### Architecture

```
Input Image → Auto Mode Detection
    ├─ Single Pills → Direct Classification (EfficientNetV2-L)
    └─ Combination Pills → YOLOv11x Detection → Crop → Classification
```

### Model Components

- **Detection:** YOLOv11x (640px input) for combination pill detection  
- **Classification:** EfficientNetV2-L (384px input) for 5000-class `edi_code` identification  
- **Target Performance:**  
  - Single pill accuracy: 92%  
  - Combination pill mAP@0.5: 0.85  

### Critical Paths

| Purpose            | Native Linux (2025-08-22)                  |
|--------------------|--------------------------------------------|
| Codebase           | `/home/max16/pillsnap`                     |
| Dataset            | `/home/max16/pillsnap_data` (분리된 경로)    |
| Virtual Environment | `/home/max16/pillsnap/.venv`               |
| Experiment Outputs | `/home/max16/pillsnap/exp`                 |

---

## Hardware Optimization Settings

### 🖥️ **Current Environment (Native Linux)**
- **GPU:** RTX 5080 (16GB)  
  - Use mixed precision (TF32)  
  - Apply `channels_last` memory format  
  - Utilize `torch.compile(model, mode='max-autotune')` for training  
- **System RAM:** 128GB  
  - **Native Linux**: num_workers=8-12 (CPU 멀티프로세싱 활성화)
  - WSL 제약 완전 해결: 안정적이고 빠른 데이터 로딩
- **Current Performance:**  
  - Stage 1: ✅ 완료 (74.9% 정확도, 1분, Native Linux)
  - Stage 2: ✅ 완료 (83.1% 정확도, Native Linux)
  - Stage 3: ✅ **학습 완료** (44.1% Classification + 25.0% Detection, 2025-08-23)
    - **Two-Stage Pipeline**: EfficientNetV2-L + YOLOv11m 통합 학습 완료
    - **Progressive Resize**: 128px→384px 점진적 해상도 증가 시스템
    - **실시간 모니터링**: WebSocket 기반 대시보드 (http://localhost:8888)
    - **OOM 방지**: 동적 배치 크기 조정 및 가비지 컬렉션
    - **Resume 기능**: 하이퍼파라미터 override + Top-5 accuracy 추적
    - **118개 테스트**: 모든 핵심 시스템 검증 완료
    - **Multi-object Detection**: JSON→YOLO 변환 99.644% 성공률
  - Stage 4: 🎯 **대기 중** (최종 프로덕션 학습)
  - 데이터 구조: Manifest 기반 로딩으로 99.7% 저장공간 절약
  - 실시간 모니터링: WebSocket 기반 학습 상태 추적 시스템 (KST 표준시 적용)
  - torch.compile 최적화 완료 (EfficientNetV2-L + YOLOv11x)
  - **Native Linux 최적화**: 128GB RAM + RTX 5080 16GB 완전 활용

### 🚀 **Planned Environment (Native Ubuntu on M.2 SSD)**
- **Storage:** Samsung 990 PRO 4TB M.2 SSD (7,450MB/s)
- **OS:** Native Ubuntu (WSL 제약 완전 해결)
- **DataLoader:** num_workers=8-12 (16 CPU 코어 활용)
- **Expected Performance:**  
  - 데이터 로딩 속도: 8-12배 향상
  - Stage 3-4 대용량 데이터셋 최적화
  - Cloud tunnel API 서비스 준비  

---

## Progressive Validation Stages (Manifest 기반)

| Stage | Images  | Classes | Purpose              | Result | Status | Method |
|-------|---------|---------|----------------------|----------|--------|---------|
| 1     | 5,000   | 50      | Pipeline verification | 74.9%    | ✅ **완료** (Native) | Config 기반 |
| 2     | 25,000  | 250     | Performance baseline  | 83.1%    | ✅ **완료** (Native) | Config 기반 |
| 3     | 100,000 | 1,000   | Scalability test      | 44.1% + 25.0% mAP | ✅ **완료** | **Two-Stage Pipeline** |
| 4     | 500,000 | 4,523   | Production deployment | 목표92%  | 🎯 **대기 중** | **Two-Stage Pipeline** |

### **⭐ Stage 3-4 핵심 변경사항:**
- **물리적 복사 없음**: 14.6GB → 50MB (manifest CSV 파일만)
- **하이브리드 스토리지**: Linux SSD + Windows SSD 심볼릭 링크 활용
- **Native Linux 최적화**: 128GB RAM + 빠른 SSD I/O로 실시간 로딩
- **용량 절약**: Stage 4까지 총 ~73GB → ~200MB 절약
- **Progressive Resize**: 동적 해상도 조정으로 GPU 메모리 최적화
- **실시간 모니터링**: WebSocket + 실시간 로그 스트리밍

---

## 🔄 Native Ubuntu Migration Plan

### **Migration Roadmap**
1. **Hardware Setup**
   - ✅ Install 4TB M.2 SSD in available slot
   - ✅ Install Native Ubuntu on M.2 SSD

2. **Data & Code Migration**
   - ✅ Windows SSD access (NTFS mount)
   - ✅ External HDD access (USB/SATA mount)
   - ✅ Copy datasets to Ubuntu M.2 SSD
   - ✅ Copy codebase to Ubuntu M.2 SSD

3. **Environment Setup**
   - ✅ Install Cursor & development tools
   - ✅ Setup Python virtual environment
   - ✅ Install PyTorch with CUDA support
   - ✅ Configure cloud tunnel for API service

4. **Performance Benefits**
   - 🎯 **CPU Utilization**: 16 cores → num_workers=8-12
   - 🎯 **Storage Speed**: 7,450MB/s (vs current 3,500MB/s)
   - 🎯 **WSL Constraints**: Completely eliminated
   - 🎯 **Production Ready**: Cloud API deployment

### **Migration Priority**
- **Stage 1-2**: ✅ Current WSL sufficient (완료됨)
  - Stage 1: 83.2% (목표 78% 초과달성)
  - Stage 2: 83.1% (목표 82% 초과달성)
- **Stage 3-4**: Native Ubuntu essential (25만-50만 이미지)
  - 현재 SSD 용량: 459GB 사용 (Stage 3 대비 부족)
  - M.2 SSD 4TB 확장 필수
- **Production API**: Cloud tunnel deployment required

---

## Project Structure

```
src/
├── data.py               # Conditional two-stage data loaders
├── models/
│   ├── detector.py       # YOLOv11x wrapper
│   ├── classifier.py     # EfficientNetV2-L implementation
│   └── pipeline.py       # Two-stage conditional pipeline
├── train.py              # GPU-optimized training loops
├── evaluate.py           # Performance metrics
├── infer.py              # Inference pipeline
└── api/                  # FastAPI serving
```

---

---

## 📝 **최근 업데이트 (2025-08-23)**

### ✅ **Stage 3 Two-Stage 학습 완료**
- **Classification 정확도**: 44.1% (1,000개 클래스 기준)
- **Detection mAP@0.5**: 25.0% (Multi-object detection)
- **Progressive Resize**: 128px→384px 점진적 해상도 증가
- **실시간 모니터링**: WebSocket 기반 대시보드 (http://localhost:8888)
- **OOM 방지**: 동적 배치 크기 조정 및 가비지 컬렉션
- **Resume 기능**: 하이퍼파라미터 오버라이드 지원
- **118개 테스트**: 모든 핵심 시스템 검증 완료

### ✅ **Multi-object Detection 완성**
- **JSON→YOLO 변환**: 12,025개 이미지 99.644% 성공률
- **실제 bounding box**: 평균 3.6개 객체/이미지 정확한 annotation
- **YOLO txt 라벨**: 11,875개 파일 생성 완료
- **Detection DataLoader**: Manifest 기반 640px 로딩 최적화
- **YOLOv11m 모델**: torch.compile 최적화 적용

### 🚀 **Stage 4 준비 완료**
모든 시스템이 완성되어 500K 샘플 대규모 학습이 준비되었습니다:
```bash
python -m src.training.train_stage3_two_stage \
  --manifest artifacts/stage4/manifest_train.csv \
  --epochs 100 --batch-size 8
```

### 🎯 **Stage 4 목표 (500K 샘플)**
- Classification Accuracy: ≥ 92% (Production 목표)
- Detection mAP@0.5: ≥ 85% (대용량 데이터 효과)
- Pipeline 추론시간: ≤ 50ms (ONNX 최적화)
- 완전 자동화: 실시간 모니터링 + OOM 방지

---

## 📝 **완성된 시스템 목록 (2025-08-23)**

### ✅ **Progressive Resize 시스템**
- **동적 해상도**: 128px→384px 점진적 증가
- **GPU 메모리 최적화**: 초기 낮은 해상도로 OOM 방지
- **성능 향상**: 점진적 fine-tuning으로 학습 안정성 증대
- **자동화**: epoch별 해상도 자동 조정

### ✅ **실시간 모니터링 시스템**
- **WebSocket 대시보드**: http://localhost:8888 실시간 로그
- **KST 표준시**: 한국 시간대 표시
- **자동 감지**: Stage 1-4 학습 상태 자동 추적
- **로그 스트리밍**: 실시간 터미널 출력 스트리밍

### ✅ **OOM 방지 & 최적화 시스템**
- **동적 배치 크기**: VRAM 사용량에 따른 자동 조정
- **가비지 컬렉션**: 메모리 누수 방지 시스템
- **torch.compile**: EfficientNetV2-L + YOLOv11m 최적화
- **Mixed Precision**: TF32 활용 성능 향상

---

By following this guide and running the session initialization command every time, Claude Code will maintain accuracy, consistency, and compliance with the PillSnap ML project standards.