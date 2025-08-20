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
- **Stage 1-2 검증 완료** 상태 및 현재 진행 상황 반영

초기화를 실행하지 않으면 일관성 없는 출력이나 프로젝트 규칙 위반이 발생할 수 있습니다.

---

## 핵심 규칙

- **언어:** 별도 지시가 없는 한 모든 응답은 **한국어**로 작성
- **경로 사용:** **SSD 기반 절대 경로만 사용** (`/home/max16/ssd_pillsnap/`). Windows 스타일 경로(예: `C:\`) 금지  
- **데이터 위치:** 모든 데이터셋과 실험은 성능을 위해 **SSD 기반 경로** 사용 (`/home/max16/ssd_pillsnap/dataset`)
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
  - **현재 WSL 제약**: num_workers=0 (DataLoader 멀티프로세싱 비활성화)
  - **Native Ubuntu 이전 계획**: CPU 멀티프로세싱 최적화를 위한 전면 이전 예정
  - Monitor VRAM usage to stay under 14GB.  

---

## Recommended Workflow

1. **Initialize session:** Run `/ .claude/commands/initial-prompts.md` first.  
2. **Environment setup:**  
   ```bash
   bash scripts/core/setup_venv.sh
   source $HOME/pillsnap/.venv/bin/activate
   ```  
3. **Training:**  
   ```bash
   # Stage별 훈련 (Manifest 기반)
   python -m src.training.train_classification_stage --manifest artifacts/stage2/manifest_ssd.csv --epochs 1 --batch-size 32
   
   # 기존 Config 기반 (일반적)
   python -m src.train --cfg config.yaml
   python -m src.train --cfg config.yaml train.resume=last
   ```  
4. **Testing & Evaluation:**  
   ```bash
   pytest tests/
   bash tests/evaluate_stage.sh 1  # Replace with appropriate stage number (1-4)
   python -m tests.stage_1_evaluator  # Replace with stage 1-4 as needed
   python -m tests.stage_progress_tracker
   ```  
5. **Inference:**  
   ```bash
   python -m src.infer --engine torch --model /mnt/data/exp/exp01/checkpoints/best.pt --inputs "/path/to/images/*.jpg" --batch 16
   python -m src.infer --engine onnx --model /mnt/data/exp/exp01/export/model.onnx --inputs "/path/to/images/*.jpg" --batch 16
   ```  
6. **API & Deployment:**  
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

| Purpose            | Current (WSL)                             | Future (Native Ubuntu)                   |
|--------------------|-------------------------------------------|-------------------------------------------|
| Codebase           | `/home/max16/pillsnap`                    | `/home/max16/pillsnap` (M.2 SSD)         |
| Dataset            | `/home/max16/ssd_pillsnap/dataset`        | `/home/max16/pillsnap/dataset` (M.2 SSD) |
| Virtual Environment | `$HOME/pillsnap/.venv`                     | `$HOME/pillsnap/.venv` (Native)          |
| Experiment Outputs | `/home/max16/ssd_pillsnap/exp`            | `/home/max16/pillsnap/exp` (M.2 SSD)     |

---

## Hardware Optimization Settings

### 🖥️ **Current Environment (WSL)**
- **GPU:** RTX 5080 (16GB)  
  - Use mixed precision (TF32)  
  - Apply `channels_last` memory format  
  - Utilize `torch.compile(model, mode='max-autotune')` for training  
- **System RAM:** 128GB  
  - **WSL 제약**: num_workers=0 (CPU 멀티프로세싱 비활성화)
  - 안정성 우선: 데드락 없는 안정적 학습
- **Current Performance:**  
  - Stage 1: ✅ 완료 (83.2% 정확도, 6분 완료)
  - Stage 2: ✅ 완료 (83.1% 정확도, 10.9분, 237클래스/23,700샘플)
  - 데이터 이전: 307,152개 이미지 + 112,365개 라벨 SSD 완료
  - 디스크 I/O 병목 해결: 35배 성능 향상 (100MB/s → 3,500MB/s)
  - Manifest 기반 훈련: Lazy Loading으로 메모리 최적화
  - Albumentations 2.0.8 완전 호환

### 🚀 **Planned Environment (Native Ubuntu on M.2 SSD)**
- **Storage:** Samsung 990 PRO 4TB M.2 SSD (7,450MB/s)
- **OS:** Native Ubuntu (WSL 제약 완전 해결)
- **DataLoader:** num_workers=8-12 (16 CPU 코어 활용)
- **Expected Performance:**  
  - 데이터 로딩 속도: 8-12배 향상
  - Stage 3-4 대용량 데이터셋 최적화
  - Cloud tunnel API 서비스 준비  

---

## Progressive Validation Stages

| Stage | Images  | Classes | Purpose              | Accuracy | Status |
|-------|---------|---------|----------------------|----------|--------|
| 1     | 5,000   | 50      | Pipeline verification | 83.2%    | ✅ **완료** |
| 2     | 23,700  | 237     | Performance baseline  | 83.1%    | ✅ **완료** |
| 3     | 100,000 | 1,000   | Scalability test      | 목표85%  | ⚠️ M.2 SSD 필요 |
| 4     | 500,000 | 4,523   | Production deployment | 목표85%  | ⏳ 대기 |

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

By following this guide and running the session initialization command every time, Claude Code will maintain accuracy, consistency, and compliance with the PillSnap ML project standards.
- === Quick Check: 데이터 루트는 /mnt/data/pillsnap_dataset 이어야 함 ===
# 0) 환경변수로 고정 (코드 변경 없이 최우선 적용)
export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset

# 1) 존재/권한/샘플 나열
ls -al /mnt/data/pillsnap_dataset | head -n 20 || echo "경로 없음"

# 2) config 로더가 해당 경로를 읽는지 확인
source $HOME/pillsnap/.venv/bin/activate && python - <<'PY'
import sys; sys.path.insert(0,'.')
import config
c = config.load_config()
print("data.root =", c.data.root)
assert c.data.root == "/mnt/data/pillsnap_dataset", "data.root mismatch"
print("✅ ok")
PY