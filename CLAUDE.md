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
   
   # Stage 3 (100K 샘플, 1000 클래스)
   python -m src.training.train_classification_stage --manifest artifacts/stage3/manifest_train.csv --epochs 50 --batch-size 16
   
   # Stage 4 (500K 샘플, 4523 클래스) 
   python -m src.training.train_classification_stage --manifest artifacts/stage4/manifest_train.csv --epochs 100 --batch-size 8
   
   # Stage 1-2 (기존 방식)
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
  - 데이터 구조: `/home/max16/pillsnap_data` 분리 완료
  - 심볼릭 링크: Windows SSD + Linux SSD 하이브리드 구성
  - Albumentations 2.0.8 업그레이드 완료

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

| Stage | Images  | Classes | Purpose              | Accuracy | Status | Method |
|-------|---------|---------|----------------------|----------|--------|---------|
| 1     | 5,000   | 50      | Pipeline verification | 74.9%    | ✅ **완료** (Native) | Config 기반 |
| 2     | 25,000  | 250     | Performance baseline  | 진행예정  | 🔄 준비됨 | Config 기반 |
| 3     | 100,000 | 1,000   | Scalability test      | 목표85%  | 🎯 **Manifest 기반** | **원본 직접로딩** |
| 4     | 500,000 | 4,523   | Production deployment | 목표92%  | 🎯 **Manifest 기반** | **원본 직접로딩** |

### **⭐ Stage 3-4 핵심 변경사항:**
- **물리적 복사 없음**: 14.6GB → 50MB (manifest CSV 파일만)
- **하이브리드 스토리지**: Linux SSD + Windows SSD 심볼릭 링크 활용
- **Native Linux 최적화**: 128GB RAM + 빠른 SSD I/O로 실시간 로딩
- **용량 절약**: Stage 4까지 총 ~73GB → ~200MB 절약

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