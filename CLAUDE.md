# CLAUDE.md

This file provides comprehensive guidance for Claude Code (claude.ai/code) when working with the PillSnap ML repository. It integrates project overview, technical details, and essential session initialization instructions to ensure consistent and optimized interactions.

---

## Session Initialization

At the start of every session, **always initialize the Claude Code environment by running the command:**

```
/.claude/commands/initial-prompts.md
```

This command sets up the context, environment variables, and project-specific configurations to enable accurate and efficient assistance. It includes loading core rules, path constraints, coding standards, and response language settings.

**Purpose:**  
- Establish baseline knowledge of the PillSnap ML project.  
- Enforce usage of absolute WSL-native paths only.  
- Activate Korean as the default language for all responses.  
- Load critical constraints such as the two-stage conditional pipeline logic.  

Failing to run this initialization may lead to inconsistent outputs or violations of project rules.

---

## Core Rules

- **Language:** All responses must be in **Korean** by default unless explicitly instructed otherwise.  
- **Path Usage:** Use **absolute paths starting with `/mnt/` only**. Never use Windows-style paths (e.g., `C:\`) within code or commands.  
- **Data Location:** All datasets and experiments must reside in WSL-native filesystems for performance (e.g., `/mnt/data/pillsnap_dataset`).  
- **Two-Stage Pipeline Enforcement:** Respect the conditional pipeline logic:  
  - Single pills → direct classification with EfficientNetV2-L  
  - Combination pills → YOLOv11x detection → crop → classification  
- **API Security:** Always assume API key authentication and rate limiting are in place (100 requests/minute).  
- **Performance Targets:**  
  - Single pill accuracy: 92%  
  - Combination pill mAP@0.5: 0.85  
- **Hardware Optimization:**  
  - Use mixed precision (TF32) and channels_last memory format on RTX 5080 (16GB) GPUs.  
  - Enable `torch.compile(model, mode='max-autotune')` for training speedups.  
  - Utilize LMDB caching and batch prefetching with 16 dataloader workers for large datasets.  
  - Monitor VRAM usage to stay under 14GB.  

---

## Recommended Workflow

1. **Initialize session:** Run `/ .claude/commands/initial-prompts.md` first.  
2. **Environment setup:**  
   ```bash
   bash scripts/bootstrap_venv.sh
   source $HOME/pillsnap/.venv/bin/activate
   ```  
3. **Training:**  
   ```bash
   python -m src.train --cfg config.yaml
   python -m src.train --cfg config.yaml train.resume=last
   python -m src.train --cfg config.yaml train.batch_size=128 dataloader.num_workers=12
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
   bash scripts/run_api.sh
   bash scripts/export_onnx.sh
   bash scripts/maintenance.sh
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

| Purpose            | Path                                      |
|--------------------|-------------------------------------------|
| Codebase           | `/mnt/c/Users/max16/Desktop/pillsnap`     |
| Dataset (English)  | `/mnt/data/pillsnap_dataset`               |
| Virtual Environment | `$HOME/pillsnap/.venv`                     |
| Experiment Outputs | `/mnt/data/exp/exp01`                      |

---

## Hardware Optimization Settings

- **GPU:** RTX 5080 (16GB)  
  - Use mixed precision (TF32)  
  - Apply `channels_last` memory format  
  - Utilize `torch.compile(model, mode='max-autotune')` for training  
- **System RAM:** 128GB  
  - Use LMDB caching for datasets  
  - Prefetch batches with `non_blocking=True`  
  - Use 16 dataloader workers for optimal throughput  
- **Batch Sizes:**  
  - Detection: 16  
  - Classification: 128 (auto-tuned based on VRAM availability)  

---

## Progressive Validation Stages

| Stage | Images  | Classes | Purpose              |
|-------|---------|---------|----------------------|
| 1     | 5,000   | 50      | Pipeline verification |
| 2     | 25,000  | 250     | Performance baseline  |
| 3     | 100,000 | 1,000   | Scalability test      |
| 4     | 500,000 | 5,000   | Production deployment |

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