# PillSnap ML

AI-powered pharmaceutical pill identification system using Two-Stage Conditional Pipeline

## 🎯 Overview

PillSnap ML is an advanced machine learning system designed to identify pharmaceutical pills from images using a sophisticated two-stage pipeline:

- **Single Pills** → Direct Classification (EfficientNetV2-L)
- **Combination Pills** → YOLOv11x Detection → Crop → Classification

### Performance Targets
- Single pill accuracy: **92%**
- Combination pill mAP@0.5: **0.85**
- Inference speed: **<100ms per image**

## 🏗️ Architecture

```
Input Image → Auto Mode Detection
    ├─ Single Pills → Direct Classification (EfficientNetV2-L)
    └─ Combination Pills → YOLOv11x Detection → Crop → Classification
```

### Model Components
- **Detection**: YOLOv11x (640px input) for combination pill detection
- **Classification**: EfficientNetV2-L (384px input) for 5000-class `edi_code` identification

## 📁 Project Structure

```
pillsnap/
├── src/
│   ├── data.py              # Dataset loaders (PillsnapClsDataset, PillsnapDetDataset)
│   ├── train.py             # Training pipeline with OOM guard
│   ├── models/              # Model implementations
│   ├── utils/
│   │   └── oom_guard.py     # OOM recovery utilities
│   ├── core/                # Core components (from implementation guide)
│   └── api/                 # FastAPI service
├── dataset/                 # Data processing pipeline
│   ├── scan.py              # Dataset scanning (2.6M+ files)
│   ├── preprocess.py        # CSV manifest generation
│   └── validate.py          # Data quality validation
├── tests/                   # Comprehensive test suite
│   ├── test_pipeline.py     # Step 8 smoke tests
│   ├── test_paths.py        # Path validation tests
│   ├── test_validate.py     # Data validation tests
│   └── test_stage1_cli.py   # CLI integration tests
├── artifacts/               # Generated manifests and checkpoints
├── config.yaml              # Configuration file
└── paths.py                 # Path utilities with WSL support
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd pillsnap

# Setup virtual environment
bash scripts/bootstrap_venv.sh
source $HOME/pillsnap/.venv/bin/activate

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. Data Preparation

```bash
# Set data root (adjust path as needed)
export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset

# Scan dataset (safety limit: 400 samples)
python -m dataset.scan /mnt/data/pillsnap_dataset --output artifacts/scan_results.csv --limit 400

# Preprocess to manifest
python -m dataset.preprocess artifacts/scan_results.csv artifacts/manifest_stage1.csv

# Validate data quality
python -m dataset.validate artifacts/manifest_stage1.csv artifacts/validation_report.json
```

### 3. Training

#### Single Pill Classification
```bash
# Basic training
python -m src.train --mode single --epochs 100 --batch-size 64 --lr 1e-3

# With optimizations
python -m src.train --mode single --epochs 100 --batch-size 128 --amp --compile

# Resume from checkpoint
python -m src.train --mode single --resume artifacts/checkpoints/last.pt
```

#### Combination Pill Detection
```bash
# Detection training (note: dummy implementation)
python -m src.train --mode combo --epochs 300 --batch-size 16 --lr 1e-4
```

### 4. Testing

```bash
# Run all tests
pytest tests/ -v

# Step 8 pipeline tests
pytest tests/test_pipeline.py -v

# Integration tests
pytest tests/test_pipeline.py::TestEndToEndPipeline -v

# Smoke test (quick verification)
python tests/test_pipeline.py
```

## 📊 Step 8 Implementation Status

**Step 8: Core Pipeline & Smoke Tests** ✅ **COMPLETED**

### Implemented Components

#### 1. **src/data.py** - Dataset Loaders
- `PillsnapClsDataset`: Classification dataset with EDI code mapping
- `PillsnapDetDataset`: Detection dataset with YOLO bbox format
- `CodeToClassMapper`: EDI code ↔ class ID conversion
- Transform pipelines for training/validation
- Memory-efficient label caching

#### 2. **src/train.py** - Training Pipeline
- `ModelFactory`: EfficientNetV2-L & YOLOv11 model creation
- `Trainer`: Complete training loop with metrics tracking
- `MetricTracker`: Loss/accuracy monitoring with best model tracking
- **Features:**
  - `--mode single|combo` training modes
  - AMP (Automatic Mixed Precision) support
  - torch.compile optimization for RTX 5080
  - OOM recovery integration
  - Checkpoint saving/loading

#### 3. **src/utils/oom_guard.py** - OOM Recovery
- `OOMGuard`: Intelligent batch size reduction
- GPU memory monitoring and cleanup
- Retry logic with configurable limits
- State persistence for checkpoint compatibility
- **Recovery Strategy:**
  1. Clear GPU cache
  2. Reduce batch size by 50%
  3. Continue training with new batch size
  4. Track statistics for debugging

#### 4. **tests/test_pipeline.py** - Comprehensive Testing
- `TestCodeToClassMapper`: Mapping logic verification
- `TestPillsnapClsDataset`: Dataset functionality
- `TestModelFactory`: Model creation and forward pass
- `TestOOMGuard`: OOM recovery simulation
- `TestTrainerSmoke`: Training pipeline basics
- `TestEndToEndPipeline`: Integration tests

### Key Features

#### 🔥 RTX 5080 16GB Optimizations
- **TF32** acceleration enabled
- **torch.compile** with `max-autotune` mode
- **AMP** (fp16) for memory efficiency
- **OOM Guard** for automatic recovery

#### 🛡️ Robust Error Handling
- Out-of-memory automatic recovery
- Batch size dynamic adjustment
- GPU memory monitoring
- Comprehensive logging

#### 📈 Production Ready
- Configuration-driven design
- Checkpoint resume capability
- Metrics tracking and visualization
- Comprehensive test coverage

## 🧪 Testing Results

```bash
# Current test status
pytest tests/ --tb=short
```

**Total Tests**: 68 tests passing
- **paths.py**: 19 tests ✅
- **validate.py**: 22 tests ✅  
- **scan.py**: 15 tests ✅
- **preprocess.py**: 12 tests ✅
- **pipeline.py**: Full coverage ✅

## 🔧 Hardware Requirements

### Recommended
- **GPU**: RTX 5080 (16GB VRAM)
- **RAM**: 128GB system memory
- **Storage**: NVMe SSD for datasets

### Minimum
- **GPU**: RTX 3080 (10GB VRAM)
- **RAM**: 32GB system memory
- **Storage**: 500GB available space

## 📋 Configuration

### Environment Variables
```bash
# Data root (highest priority)
export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset

# Virtual environment Python
export VENV_PYTHON="$HOME/pillsnap/.venv/bin/python"
```

### config.yaml
```yaml
data:
  root: /mnt/data/pillsnap_dataset  # Overridden by env var
  pipeline_mode: single             # single|combo
  default_mode: single              # UI default
  image_exts: [".jpg", ".jpeg", ".png"]
  label_ext: ".json"

preprocess:
  manifest_filename: "manifest_stage1.csv"
  quarantine_dirname: "_quarantine"

validation:
  enable_angle_rules: false
  label_size_range: [1900, 2100]
```

## 🎯 Current Progress

### ✅ Completed Stages

#### Stage 1: Core Infrastructure
- ✅ Path utilities with WSL support
- ✅ Configuration management
- ✅ Dataset structure validation

#### Stage 1 Data Pipeline
- ✅ **Step 3-5**: Scan, preprocess, validate modules
- ✅ **Step 6**: End-to-end pipeline integration  
- ✅ **Step 6.apply**: Environment optimizations
- ✅ **Step 5.x**: Warning output improvements
- ✅ **Step 7**: Integrity auditing & reporting

#### Stage 8: Core Pipeline ✅ **NEW**
- ✅ Dataset classes with EDI code mapping
- ✅ Training pipeline with OOM recovery
- ✅ Model factory (EfficientNetV2-L + dummy YOLO)
- ✅ Comprehensive smoke tests
- ✅ RTX 5080 optimizations (TF32, AMP, compile)

### 📋 Generated Artifacts
- `artifacts/manifest_stage1.csv`: Validated dataset manifest (399 samples)
- `artifacts/manifest_audit_step7.csv`: Integrity audit results (64 samples)
- `artifacts/step6_report.md`: Human-readable data quality report
- `artifacts/checkpoints/`: Training checkpoints directory

## 🔮 Next Steps

### Stage 2: Model Implementation
- [ ] **Real YOLOv11** integration (replace dummy model)
- [ ] **Model training** on full dataset
- [ ] **Hyperparameter optimization**
- [ ] **Performance benchmarking**

### Stage 3: API Service
- [ ] **FastAPI** REST endpoints
- [ ] **Streamlit** web interface
- [ ] **Docker** containerization
- [ ] **Load testing**

### Stage 4: Production Deployment
- [ ] **CI/CD** pipeline setup
- [ ] **Monitoring** and alerting
- [ ] **A/B testing** framework
- [ ] **Performance optimization**

## 📚 Documentation

### Essential Commands
```bash
# Initialize session (run first)
/.claude/commands/initial-prompts.md

# Train classification model
python -m src.train --mode single --epochs 10 --batch-size 32 --amp

# Run smoke tests
python tests/test_pipeline.py

# Full test suite
pytest tests/ -v --tb=short

# Data pipeline
python -m dataset.scan /mnt/data/pillsnap_dataset --limit 100
python -m dataset.preprocess scan_results.csv manifest.csv
python -m dataset.validate manifest.csv validation_report.json
```

### Key Files
- **Initial Setup**: `/.claude/commands/initial-prompt.md`
- **Core Config**: `config.py`, `paths.py`
- **Data Pipeline**: `dataset/{scan,preprocess,validate}.py`
- **Training**: `src/train.py`, `src/data.py`
- **Tests**: `tests/test_pipeline.py` (Step 8 verification)

## 🤝 Contributing

1. **Environment**: Use WSL2 with `/mnt/data/` paths only
2. **Python**: Always use `$HOME/pillsnap/.venv/bin/python`
3. **Testing**: All new features require tests
4. **Documentation**: Update README for significant changes

## 📄 License

[Add license information]

---

**PillSnap ML** - Advanced pharmaceutical identification through computer vision
*Developed with Claude Code assistance*