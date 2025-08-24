#!/usr/bin/env python3
"""
Stage 3 Two-Stage Setup Integration Tests

Tests the setup and initialization of the Two-Stage Pipeline system,
including model loading, data loading, and configuration validation.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.training.train_stage3_two_stage import Stage3TwoStageTrainer, TwoStageTrainingConfig


class TestStage3TwoStageSetup:
    """Two-Stage Pipeline Setup Test Suite"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization with default parameters"""
        
        trainer = Stage3TwoStageTrainer(
            config_path="/home/max16/pillsnap/config.yaml",
            manifest_train="/home/max16/pillsnap/artifacts/stage3/manifest_train.csv",
            manifest_val="/home/max16/pillsnap/artifacts/stage3/manifest_val.csv"
        )
        
        assert trainer.device.type in ['cuda', 'cpu'], "Device should be cuda or cpu"
        assert trainer.manifest_train.exists(), "Training manifest should exist"
        assert trainer.manifest_val.exists(), "Validation manifest should exist"
        
    def test_training_config(self):
        """Test training configuration parameters"""
        
        config = TwoStageTrainingConfig()
        
        # Verify default values
        assert config.max_epochs > 0, "Max epochs should be positive"
        assert config.batch_size > 0, "Batch size should be positive"
        assert 0 < config.learning_rate_classifier < 1, "Classifier LR should be in valid range"
        assert 0 < config.learning_rate_detector < 1, "Detector LR should be in valid range"
        
        # Verify optimization flags
        assert isinstance(config.mixed_precision, bool), "Mixed precision should be boolean"
        assert isinstance(config.torch_compile, bool), "Torch compile should be boolean"
        
    def test_models_setup(self):
        """Test model initialization"""
        
        config = TwoStageTrainingConfig()
        config.max_epochs = 1  # Minimal for testing
        config.batch_size = 4
        
        trainer = Stage3TwoStageTrainer()
        trainer.training_config = config
        
        # Initialize models
        trainer.setup_models()
        
        assert trainer.classifier is not None, "Classifier should be initialized"
        assert trainer.detector is not None, "Detector should be initialized"
        
        # Verify models are on correct device
        assert next(trainer.classifier.parameters()).device.type == trainer.device.type
        
    def test_data_loaders_setup(self):
        """Test data loader initialization"""
        
        trainer = Stage3TwoStageTrainer()
        trainer.setup_data_loaders()
        
        assert trainer.classification_dataloader is not None, "Classification dataloader should exist"
        assert trainer.detection_dataloader is not None, "Detection dataloader should exist"
        
        # Verify data loader can provide samples
        train_loader = trainer.classification_dataloader.get_train_loader()
        assert len(train_loader) > 0, "Training loader should have samples"
        
        val_loader = trainer.classification_dataloader.get_val_loader()
        assert len(val_loader) > 0, "Validation loader should have samples"


def test_two_stage_setup_minimal():
    """Minimal setup test for direct execution"""
    
    try:
        # Test basic initialization
        trainer = Stage3TwoStageTrainer()
        print("✅ Trainer initialization successful")
        
        # Test model setup
        trainer.setup_models()
        print("✅ Models setup successful")
        
        # Test data setup
        trainer.setup_data_loaders() 
        print("✅ Data loaders setup successful")
        
        # Verify GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
        else:
            print("⚠️ GPU not available, using CPU")
        
        print("🎉 Two-Stage setup test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Two-Stage setup test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_two_stage_setup_minimal()
    exit(0 if success else 1)