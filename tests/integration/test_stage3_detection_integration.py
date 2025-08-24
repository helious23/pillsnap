#!/usr/bin/env python3
"""
Stage 3 Detection Integration Tests

Tests the integration between Stage 3 Two-Stage Pipeline and YOLO Detection,
including dataset configuration and data structure validation.
"""

import yaml
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.training.train_stage3_two_stage import Stage3TwoStageTrainer


class TestStage3DetectionIntegration:
    """Stage 3 Detection Integration Test Suite"""
    
    def test_yolo_dataset_config_generation(self):
        """Test YOLO dataset configuration generation"""
        
        # Initialize trainer
        trainer = Stage3TwoStageTrainer()
        
        # Generate YOLO dataset configuration
        dataset_yaml = trainer._create_yolo_dataset_config()
        
        # Verify configuration file exists
        assert dataset_yaml.exists(), f"YOLO config file not created: {dataset_yaml}"
        
        # Load and verify configuration content
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'names', 'nc']
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"
        
        # Verify configuration values
        assert config['nc'] == 1, "Should have 1 class (pill)"
        assert config['names'] == {0: 'pill'}, "Should have pill class mapping"
        assert 'yolo_dataset' in config['path'], "Should point to YOLO dataset directory"
        
    def test_yolo_dataset_structure(self):
        """Test YOLO dataset directory structure"""
        
        trainer = Stage3TwoStageTrainer()
        dataset_yaml = trainer._create_yolo_dataset_config()
        
        # Load configuration
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
        
        dataset_root = Path(config['path'])
        images_dir = dataset_root / 'images'
        labels_dir = dataset_root / 'labels'
        
        # Verify directory structure
        assert dataset_root.exists(), "YOLO dataset root should exist"
        assert images_dir.exists(), "Images directory should exist"
        assert labels_dir.exists(), "Labels directory should exist"
        
        # Count files
        image_files = list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        # Verify file counts are reasonable (should be > 10,000)
        assert len(image_files) > 10000, f"Expected >10k images, got {len(image_files)}"
        assert len(label_files) > 10000, f"Expected >10k labels, got {len(label_files)}"
        
        # Verify matching rate is high (>99%)
        matching_rate = len(label_files) / len(image_files)
        assert matching_rate > 0.99, f"Low matching rate: {matching_rate:.3f}"
        
    def test_image_label_correspondence(self):
        """Test that images and labels correspond correctly"""
        
        trainer = Stage3TwoStageTrainer()
        dataset_yaml = trainer._create_yolo_dataset_config()
        
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
        
        dataset_root = Path(config['path'])
        images_dir = dataset_root / 'images'
        labels_dir = dataset_root / 'labels'
        
        # Sample check: verify first 10 image-label pairs
        image_files = sorted(images_dir.glob('*.png'))[:10]
        
        for img_file in image_files:
            expected_label = labels_dir / f"{img_file.stem}.txt"
            assert expected_label.exists(), f"Missing label for {img_file.name}"
            
            # Verify label file has content
            assert expected_label.stat().st_size > 0, f"Empty label file: {expected_label.name}"


def test_detection_integration_standalone():
    """Standalone test function for direct execution"""
    
    try:
        trainer = Stage3TwoStageTrainer()
        dataset_yaml = trainer._create_yolo_dataset_config()
        
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
        
        dataset_root = Path(config['path'])
        images_dir = dataset_root / 'images'
        labels_dir = dataset_root / 'labels'
        
        image_count = len(list(images_dir.glob('*.png')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"✅ YOLO 데이터셋 설정 성공")
        print(f"   - 설정 파일: {dataset_yaml}")
        print(f"   - 데이터셋 경로: {dataset_root}")
        print(f"   - 이미지: {image_count:,}개")
        print(f"   - 라벨: {label_count:,}개")
        print(f"   - 매칭률: {(label_count/image_count*100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection 통합 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    success = test_detection_integration_standalone()
    exit(0 if success else 1)