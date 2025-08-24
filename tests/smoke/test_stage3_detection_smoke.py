#!/usr/bin/env python3
"""
Stage 3 Detection Smoke Tests

Quick smoke tests to verify YOLO detection functionality works with minimal data.
"""

import torch
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.core import PillSnapLogger


class DetectionSmokeTest:
    """Detection smoke test runner"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        
    def test_minimal_detection_training(self):
        """Test YOLO detection with single image"""
        
        self.logger.info("🎯 Starting minimal detection training test")
        
        # Setup test directory
        test_root = Path('/home/max16/pillsnap_data/test_detection_smoke')
        test_root.mkdir(exist_ok=True)
        
        images_dir = test_root / 'images'
        labels_dir = test_root / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Clean up previous test data
        for f in images_dir.glob('*'):
            f.unlink()
        for f in labels_dir.glob('*'):
            f.unlink()
        
        # Copy single image and label for testing
        source_img = Path('/home/max16/pillsnap_data/train/images/combination/TS_1_combo/K-000573-001866-006192-044834/K-000573-001866-006192-044834_0_2_0_2_70_000_200.png')
        source_label = Path('/home/max16/pillsnap_data/train/labels/combination_yolo/K-000573-001866-006192-044834_0_2_0_2_70_000_200.txt')
        
        if not source_img.exists() or not source_label.exists():
            self.logger.warning("Source files not found, using alternative...")
            # Find any available combination image-label pair
            combo_base = Path('/home/max16/pillsnap_data/train/images/combination')
            label_base = Path('/home/max16/pillsnap_data/train/labels/combination_yolo')
            
            for ts_dir in combo_base.glob('TS_*_combo'):
                for k_dir in ts_dir.iterdir():
                    if not k_dir.is_dir():
                        continue
                    for img in k_dir.glob('*_0_2_0_2_*.png'):
                        label = label_base / f"{img.stem}.txt"
                        if label.exists():
                            source_img = img
                            source_label = label
                            break
                    if source_img.exists():
                        break
                if source_img.exists():
                    break
        
        if not source_img.exists():
            raise FileNotFoundError("No suitable test image found")
            
        # Copy to test directory
        target_img = images_dir / source_img.name
        target_label = labels_dir / source_label.name
        
        shutil.copy2(source_img, target_img)
        shutil.copy2(source_label, target_label)
        
        self.logger.info(f"Test data prepared: {target_img.name}")
        
        # Create YOLO dataset configuration
        config = {
            'path': str(test_root),
            'train': 'images',
            'val': 'images',
            'names': {0: 'pill'},
            'nc': 1
        }
        
        config_path = test_root / 'dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run YOLO training (1 epoch)
        try:
            model = YOLO('yolo11m.pt')
            results = model.train(
                data=str(config_path),
                epochs=1,
                batch=1,
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save=False,
                verbose=False,
                workers=0,
                rect=False,
                cache=False,
                plots=False,
                exist_ok=True,
                project=None,
                name=None
            )
            
            self.logger.info("✅ Detection smoke test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Detection smoke test failed: {e}")
            return False
        
        finally:
            # Cleanup
            if test_root.exists():
                shutil.rmtree(test_root, ignore_errors=True)


def run_detection_smoke_test():
    """Run detection smoke test"""
    
    tester = DetectionSmokeTest()
    return tester.test_minimal_detection_training()


if __name__ == "__main__":
    success = run_detection_smoke_test()
    print(f"Detection smoke test: {'✅ PASSED' if success else '❌ FAILED'}")
    exit(0 if success else 1)