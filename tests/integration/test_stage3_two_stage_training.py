#!/usr/bin/env python3
"""
Stage 3 Two-Stage Pipeline 통합 테스트

Detection + Classification 통합 학습 시스템 검증:
- YOLOv11x Detection 기능 검증
- EfficientNetV2-L Classification 성능 유지
- 교차 학습 파이프라인 안정성
- GPU 메모리 관리 및 최적화 검증
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.train_stage3_two_stage import Stage3TwoStageTrainer, TwoStageTrainingConfig
from src.data.dataloader_manifest_training import ManifestTrainingDataLoader
from src.utils.core import PillSnapLogger


class TestStage3TwoStageTrainingIntegration:
    """Stage 3 Two-Stage Pipeline 통합 테스트"""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_manifest_data(self, temp_dir):
        """Mock manifest 데이터"""
        # 훈련 manifest
        train_data = {
            'image_path': [f'/fake/path/img_{i}.jpg' for i in range(100)],
            'edi_code': [f'K-{i//10:06d}' for i in range(100)],  # 10개 클래스
            'mapping_code': [f'K-{i//10:06d}' for i in range(100)],
            'image_type': ['single'] * 95 + ['combination'] * 5  # 95:5 비율
        }
        train_manifest = temp_dir / "train.csv"
        pd.DataFrame(train_data).to_csv(train_manifest, index=False)

        # 검증 manifest
        val_data = {
            'image_path': [f'/fake/path/val_img_{i}.jpg' for i in range(20)],
            'edi_code': [f'K-{i//2:06d}' for i in range(20)],
            'mapping_code': [f'K-{i//2:06d}' for i in range(20)],
            'image_type': ['single'] * 19 + ['combination'] * 1
        }
        val_manifest = temp_dir / "val.csv"
        pd.DataFrame(val_data).to_csv(val_manifest, index=False)

        return {
            'train_manifest': str(train_manifest),
            'val_manifest': str(val_manifest),
            'num_classes': 10,
            'total_samples': 100
        }

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Mock config.yaml"""
        config_content = {
            'paths': {
                'data_root': '/tmp/fake_data',
                'exp_dir': '/tmp/fake_exp'
            },
            'pipeline': {
                'mode': 'single',
                'single_mode': {
                    'model': 'efficientnetv2_l'
                },
                'combo_mode': {
                    'detector': 'yolov11x',
                    'classifier': 'efficientnetv2_l'
                }
            },
            'data': {
                'root': '/tmp/fake_data'
            },
            'progressive_validation': {
                'enabled': True,
                'current_stage': 3,
                'stage_configs': {
                    'stage_3': {
                        'purpose': 'two_stage_pipeline_validation',
                        'max_samples': 100000,
                        'max_classes': 1000,
                        'target_ratio': {'single': 0.95, 'combination': 0.05},
                        'focus': 'two_stage_pipeline',
                        'skip_detection': False,
                        'target_metrics': {
                            'classification_accuracy': 0.85,
                            'detection_map_0_5': 0.30
                        }
                    }
                }
            }
        }
        
        config_path = temp_dir / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        return str(config_path)

    def test_two_stage_trainer_initialization(self, mock_manifest_data, mock_config):
        """Two-Stage 학습기 초기화 테스트"""
        
        trainer = Stage3TwoStageTrainer(
            config_path=mock_config,
            manifest_train=mock_manifest_data['train_manifest'],
            manifest_val=mock_manifest_data['val_manifest'],
            device='cpu'  # CI/CD 환경 고려
        )
        
        # 초기화 검증
        assert trainer.device.type == 'cpu'
        assert trainer.seed == 42
        assert trainer.best_classification_accuracy == 0.0
        assert trainer.best_detection_map == 0.0
        assert trainer.training_config.target_classification_accuracy == 0.85
        assert trainer.training_config.target_detection_map == 0.30
        
        print("✅ Two-Stage Trainer 초기화 성공")

    @patch('src.training.train_stage3_two_stage.create_pillsnap_classifier')
    @patch('src.training.train_stage3_two_stage.create_pillsnap_detector')
    def test_models_setup(self, mock_detector, mock_classifier, mock_manifest_data, mock_config):
        """모델 설정 테스트"""
        
        # Mock 모델 생성
        mock_classifier_model = Mock()
        mock_classifier_model.to.return_value = mock_classifier_model
        mock_classifier.return_value = mock_classifier_model
        
        mock_detector_model = Mock()
        mock_detector_model.to.return_value = mock_detector_model
        mock_detector.return_value = mock_detector_model
        
        trainer = Stage3TwoStageTrainer(
            config_path=mock_config,
            manifest_train=mock_manifest_data['train_manifest'],
            manifest_val=mock_manifest_data['val_manifest'],
            device='cpu'
        )
        
        trainer.setup_models()
        
        # Classification 모델 검증
        mock_classifier.assert_called_once_with(
            num_classes=10,
            model_name="efficientnetv2_l",
            pretrained=True,
            device=trainer.device
        )
        
        # Detection 모델 검증  
        mock_detector.assert_called_once_with(
            num_classes=1,
            model_size="yolo11x",
            input_size=640,
            device=trainer.device
        )
        
        assert trainer.classifier is not None
        assert trainer.detector is not None
        
        print("✅ Two-Stage 모델 설정 성공")

    def test_data_loaders_setup(self, mock_manifest_data, mock_config):
        """데이터 로더 설정 테스트"""
        
        with patch('src.training.train_stage3_two_stage.ManifestTrainingDataLoader') as mock_loader:
            # Mock 데이터 로더
            mock_classification_loader = Mock()
            mock_detection_loader = Mock()
            
            def loader_side_effect(*args, **kwargs):
                if kwargs.get('task') == 'classification':
                    return mock_classification_loader
                else:
                    return mock_detection_loader
            
            mock_loader.side_effect = loader_side_effect
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_data_loaders()
            
            # 두 번 호출되어야 함 (classification + detection)
            assert mock_loader.call_count == 2
            assert trainer.classification_dataloader is not None
            assert trainer.detection_dataloader is not None
            
        print("✅ Two-Stage 데이터 로더 설정 성공")

    @patch('src.training.train_stage3_two_stage.torch.optim.AdamW')
    def test_optimizers_setup(self, mock_optimizer, mock_manifest_data, mock_config):
        """옵티마이저 설정 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier'), \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector'):
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_models()
            classifier_optimizer, detector_optimizer = trainer.setup_optimizers()
            
            # 두 개의 옵티마이저가 생성되어야 함
            assert mock_optimizer.call_count == 2
            
        print("✅ Two-Stage 옵티마이저 설정 성공")

    def test_classification_training_epoch(self, mock_manifest_data, mock_config):
        """Classification 에포크 학습 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier') as mock_create_cls, \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector'):
            
            # Mock 분류기
            mock_classifier = Mock()
            mock_classifier.train = Mock()
            mock_classifier.return_value = torch.randn(4, 10)  # 4 batch, 10 classes
            mock_create_cls.return_value = mock_classifier
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_models()
            
            # Mock 데이터 로더
            mock_data = [(torch.randn(4, 3, 384, 384), torch.randint(0, 10, (4,))) for _ in range(2)]
            trainer.classification_dataloader = Mock()
            trainer.classification_dataloader.get_train_loader.return_value = mock_data
            
            # Mock 옵티마이저
            optimizer = Mock()
            scaler = Mock()
            
            results = trainer.train_classification_epoch(optimizer, scaler, epoch=1)
            
            assert 'classification_loss' in results
            assert 'classification_accuracy' in results
            assert isinstance(results['classification_loss'], float)
            assert isinstance(results['classification_accuracy'], float)
            
        print("✅ Classification 에포크 학습 테스트 성공")

    def test_detection_training_epoch(self, mock_manifest_data, mock_config):
        """Detection 에포크 학습 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier'), \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector') as mock_create_det:
            
            # Mock 검출기
            mock_detector = Mock()
            mock_detector.train.return_value = {'loss': 0.1, 'map': 0.15}
            mock_create_det.return_value = mock_detector
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_models()
            
            # Mock 데이터 로더
            trainer.detection_dataloader = Mock()
            trainer.detection_dataloader.get_train_loader.return_value = []
            
            optimizer = Mock()
            
            results = trainer.train_detection_epoch(optimizer, epoch=1)
            
            assert 'detection_loss' in results
            assert 'detection_map' in results
            
        print("✅ Detection 에포크 학습 테스트 성공")

    def test_model_validation(self, mock_manifest_data, mock_config):
        """모델 검증 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier') as mock_create_cls, \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector'):
            
            # Mock 분류기
            mock_classifier = Mock()
            mock_classifier.eval = Mock()
            mock_classifier.return_value = torch.randn(4, 10)
            mock_create_cls.return_value = mock_classifier
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_models()
            
            # Mock 검증 데이터
            mock_val_data = [(torch.randn(4, 3, 384, 384), torch.randint(0, 10, (4,))) for _ in range(2)]
            trainer.classification_dataloader = Mock()
            trainer.classification_dataloader.get_val_loader.return_value = mock_val_data
            
            results = trainer.validate_models()
            
            assert 'val_classification_accuracy' in results
            assert 'val_detection_map' in results
            assert 0 <= results['val_classification_accuracy'] <= 1
            assert 0 <= results['val_detection_map'] <= 1
            
        print("✅ 모델 검증 테스트 성공")

    def test_checkpoint_saving(self, mock_manifest_data, mock_config, temp_dir):
        """체크포인트 저장 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier') as mock_create_cls, \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector') as mock_create_det:
            
            mock_classifier = Mock()
            mock_classifier.state_dict.return_value = {'layer': torch.randn(10, 10)}
            mock_create_cls.return_value = mock_classifier
            
            mock_detector = Mock()
            mock_detector.save = Mock()
            mock_create_det.return_value = mock_detector
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            trainer.setup_models()
            trainer.best_classification_accuracy = 0.85
            
            # 임시 체크포인트 디렉토리 설정
            with patch('pathlib.Path.mkdir'):
                with patch('torch.save') as mock_torch_save:
                    trainer.save_checkpoint('classification', 'best')
                    mock_torch_save.assert_called_once()
                
                trainer.save_checkpoint('detection', 'best')
                mock_detector.save.assert_called_once()
                
        print("✅ 체크포인트 저장 테스트 성공")

    @pytest.mark.slow
    def test_short_training_cycle(self, mock_manifest_data, mock_config):
        """짧은 학습 사이클 테스트"""
        
        with patch('src.training.train_stage3_two_stage.create_pillsnap_classifier') as mock_create_cls, \
             patch('src.training.train_stage3_two_stage.create_pillsnap_detector') as mock_create_det:
            
            # Mock 모델들
            mock_classifier = Mock()
            mock_classifier.train = Mock()
            mock_classifier.eval = Mock()
            mock_classifier.to.return_value = mock_classifier
            mock_classifier.return_value = torch.randn(2, 10)
            mock_create_cls.return_value = mock_classifier
            
            mock_detector = Mock()
            mock_detector.parameters.return_value = [torch.randn(10, 10)]
            mock_detector.train.return_value = {'loss': 0.1}
            mock_detector.save = Mock()
            mock_create_det.return_value = mock_detector
            
            trainer = Stage3TwoStageTrainer(
                config_path=mock_config,
                manifest_train=mock_manifest_data['train_manifest'],
                manifest_val=mock_manifest_data['val_manifest'],
                device='cpu'
            )
            
            # 짧은 학습 설정
            trainer.training_config.max_epochs = 2
            trainer.training_config.target_classification_accuracy = 0.1  # 낮은 목표
            trainer.training_config.target_detection_map = 0.1
            
            # Mock 데이터 로더
            mock_train_data = [(torch.randn(2, 3, 384, 384), torch.randint(0, 10, (2,))) for _ in range(2)]
            mock_val_data = [(torch.randn(2, 3, 384, 384), torch.randint(0, 10, (2,))) for _ in range(2)]
            
            trainer.classification_dataloader = Mock()
            trainer.classification_dataloader.get_train_loader.return_value = mock_train_data
            trainer.classification_dataloader.get_val_loader.return_value = mock_val_data
            
            trainer.detection_dataloader = Mock()
            trainer.detection_dataloader.get_train_loader.return_value = []
            
            with patch('torch.manual_seed'), \
                 patch('pathlib.Path.mkdir'), \
                 patch('torch.save'):
                
                results = trainer.train()
                
                assert results['training_completed'] == True
                assert results['epochs_completed'] >= 1
                assert 'best_classification_accuracy' in results
                assert 'best_detection_map' in results
                assert 'target_achieved' in results
                
        print("✅ 짧은 학습 사이클 테스트 성공")


class TestStage3TwoStageProductionReadiness:
    """Stage 3 Two-Stage 프로덕션 준비 테스트"""

    def test_training_config_validation(self):
        """학습 설정 검증"""
        
        config = TwoStageTrainingConfig()
        
        # 기본값 검증
        assert config.max_epochs == 20
        assert config.learning_rate_classifier == 2e-4
        assert config.learning_rate_detector == 1e-3
        assert config.batch_size == 16
        assert config.interleaved_training == True
        assert config.mixed_precision == True
        assert config.torch_compile == True
        assert config.target_classification_accuracy == 0.85
        assert config.target_detection_map == 0.30
        
        print("✅ 학습 설정 검증 성공")

    def test_memory_optimization_settings(self):
        """메모리 최적화 설정 검증"""
        
        config = TwoStageTrainingConfig()
        
        # RTX 5080 최적화 설정 검증
        assert config.mixed_precision == True, "Mixed Precision 필수"
        assert config.channels_last == True, "Channels Last 최적화 필수"
        assert config.torch_compile == True, "torch.compile 최적화 필수"
        
        # 배치 크기 적정성
        assert 8 <= config.batch_size <= 32, "배치 크기가 RTX 5080에 적합해야 함"
        
        print("✅ 메모리 최적화 설정 검증 성공")

    def test_target_metrics_realistic(self):
        """목표 지표 현실성 검증"""
        
        config = TwoStageTrainingConfig()
        
        # Classification 목표 (Stage 1-2 성과 기반)
        assert 0.8 <= config.target_classification_accuracy <= 0.9, "분류 정확도 목표가 현실적이어야 함"
        
        # Detection 목표 (5% 데이터, 기능 검증용)
        assert 0.2 <= config.target_detection_map <= 0.4, "검출 mAP 목표가 5% 데이터에 적합해야 함"
        
        print("✅ 목표 지표 현실성 검증 성공")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])