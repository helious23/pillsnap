"""
PillSnap 파이프라인 스모크 테스트

목적: Step 8 핵심 파이프라인의 기본 동작 검증
테스트 범위:
- 데이터셋 로딩 및 변환
- 모델 생성 및 forward 패스
- 학습 루프 기본 동작
- OOM 가드 기능
- 체크포인트 저장/로드

주의: 실제 학습이 아닌 최소한의 forward/backward 테스트만 수행
"""

import os
import tempfile
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image

# 프로젝트 모듈
import config
from src.data import (
    PillsnapClsDataset, 
    CodeToClassMapper,
    create_classification_transforms,
    create_dataloader
)
from src.train import ModelFactory, Trainer, MetricTracker
from src.utils.oom_guard import OOMGuard, handle_oom_error


class TestCodeToClassMapper:
    """CodeToClassMapper 테스트"""
    
    def test_mapper_creation(self):
        """매퍼 생성 및 기본 기능"""
        codes = ["K-001", "K-002", "K-003"]
        mapper = CodeToClassMapper(codes)
        
        assert mapper.num_classes == 3
        assert mapper.encode("K-001") == 0
        assert mapper.encode("K-002") == 1
        assert mapper.decode(0) == "K-001"
        assert mapper.get_all_codes() == ["K-001", "K-002", "K-003"]
    
    def test_mapper_sorting(self):
        """코드 정렬 확인"""
        codes = ["K-003", "K-001", "K-002"]  # 비정렬 입력
        mapper = CodeToClassMapper(codes)
        
        # 정렬된 순서로 ID 할당되어야 함
        assert mapper.encode("K-001") == 0
        assert mapper.encode("K-002") == 1
        assert mapper.encode("K-003") == 2
    
    def test_mapper_duplicates(self):
        """중복 코드 처리"""
        codes = ["K-001", "K-002", "K-001", "K-003"]
        mapper = CodeToClassMapper(codes)
        
        assert mapper.num_classes == 3  # 중복 제거
        assert "K-001" in mapper.code_to_id
    
    def test_mapper_errors(self):
        """에러 케이스"""
        mapper = CodeToClassMapper(["K-001", "K-002"])
        
        with pytest.raises(ValueError):
            mapper.encode("K-999")  # 없는 코드
        
        with pytest.raises(ValueError):
            mapper.decode(999)  # 없는 ID


class TestPillsnapClsDataset:
    """PillsnapClsDataset 테스트"""
    
    @pytest.fixture
    def temp_manifest(self, tmp_path):
        """임시 매니페스트 및 데이터 생성"""
        # 임시 이미지 생성
        img_paths = []
        label_paths = []
        codes = []
        
        for i in range(5):
            # 이미지 파일 생성
            img_path = tmp_path / f"train/image_{i:03d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
            img.save(img_path)
            
            # 라벨 파일 생성
            label_path = tmp_path / f"train/label_{i:03d}.json"
            label_data = {
                "code": f"K-{i:03d}",
                "metadata": {"test": True}
            }
            with open(label_path, 'w') as f:
                json.dump(label_data, f)
            
            img_paths.append(str(img_path))
            label_paths.append(str(label_path))
            codes.append(f"K-{i:03d}")
        
        # 매니페스트 CSV 생성
        manifest_df = pd.DataFrame({
            'image_path': img_paths,
            'label_path': label_paths,
            'code': codes,
            'is_pair': [True] * 5
        })
        
        manifest_path = tmp_path / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        
        return str(manifest_path), tmp_path
    
    def test_dataset_creation(self, temp_manifest):
        """데이터셋 생성 테스트"""
        manifest_path, _ = temp_manifest
        
        # 더미 config 객체
        class DummyConfig:
            pass
        
        cfg = DummyConfig()
        
        dataset = PillsnapClsDataset(
            manifest_path=manifest_path,
            config=cfg,
            split="train"
        )
        
        assert len(dataset) == 5
        assert dataset.code_mapper.num_classes == 5
    
    def test_dataset_getitem(self, temp_manifest):
        """데이터셋 아이템 접근 테스트"""
        manifest_path, _ = temp_manifest
        
        class DummyConfig:
            pass
        
        cfg = DummyConfig()
        transform = create_classification_transforms(augment=False)
        
        dataset = PillsnapClsDataset(
            manifest_path=manifest_path,
            config=cfg,
            split="train",
            transform=transform
        )
        
        # 첫 번째 샘플 로드
        image, target = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 384, 384)  # transform 후 크기
        assert isinstance(target, int)
        assert 0 <= target < dataset.code_mapper.num_classes
    
    def test_dataset_sample_info(self, temp_manifest):
        """샘플 정보 조회 테스트"""
        manifest_path, _ = temp_manifest
        
        class DummyConfig:
            pass
        
        cfg = DummyConfig()
        dataset = PillsnapClsDataset(manifest_path=manifest_path, config=cfg, split="train")
        
        info = dataset.get_sample_info(0)
        
        assert 'idx' in info
        assert 'code' in info
        assert 'class_id' in info
        assert 'image_path' in info
        assert 'label_path' in info
        assert 'label_data' in info


class TestModelFactory:
    """ModelFactory 테스트"""
    
    def test_classification_model(self):
        """분류 모델 생성 테스트"""
        num_classes = 10
        model = ModelFactory.create_classification_model(num_classes, pretrained=False)
        
        assert isinstance(model, nn.Module)
        
        # 테스트 입력
        x = torch.randn(2, 3, 384, 384)
        output = model(x)
        
        assert output.shape == (2, num_classes)
    
    def test_detection_model(self):
        """검출 모델 생성 테스트 (더미)"""
        num_classes = 5
        model = ModelFactory.create_detection_model(num_classes)
        
        assert isinstance(model, nn.Module)
        
        # 테스트 입력
        x = torch.randn(2, 3, 640, 640)
        output = model(x)
        
        assert output.shape == (2, num_classes, 5)  # batch, classes, (x,y,w,h,conf)


class TestOOMGuard:
    """OOMGuard 테스트"""
    
    def test_oom_guard_init(self):
        """OOMGuard 초기화"""
        oom_guard = OOMGuard(initial_batch_size=64, min_batch_size=4)
        
        assert oom_guard.current_batch_size == 64
        assert oom_guard.can_retry() == True
    
    def test_batch_size_reduction(self):
        """배치 크기 감소 테스트"""
        oom_guard = OOMGuard(initial_batch_size=64, min_batch_size=4)
        
        # 첫 번째 감소
        new_size = oom_guard.reduce_batch_size()
        assert new_size == 32  # 64 * 0.5
        
        # 두 번째 감소
        new_size = oom_guard.reduce_batch_size()
        assert new_size == 16  # 32 * 0.5
    
    def test_oom_handling(self):
        """OOM 처리 테스트"""
        oom_guard = OOMGuard(initial_batch_size=32, min_batch_size=4)
        
        # OOM 에러 시뮬레이션
        fake_error = RuntimeError("CUDA out of memory")
        success = handle_oom_error(fake_error, oom_guard, epoch=1, batch_idx=5)
        
        assert success == True
        assert oom_guard.current_batch_size == 16  # 감소됨
        assert oom_guard.state.total_oom_events == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_clearing(self):
        """메모리 정리 테스트"""
        oom_guard = OOMGuard(initial_batch_size=32)
        
        # 메모리 정리 실행
        memory_info = oom_guard.clear_gpu_memory()
        
        assert 'cleared_mb' in memory_info
        assert 'total_mb' in memory_info
        assert 'free_mb' in memory_info
        assert oom_guard.state.memory_cleared_count == 1


class TestMetricTracker:
    """MetricTracker 테스트"""
    
    def test_metric_tracking(self):
        """메트릭 추적 테스트"""
        tracker = MetricTracker()
        
        # 첫 번째 에포크
        tracker.update({'train_loss': 1.5, 'val_loss': 1.8, 'val_acc': 65.0}, epoch=0)
        
        # 두 번째 에포크 (개선)
        tracker.update({'train_loss': 1.2, 'val_loss': 1.4, 'val_acc': 70.0}, epoch=1)
        
        summary = tracker.get_summary()
        
        assert summary['best_epoch'] == 1  # val_loss 기준 베스트
        assert summary['val_loss']['best'] == 1.4
        assert summary['val_acc']['best'] == 70.0


class TestTrainerSmoke:
    """Trainer 스모크 테스트"""
    
    @pytest.fixture
    def temp_setup(self, tmp_path):
        """임시 환경 설정"""
        # 임시 매니페스트 생성
        img_paths = []
        label_paths = []
        codes = []
        
        for i in range(10):  # 작은 데이터셋
            img_path = tmp_path / f"train/img_{i}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img.save(img_path)
            
            label_path = tmp_path / f"train/label_{i}.json"
            with open(label_path, 'w') as f:
                json.dump({"code": f"K-{i:03d}"}, f)
            
            img_paths.append(str(img_path))
            label_paths.append(str(label_path))
            codes.append(f"K-{i:03d}")
        
        manifest_df = pd.DataFrame({
            'image_path': img_paths,
            'label_path': label_paths,
            'code': codes,
            'is_pair': [True] * 10
        })
        
        manifest_path = tmp_path / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        
        return str(manifest_path), tmp_path
    
    @patch('artifacts/manifest_stage1.csv')  # 실제 매니페스트 패치
    def test_trainer_initialization(self, temp_setup):
        """Trainer 초기화 테스트"""
        manifest_path, tmp_path = temp_setup
        
        # Mock args
        class MockArgs:
            mode = "single"
            batch_size = 4
            epochs = 1
            lr = 1e-3
            weight_decay = 1e-4
            workers = 0
            amp = False
            compile = False
            resume = None
        
        args = MockArgs()
        
        # Config 생성
        cfg = config.load_config()
        
        # 매니페스트 경로 패치
        with patch('src.train.PillsnapClsDataset') as mock_dataset:
            # 더미 데이터셋 설정
            mock_instance = MagicMock()
            mock_instance.code_mapper.num_classes = 5
            mock_instance.__len__.return_value = 10
            mock_dataset.return_value = mock_instance
            
            trainer = Trainer(cfg, args)
            
            assert trainer.device in [torch.device('cuda'), torch.device('cpu')]
            assert trainer.oom_guard.current_batch_size == 4
    
    def test_model_forward_pass(self):
        """모델 포워드 패스 테스트"""
        # 작은 분류 모델 생성
        model = ModelFactory.create_classification_model(num_classes=5, pretrained=False)
        model.eval()
        
        # 테스트 입력
        x = torch.randn(2, 3, 384, 384)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
    
    def test_loss_backward_pass(self):
        """손실 역전파 테스트"""
        model = ModelFactory.create_classification_model(num_classes=3, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 더미 데이터
        x = torch.randn(2, 3, 384, 384)
        targets = torch.tensor([0, 1])
        
        # Forward + Backward
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_transforms_pipeline(self):
        """변환 파이프라인 테스트"""
        transform = create_classification_transforms(augment=True)
        
        # PIL 이미지 생성
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # 변환 적용
        tensor = transform(img)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 384, 384)
        assert tensor.dtype == torch.float32
    
    def test_dataloader_creation(self):
        """데이터로더 생성 테스트"""
        # 더미 데이터셋
        from torch.utils.data import TensorDataset
        dummy_data = torch.randn(20, 3, 384, 384)
        dummy_targets = torch.randint(0, 5, (20,))
        dataset = TensorDataset(dummy_data, dummy_targets)
        
        # 데이터로더 생성
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 테스트에서는 0
        )
        
        assert len(dataloader) == 5  # 20 / 4
        
        # 첫 번째 배치 확인
        batch = next(iter(dataloader))
        images, targets = batch
        
        assert images.shape == (4, 3, 384, 384)
        assert targets.shape == (4,)


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-End 파이프라인 테스트"""
    
    def test_minimal_training_loop(self, tmp_path):
        """최소한의 학습 루프 테스트"""
        # 더미 모델 (매우 작은)
        class TinyModel(nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.fc = nn.Linear(384*384*3, num_classes)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        model = TinyModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 더미 데이터
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = torch.randn(8, 3, 384, 384)
        dummy_targets = torch.randint(0, 3, (8,))
        dataset = TensorDataset(dummy_data, dummy_targets)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # 미니 학습 루프
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        assert avg_loss > 0
        assert not torch.isnan(torch.tensor(avg_loss))
        
        # 체크포인트 저장 테스트
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # 체크포인트 로드 테스트
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'loss' in checkpoint


if __name__ == "__main__":
    # 스모크 테스트 실행
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running PillSnap pipeline smoke tests...")
    
    # 기본 컴포넌트 테스트
    print("✓ Testing CodeToClassMapper...")
    codes = ["K-001", "K-002", "K-003"]
    mapper = CodeToClassMapper(codes)
    assert mapper.num_classes == 3
    
    print("✓ Testing ModelFactory...")
    model = ModelFactory.create_classification_model(5, pretrained=False)
    x = torch.randn(1, 3, 384, 384)
    output = model(x)
    assert output.shape == (1, 5)
    
    print("✓ Testing OOMGuard...")
    oom_guard = OOMGuard(32, min_batch_size=4)
    assert oom_guard.current_batch_size == 32
    
    print("✓ Testing transforms...")
    transform = create_classification_transforms(augment=False)
    img = Image.new('RGB', (256, 256))
    tensor = transform(img)
    assert tensor.shape == (3, 384, 384)
    
    print("✅ All smoke tests passed!")
    print("\nTo run full test suite:")
    print("  pytest tests/test_pipeline.py -v")
    print("  pytest tests/test_pipeline.py::TestEndToEndPipeline -v  # integration tests")