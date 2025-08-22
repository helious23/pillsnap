"""
Single Pill Training DataLoader
단일 약품 학습용 데이터로더

EfficientNetV2-S 분류용:
- 384x384 이미지 전처리
- Progressive Validation 샘플링 통합
- RTX 5080 최적화 (배치 처리, 메모리 효율성)
- 데이터 증강 및 캐싱 지원
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.progressive_validation_sampler import ProgressiveValidationSampler, Stage1SamplingStrategy
from src.data.image_preprocessing_factory import TwoStageImagePreprocessor, PipelineStage
from src.utils.core import PillSnapLogger, load_config
from src.utils.system_optimization import get_dataloader_kwargs, log_system_optimization


class SinglePillDataset(Dataset):
    """단일 약품 데이터셋"""
    
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        class_to_idx: Dict[str, int],
        preprocessor: TwoStageImagePreprocessor,
        is_training: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.logger = PillSnapLogger(__name__)
        
        assert len(image_paths) == len(labels), "이미지와 라벨 수가 일치하지 않음"
        
        self.logger.info(f"SinglePillDataset 초기화: {len(image_paths)}개 이미지, {len(class_to_idx)}개 클래스")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 분류용 전처리 (384x384)
            success, processed_tensor, info = self.preprocessor.preprocess_for_classification(
                image_path, 
                is_training=self.is_training
            )
            
            if not success:
                # 전처리 실패 시 기본 텐서 반환
                self.logger.warning(f"전처리 실패: {image_path}, 기본 텐서 사용")
                processed_tensor = torch.zeros(3, 384, 384)
            
            return processed_tensor, label
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패 {image_path}: {e}")
            # 에러 시 기본값 반환
            return torch.zeros(3, 384, 384), label
    
    def get_class_name(self, class_idx: int) -> str:
        """클래스 인덱스를 클래스명으로 변환"""
        return self.idx_to_class.get(class_idx, f"Unknown_{class_idx}")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """클래스별 데이터 분포 반환"""
        distribution = {}
        for label in self.labels:
            class_name = self.get_class_name(label)
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


class SinglePillTrainingDataLoader:
    """단일 약품 학습용 데이터로더 매니저"""
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        stage: int = 1,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):
        # 설정에서 데이터 루트 가져오기
        if data_root is None:
            config = load_config()
            data_root = config['data']['root']
        
        self.data_root = Path(data_root)
        self.stage = stage
        self.batch_size = batch_size
        
        # 시스템 최적화 설정 적용
        dataloader_config = get_dataloader_kwargs(stage)
        self.num_workers = num_workers if num_workers is not None else dataloader_config['num_workers']
        self.pin_memory = pin_memory if pin_memory is not None else dataloader_config['pin_memory']
        self.persistent_workers = dataloader_config.get('persistent_workers', False)
        self.prefetch_factor = dataloader_config.get('prefetch_factor', 2)
        self.drop_last = dataloader_config.get('drop_last', True)
        self.multiprocessing_context = dataloader_config.get('multiprocessing_context', None)
        
        self.logger = PillSnapLogger(__name__)
        
        # 시스템 최적화 정보 로깅 (첫 번째 인스턴스에서만)
        if not hasattr(SinglePillTrainingDataLoader, '_optimization_logged'):
            log_system_optimization()
            SinglePillTrainingDataLoader._optimization_logged = True
        
        # 전처리기 초기화
        self.preprocessor = TwoStageImagePreprocessor()
        
        # 샘플링 전략
        if stage == 1:
            self.strategy = Stage1SamplingStrategy(target_images=5000, target_classes=50)
        elif stage == 2:
            # Stage 2: 25K 샘플, 250 클래스
            from src.data.progressive_validation_sampler import Stage2SamplingStrategy
            self.strategy = Stage2SamplingStrategy(target_images=25000, target_classes=250)
        elif stage == 3:
            # Stage 3: 100K 샘플, 1000 클래스  
            from src.data.progressive_validation_sampler import Stage3SamplingStrategy
            self.strategy = Stage3SamplingStrategy(target_images=100000, target_classes=1000)
        elif stage == 4:
            # Stage 4: 500K 샘플, 4523 클래스
            from src.data.progressive_validation_sampler import Stage4SamplingStrategy
            self.strategy = Stage4SamplingStrategy(target_images=500000, target_classes=4523)
        else:
            raise NotImplementedError(f"Stage {stage} 지원하지 않음 (1-4만 지원)")
        
        self.logger.info(f"SinglePillTrainingDataLoader 초기화 (Stage {stage})")
        
    def prepare_datasets(
        self,
        validation_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[SinglePillDataset, SinglePillDataset]:
        """학습/검증 데이터셋 준비"""
        
        self.logger.step("데이터셋 준비", f"Stage {self.stage} 단일 약품 샘플링 및 분할")
        
        try:
            # Progressive Validation 샘플러로 데이터 생성
            sampler = ProgressiveValidationSampler(str(self.data_root), self.strategy)
            stage_sample = sampler.generate_stage1_sample()
            
            # 단일 약품만 필터링
            single_samples = []
            for k_code, sample_data in stage_sample['samples'].items():
                single_images = sample_data.get('single_images', [])
                single_samples.extend([(Path(img_path), k_code) for img_path in single_images])
            
            # 클래스 매핑 생성
            unique_classes = sorted(list(set([k_code for _, k_code in single_samples])))
            class_to_idx = {k_code: idx for idx, k_code in enumerate(unique_classes)}
            
            # 이미지 경로와 라벨 분리
            image_paths = [img_path for img_path, _ in single_samples]
            labels = [class_to_idx[k_code] for _, k_code in single_samples]
            
            self.logger.info(f"총 이미지: {len(image_paths)}개, 클래스: {len(unique_classes)}개")
            
            # 학습/검증 분할
            random.seed(random_seed)
            indices = list(range(len(image_paths)))
            random.shuffle(indices)
            
            split_idx = int(len(indices) * (1 - validation_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # 학습 데이터셋
            train_image_paths = [image_paths[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]
            
            train_dataset = SinglePillDataset(
                image_paths=train_image_paths,
                labels=train_labels,
                class_to_idx=class_to_idx,
                preprocessor=self.preprocessor,
                is_training=True
            )
            
            # 검증 데이터셋
            val_image_paths = [image_paths[i] for i in val_indices]
            val_labels = [labels[i] for i in val_indices]
            
            val_dataset = SinglePillDataset(
                image_paths=val_image_paths,
                labels=val_labels,
                class_to_idx=class_to_idx,
                preprocessor=self.preprocessor,
                is_training=False
            )
            
            self.logger.info(f"학습 데이터: {len(train_dataset)}개")
            self.logger.info(f"검증 데이터: {len(val_dataset)}개")
            self.logger.success("데이터셋 준비 완료")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.error(f"데이터셋 준비 실패: {e}")
            raise
    
    def create_dataloaders(
        self,
        train_dataset: SinglePillDataset,
        val_dataset: SinglePillDataset,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """데이터로더 생성"""
        
        try:
            # 학습 데이터로더 (시스템 최적화 적용)
            train_kwargs = {
                'dataset': train_dataset,
                'batch_size': self.batch_size,
                'shuffle': shuffle_train,
                'num_workers': self.num_workers,
                'pin_memory': self.pin_memory,
                'drop_last': self.drop_last
            }
            
            # 추가 최적화 옵션 적용 (num_workers > 0일 때만)
            if self.num_workers > 0:
                if self.persistent_workers:
                    train_kwargs['persistent_workers'] = True
                if self.prefetch_factor is not None:
                    train_kwargs['prefetch_factor'] = self.prefetch_factor
                if self.multiprocessing_context is not None:
                    import multiprocessing as mp
                    train_kwargs['multiprocessing_context'] = mp.get_context(self.multiprocessing_context)
            
            train_loader = DataLoader(**train_kwargs)
            
            # 검증 데이터로더 (동일한 최적화 적용, 셔플 제외)
            val_kwargs = train_kwargs.copy()
            val_kwargs.update({
                'dataset': val_dataset,
                'shuffle': False,
                'drop_last': False  # 검증에서는 모든 데이터 사용
            })
            
            val_loader = DataLoader(**val_kwargs)
            
            self.logger.info(f"데이터로더 생성 완료")
            self.logger.info(f"  학습 배치 수: {len(train_loader)}")
            self.logger.info(f"  검증 배치 수: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"데이터로더 생성 실패: {e}")
            raise
    
    def get_stage_dataloaders(
        self,
        validation_split: float = 0.2,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Stage별 데이터로더 원스톱 생성"""
        
        # 데이터셋 준비
        train_dataset, val_dataset = self.prepare_datasets(validation_split)
        
        # 데이터로더 생성
        train_loader, val_loader = self.create_dataloaders(
            train_dataset, val_dataset, shuffle_train
        )
        
        # 메타정보
        metadata = {
            'num_classes': len(train_dataset.class_to_idx),
            'class_to_idx': train_dataset.class_to_idx,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'batch_size': self.batch_size,
            'stage': self.stage
        }
        
        return train_loader, val_loader, metadata


def main():
    """단일 약품 데이터로더 테스트"""
    print("📊 Single Pill Training DataLoader Test")
    print("=" * 60)
    
    try:
        # 데이터로더 매니저 생성
        dataloader_manager = SinglePillTrainingDataLoader(
            stage=1,
            batch_size=16,  # 테스트용 작은 배치
            num_workers=2
        )
        
        # 데이터로더 생성
        train_loader, val_loader, metadata = dataloader_manager.get_stage_dataloaders()
        
        print(f"✅ 데이터로더 생성 성공")
        print(f"   클래스 수: {metadata['num_classes']}")
        print(f"   학습 데이터: {metadata['train_size']}개")
        print(f"   검증 데이터: {metadata['val_size']}개")
        print(f"   배치 수: 학습 {len(train_loader)}, 검증 {len(val_loader)}")
        
        # 첫 번째 배치 테스트
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        print(f"   배치 모양: 이미지 {images.shape}, 라벨 {labels.shape}")
        print(f"   이미지 범위: [{images.min():.3f}, {images.max():.3f}]")
        
        print("\n✅ 단일 약품 데이터로더 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()