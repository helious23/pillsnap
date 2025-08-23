"""
Manifest 기반 훈련 데이터로더 - Two-Stage Pipeline 지원

CSV manifest 파일을 사용하여 Classification과 Detection을 모두 지원하는 데이터로더를 생성합니다.
- Classification: EfficientNetV2-L (384px)
- Detection: YOLOv11x (640px) + YOLO 형식 어노테이션
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import torchvision.transforms as transforms

from src.utils.core import PillSnapLogger


class ManifestDataset(Dataset):
    """Manifest CSV 기반 데이터셋"""
    
    def __init__(self, manifest_df: pd.DataFrame, transform=None):
        self.data = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.logger = PillSnapLogger(__name__)
        
        # K-코드를 정수 라벨로 매핑
        unique_k_codes = sorted(self.data['mapping_code'].unique())
        self.k_code_to_label = {k_code: idx for idx, k_code in enumerate(unique_k_codes)}
        self.label_to_k_code = {idx: k_code for k_code, idx in self.k_code_to_label.items()}
        
        self.logger.info(f"ManifestDataset 생성: {len(self.data)}개 샘플, {len(unique_k_codes)}개 클래스")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 이미지 로드
        image_path = Path(row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"이미지 로드 실패: {image_path} - {e}")
            # 기본 이미지 생성 (검은색 384x384)
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        # 라벨 (K-코드를 정수로 변환)
        k_code = row['mapping_code']
        label = self.k_code_to_label[k_code]
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ManifestDetectionDataset(Dataset):
    """Detection용 Manifest 데이터셋 (YOLO 형식)"""
    
    def __init__(
        self, 
        manifest_df: pd.DataFrame, 
        transform=None,
        image_size: int = 640
    ):
        # Combination 데이터만 필터링
        self.data = manifest_df[manifest_df['pill_type'] == 'combination'].reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.logger = PillSnapLogger(__name__)
        
        self.logger.info(f"ManifestDetectionDataset 생성: {len(self.data)}개 Combination 샘플")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 이미지 로드
        image_path = Path(row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"이미지 로드 실패: {image_path} - {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # YOLO 형식 라벨 생성 (간단화 - 실제로는 어노테이션 파일 필요)
        # 임시로 전체 이미지를 bbox로 설정 (실제 구현에서는 정확한 bbox 필요)
        boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])  # [x_center, y_center, width, height] normalized
        labels = torch.tensor([0])  # 클래스 0 (pill)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # YOLO 형식으로 반환
        targets = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, targets


class ManifestTrainingDataLoader:
    """Manifest 기반 훈련 데이터로더 관리자 - Two-Stage Pipeline 지원"""
    
    def __init__(
        self, 
        manifest_train_path: str, 
        manifest_val_path: str,
        batch_size: int = 32,
        image_size: int = 384,
        num_workers: int = 4,
        task: str = "classification"  # "classification" or "detection"
    ):
        self.manifest_train_path = Path(manifest_train_path)
        self.manifest_val_path = Path(manifest_val_path)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.task = task
        self.logger = PillSnapLogger(__name__)
        
        # Task에 따른 변환 설정
        if task == "classification":
            # EfficientNetV2-L용 (384px)
            self.train_transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.val_transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif task == "detection":
            # YOLOv11x용 (640px)
            self.train_transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
            
            self.val_transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
        
        else:
            raise ValueError(f"지원하지 않는 작업: {task}. 'classification' 또는 'detection'을 사용하세요.")
    
    def load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        """Manifest CSV 파일 로드"""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest 파일을 찾을 수 없습니다: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        self.logger.info(f"Manifest 로드: {len(df)}개 샘플")
        
        # 필수 컬럼 확인
        required_cols = ['image_path']
        if self.task == "classification":
            required_cols.append('mapping_code')
        elif self.task == "detection":
            required_cols.append('pill_type')
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        
        return df
    
    def get_train_loader(self) -> DataLoader:
        """학습 데이터로더 생성"""
        train_df = self.load_manifest(self.manifest_train_path)
        
        if self.task == "classification":
            train_dataset = ManifestDataset(train_df, transform=self.train_transform)
        elif self.task == "detection":
            train_dataset = ManifestDetectionDataset(
                train_df, 
                transform=self.train_transform,
                image_size=self.image_size
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        
        return train_loader
    
    def get_val_loader(self) -> DataLoader:
        """검증 데이터로더 생성"""
        val_df = self.load_manifest(self.manifest_val_path)
        
        if self.task == "classification":
            val_dataset = ManifestDataset(val_df, transform=self.val_transform)
        elif self.task == "detection":
            val_dataset = ManifestDetectionDataset(
                val_df,
                transform=self.val_transform,
                image_size=self.image_size
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        
        return val_loader
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """훈련/검증 데이터로더 생성 (레거시 호환성)"""
        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()
        
        # 메타데이터
        train_df = self.load_manifest(self.manifest_train_path)
        metadata = {
            'task': self.task,
            'num_train_samples': len(train_df),
            'num_val_samples': len(self.load_manifest(self.manifest_val_path)),
            'batch_size': self.batch_size,
            'image_size': self.image_size
        }
        
        if self.task == "classification":
            unique_classes = sorted(train_df['mapping_code'].unique())
            metadata['num_classes'] = len(unique_classes)
            metadata['classes'] = unique_classes
        
        return train_loader, val_loader, metadata


def test_manifest_dataloader():
    """ManifestTrainingDataLoader 테스트"""
    logger = PillSnapLogger(__name__)
    
    try:
        # Stage 2 manifest로 테스트
        manifest_path = "artifacts/stage2/manifest_ssd.csv"
        
        dataloader_manager = ManifestTrainingDataLoader(
            manifest_path=manifest_path,
            batch_size=4,
            val_split=0.2
        )
        
        train_loader, val_loader, metadata = dataloader_manager.get_dataloaders()
        
        logger.info("ManifestTrainingDataLoader 테스트 결과:")
        logger.info(f"  클래스 수: {metadata['num_classes']}")
        logger.info(f"  훈련 샘플: {metadata['train_size']}")
        logger.info(f"  검증 샘플: {metadata['val_size']}")
        
        # 첫 번째 배치 테스트
        for images, labels in train_loader:
            logger.info(f"  배치 shape: images={images.shape}, labels={labels.shape}")
            logger.info(f"  라벨 범위: {labels.min().item()} ~ {labels.max().item()}")
            break
            
        return True
        
    except Exception as e:
        logger.error(f"ManifestTrainingDataLoader 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    test_manifest_dataloader()