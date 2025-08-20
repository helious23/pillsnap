"""
Manifest 기반 훈련 데이터로더

CSV manifest 파일을 사용하여 정확한 이미지-라벨 쌍으로 데이터로더를 생성합니다.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import json
from typing import Tuple, Dict, Any
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


class ManifestTrainingDataLoader:
    """Manifest 기반 훈련 데이터로더 관리자"""
    
    def __init__(self, manifest_path: str, batch_size: int = 32, val_split: float = 0.2):
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.val_split = val_split
        self.logger = PillSnapLogger(__name__)
        
        # 데이터 변환 (EfficientNetV2 표준)
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
    
    def load_manifest(self) -> pd.DataFrame:
        """Manifest CSV 파일 로드"""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest 파일을 찾을 수 없습니다: {self.manifest_path}")
        
        df = pd.read_csv(self.manifest_path)
        self.logger.info(f"Manifest 로드: {len(df)}개 샘플")
        
        # 필수 컬럼 확인
        required_cols = ['image_path', 'mapping_code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        
        return df
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """훈련/검증 데이터로더 생성"""
        
        # Manifest 로드
        df = self.load_manifest()
        
        # 클래스별 균등 분할을 위한 stratified split
        train_data = []
        val_data = []
        
        for k_code in df['mapping_code'].unique():
            k_code_data = df[df['mapping_code'] == k_code]
            n_samples = len(k_code_data)
            n_val = max(1, int(n_samples * self.val_split))  # 최소 1개는 검증용
            
            # 셔플 후 분할
            k_code_shuffled = k_code_data.sample(frac=1, random_state=42).reset_index(drop=True)
            val_data.append(k_code_shuffled[:n_val])
            train_data.append(k_code_shuffled[n_val:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        
        self.logger.info(f"데이터 분할: 훈련 {len(train_df)}개, 검증 {len(val_df)}개")
        
        # 데이터셋 생성
        train_dataset = ManifestDataset(train_df, transform=self.train_transform)
        val_dataset = ManifestDataset(val_df, transform=self.val_transform)
        
        # WSL 환경에서는 num_workers=0 사용
        num_workers = 0
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 메타데이터
        metadata = {
            'num_classes': df['mapping_code'].nunique(),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'total_size': len(df),
            'class_names': sorted(df['mapping_code'].unique()),
            'k_code_to_label': train_dataset.k_code_to_label,
            'label_to_k_code': train_dataset.label_to_k_code
        }
        
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