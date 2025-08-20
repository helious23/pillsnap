"""
Manifest 기반 훈련 데이터로더 (Lazy Loading 버전)

CSV manifest 파일을 사용하여 정확한 이미지-라벨 쌍으로 데이터로더를 생성합니다.
메모리 효율성을 위해 이미지 경로만 저장하고 실제 로딩은 필요시에만 수행합니다.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import json
from typing import Tuple, Dict, Any, List
import torchvision.transforms as transforms
import numpy as np

from src.utils.core import PillSnapLogger


class ManifestDatasetLazy(Dataset):
    """Manifest CSV 기반 데이터셋 (Lazy Loading)"""
    
    def __init__(self, image_paths: List[str], k_codes: List[str], transform=None):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            k_codes: K-코드 리스트 (각 이미지에 대응)
            transform: 이미지 변환
        """
        self.image_paths = image_paths
        self.k_codes = k_codes
        self.transform = transform
        self.logger = PillSnapLogger(__name__)
        
        # K-코드를 정수 라벨로 매핑
        unique_k_codes = sorted(set(k_codes))
        self.k_code_to_label = {k_code: idx for idx, k_code in enumerate(unique_k_codes)}
        self.label_to_k_code = {idx: k_code for k_code, idx in self.k_code_to_label.items()}
        
        self.logger.info(f"ManifestDatasetLazy 생성: {len(self.image_paths)}개 샘플, {len(unique_k_codes)}개 클래스")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드 (필요시에만)
        image_path = Path(self.image_paths[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            self.logger.error(f"이미지 로드 실패: {image_path} - {e}")
            # 기본 이미지 생성 (검은색 384x384)
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        # 라벨 (K-코드를 정수로 변환)
        k_code = self.k_codes[idx]
        label = self.k_code_to_label[k_code]
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ManifestTrainingDataLoaderLazy:
    """Manifest 기반 훈련 데이터로더 관리자 (Lazy Loading)"""
    
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
        """훈련/검증 데이터로더 생성 (메모리 효율적)"""
        
        # Manifest 로드
        df = self.load_manifest()
        
        # 경로와 라벨만 추출 (DataFrame 전체를 메모리에 유지하지 않음)
        train_paths = []
        train_k_codes = []
        val_paths = []
        val_k_codes = []
        
        # 클래스별 균등 분할
        for k_code in df['mapping_code'].unique():
            k_code_indices = df[df['mapping_code'] == k_code].index.tolist()
            n_samples = len(k_code_indices)
            n_val = max(1, int(n_samples * self.val_split))
            
            # 인덱스 셔플
            np.random.seed(42)
            np.random.shuffle(k_code_indices)
            
            # 검증 데이터
            for idx in k_code_indices[:n_val]:
                val_paths.append(df.loc[idx, 'image_path'])
                val_k_codes.append(k_code)
            
            # 훈련 데이터
            for idx in k_code_indices[n_val:]:
                train_paths.append(df.loc[idx, 'image_path'])
                train_k_codes.append(k_code)
        
        self.logger.info(f"데이터 분할: 훈련 {len(train_paths)}개, 검증 {len(val_paths)}개")
        
        # DataFrame 메모리 해제
        del df
        
        # 데이터셋 생성 (경로만 전달)
        train_dataset = ManifestDatasetLazy(
            image_paths=train_paths,
            k_codes=train_k_codes,
            transform=self.train_transform
        )
        
        val_dataset = ManifestDatasetLazy(
            image_paths=val_paths,
            k_codes=val_k_codes,
            transform=self.val_transform
        )
        
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
        all_k_codes = train_k_codes + val_k_codes
        unique_k_codes = sorted(set(all_k_codes))
        
        metadata = {
            'num_classes': len(unique_k_codes),
            'train_size': len(train_paths),
            'val_size': len(val_paths),
            'total_size': len(train_paths) + len(val_paths),
            'class_names': unique_k_codes,
            'k_code_to_label': train_dataset.k_code_to_label,
            'label_to_k_code': train_dataset.label_to_k_code
        }
        
        return train_loader, val_loader, metadata


def test_manifest_dataloader_lazy():
    """ManifestTrainingDataLoaderLazy 테스트"""
    logger = PillSnapLogger(__name__)
    
    try:
        # Stage 2 manifest로 테스트
        manifest_path = "artifacts/stage2/manifest_ssd.csv"
        
        dataloader_manager = ManifestTrainingDataLoaderLazy(
            manifest_path=manifest_path,
            batch_size=32,
            val_split=0.2
        )
        
        train_loader, val_loader, metadata = dataloader_manager.get_dataloaders()
        
        logger.info("ManifestTrainingDataLoaderLazy 테스트 결과:")
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
        logger.error(f"ManifestTrainingDataLoaderLazy 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    test_manifest_dataloader_lazy()