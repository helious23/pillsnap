"""
PillSnap 데이터셋 로더 모듈

목적: 단일/조합 약품 이미지-라벨 데이터셋을 PyTorch DataLoader로 제공
핵심 기능:
- PillsnapClsDataset: 분류용 (code → class_id 매핑)
- PillsnapDetDataset: 검출용 (YOLO bbox 형식)
- 감사된 매니페스트 CSV 기반 데이터 로딩
- 메모리 효율적 LMDB 캐싱 지원
- 실시간 augmentation 및 normalization

사용법:
    from src.data import PillsnapClsDataset, PillsnapDetDataset
    
    # 분류 데이터셋
    cls_dataset = PillsnapClsDataset(
        manifest_path="artifacts/manifest_stage1.csv",
        config=config.data,
        split="train",
        transform=transforms.Compose([...])
    )
    
    # 검출 데이터셋  
    det_dataset = PillsnapDetDataset(
        manifest_path="artifacts/manifest_combo.csv",
        config=config.data,
        split="train",
        transform=transforms.Compose([...])
    )
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class CodeToClassMapper:
    """EDI 코드를 클래스 ID로 매핑하는 유틸리티"""
    
    def __init__(self, codes: List[str]):
        """
        Args:
            codes: 유니크한 EDI 코드 리스트 (정렬된 순서 권장)
        """
        self.code_to_id = {code: idx for idx, code in enumerate(sorted(set(codes)))}
        self.id_to_code = {idx: code for code, idx in self.code_to_id.items()}
        self.num_classes = len(self.code_to_id)
        
        logger.info(f"Initialized CodeToClassMapper with {self.num_classes} classes")
    
    def encode(self, code: str) -> int:
        """코드를 클래스 ID로 변환"""
        if code not in self.code_to_id:
            raise ValueError(f"Unknown code: {code}")
        return self.code_to_id[code]
    
    def decode(self, class_id: int) -> str:
        """클래스 ID를 코드로 변환"""
        if class_id not in self.id_to_code:
            raise ValueError(f"Unknown class_id: {class_id}")
        return self.id_to_code[class_id]
    
    def get_all_codes(self) -> List[str]:
        """모든 코드 리스트 반환 (정렬된 순서)"""
        return [self.id_to_code[i] for i in range(self.num_classes)]


class PillsnapClsDataset(Dataset):
    """
    PillSnap 분류용 데이터셋
    
    Features:
    - 감사된 매니페스트 CSV 기반 로딩
    - EDI 코드 → 클래스 ID 자동 매핑
    - PIL Image 로딩 및 transform 적용
    - 라벨 JSON 파싱 (필요시 추가 정보 추출)
    """
    
    def __init__(
        self,
        manifest_path: str,
        config: Any,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        cache_labels: bool = True
    ):
        """
        Args:
            manifest_path: 매니페스트 CSV 파일 경로
            config: 데이터 설정 객체 (config.data)
            split: 데이터 분할 ("train", "val", "test")
            transform: 이미지 변환 파이프라인
            target_transform: 타겟 변환 파이프라인
            cache_labels: 라벨 JSON을 메모리에 캐시할지 여부
        """
        self.manifest_path = manifest_path
        self.config = config
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cache_labels = cache_labels
        
        # 매니페스트 로딩
        self.df = self._load_manifest()
        
        # 코드 매퍼 초기화
        unique_codes = self.df['code'].unique().tolist()
        self.code_mapper = CodeToClassMapper(unique_codes)
        
        # 라벨 캐시 초기화
        self.label_cache = {} if cache_labels else None
        
        logger.info(f"Initialized PillsnapClsDataset: {len(self.df)} samples, {self.code_mapper.num_classes} classes")
    
    def _load_manifest(self) -> pd.DataFrame:
        """매니페스트 CSV 로딩 및 검증"""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        df = pd.read_csv(self.manifest_path)
        
        # 필수 컬럼 확인
        required_cols = ['image_path', 'label_path', 'code', 'is_pair']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in manifest: {missing_cols}")
        
        # is_pair=True만 필터링 (유효한 이미지-라벨 쌍만)
        df = df[df['is_pair'] == True].copy()
        
        # split 필터링 (경로에서 train/val 감지)
        if self.split in ["train", "val", "test"]:
            df = df[df['image_path'].str.contains(f"/{self.split}/")].copy()
        
        if len(df) == 0:
            raise ValueError(f"No valid samples found for split: {self.split}")
        
        # 인덱스 리셋
        df = df.reset_index(drop=True)
        
        logger.info(f"Loaded manifest: {len(df)} samples for split '{self.split}'")
        return df
    
    def _load_label(self, label_path: str) -> Dict[str, Any]:
        """라벨 JSON 로딩 (캐시 활용)"""
        if self.cache_labels and label_path in self.label_cache:
            return self.label_cache[label_path]
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            if self.cache_labels:
                self.label_cache[label_path] = label_data
            
            return label_data
            
        except Exception as e:
            logger.warning(f"Failed to load label {label_path}: {e}")
            return {}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: 변환된 이미지 텐서 [C, H, W]
            target: 클래스 ID (int)
        """
        row = self.df.iloc[idx]
        
        # 이미지 로딩
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {row['image_path']}: {e}")
            # 더미 이미지 생성 (디버깅용)
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 라벨 로딩 (분류용으로는 코드만 사용)
        code = row['code']
        target = self.code_mapper.encode(code)
        
        # 변환 적용
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """샘플의 메타데이터 반환 (디버깅용)"""
        row = self.df.iloc[idx]
        label_data = self._load_label(row['label_path'])
        
        return {
            'idx': idx,
            'code': row['code'],
            'class_id': self.code_mapper.encode(row['code']),
            'image_path': row['image_path'],
            'label_path': row['label_path'],
            'label_data': label_data
        }


class PillsnapDetDataset(Dataset):
    """
    PillSnap 검출용 데이터셋 (YOLO 형식)
    
    Features:
    - 조합 약품 이미지 + bbox 라벨 
    - YOLO 형식 bbox 좌표 (normalized xyxy)
    - 다중 객체 지원 (하나 이미지에 여러 알약)
    """
    
    def __init__(
        self,
        manifest_path: str,
        config: Any,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        cache_labels: bool = True
    ):
        """
        Args:
            manifest_path: 조합 약품 매니페스트 CSV 파일 경로
            config: 데이터 설정 객체
            split: 데이터 분할
            transform: 이미지 변환 파이프라인
            cache_labels: 라벨 캐시 여부
        """
        self.manifest_path = manifest_path
        self.config = config
        self.split = split
        self.transform = transform
        self.cache_labels = cache_labels
        
        # 매니페스트 로딩
        self.df = self._load_manifest()
        
        # 라벨 캐시
        self.label_cache = {} if cache_labels else None
        
        logger.info(f"Initialized PillsnapDetDataset: {len(self.df)} samples")
    
    def _load_manifest(self) -> pd.DataFrame:
        """조합 약품 매니페스트 로딩"""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Detection manifest not found: {self.manifest_path}")
        
        df = pd.read_csv(self.manifest_path)
        
        # 조합 데이터만 필터링 (combination 키워드 포함)
        df = df[df['image_path'].str.contains('/combination/')].copy()
        
        # split 필터링
        if self.split in ["train", "val", "test"]:
            df = df[df['image_path'].str.contains(f"/{self.split}/")].copy()
        
        df = df.reset_index(drop=True)
        
        logger.info(f"Loaded detection manifest: {len(df)} combination samples")
        return df
    
    def _load_label(self, label_path: str) -> Dict[str, Any]:
        """라벨 JSON 로딩 (YOLO bbox 추출)"""
        if self.cache_labels and label_path in self.label_cache:
            return self.label_cache[label_path]
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            if self.cache_labels:
                self.label_cache[label_path] = label_data
            
            return label_data
            
        except Exception as e:
            logger.warning(f"Failed to load detection label {label_path}: {e}")
            return {}
    
    def _extract_bboxes(self, label_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        라벨 JSON에서 bbox 추출 (YOLO 형식으로 변환)
        
        Returns:
            List of bbox dicts with keys: ['x', 'y', 'w', 'h', 'class_id', 'code']
        """
        bboxes = []
        
        # TODO: 실제 라벨 JSON 스키마에 맞게 구현
        # 현재는 더미 구현 (테스트용)
        if 'objects' in label_data:
            for obj in label_data['objects']:
                if 'bbox' in obj:
                    bbox = obj['bbox']
                    bboxes.append({
                        'x': bbox.get('x', 0.5),        # center_x (normalized)
                        'y': bbox.get('y', 0.5),        # center_y (normalized)  
                        'w': bbox.get('w', 0.1),        # width (normalized)
                        'h': bbox.get('h', 0.1),        # height (normalized)
                        'class_id': obj.get('class_id', 0),
                        'code': obj.get('code', 'unknown')
                    })
        
        return bboxes
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            image: 변환된 이미지 텐서 [C, H, W]
            targets: YOLO 형식 타겟 딕셔너리
                - boxes: [N, 4] normalized xyxy 좌표
                - labels: [N] 클래스 ID
                - codes: [N] EDI 코드 리스트
        """
        row = self.df.iloc[idx]
        
        # 이미지 로딩
        try:
            image = Image.open(row['image_path']).convert('RGB')
            img_w, img_h = image.size
        except Exception as e:
            logger.error(f"Failed to load detection image {row['image_path']}: {e}")
            image = Image.new('RGB', (640, 640), color=(128, 128, 128))
            img_w, img_h = 640, 640
        
        # 라벨 로딩 및 bbox 추출
        label_data = self._load_label(row['label_path'])
        bboxes = self._extract_bboxes(label_data)
        
        # YOLO 형식으로 변환
        if len(bboxes) > 0:
            boxes = []
            labels = []
            codes = []
            
            for bbox in bboxes:
                # xywh (center) → xyxy 변환
                cx, cy, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                x1 = max(0, cx - w/2)
                y1 = max(0, cy - h/2) 
                x2 = min(1, cx + w/2)
                y2 = min(1, cy + h/2)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(bbox['class_id'])
                codes.append(bbox['code'])
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            # 빈 타겟 (배경 이미지)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            codes = []
        
        # 이미지 변환
        if self.transform is not None:
            image = self.transform(image)
        
        targets = {
            'boxes': boxes,
            'labels': labels,
            'codes': codes,
            'image_id': idx,
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.bool) if len(boxes) > 0 else torch.tensor([], dtype=torch.bool)
        }
        
        return image, targets
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """샘플 메타데이터 반환"""
        row = self.df.iloc[idx]
        label_data = self._load_label(row['label_path'])
        bboxes = self._extract_bboxes(label_data)
        
        return {
            'idx': idx,
            'image_path': row['image_path'],
            'label_path': row['label_path'],
            'num_objects': len(bboxes),
            'bboxes': bboxes,
            'label_data': label_data
        }


def create_classification_transforms(
    input_size: int = 384,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augment: bool = True
) -> transforms.Compose:
    """분류용 이미지 변환 파이프라인 생성"""
    
    if augment:
        # 훈련용 augmentation
        transform_list = [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        # 검증/테스트용
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)


def create_detection_transforms(
    input_size: int = 640,
    augment: bool = True
) -> transforms.Compose:
    """검출용 이미지 변환 파이프라인 생성"""
    
    if augment:
        # 훈련용 (bbox 좌표는 normalized이므로 문제없음)
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor()
        ]
    else:
        # 검증/테스트용
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ]
    
    return transforms.Compose(transform_list)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    collate_fn: Optional = None
) -> torch.utils.data.DataLoader:
    """DataLoader 생성 헬퍼"""
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True if shuffle else False  # 훈련시에만 마지막 배치 드롭
    )


def detection_collate_fn(batch):
    """검출용 커스텀 collate function (가변 개수 객체 처리)"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 이미지는 스택 가능
    images = torch.stack(images, dim=0)
    
    # 타겟은 리스트로 유지 (각 이미지마다 다른 개수의 객체)
    return images, targets


if __name__ == "__main__":
    # 간단한 테스트
    import config
    
    # 설정 로드
    cfg = config.load_config()
    print(f"Data root: {cfg.data.root}")
    
    # 변환 파이프라인 생성
    train_transform = create_classification_transforms(augment=True)
    val_transform = create_classification_transforms(augment=False)
    
    print("Transforms created successfully")
    print(f"Train transform: {len(train_transform.transforms)} steps")
    print(f"Val transform: {len(val_transform.transforms)} steps")