"""
Combination Pill Training DataLoader
조합 약품 학습용 데이터로더

YOLOv11m 검출용:
- 640x640 이미지 전처리
- YOLO 어노테이션 포맷 지원
- 검출 + 분류 Two-Stage 파이프라인 지원
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.progressive_validation_sampler import ProgressiveValidationSampler, Stage1SamplingStrategy
from src.data.image_preprocessing_factory import TwoStageImagePreprocessor, PipelineStage
from src.utils.core import PillSnapLogger


class CombinationPillDataset(Dataset):
    """조합 약품 데이터셋 (YOLO 검출용)"""
    
    def __init__(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        preprocessor: TwoStageImagePreprocessor,
        is_training: bool = True
    ):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.logger = PillSnapLogger(__name__)
        
        assert len(image_paths) == len(annotation_paths), "이미지와 어노테이션 수가 일치하지 않음"
        
        self.logger.info(f"CombinationPillDataset 초기화: {len(image_paths)}개 이미지")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        try:
            # 검출용 전처리 (640x640)
            success, processed_tensor, info = self.preprocessor.preprocess_for_detection(
                image_path,
                is_training=self.is_training
            )
            
            if not success:
                self.logger.warning(f"전처리 실패: {image_path}")
                processed_tensor = torch.zeros(3, 640, 640)
            
            # YOLO 어노테이션 로드
            targets = self._load_yolo_annotation(annotation_path)
            
            return processed_tensor, targets
            
        except Exception as e:
            self.logger.error(f"데이터 로드 실패 {image_path}: {e}")
            return torch.zeros(3, 640, 640), torch.zeros(0, 5)  # 빈 타겟
    
    def _load_yolo_annotation(self, annotation_path: Path) -> torch.Tensor:
        """YOLO 형식 어노테이션 로드"""
        try:
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                
                targets = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO 형식: class_id, center_x, center_y, width, height
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        targets.append([class_id, center_x, center_y, width, height])
                
                if targets:
                    return torch.tensor(targets, dtype=torch.float32)
            
            # 어노테이션이 없거나 빈 경우
            return torch.zeros(0, 5)
            
        except Exception as e:
            self.logger.warning(f"어노테이션 로드 실패 {annotation_path}: {e}")
            return torch.zeros(0, 5)


class CombinationPillTrainingDataLoader:
    """조합 약품 학습용 데이터로더 매니저"""
    
    def __init__(
        self,
        data_root: str = "/mnt/data/pillsnap_dataset",
        stage: int = 1,
        batch_size: int = 16,  # YOLO는 일반적으로 더 작은 배치
        num_workers: int = 8,
        pin_memory: bool = True
    ):
        self.data_root = Path(data_root)
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.logger = PillSnapLogger(__name__)
        
        # 전처리기 초기화
        self.preprocessor = TwoStageImagePreprocessor()
        
        # 샘플링 전략
        if stage == 1:
            self.strategy = Stage1SamplingStrategy(target_images=5000, target_classes=50)
        else:
            raise NotImplementedError(f"Stage {stage} 아직 구현되지 않음")
        
        self.logger.info(f"CombinationPillTrainingDataLoader 초기화 (Stage {stage})")
    
    def prepare_datasets(
        self,
        validation_split: float = 0.2
    ) -> Tuple[CombinationPillDataset, CombinationPillDataset]:
        """학습/검증 데이터셋 준비"""
        
        self.logger.step("조합 약품 데이터셋 준비", f"Stage {self.stage} 검출용 샘플링")
        
        try:
            # Progressive Validation 샘플러로 데이터 생성
            sampler = ProgressiveValidationSampler(str(self.data_root), self.strategy)
            stage_sample = sampler.generate_stage1_sample()
            
            # 조합 약품만 필터링
            combo_samples = []
            for k_code, sample_data in stage_sample['samples'].items():
                combo_images = sample_data.get('combo_images', [])
                combo_samples.extend([(Path(img_path), k_code) for img_path in combo_images])
            
            if not combo_samples:
                self.logger.warning("조합 약품 이미지가 없음 - 단일 약품으로 대체")
                # 단일 약품으로 대체 (검출 학습용)
                for k_code, sample_data in stage_sample['samples'].items():
                    single_images = sample_data.get('single_images', [])[:10]  # 제한적으로 사용
                    combo_samples.extend([(Path(img_path), k_code) for img_path in single_images])
            
            # 이미지 경로 분리
            image_paths = [img_path for img_path, _ in combo_samples]
            
            # 어노테이션 경로 생성 (실제로는 YOLO 어노테이션 파일이 있어야 함)
            annotation_paths = []
            for img_path in image_paths:
                # 어노테이션 파일 경로 추정
                ann_path = img_path.parent / f"{img_path.stem}.txt"
                if not ann_path.exists():
                    # 더미 어노테이션 생성
                    ann_path = self._create_dummy_annotation(img_path)
                annotation_paths.append(ann_path)
            
            self.logger.info(f"조합 약품 이미지: {len(image_paths)}개")
            
            # 학습/검증 분할
            import random
            random.seed(42)
            indices = list(range(len(image_paths)))
            random.shuffle(indices)
            
            split_idx = int(len(indices) * (1 - validation_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # 학습 데이터셋
            train_image_paths = [image_paths[i] for i in train_indices]
            train_annotation_paths = [annotation_paths[i] for i in train_indices]
            
            train_dataset = CombinationPillDataset(
                image_paths=train_image_paths,
                annotation_paths=train_annotation_paths,
                preprocessor=self.preprocessor,
                is_training=True
            )
            
            # 검증 데이터셋
            val_image_paths = [image_paths[i] for i in val_indices]
            val_annotation_paths = [annotation_paths[i] for i in val_indices]
            
            val_dataset = CombinationPillDataset(
                image_paths=val_image_paths,
                annotation_paths=val_annotation_paths,
                preprocessor=self.preprocessor,
                is_training=False
            )
            
            self.logger.info(f"학습 데이터: {len(train_dataset)}개")
            self.logger.info(f"검증 데이터: {len(val_dataset)}개")
            self.logger.success("조합 약품 데이터셋 준비 완료")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.error(f"조합 약품 데이터셋 준비 실패: {e}")
            raise
    
    def _create_dummy_annotation(self, image_path: Path) -> Path:
        """더미 YOLO 어노테이션 생성"""
        try:
            ann_path = image_path.parent / f"{image_path.stem}.txt"
            
            # 더미 바운딩 박스 (이미지 중앙에 약품 하나)
            dummy_annotation = "0 0.5 0.5 0.3 0.3\n"  # class_id=0, 중앙 위치, 30% 크기
            
            with open(ann_path, 'w') as f:
                f.write(dummy_annotation)
            
            return ann_path
            
        except Exception as e:
            self.logger.warning(f"더미 어노테이션 생성 실패: {e}")
            return image_path.parent / "dummy.txt"
    
    def create_dataloaders(
        self,
        train_dataset: CombinationPillDataset,
        val_dataset: CombinationPillDataset
    ) -> Tuple[DataLoader, DataLoader]:
        """YOLO용 데이터로더 생성"""
        
        try:
            # 커스텀 collate function (YOLO는 가변 길이 타겟)
            def yolo_collate_fn(batch):
                images, targets = zip(*batch)
                images = torch.stack(images, 0)
                
                # 타겟을 배치 인덱스와 함께 패킹
                batch_targets = []
                for i, target in enumerate(targets):
                    if target.size(0) > 0:
                        # 배치 인덱스 추가
                        batch_idx = torch.full((target.size(0), 1), i)
                        target_with_batch = torch.cat([batch_idx, target], dim=1)
                        batch_targets.append(target_with_batch)
                
                if batch_targets:
                    targets = torch.cat(batch_targets, 0)
                else:
                    targets = torch.zeros(0, 6)  # batch_idx + 5 YOLO params
                
                return images, targets
            
            # 학습 데이터로더
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=yolo_collate_fn,
                drop_last=True
            )
            
            # 검증 데이터로더
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=yolo_collate_fn,
                drop_last=False
            )
            
            self.logger.info(f"YOLO 데이터로더 생성 완료")
            self.logger.info(f"  학습 배치 수: {len(train_loader)}")
            self.logger.info(f"  검증 배치 수: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"YOLO 데이터로더 생성 실패: {e}")
            raise
    
    def get_stage_dataloaders(
        self,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Stage별 조합 약품 데이터로더 생성"""
        
        # 데이터셋 준비
        train_dataset, val_dataset = self.prepare_datasets(validation_split)
        
        # 데이터로더 생성
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        # 메타정보
        metadata = {
            'num_classes': 1,  # 단일 클래스 (pill)
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'batch_size': self.batch_size,
            'stage': self.stage,
            'annotation_format': 'YOLO'
        }
        
        return train_loader, val_loader, metadata


def main():
    """조합 약품 데이터로더 테스트"""
    print("📊 Combination Pill Training DataLoader Test")
    print("=" * 60)
    
    try:
        # 데이터로더 매니저 생성
        dataloader_manager = CombinationPillTrainingDataLoader(
            stage=1,
            batch_size=8,  # 테스트용 작은 배치
            num_workers=2
        )
        
        # 데이터로더 생성
        train_loader, val_loader, metadata = dataloader_manager.get_stage_dataloaders()
        
        print(f"✅ 조합 약품 데이터로더 생성 성공")
        print(f"   학습 데이터: {metadata['train_size']}개")
        print(f"   검증 데이터: {metadata['val_size']}개")
        print(f"   배치 수: 학습 {len(train_loader)}, 검증 {len(val_loader)}")
        
        # 첫 번째 배치 테스트
        if len(train_loader) > 0:
            train_batch = next(iter(train_loader))
            images, targets = train_batch
            
            print(f"   배치 모양: 이미지 {images.shape}, 타겟 {targets.shape}")
            print(f"   이미지 범위: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   타겟 수: {targets.size(0)}개 바운딩 박스")
        
        print("\n✅ 조합 약품 데이터로더 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()