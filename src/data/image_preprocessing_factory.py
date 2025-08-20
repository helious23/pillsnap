"""
PillSnap ML 이미지 전처리 파이프라인

Two-Stage Conditional Pipeline 지원:
- Single Pills: 384px 분류용 전처리
- Combo Pills: 640px 검출용 + 384px 분류용 전처리  
- GPU 최적화 (RTX 5080 16GB)
- 메모리 효율적 배치 처리
- Albumentation 기반 고속 증강
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger, load_config


class PipelineStage(Enum):
    """파이프라인 단계 정의"""
    DETECTION = "detection"    # 640px, YOLOv11x용
    CLASSIFICATION = "classification"  # 384px, EfficientNetV2-L용
    AUGMENTATION = "augmentation"  # 데이터 증강


class ImageQualityLevel(Enum):
    """이미지 품질 수준"""
    HIGH = "high"      # 원본 해상도 유지
    MEDIUM = "medium"  # 적당한 압축
    LOW = "low"        # 높은 압축


@dataclass
class ImageProcessingConfig:
    """이미지 처리 설정"""
    
    # 기본 해상도 (Two-Stage Pipeline 표준)
    detection_size: Tuple[int, int] = (640, 640)    # YOLOv11x
    classification_size: Tuple[int, int] = (384, 384)  # EfficientNetV2-L
    
    # 품질 및 성능
    jpeg_quality: int = 95
    interpolation: int = cv2.INTER_LANCZOS4
    memory_format: str = "channels_last"  # RTX 5080 최적화
    
    # 배치 처리
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    
    # 증강 설정
    augmentation_probability: float = 0.8
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageFormatValidator:
    """이미지 포맷 검증기"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_RESOLUTION = (32, 32)
    MAX_RESOLUTION = (4096, 4096)
    
    @classmethod
    def validate_image_file(cls, image_path: Path) -> Dict[str, Any]:
        """이미지 파일 유효성 검증"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # 1. 파일 존재 확인
            if not image_path.exists():
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"파일이 존재하지 않음: {image_path}")
                return validation_result
            
            # 2. 확장자 확인
            if image_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"지원하지 않는 포맷: {image_path.suffix}")
                return validation_result
            
            # 3. 파일 크기 확인
            file_size = image_path.stat().st_size
            if file_size > cls.MAX_FILE_SIZE:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"파일 크기 초과: {file_size/1024/1024:.1f}MB > 50MB")
                return validation_result
            
            # 4. 이미지 로드 및 메타데이터 추출
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                
                validation_result['metadata'] = {
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'file_size': file_size,
                    'format': img.format
                }
                
                # 5. 해상도 확인
                if width < cls.MIN_RESOLUTION[0] or height < cls.MIN_RESOLUTION[1]:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"해상도 너무 낮음: {width}x{height}")
                
                if width > cls.MAX_RESOLUTION[0] or height > cls.MAX_RESOLUTION[1]:
                    validation_result['warnings'].append(f"고해상도 이미지: {width}x{height}")
                
                # 6. 컬러 모드 확인
                if mode not in ['RGB', 'RGBA', 'L']:
                    validation_result['warnings'].append(f"비표준 컬러 모드: {mode}")
        
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"이미지 로드 실패: {str(e)}")
        
        return validation_result


class ImageResizeProcessor:
    """이미지 리사이즈 처리기"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.logger = PillSnapLogger(__name__)
    
    def resize_for_detection(self, image: np.ndarray) -> np.ndarray:
        """검출용 640x640 리사이즈 (패딩 적용)"""
        target_size = self.config.detection_size
        h, w = image.shape[:2]
        
        # 종횡비 유지하면서 리사이즈
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.config.interpolation)
        
        # 검은색 패딩 추가 (YOLO 표준)
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, 
            pad_h, target_size[1] - new_h - pad_h,
            pad_w, target_size[0] - new_w - pad_w,
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        return padded
    
    def resize_for_classification(self, image: np.ndarray) -> np.ndarray:
        """분류용 384x384 리사이즈 (중앙 크롭)"""
        target_size = self.config.classification_size
        h, w = image.shape[:2]
        
        # 짧은 변을 목표 크기에 맞춤
        scale = max(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 최소 크기 보장 (target_size보다 작으면 안됨)
        new_w = max(new_w, target_size[0])
        new_h = max(new_h, target_size[1])
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.config.interpolation)
        
        # 중앙 크롭 - 안전한 경계 확인
        start_x = max(0, (new_w - target_size[0]) // 2)
        start_y = max(0, (new_h - target_size[1]) // 2)
        end_x = min(new_w, start_x + target_size[0])
        end_y = min(new_h, start_y + target_size[1])
        
        cropped = resized[start_y:end_y, start_x:end_x]
        
        # 크기 검증 및 필요시 다시 리사이즈
        if cropped.shape[:2] != target_size:
            cropped = cv2.resize(cropped, target_size, interpolation=self.config.interpolation)
        
        return cropped
    
    def smart_resize(self, image: np.ndarray, target_stage: PipelineStage) -> np.ndarray:
        """지능형 리사이즈 (단계별 최적화)"""
        if target_stage == PipelineStage.DETECTION:
            return self.resize_for_detection(image)
        elif target_stage == PipelineStage.CLASSIFICATION:
            return self.resize_for_classification(image)
        else:
            raise ValueError(f"지원하지 않는 파이프라인 단계: {target_stage}")


class ImageAugmentationPipeline:
    """이미지 증강 파이프라인"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 검출용 증강 파이프라인
        self.detection_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=0.5
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
        
        # 분류용 증강 파이프라인
        self.classification_augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=15, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.03)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
        
        # 검증용 (증강 없음)
        self.validation_transform = A.Compose([
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
    
    def apply_augmentation(
        self, 
        image: np.ndarray, 
        stage: PipelineStage, 
        is_training: bool = True
    ) -> torch.Tensor:
        """증강 적용"""
        
        if not is_training:
            # 검증 시에는 증강 없음
            transformed = self.validation_transform(image=image)
            return transformed['image']
        
        # 훈련 시 단계별 증강 적용
        if stage == PipelineStage.DETECTION:
            transformed = self.detection_augmentation(image=image)
        elif stage == PipelineStage.CLASSIFICATION:
            transformed = self.classification_augmentation(image=image)
        else:
            raise ValueError(f"지원하지 않는 파이프라인 단계: {stage}")
        
        return transformed['image']


class TwoStageImagePreprocessor:
    """Two-Stage Conditional Pipeline 이미지 전처리기"""
    
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or ImageProcessingConfig()
        self.logger = PillSnapLogger(__name__)
        
        # 컴포넌트 초기화
        self.validator = ImageFormatValidator()
        self.resizer = ImageResizeProcessor(self.config)
        self.augmenter = ImageAugmentationPipeline(self.config)
        
        # 성능 통계
        self.processing_stats = {
            'processed_images': 0,
            'validation_failures': 0,
            'processing_time_ms': [],
            'memory_usage_mb': []
        }
        
        self.logger.info(f"TwoStageImagePreprocessor 초기화")
        self.logger.info(f"  검출 해상도: {self.config.detection_size}")
        self.logger.info(f"  분류 해상도: {self.config.classification_size}")
        self.logger.info(f"  배치 크기: {self.config.batch_size}")
    
    def load_and_validate_image(self, image_path: Path) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """이미지 로드 및 검증"""
        
        # 1. 파일 검증
        validation_result = self.validator.validate_image_file(image_path)
        
        if not validation_result['is_valid']:
            self.processing_stats['validation_failures'] += 1
            return False, None, validation_result
        
        # 2. 이미지 로드
        try:
            # PIL로 로드 후 OpenCV 형식으로 변환
            with Image.open(image_path) as pil_img:
                # RGBA -> RGB 변환
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'L':
                    pil_img = pil_img.convert('RGB')
                
                # OpenCV BGR -> RGB 형식으로 변환
                image_array = np.array(pil_img)  # RGB
                
            return True, image_array, validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"이미지 로드 실패: {str(e)}")
            self.processing_stats['validation_failures'] += 1
            return False, None, validation_result
    
    def preprocess_for_detection(
        self, 
        image_path: Path, 
        is_training: bool = True
    ) -> Tuple[bool, Optional[torch.Tensor], Dict]:
        """검출용 전처리 (640x640)"""
        import time
        start_time = time.time()
        
        # 1. 이미지 로드 및 검증
        is_valid, image_array, validation_info = self.load_and_validate_image(image_path)
        
        if not is_valid:
            return False, None, validation_info
        
        try:
            # 2. 검출용 리사이즈
            resized_image = self.resizer.resize_for_detection(image_array)
            
            # 3. 증강 적용 및 텐서 변환
            processed_tensor = self.augmenter.apply_augmentation(
                resized_image, 
                PipelineStage.DETECTION, 
                is_training
            )
            
            # 4. 메모리 포맷 최적화는 배치 처리 시에만 적용 (3D 텐서는 지원 안함)
            
            # 5. 성능 통계 업데이트
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_stats['processed_images'] += 1
            self.processing_stats['processing_time_ms'].append(processing_time)
            
            processing_info = {
                'stage': 'detection',
                'input_size': f"{validation_info['metadata']['width']}x{validation_info['metadata']['height']}",
                'output_size': self.config.detection_size,
                'processing_time_ms': processing_time,
                'is_training': is_training
            }
            
            return True, processed_tensor, processing_info
            
        except Exception as e:
            error_info = {
                'is_valid': False,
                'errors': [f"검출용 전처리 실패: {str(e)}"],
                'stage': 'detection'
            }
            return False, None, error_info
    
    def preprocess_for_classification(
        self, 
        image_path: Path, 
        is_training: bool = True
    ) -> Tuple[bool, Optional[torch.Tensor], Dict]:
        """분류용 전처리 (384x384)"""
        import time
        start_time = time.time()
        
        # 1. 이미지 로드 및 검증
        is_valid, image_array, validation_info = self.load_and_validate_image(image_path)
        
        if not is_valid:
            return False, None, validation_info
        
        try:
            # 2. 분류용 리사이즈
            resized_image = self.resizer.resize_for_classification(image_array)
            
            # 3. 증강 적용 및 텐서 변환
            processed_tensor = self.augmenter.apply_augmentation(
                resized_image, 
                PipelineStage.CLASSIFICATION, 
                is_training
            )
            
            # 4. 메모리 포맷 최적화는 배치 처리 시에만 적용 (3D 텐서는 지원 안함)
            
            # 5. 성능 통계 업데이트
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_stats['processed_images'] += 1
            self.processing_stats['processing_time_ms'].append(processing_time)
            
            processing_info = {
                'stage': 'classification',
                'input_size': f"{validation_info['metadata']['width']}x{validation_info['metadata']['height']}",
                'output_size': self.config.classification_size,
                'processing_time_ms': processing_time,
                'is_training': is_training
            }
            
            return True, processed_tensor, processing_info
            
        except Exception as e:
            error_info = {
                'is_valid': False,
                'errors': [f"분류용 전처리 실패: {str(e)}"],
                'stage': 'classification'
            }
            return False, None, error_info
    
    def batch_preprocess(
        self, 
        image_paths: List[Path], 
        target_stage: PipelineStage,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """배치 전처리"""
        
        results = {
            'successful_tensors': [],
            'failed_paths': [],
            'processing_info': [],
            'batch_stats': {
                'total_images': len(image_paths),
                'successful': 0,
                'failed': 0,
                'avg_processing_time_ms': 0,
                'total_processing_time_ms': 0
            }
        }
        
        self.logger.info(f"배치 전처리 시작: {len(image_paths)}개 이미지, 단계={target_stage.value}")
        
        import time
        batch_start_time = time.time()
        
        for image_path in image_paths:
            if target_stage == PipelineStage.DETECTION:
                success, tensor, info = self.preprocess_for_detection(image_path, is_training)
            elif target_stage == PipelineStage.CLASSIFICATION:
                success, tensor, info = self.preprocess_for_classification(image_path, is_training)
            else:
                raise ValueError(f"지원하지 않는 파이프라인 단계: {target_stage}")
            
            if success:
                results['successful_tensors'].append(tensor)
                results['processing_info'].append(info)
                results['batch_stats']['successful'] += 1
            else:
                results['failed_paths'].append(str(image_path))
                results['batch_stats']['failed'] += 1
        
        # 배치 통계 계산
        batch_processing_time = (time.time() - batch_start_time) * 1000  # ms
        results['batch_stats']['total_processing_time_ms'] = batch_processing_time
        
        if results['batch_stats']['successful'] > 0:
            avg_time = batch_processing_time / results['batch_stats']['successful']
            results['batch_stats']['avg_processing_time_ms'] = avg_time
        
        self.logger.info(f"배치 전처리 완료: {results['batch_stats']['successful']}/{len(image_paths)} 성공")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.processing_stats.copy()
        
        if stats['processing_time_ms']:
            stats['avg_processing_time_ms'] = sum(stats['processing_time_ms']) / len(stats['processing_time_ms'])
            stats['max_processing_time_ms'] = max(stats['processing_time_ms'])
            stats['min_processing_time_ms'] = min(stats['processing_time_ms'])
        
        if stats['processed_images'] > 0:
            stats['failure_rate'] = stats['validation_failures'] / (stats['processed_images'] + stats['validation_failures'])
        
        return stats


def create_preprocessing_config_for_stage1() -> ImageProcessingConfig:
    """Stage 1용 전처리 설정 생성"""
    return ImageProcessingConfig(
        detection_size=(640, 640),
        classification_size=(384, 384),
        jpeg_quality=95,
        batch_size=16,  # Stage 1용 작은 배치
        num_workers=4,
        augmentation_probability=0.8,
        memory_format="channels_last"
    )


def image_preprocessing_factory(stage: str = "stage1") -> TwoStageImagePreprocessor:
    """이미지 전처리기 팩토리 함수"""
    if stage == "stage1":
        config = create_preprocessing_config_for_stage1()
    else:
        config = ImageProcessingConfig()
    
    return TwoStageImagePreprocessor(config)


if __name__ == "__main__":
    # 기본 사용 예제
    preprocessor = image_preprocessing_factory("stage1")
    
    # 테스트 이미지 경로 (실제 경로로 변경 필요)
    test_image_path = Path("/mnt/data/pillsnap_dataset/data/train/images/single/K-000001/image_001.jpg")
    
    if test_image_path.exists():
        # 검출용 전처리
        success, tensor, info = preprocessor.preprocess_for_detection(test_image_path, is_training=True)
        
        if success:
            print(f"검출용 전처리 성공: {tensor.shape}")
            print(f"처리 정보: {info}")
        else:
            print(f"검출용 전처리 실패: {info}")
        
        # 분류용 전처리
        success, tensor, info = preprocessor.preprocess_for_classification(test_image_path, is_training=False)
        
        if success:
            print(f"분류용 전처리 성공: {tensor.shape}")
            print(f"처리 정보: {info}")
        else:
            print(f"분류용 전처리 실패: {info}")
    
    # 성능 통계 출력
    stats = preprocessor.get_performance_stats()
    print(f"\n성능 통계: {stats}")