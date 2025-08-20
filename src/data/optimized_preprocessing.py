"""
PillSnap ML 최적화된 이미지 전처리 파이프라인

고정 해상도 (976x1280) 특화 전처리:
- 동적 크기 계산 제거
- 하드코딩된 변환으로 성능 향상
- 메모리 효율적인 배치 처리
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

from src.utils.core import PillSnapLogger


@dataclass
class OptimizedPreprocessingConfig:
    """최적화된 전처리 설정"""
    
    # 고정 입력 해상도 (실제 데이터)
    input_size: Tuple[int, int] = (976, 1280)  # W x H
    
    # 목표 해상도
    detection_size: Tuple[int, int] = (640, 640)      # YOLOv11x
    classification_size: Tuple[int, int] = (384, 384)  # EfficientNetV2-L
    
    # 성능 최적화
    interpolation: int = cv2.INTER_LINEAR  # LANCZOS4 대신 LINEAR (더 빠름)
    memory_format: str = "channels_last"
    
    # 정규화 상수 (미리 계산)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class OptimizedImagePreprocessor:
    """976x1280 고정 해상도 특화 전처리기"""
    
    def __init__(self, config: Optional[OptimizedPreprocessingConfig] = None):
        self.config = config or OptimizedPreprocessingConfig()
        self.logger = PillSnapLogger(__name__)
        
        # 고정 변환 매트릭스 미리 계산
        self._precalculate_transforms()
        
        # 증강 파이프라인 (최적화된 버전)
        self._setup_augmentation_pipelines()
        
        # 성능 통계
        self.stats = {
            'processed_images': 0,
            'total_time_ms': 0,
            'avg_time_ms': 0
        }
        
        self.logger.info(f"최적화된 전처리기 초기화 (976x1280 특화)")
        self.logger.info(f"  검출 목표: {self.config.detection_size}")
        self.logger.info(f"  분류 목표: {self.config.classification_size}")
    
    def _precalculate_transforms(self):
        """고정 해상도용 변환 매트릭스 미리 계산"""
        
        # 분류용 변환 (976x1280 → 384x384)
        # 최적: 중앙 영역을 384x503으로 크롭 후 384x384로 리사이즈
        input_w, input_h = self.config.input_size
        target_w, target_h = self.config.classification_size
        
        # 중앙 크롭 영역 계산 (종횡비 맞춤)
        crop_w = min(input_w, int(input_h * target_w / target_h))
        crop_h = min(input_h, int(input_w * target_h / target_w))
        
        self.cls_crop_x = (input_w - crop_w) // 2
        self.cls_crop_y = (input_h - crop_h) // 2
        self.cls_crop_w = crop_w
        self.cls_crop_h = crop_h
        
        # 검출용 변환 (976x1280 → 640x640)
        # 최적: 488x640으로 리사이즈 후 양쪽에 76px 패딩
        det_w, det_h = self.config.detection_size
        scale = min(det_w / input_w, det_h / input_h)
        
        self.det_new_w = int(input_w * scale)
        self.det_new_h = int(input_h * scale)
        self.det_pad_x = (det_w - self.det_new_w) // 2
        self.det_pad_y = (det_h - self.det_new_h) // 2
        
        self.logger.debug(f"분류용 크롭: ({self.cls_crop_x}, {self.cls_crop_y}, {self.cls_crop_w}, {self.cls_crop_h})")
        self.logger.debug(f"검출용 리사이즈: {self.det_new_w}x{self.det_new_h}, 패딩: {self.det_pad_x}x{self.det_pad_y}")
    
    def _setup_augmentation_pipelines(self):
        """최적화된 증강 파이프라인 설정"""
        
        # 검출용 (빠른 버전)
        self.detection_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
            A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
            ToTensorV2()
        ])
        
        # 분류용 (빠른 버전)
        self.classification_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.4),
            A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
            ToTensorV2()
        ])
        
        # 검증용 (증강 없음)
        self.validation_transform = A.Compose([
            A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
            ToTensorV2()
        ])
    
    def preprocess_for_classification(
        self, 
        image_path: Path, 
        is_training: bool = True
    ) -> Tuple[bool, Optional[torch.Tensor], Dict]:
        """최적화된 분류용 전처리 (976x1280 → 384x384)"""
        start_time = time.time()
        
        try:
            # 1. 이미지 로드 (PIL 대신 OpenCV 직접 사용)
            image = cv2.imread(str(image_path))
            if image is None:
                return False, None, {'error': f'이미지 로드 실패: {image_path}'}
            
            # BGR → RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. 고정 크롭 (중앙 영역)
            cropped = image[
                self.cls_crop_y:self.cls_crop_y + self.cls_crop_h,
                self.cls_crop_x:self.cls_crop_x + self.cls_crop_w
            ]
            
            # 3. 목표 크기로 리사이즈
            resized = cv2.resize(
                cropped, 
                self.config.classification_size, 
                interpolation=self.config.interpolation
            )
            
            # 4. 증강 및 텐서 변환
            if is_training:
                transformed = self.classification_augmentation(image=resized)
            else:
                transformed = self.validation_transform(image=resized)
            
            tensor = transformed['image']
            
            # 5. 통계 업데이트
            processing_time = (time.time() - start_time) * 1000
            self.stats['processed_images'] += 1
            self.stats['total_time_ms'] += processing_time
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['processed_images']
            
            return True, tensor, {
                'stage': 'classification',
                'processing_time_ms': processing_time,
                'input_size': f"{self.config.input_size[0]}x{self.config.input_size[1]}",
                'output_size': self.config.classification_size,
                'is_training': is_training
            }
            
        except Exception as e:
            return False, None, {'error': f'전처리 실패: {str(e)}'}
    
    def preprocess_for_detection(
        self, 
        image_path: Path, 
        is_training: bool = True
    ) -> Tuple[bool, Optional[torch.Tensor], Dict]:
        """최적화된 검출용 전처리 (976x1280 → 640x640)"""
        start_time = time.time()
        
        try:
            # 1. 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                return False, None, {'error': f'이미지 로드 실패: {image_path}'}
            
            # BGR → RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. 고정 스케일 리사이즈
            resized = cv2.resize(
                image, 
                (self.det_new_w, self.det_new_h), 
                interpolation=self.config.interpolation
            )
            
            # 3. 고정 패딩 추가
            padded = cv2.copyMakeBorder(
                resized,
                self.det_pad_y, 
                self.config.detection_size[1] - self.det_new_h - self.det_pad_y,
                self.det_pad_x, 
                self.config.detection_size[0] - self.det_new_w - self.det_pad_x,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
            # 4. 증강 및 텐서 변환
            if is_training:
                transformed = self.detection_augmentation(image=padded)
            else:
                transformed = self.validation_transform(image=padded)
            
            tensor = transformed['image']
            
            # 5. 통계 업데이트
            processing_time = (time.time() - start_time) * 1000
            self.stats['processed_images'] += 1
            self.stats['total_time_ms'] += processing_time
            self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['processed_images']
            
            return True, tensor, {
                'stage': 'detection',
                'processing_time_ms': processing_time,
                'input_size': f"{self.config.input_size[0]}x{self.config.input_size[1]}",
                'output_size': self.config.detection_size,
                'is_training': is_training
            }
            
        except Exception as e:
            return False, None, {'error': f'전처리 실패: {str(e)}'}
    
    def preprocess_pil_for_detection(
        self, 
        pil_image, 
        is_training: bool = True
    ) -> torch.Tensor:
        """PIL Image를 Detection용으로 전처리 (640x640)"""
        import numpy as np
        
        # PIL → numpy 변환
        image_np = np.array(pil_image)
        
        # RGB 확인 (PIL은 이미 RGB)
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            raise ValueError("RGB 이미지가 아닙니다")
        
        # 640x640으로 리사이즈
        resized = cv2.resize(image_np, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # 정규화 및 텐서 변환
        tensor = torch.from_numpy(resized).float()
        tensor = tensor / 255.0  # [0, 1] 정규화
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        tensor = tensor.unsqueeze(0)  # 배치 차원 추가: [1, 3, 640, 640]
        
        return tensor
    
    def preprocess_pil_for_classification(
        self, 
        pil_image, 
        is_training: bool = True
    ) -> torch.Tensor:
        """PIL Image를 Classification용으로 전처리 (384x384)"""
        import numpy as np
        
        # PIL → numpy 변환
        image_np = np.array(pil_image)
        
        # RGB 확인
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            raise ValueError("RGB 이미지가 아닙니다")
        
        # 384x384로 리사이즈
        resized = cv2.resize(image_np, (384, 384), interpolation=cv2.INTER_LINEAR)
        
        # 정규화 및 텐서 변환
        tensor = torch.from_numpy(resized).float()
        tensor = tensor / 255.0  # [0, 1] 정규화
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        tensor = tensor.unsqueeze(0)  # 배치 차원 추가: [1, 3, 384, 384]
        
        return tensor

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.stats.copy()


def create_optimized_preprocessor_for_stage1() -> OptimizedImagePreprocessor:
    """Stage 1용 최적화된 전처리기 생성"""
    config = OptimizedPreprocessingConfig(
        detection_size=(640, 640),
        classification_size=(384, 384),
        interpolation=cv2.INTER_LINEAR,  # 성능 우선
    )
    return OptimizedImagePreprocessor(config)


if __name__ == "__main__":
    # 성능 비교 테스트
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.image_preprocessing import TwoStageImagePreprocessor
    
    # 테스트 이미지
    test_image = Path("/mnt/data/pillsnap_dataset/data/train/images/single/TS_66_single/K-030552/K-030552_0_0_1_0_75_000_200.png")
    
    if test_image.exists():
        print("=== 성능 비교 테스트 ===")
        
        # 기존 전처리기
        old_preprocessor = TwoStageImagePreprocessor()
        
        # 최적화된 전처리기
        new_preprocessor = create_optimized_preprocessor_for_stage1()
        
        # 성능 측정
        n_tests = 20
        
        print(f"\n🔬 {n_tests}회 반복 테스트 (분류용)")
        
        # 기존 방식
        old_times = []
        for _ in range(n_tests):
            start = time.time()
            success, tensor, info = old_preprocessor.preprocess_for_classification(test_image, is_training=False)
            old_times.append((time.time() - start) * 1000)
        
        # 최적화된 방식
        new_times = []
        for _ in range(n_tests):
            start = time.time()
            success, tensor, info = new_preprocessor.preprocess_for_classification(test_image, is_training=False)
            new_times.append((time.time() - start) * 1000)
        
        print(f"기존 방식: {sum(old_times)/len(old_times):.2f}ms (평균)")
        print(f"최적화 방식: {sum(new_times)/len(new_times):.2f}ms (평균)")
        print(f"성능 향상: {((sum(old_times)/len(old_times)) / (sum(new_times)/len(new_times)) - 1) * 100:.1f}%")
        print(f"처리량 향상: {1000/(sum(new_times)/len(new_times)):.1f} vs {1000/(sum(old_times)/len(old_times)):.1f} images/sec")