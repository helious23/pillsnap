"""
PillSnap ML Two-Stage Conditional Pipeline 데이터 로더

Single Pills와 Combination Pills을 위한 전용 데이터 로더:
- SinglePillDatasetLoader: 분류 전용 (384px, EfficientNetV2-L)
- CombinationPillDatasetLoader: 검출 전용 (640px, YOLOv11x)  
- BatchDataProcessingManager: 배치 처리 최적화
- MemoryOptimizedDataFeeder: LMDB 캐싱 및 메모리 효율성
- ProgressiveValidationDataProvider: Stage별 데이터 제공
"""

import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from collections import defaultdict
import json

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger, load_config
from src.data.image_preprocessing import (
    TwoStageImagePreprocessor, 
    image_preprocessing_factory
)
from src.data.pharmaceutical_code_registry import (
    load_pharmaceutical_registry_from_artifacts
)


@dataclass
class DataLoadingConfiguration:
    """데이터 로딩 설정"""
    
    # 배치 처리 설정
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # 셔플링 및 샘플링
    shuffle_training: bool = True
    drop_last_batch: bool = False
    
    # 메모리 최적화
    memory_format: str = "channels_last"  # RTX 5080 최적화
    use_lmdb_cache: bool = False  # 추후 LMDB 구현시 활성화
    
    # 성능 모니터링
    enable_profiling: bool = False
    max_batch_size_auto_tune: bool = True
    
    # 검증 및 디버깅
    validate_data_integrity: bool = True
    log_batch_statistics: bool = False
    
    def __post_init__(self):
        """설정 유효성 검증"""
        if self.batch_size <= 0:
            raise ValueError(f"배치 크기는 양수여야 함: {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"워커 수는 음수일 수 없음: {self.num_workers}")
        
        if self.prefetch_factor < 1:
            raise ValueError(f"프리페치 팩터는 1 이상이어야 함: {self.prefetch_factor}")


class ImagePathValidationError(Exception):
    """이미지 경로 검증 오류"""
    pass


class DataIntegrityValidationError(Exception):
    """데이터 무결성 검증 오류"""
    pass


class SinglePillDatasetHandler(Dataset):
    """Single Pill 분류용 데이터셋 핸들러"""
    
    def __init__(
        self,
        image_paths: List[Path],
        class_labels: List[int],
        edi_codes: List[str],
        preprocessor: TwoStageImagePreprocessor,
        is_training: bool = True,
        enable_data_validation: bool = True
    ):
        self.image_paths = image_paths
        self.class_labels = class_labels
        self.edi_codes = edi_codes
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.enable_data_validation = enable_data_validation
        self.logger = PillSnapLogger(__name__)
        
        # 데이터 무결성 검증
        if enable_data_validation:
            self._validate_dataset_integrity()
        
        # 통계 정보
        self.dataset_statistics = self._compute_dataset_statistics()
        
        self.logger.info(f"SinglePillDatasetHandler 초기화 완료")
        self.logger.info(f"  총 샘플: {len(self.image_paths)}개")
        self.logger.info(f"  고유 클래스: {self.dataset_statistics['unique_classes']}개")
        self.logger.info(f"  훈련 모드: {self.is_training}")
    
    def _validate_dataset_integrity(self):
        """데이터셋 무결성 검증"""
        # 1. 길이 일치 확인
        if not (len(self.image_paths) == len(self.class_labels) == len(self.edi_codes)):
            raise DataIntegrityValidationError(
                f"데이터 길이 불일치: images={len(self.image_paths)}, "
                f"labels={len(self.class_labels)}, edi_codes={len(self.edi_codes)}"
            )
        
        # 2. 빈 데이터셋 확인
        if len(self.image_paths) == 0:
            raise DataIntegrityValidationError("빈 데이터셋")
        
        # 3. 이미지 파일 존재 확인 (샘플링)
        sample_size = min(100, len(self.image_paths))
        sample_indices = random.sample(range(len(self.image_paths)), sample_size)
        
        missing_files = []
        for idx in sample_indices:
            if not self.image_paths[idx].exists():
                missing_files.append(str(self.image_paths[idx]))
        
        if missing_files:
            raise ImagePathValidationError(f"존재하지 않는 이미지 파일들: {missing_files[:5]}")
        
        # 4. 클래스 라벨 유효성 확인
        invalid_labels = [label for label in self.class_labels if not isinstance(label, int) or label < 0]
        if invalid_labels:
            raise DataIntegrityValidationError(f"유효하지 않은 클래스 라벨: {invalid_labels[:5]}")
        
        # 5. EDI 코드 유효성 확인
        invalid_edi_codes = [edi for edi in self.edi_codes if not isinstance(edi, str) or len(edi) == 0]
        if invalid_edi_codes:
            raise DataIntegrityValidationError(f"유효하지 않은 EDI 코드: {invalid_edi_codes[:5]}")
    
    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 계산"""
        from collections import Counter
        
        class_distribution = Counter(self.class_labels)
        edi_distribution = Counter(self.edi_codes)
        
        return {
            'total_samples': len(self.image_paths),
            'unique_classes': len(set(self.class_labels)),
            'unique_edi_codes': len(set(self.edi_codes)),
            'class_distribution': dict(class_distribution),
            'edi_distribution': dict(edi_distribution),
            'min_class_samples': min(class_distribution.values()) if class_distribution else 0,
            'max_class_samples': max(class_distribution.values()) if class_distribution else 0,
            'mean_class_samples': sum(class_distribution.values()) / len(class_distribution) if class_distribution else 0
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """데이터셋 아이템 반환"""
        try:
            # 인덱스 유효성 확인
            if index < 0 or index >= len(self.image_paths):
                raise IndexError(f"인덱스 범위 초과: {index} (데이터셋 크기: {len(self.image_paths)})")
            
            image_path = self.image_paths[index]
            class_label = self.class_labels[index]
            edi_code = self.edi_codes[index]
            
            # 이미지 전처리
            success, processed_image, preprocessing_info = self.preprocessor.preprocess_for_classification(
                image_path, 
                is_training=self.is_training
            )
            
            if not success:
                # 전처리 실패시 로그 및 예외 발생
                self.logger.error(f"이미지 전처리 실패: {image_path}, 오류: {preprocessing_info}")
                raise RuntimeError(f"이미지 전처리 실패: {image_path}")
            
            return {
                'image': processed_image,
                'class_label': torch.tensor(class_label, dtype=torch.long),
                'edi_code': edi_code,
                'image_path': str(image_path),
                'preprocessing_info': preprocessing_info
            }
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 실패 (index={index}): {e}")
            raise
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 가중치 계산 (불균형 클래스 대응)"""
        class_counts = np.bincount(self.class_labels)
        total_samples = len(self.class_labels)
        num_classes = len(class_counts)
        
        # 역빈도 가중치 계산
        class_weights = total_samples / (num_classes * class_counts)
        return torch.FloatTensor(class_weights)
    
    def get_sample_by_edi_code(self, target_edi_code: str) -> List[int]:
        """특정 EDI 코드의 샘플 인덱스 반환"""
        return [i for i, edi_code in enumerate(self.edi_codes) if edi_code == target_edi_code]
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        return self.dataset_statistics.copy()


class CombinationPillDatasetHandler(Dataset):
    """Combination Pill 검출용 데이터셋 핸들러"""
    
    def __init__(
        self,
        image_paths: List[Path],
        annotation_paths: List[Path],
        preprocessor: TwoStageImagePreprocessor,
        class_id_mapping: Dict[str, int],
        is_training: bool = True,
        enable_data_validation: bool = True,
        max_detections_per_image: int = 50
    ):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.preprocessor = preprocessor
        self.class_id_mapping = class_id_mapping
        self.is_training = is_training
        self.enable_data_validation = enable_data_validation
        self.max_detections_per_image = max_detections_per_image
        self.logger = PillSnapLogger(__name__)
        
        # 데이터 무결성 검증
        if enable_data_validation:
            self._validate_dataset_integrity()
        
        # 어노테이션 캐시 (성능 최적화)
        self.annotation_cache = {}
        self._preload_annotations()
        
        # 통계 정보
        self.dataset_statistics = self._compute_dataset_statistics()
        
        self.logger.info(f"CombinationPillDatasetHandler 초기화 완료")
        self.logger.info(f"  총 샘플: {len(self.image_paths)}개")
        self.logger.info(f"  총 검출 객체: {self.dataset_statistics['total_detections']}개")
        self.logger.info(f"  평균 객체/이미지: {self.dataset_statistics['avg_detections_per_image']:.1f}개")
    
    def _validate_dataset_integrity(self):
        """데이터셋 무결성 검증"""
        # 1. 길이 일치 확인
        if len(self.image_paths) != len(self.annotation_paths):
            raise DataIntegrityValidationError(
                f"이미지와 어노테이션 개수 불일치: images={len(self.image_paths)}, "
                f"annotations={len(self.annotation_paths)}"
            )
        
        # 2. 빈 데이터셋 확인
        if len(self.image_paths) == 0:
            raise DataIntegrityValidationError("빈 데이터셋")
        
        # 3. 파일 존재 확인 (샘플링)
        sample_size = min(50, len(self.image_paths))
        sample_indices = random.sample(range(len(self.image_paths)), sample_size)
        
        missing_files = []
        for idx in sample_indices:
            if not self.image_paths[idx].exists():
                missing_files.append(f"이미지: {self.image_paths[idx]}")
            if not self.annotation_paths[idx].exists():
                missing_files.append(f"어노테이션: {self.annotation_paths[idx]}")
        
        if missing_files:
            raise ImagePathValidationError(f"존재하지 않는 파일들: {missing_files[:5]}")
        
        # 4. 어노테이션 형식 검증 (샘플링)
        for idx in sample_indices[:10]:  # 10개만 검증
            try:
                self._load_yolo_annotation(self.annotation_paths[idx])
            except Exception as e:
                raise DataIntegrityValidationError(f"어노테이션 파일 형식 오류 {self.annotation_paths[idx]}: {e}")
    
    def _load_yolo_annotation(self, annotation_path: Path) -> List[Dict[str, float]]:
        """YOLO 어노테이션 파일 로드"""
        annotations = []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"라인 {line_num}: YOLO 형식 오류 (5개 값 필요, {len(parts)}개 발견)")
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError as e:
                    raise ValueError(f"라인 {line_num}: 숫자 변환 오류 - {e}")
                
                # 좌표 유효성 검증
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"라인 {line_num}: 좌표 범위 오류 - ({x_center}, {y_center}, {width}, {height})")
                
                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            
            # 최대 검출 수 제한
            if len(annotations) > self.max_detections_per_image:
                self.logger.warning(
                    f"검출 수 제한 초과 ({len(annotations)} > {self.max_detections_per_image}): {annotation_path}"
                )
                annotations = annotations[:self.max_detections_per_image]
            
            return annotations
            
        except Exception as e:
            raise RuntimeError(f"어노테이션 로딩 실패 {annotation_path}: {e}")
    
    def _preload_annotations(self):
        """어노테이션 사전 로딩 (캐싱)"""
        self.logger.info("어노테이션 사전 로딩 중...")
        
        failed_count = 0
        for i, annotation_path in enumerate(self.annotation_paths):
            try:
                self.annotation_cache[i] = self._load_yolo_annotation(annotation_path)
            except Exception as e:
                self.logger.error(f"어노테이션 로딩 실패 (index={i}): {e}")
                self.annotation_cache[i] = []  # 빈 어노테이션
                failed_count += 1
        
        if failed_count > 0:
            self.logger.warning(f"어노테이션 로딩 실패: {failed_count}개")
        
        self.logger.info(f"어노테이션 사전 로딩 완료: {len(self.annotation_cache)}개")
    
    def _compute_dataset_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 계산"""
        total_detections = 0
        class_distribution = defaultdict(int)
        detections_per_image = []
        
        for annotations in self.annotation_cache.values():
            detections_per_image.append(len(annotations))
            total_detections += len(annotations)
            
            for annotation in annotations:
                class_distribution[annotation['class_id']] += 1
        
        return {
            'total_samples': len(self.image_paths),
            'total_detections': total_detections,
            'unique_classes': len(class_distribution),
            'class_distribution': dict(class_distribution),
            'avg_detections_per_image': total_detections / len(self.image_paths) if self.image_paths else 0,
            'max_detections_per_image': max(detections_per_image) if detections_per_image else 0,
            'min_detections_per_image': min(detections_per_image) if detections_per_image else 0
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """데이터셋 아이템 반환"""
        try:
            # 인덱스 유효성 확인
            if index < 0 or index >= len(self.image_paths):
                raise IndexError(f"인덱스 범위 초과: {index} (데이터셋 크기: {len(self.image_paths)})")
            
            image_path = self.image_paths[index]
            annotations = self.annotation_cache.get(index, [])
            
            # 이미지 전처리
            success, processed_image, preprocessing_info = self.preprocessor.preprocess_for_detection(
                image_path, 
                is_training=self.is_training
            )
            
            if not success:
                self.logger.error(f"이미지 전처리 실패: {image_path}, 오류: {preprocessing_info}")
                raise RuntimeError(f"이미지 전처리 실패: {image_path}")
            
            # 어노테이션을 텐서로 변환
            if annotations:
                # [num_objects, 5] 형태: [class_id, x_center, y_center, width, height]
                targets = torch.zeros((len(annotations), 5), dtype=torch.float32)
                for i, annotation in enumerate(annotations):
                    targets[i] = torch.tensor([
                        annotation['class_id'],
                        annotation['x_center'],
                        annotation['y_center'],
                        annotation['width'],
                        annotation['height']
                    ])
            else:
                # 빈 어노테이션
                targets = torch.zeros((0, 5), dtype=torch.float32)
            
            return {
                'image': processed_image,
                'targets': targets,
                'image_path': str(image_path),
                'annotation_path': str(self.annotation_paths[index]),
                'preprocessing_info': preprocessing_info,
                'num_detections': len(annotations)
            }
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 실패 (index={index}): {e}")
            raise
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        return self.dataset_statistics.copy()


class BalancedClassSampler(Sampler):
    """균형 잡힌 클래스 샘플러 (불균형 데이터셋 대응)"""
    
    def __init__(
        self, 
        dataset: SinglePillDatasetHandler, 
        samples_per_class: Optional[int] = None,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.replacement = replacement
        self.logger = PillSnapLogger(__name__)
        
        # 클래스별 인덱스 구축
        self.class_to_indices = defaultdict(list)
        for idx, class_label in enumerate(dataset.class_labels):
            self.class_to_indices[class_label].append(idx)
        
        self.unique_classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.unique_classes)
        
        # 클래스당 샘플 수 결정
        if samples_per_class is None:
            # 가장 적은 클래스의 샘플 수를 기준으로 설정
            min_samples = min(len(indices) for indices in self.class_to_indices.values())
            self.samples_per_class = min_samples
        else:
            self.samples_per_class = samples_per_class
        
        self.total_samples = self.num_classes * self.samples_per_class
        
        self.logger.info(f"BalancedClassSampler 초기화")
        self.logger.info(f"  클래스 수: {self.num_classes}")
        self.logger.info(f"  클래스당 샘플 수: {self.samples_per_class}")
        self.logger.info(f"  총 샘플 수: {self.total_samples}")
        self.logger.info(f"  복원 추출: {self.replacement}")
    
    def __iter__(self) -> Iterator[int]:
        """샘플 인덱스 반복자"""
        sampled_indices = []
        
        for class_label in self.unique_classes:
            class_indices = self.class_to_indices[class_label]
            
            if self.replacement or len(class_indices) >= self.samples_per_class:
                # 복원 추출 또는 충분한 샘플이 있는 경우
                sampled = random.choices(class_indices, k=self.samples_per_class)
            else:
                # 복원 없이 부족한 경우 모든 샘플 사용
                sampled = class_indices.copy()
                
                # 부족한 만큼 무작위 복제
                shortage = self.samples_per_class - len(sampled)
                additional = random.choices(class_indices, k=shortage)
                sampled.extend(additional)
            
            sampled_indices.extend(sampled)
        
        # 전체 샘플을 셞플
        random.shuffle(sampled_indices)
        return iter(sampled_indices)
    
    def __len__(self) -> int:
        return self.total_samples


class BatchDataProcessingManager:
    """배치 데이터 처리 관리자"""
    
    def __init__(self, configuration: DataLoadingConfiguration):
        self.config = configuration
        self.logger = PillSnapLogger(__name__)
        
        # 성능 통계
        self.performance_stats = {
            'batches_processed': 0,
            'total_samples_processed': 0,
            'avg_batch_processing_time_ms': 0,
            'memory_usage_mb': [],
            'gpu_utilization': []
        }
        
        self.logger.info(f"BatchDataProcessingManager 초기화")
        self.logger.info(f"  배치 크기: {self.config.batch_size}")
        self.logger.info(f"  워커 수: {self.config.num_workers}")
        self.logger.info(f"  메모리 고정: {self.config.pin_memory}")
    
    def create_single_pill_dataloader(
        self,
        dataset: SinglePillDatasetHandler,
        use_balanced_sampling: bool = False
    ) -> DataLoader:
        """Single Pill 데이터로더 생성"""
        
        # 샘플러 선택
        if use_balanced_sampling and dataset.is_training:
            sampler = BalancedClassSampler(dataset)
            shuffle = False  # 샘플러 사용시 shuffle=False
        else:
            sampler = None
            shuffle = self.config.shuffle_training and dataset.is_training
        
        # 커스텀 배치 콜레이터
        def single_pill_collate_fn(batch):
            """Single Pill 배치 콜레이터"""
            images = torch.stack([item['image'] for item in batch])
            class_labels = torch.stack([item['class_label'] for item in batch])
            edi_codes = [item['edi_code'] for item in batch]
            image_paths = [item['image_path'] for item in batch]
            preprocessing_infos = [item['preprocessing_info'] for item in batch]
            
            # 메모리 포맷 최적화
            if self.config.memory_format == "channels_last":
                images = images.to(memory_format=torch.channels_last)
            
            return {
                'images': images,
                'class_labels': class_labels,
                'edi_codes': edi_codes,
                'image_paths': image_paths,
                'preprocessing_infos': preprocessing_infos,
                'batch_size': len(batch)
            }
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=single_pill_collate_fn,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last_batch,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.logger.info(f"Single Pill 데이터로더 생성 완료")
        self.logger.info(f"  데이터셋 크기: {len(dataset)}")
        self.logger.info(f"  배치 수: {len(dataloader)}")
        self.logger.info(f"  균형 샘플링: {use_balanced_sampling}")
        
        return dataloader
    
    def create_combination_pill_dataloader(
        self,
        dataset: CombinationPillDatasetHandler
    ) -> DataLoader:
        """Combination Pill 데이터로더 생성"""
        
        def combination_pill_collate_fn(batch):
            """Combination Pill 배치 콜레이터"""
            images = torch.stack([item['image'] for item in batch])
            
            # 가변 길이 타겟 처리
            targets = []
            for i, item in enumerate(batch):
                # 배치 인덱스 추가: [batch_idx, class_id, x_center, y_center, width, height]
                if len(item['targets']) > 0:
                    batch_targets = torch.cat([
                        torch.full((len(item['targets']), 1), i, dtype=torch.float32),
                        item['targets']
                    ], dim=1)
                    targets.append(batch_targets)
            
            # 모든 타겟을 하나의 텐서로 결합
            if targets:
                all_targets = torch.cat(targets, dim=0)
            else:
                all_targets = torch.zeros((0, 6), dtype=torch.float32)
            
            image_paths = [item['image_path'] for item in batch]
            annotation_paths = [item['annotation_path'] for item in batch]
            preprocessing_infos = [item['preprocessing_info'] for item in batch]
            num_detections = [item['num_detections'] for item in batch]
            
            # 메모리 포맷 최적화
            if self.config.memory_format == "channels_last":
                images = images.to(memory_format=torch.channels_last)
            
            return {
                'images': images,
                'targets': all_targets,  # [N, 6] where N is total detections across batch
                'image_paths': image_paths,
                'annotation_paths': annotation_paths,
                'preprocessing_infos': preprocessing_infos,
                'num_detections': num_detections,
                'batch_size': len(batch)
            }
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_training and dataset.is_training,
            num_workers=self.config.num_workers,
            collate_fn=combination_pill_collate_fn,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last_batch,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.logger.info(f"Combination Pill 데이터로더 생성 완료")
        self.logger.info(f"  데이터셋 크기: {len(dataset)}")
        self.logger.info(f"  배치 수: {len(dataloader)}")
        
        return dataloader
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.performance_stats.copy()


class ProgressiveValidationDataProvider:
    """Progressive Validation 단계별 데이터 제공자"""
    
    def __init__(
        self, 
        data_root: str,
        stage1_sample_path: Optional[Path] = None,
        registry_path: Optional[Path] = None
    ):
        self.data_root = Path(data_root)
        self.stage1_sample_path = stage1_sample_path
        self.registry_path = registry_path
        self.logger = PillSnapLogger(__name__)
        
        # 레지스트리 로드
        self.registry = None
        if registry_path and registry_path.exists():
            try:
                self.registry = load_pharmaceutical_registry_from_artifacts(registry_path)
                self.logger.info(f"의약품 레지스트리 로드 완료: {len(self.registry.drug_records)}개 레코드")
            except Exception as e:
                self.logger.error(f"레지스트리 로드 실패: {e}")
        
        # 전처리기 초기화
        self.preprocessor = image_preprocessing_factory("stage1")
        
        self.logger.info(f"ProgressiveValidationDataProvider 초기화 완료")
        self.logger.info(f"  데이터 루트: {self.data_root}")
        self.logger.info(f"  Stage 1 샘플: {self.stage1_sample_path}")
        self.logger.info(f"  레지스트리: {self.registry_path}")
    
    def create_stage1_single_pill_dataset(
        self, 
        train_split: float = 0.8,
        enable_validation: bool = True
    ) -> Tuple[SinglePillDatasetHandler, Optional[SinglePillDatasetHandler]]:
        """Stage 1 Single Pill 데이터셋 생성"""
        
        if not self.stage1_sample_path or not self.stage1_sample_path.exists():
            raise FileNotFoundError(f"Stage 1 샘플 파일 없음: {self.stage1_sample_path}")
        
        if not self.registry:
            raise RuntimeError("의약품 레지스트리가 로드되지 않음")
        
        # Stage 1 샘플 데이터 로드
        with open(self.stage1_sample_path, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        
        # 이미지 경로와 라벨 수집
        all_image_paths = []
        all_class_labels = []
        all_edi_codes = []
        
        selected_k_codes = stage1_data['metadata']['selected_classes']
        
        for k_code in selected_k_codes:
            sample_info = stage1_data['samples'][k_code]
            
            # 레지스트리에서 클래스 ID와 EDI 코드 가져오기
            drug_record = self.registry.lookup_drug_by_k_code(k_code)
            if not drug_record:
                self.logger.warning(f"레지스트리에서 {k_code} 찾을 수 없음")
                continue
            
            class_id = drug_record.stage1_class_id
            edi_code = drug_record.edi_code
            
            # Single 이미지 경로 추가
            for img_path_str in sample_info['single_images']:
                img_path = Path(img_path_str)
                if img_path.exists():
                    all_image_paths.append(img_path)
                    all_class_labels.append(class_id)
                    all_edi_codes.append(edi_code)
                else:
                    self.logger.warning(f"이미지 파일 없음: {img_path}")
        
        if not all_image_paths:
            raise RuntimeError("유효한 이미지 데이터가 없음")
        
        self.logger.info(f"Stage 1 Single Pill 데이터 수집 완료: {len(all_image_paths)}개")
        
        # Train/Validation 분할
        if enable_validation:
            # 클래스별로 분할하여 균등 분포 유지
            train_indices = []
            val_indices = []
            
            class_to_indices = defaultdict(list)
            for idx, class_label in enumerate(all_class_labels):
                class_to_indices[class_label].append(idx)
            
            for class_indices in class_to_indices.values():
                random.shuffle(class_indices)
                split_point = int(len(class_indices) * train_split)
                train_indices.extend(class_indices[:split_point])
                val_indices.extend(class_indices[split_point:])
            
            # 트레이닝 데이터셋
            train_image_paths = [all_image_paths[i] for i in train_indices]
            train_class_labels = [all_class_labels[i] for i in train_indices]
            train_edi_codes = [all_edi_codes[i] for i in train_indices]
            
            train_dataset = SinglePillDatasetHandler(
                image_paths=train_image_paths,
                class_labels=train_class_labels,
                edi_codes=train_edi_codes,
                preprocessor=self.preprocessor,
                is_training=True,
                enable_data_validation=enable_validation
            )
            
            # 검증 데이터셋
            if val_indices:
                val_image_paths = [all_image_paths[i] for i in val_indices]
                val_class_labels = [all_class_labels[i] for i in val_indices]
                val_edi_codes = [all_edi_codes[i] for i in val_indices]
                
                val_dataset = SinglePillDatasetHandler(
                    image_paths=val_image_paths,
                    class_labels=val_class_labels,
                    edi_codes=val_edi_codes,
                    preprocessor=self.preprocessor,
                    is_training=False,
                    enable_data_validation=enable_validation
                )
            else:
                val_dataset = None
            
            return train_dataset, val_dataset
        
        else:
            # 검증 없이 전체를 트레이닝용으로
            dataset = SinglePillDatasetHandler(
                image_paths=all_image_paths,
                class_labels=all_class_labels,
                edi_codes=all_edi_codes,
                preprocessor=self.preprocessor,
                is_training=True,
                enable_data_validation=enable_validation
            )
            
            return dataset, None
    
    def create_stage1_combination_pill_dataset(
        self,
        yolo_dataset_path: Path,
        enable_validation: bool = True
    ) -> Tuple[CombinationPillDatasetHandler, Optional[CombinationPillDatasetHandler]]:
        """Stage 1 Combination Pill 데이터셋 생성"""
        
        if not yolo_dataset_path.exists():
            raise FileNotFoundError(f"YOLO 데이터셋 경로 없음: {yolo_dataset_path}")
        
        # 이미지와 어노테이션 파일 찾기
        train_images_dir = yolo_dataset_path / 'images' / 'train'
        train_labels_dir = yolo_dataset_path / 'labels' / 'train'
        val_images_dir = yolo_dataset_path / 'images' / 'val'
        val_labels_dir = yolo_dataset_path / 'labels' / 'val'
        
        def collect_image_annotation_pairs(images_dir: Path, labels_dir: Path):
            """이미지-어노테이션 쌍 수집"""
            image_paths = []
            annotation_paths = []
            
            if not images_dir.exists() or not labels_dir.exists():
                return image_paths, annotation_paths
            
            for img_file in images_dir.glob('*.jpg'):
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    image_paths.append(img_file)
                    annotation_paths.append(label_file)
                else:
                    self.logger.warning(f"어노테이션 파일 없음: {label_file}")
            
            return image_paths, annotation_paths
        
        # 트레이닝 데이터 수집
        train_image_paths, train_annotation_paths = collect_image_annotation_pairs(
            train_images_dir, train_labels_dir
        )
        
        if not train_image_paths:
            raise RuntimeError(f"트레이닝 이미지가 없음: {train_images_dir}")
        
        # 클래스 ID 매핑 구성 (레지스트리에서)
        class_id_mapping = {}
        if self.registry:
            mapping = self.registry.get_classification_mapping_for_stage1()
            class_id_mapping = mapping.k_code_to_class_id
        
        # 트레이닝 데이터셋
        train_dataset = CombinationPillDatasetHandler(
            image_paths=train_image_paths,
            annotation_paths=train_annotation_paths,
            preprocessor=self.preprocessor,
            class_id_mapping=class_id_mapping,
            is_training=True,
            enable_data_validation=enable_validation
        )
        
        # 검증 데이터셋
        val_dataset = None
        if enable_validation:
            val_image_paths, val_annotation_paths = collect_image_annotation_pairs(
                val_images_dir, val_labels_dir
            )
            
            if val_image_paths:
                val_dataset = CombinationPillDatasetHandler(
                    image_paths=val_image_paths,
                    annotation_paths=val_annotation_paths,
                    preprocessor=self.preprocessor,
                    class_id_mapping=class_id_mapping,
                    is_training=False,
                    enable_data_validation=enable_validation
                )
        
        self.logger.info(f"Stage 1 Combination Pill 데이터셋 생성 완료")
        self.logger.info(f"  트레이닝: {len(train_dataset)}개")
        self.logger.info(f"  검증: {len(val_dataset) if val_dataset else 0}개")
        
        return train_dataset, val_dataset


def create_dataloader_configuration_for_stage1() -> DataLoadingConfiguration:
    """Stage 1용 데이터로더 설정 생성"""
    return DataLoadingConfiguration(
        batch_size=16,          # Stage 1용 작은 배치
        num_workers=4,          # Stage 1용 적은 워커
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        shuffle_training=True,
        drop_last_batch=False,
        memory_format="channels_last",
        use_lmdb_cache=False,   # 추후 구현
        enable_profiling=False,
        max_batch_size_auto_tune=True,
        validate_data_integrity=True,
        log_batch_statistics=False
    )


if __name__ == "__main__":
    # 사용 예제
    config = load_config()
    
    # 데이터 제공자 초기화
    data_provider = ProgressiveValidationDataProvider(
        data_root=config['data']['root'],
        stage1_sample_path=Path("artifacts/stage1/sampling/stage1_sample_test.json"),
        registry_path=Path("artifacts/stage1/registry/pharmaceutical_registry.json")
    )
    
    try:
        # Stage 1 Single Pill 데이터셋 생성
        train_dataset, val_dataset = data_provider.create_stage1_single_pill_dataset()
        
        print(f"Single Pill 데이터셋 생성 완료")
        print(f"  트레이닝: {len(train_dataset)}개")
        print(f"  검증: {len(val_dataset) if val_dataset else 0}개")
        
        # 데이터로더 생성
        config_loader = create_dataloader_configuration_for_stage1()
        batch_manager = BatchDataProcessingManager(config_loader)
        
        train_dataloader = batch_manager.create_single_pill_dataloader(train_dataset)
        print(f"데이터로더 생성 완료: {len(train_dataloader)}개 배치")
        
        # 첫 번째 배치 테스트
        first_batch = next(iter(train_dataloader))
        print(f"배치 테스트:")
        print(f"  이미지 형태: {first_batch['images'].shape}")
        print(f"  클래스 라벨 형태: {first_batch['class_labels'].shape}")
        print(f"  EDI 코드 수: {len(first_batch['edi_codes'])}")
        
    except Exception as e:
        print(f"데이터로더 테스트 실패: {e}")
        import traceback
        traceback.print_exc()