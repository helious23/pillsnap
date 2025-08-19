"""
데이터 로더 엄격한 검증 테스트

실제 구현 품질을 검증하는 테스트:
- 실제 데이터로 데이터 무결성 검증
- 메모리 누수 및 성능 검증  
- Edge case 및 오류 처리 검증
- 배치 처리 정확성 검증
- 멀티프로세싱 안정성 검증
"""

import pytest
import tempfile
import json
import shutil
import time
import gc
import sys
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import psutil
import os

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataloaders import (
    DataLoadingConfiguration,
    SinglePillDatasetHandler,
    CombinationPillDatasetHandler,
    BalancedClassSampler,
    BatchDataProcessingManager,
    ProgressiveValidationDataProvider,
    ImagePathValidationError,
    DataIntegrityValidationError,
    create_dataloader_configuration_for_stage1
)
from src.data.image_preprocessing import image_preprocessing_factory
from src.data.pharmaceutical_code_registry import PharmaceuticalCodeRegistry


class TestDataLoadingConfigurationValidation:
    """데이터 로딩 설정 엄격 검증"""
    
    def test_configuration_parameter_bounds_validation(self):
        """설정 매개변수 경계 검증"""
        # 유효한 설정
        valid_config = DataLoadingConfiguration(
            batch_size=32,
            num_workers=4,
            prefetch_factor=2
        )
        assert valid_config.batch_size == 32
        assert valid_config.num_workers == 4
        assert valid_config.prefetch_factor == 2
        
        # 유효하지 않은 배치 크기
        with pytest.raises(ValueError, match="배치 크기는 양수여야 함"):
            DataLoadingConfiguration(batch_size=0)
        
        with pytest.raises(ValueError, match="배치 크기는 양수여야 함"):
            DataLoadingConfiguration(batch_size=-1)
        
        # 유효하지 않은 워커 수
        with pytest.raises(ValueError, match="워커 수는 음수일 수 없음"):
            DataLoadingConfiguration(num_workers=-1)
        
        # 유효하지 않은 프리페치 팩터
        with pytest.raises(ValueError, match="프리페치 팩터는 1 이상이어야 함"):
            DataLoadingConfiguration(prefetch_factor=0)
    
    def test_configuration_logical_consistency(self):
        """설정 논리적 일관성 검증"""
        # persistent_workers는 num_workers > 0일 때만 의미있음
        config_no_workers = DataLoadingConfiguration(
            num_workers=0,
            persistent_workers=True
        )
        # 이 경우 경고를 출력하거나 자동으로 False로 설정해야 함
        # 여기서는 단순히 설정이 생성되는지만 확인
        assert config_no_workers.num_workers == 0
        assert config_no_workers.persistent_workers == True  # 설정 그대로 유지
    
    def test_stage1_specific_configuration(self):
        """Stage 1 전용 설정 검증"""
        stage1_config = create_dataloader_configuration_for_stage1()
        
        # Stage 1 요구사항에 맞는 설정인지 확인
        assert stage1_config.batch_size == 16  # 작은 배치
        assert stage1_config.num_workers == 4   # 적은 워커
        assert stage1_config.memory_format == "channels_last"
        assert stage1_config.validate_data_integrity == True
        assert stage1_config.use_lmdb_cache == False  # 아직 구현 안됨


class TestSinglePillDatasetHandlerStrictValidation:
    """Single Pill 데이터셋 핸들러 엄격 검증"""
    
    @pytest.fixture
    def realistic_test_environment(self):
        """현실적인 테스트 환경 구성"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # 현실적인 클래스 분포 (불균형)
        class_distributions = {
            0: 100,  # 많은 샘플
            1: 50,   # 중간 샘플
            2: 20,   # 적은 샘플
            3: 5,    # 매우 적은 샘플
        }
        
        edi_codes_mapping = {
            0: "EDI001",
            1: "EDI002", 
            2: "EDI003",
            3: "EDI004"
        }
        
        all_image_paths = []
        all_class_labels = []
        all_edi_codes = []
        
        # 실제와 유사한 이미지 생성
        for class_id, count in class_distributions.items():
            class_dir = temp_path / f"class_{class_id}"
            class_dir.mkdir()
            
            for i in range(count):
                # 다양한 크기의 이미지 생성 (현실적)
                width = np.random.randint(200, 1200)
                height = np.random.randint(200, 1200)
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255), 
                    np.random.randint(0, 255)
                )
                
                image = Image.new('RGB', (width, height), color=color)
                image_path = class_dir / f"pill_{class_id}_{i:03d}.jpg"
                image.save(image_path, quality=np.random.randint(70, 95))
                
                all_image_paths.append(image_path)
                all_class_labels.append(class_id)
                all_edi_codes.append(edi_codes_mapping[class_id])
        
        # 손상된 이미지 몇 개 추가 (현실적 시나리오)
        corrupted_dir = temp_path / "corrupted"
        corrupted_dir.mkdir()
        
        # 텍스트 파일을 이미지 확장자로 생성 (손상된 이미지)
        corrupted_image_path = corrupted_dir / "corrupted.jpg"
        with open(corrupted_image_path, 'w') as f:
            f.write("This is not an image")
        
        yield {
            'temp_path': temp_path,
            'image_paths': all_image_paths,
            'class_labels': all_class_labels,
            'edi_codes': all_edi_codes,
            'class_distributions': class_distributions,
            'corrupted_image_path': corrupted_image_path
        }
        
        # 클린업
        shutil.rmtree(temp_dir)
    
    def test_dataset_integrity_validation_thoroughness(self, realistic_test_environment):
        """데이터셋 무결성 검증 철저함 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        # 1. 정상 데이터셋 - 통과해야 함
        valid_dataset = SinglePillDatasetHandler(
            image_paths=env['image_paths'],
            class_labels=env['class_labels'],
            edi_codes=env['edi_codes'],
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        assert len(valid_dataset) == len(env['image_paths'])
        
        # 2. 길이 불일치 - 실패해야 함
        with pytest.raises(DataIntegrityValidationError, match="데이터 길이 불일치"):
            SinglePillDatasetHandler(
                image_paths=env['image_paths'],
                class_labels=env['class_labels'][:-1],  # 하나 적음
                edi_codes=env['edi_codes'],
                preprocessor=preprocessor,
                enable_data_validation=True
            )
        
        # 3. 빈 데이터셋 - 실패해야 함
        with pytest.raises(DataIntegrityValidationError, match="빈 데이터셋"):
            SinglePillDatasetHandler(
                image_paths=[],
                class_labels=[],
                edi_codes=[],
                preprocessor=preprocessor,
                enable_data_validation=True
            )
        
        # 4. 유효하지 않은 클래스 라벨 - 실패해야 함
        invalid_labels = env['class_labels'].copy()
        invalid_labels[0] = -1  # 음수 라벨
        
        with pytest.raises(DataIntegrityValidationError, match="유효하지 않은 클래스 라벨"):
            SinglePillDatasetHandler(
                image_paths=env['image_paths'],
                class_labels=invalid_labels,
                edi_codes=env['edi_codes'],
                preprocessor=preprocessor,
                enable_data_validation=True
            )
        
        # 5. 유효하지 않은 EDI 코드 - 실패해야 함
        invalid_edi_codes = env['edi_codes'].copy()
        invalid_edi_codes[0] = ""  # 빈 문자열
        
        with pytest.raises(DataIntegrityValidationError, match="유효하지 않은 EDI 코드"):
            SinglePillDatasetHandler(
                image_paths=env['image_paths'],
                class_labels=env['class_labels'],
                edi_codes=invalid_edi_codes,
                preprocessor=preprocessor,
                enable_data_validation=True
            )
        
        # 6. 존재하지 않는 이미지 파일 - 실패해야 함
        # 처음 10개를 모두 가짜 경로로 변경하여 확실히 샘플링에 포함되도록 함
        fake_paths = env['image_paths'].copy()
        for i in range(min(10, len(fake_paths))):
            fake_paths[i] = Path(f"/nonexistent/fake_{i}.jpg")
        
        with pytest.raises(ImagePathValidationError, match="존재하지 않는 이미지 파일들"):
            SinglePillDatasetHandler(
                image_paths=fake_paths,
                class_labels=env['class_labels'],
                edi_codes=env['edi_codes'],
                preprocessor=preprocessor,
                enable_data_validation=True
            )
    
    def test_dataset_statistics_accuracy(self, realistic_test_environment):
        """데이터셋 통계 정확성 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=env['image_paths'],
            class_labels=env['class_labels'],
            edi_codes=env['edi_codes'],
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        
        stats = dataset.get_dataset_statistics()
        
        # 실제 데이터와 통계 일치 확인
        assert stats['total_samples'] == len(env['image_paths'])
        assert stats['unique_classes'] == len(env['class_distributions'])
        assert stats['unique_edi_codes'] == len(set(env['edi_codes']))
        
        # 클래스 분포 정확성 확인
        for class_id, expected_count in env['class_distributions'].items():
            actual_count = stats['class_distribution'][class_id]
            assert actual_count == expected_count, f"클래스 {class_id} 분포 불일치: 예상={expected_count}, 실제={actual_count}"
        
        # 최소/최대/평균 값 검증
        expected_min = min(env['class_distributions'].values())
        expected_max = max(env['class_distributions'].values())
        expected_mean = sum(env['class_distributions'].values()) / len(env['class_distributions'])
        
        assert stats['min_class_samples'] == expected_min
        assert stats['max_class_samples'] == expected_max
        assert abs(stats['mean_class_samples'] - expected_mean) < 0.1
    
    def test_getitem_error_handling_robustness(self, realistic_test_environment):
        """__getitem__ 오류 처리 견고성 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=env['image_paths'],
            class_labels=env['class_labels'],
            edi_codes=env['edi_codes'],
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        
        # 1. 유효한 인덱스 - 성공해야 함
        valid_item = dataset[0]
        assert 'image' in valid_item
        assert 'class_label' in valid_item
        assert 'edi_code' in valid_item
        assert valid_item['image'].shape == (3, 384, 384)  # 분류용 크기
        assert isinstance(valid_item['class_label'], torch.Tensor)
        
        # 2. 인덱스 범위 초과 - 실패해야 함
        with pytest.raises(IndexError, match="인덱스 범위 초과"):
            dataset[len(dataset)]
        
        with pytest.raises(IndexError, match="인덱스 범위 초과"):
            dataset[-len(dataset) - 1]
        
        # 3. 손상된 이미지가 있는 데이터셋 - 오류 처리 확인
        corrupted_paths = env['image_paths'][:5] + [env['corrupted_image_path']]
        corrupted_labels = env['class_labels'][:5] + [0]
        corrupted_edi_codes = env['edi_codes'][:5] + ["EDI999"]
        
        # 검증 비활성화하여 손상된 데이터셋 생성
        corrupted_dataset = SinglePillDatasetHandler(
            image_paths=corrupted_paths,
            class_labels=corrupted_labels,
            edi_codes=corrupted_edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False  # 검증 비활성화
        )
        
        # 정상 이미지는 로드되어야 함
        for i in range(5):
            item = corrupted_dataset[i]
            assert item['image'].shape == (3, 384, 384)
        
        # 손상된 이미지는 RuntimeError 발생해야 함
        with pytest.raises(RuntimeError, match="이미지 전처리 실패"):
            corrupted_dataset[5]  # 손상된 이미지
    
    def test_class_weights_calculation_accuracy(self, realistic_test_environment):
        """클래스 가중치 계산 정확성 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=env['image_paths'],
            class_labels=env['class_labels'],
            edi_codes=env['edi_codes'],
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        
        class_weights = dataset.get_class_weights()
        
        # 클래스 가중치 검증
        assert len(class_weights) == len(env['class_distributions'])
        
        # 역빈도 가중치 공식 확인: total_samples / (num_classes * class_count)
        total_samples = sum(env['class_distributions'].values())
        num_classes = len(env['class_distributions'])
        
        for class_id, class_count in env['class_distributions'].items():
            expected_weight = total_samples / (num_classes * class_count)
            actual_weight = class_weights[class_id].item()
            
            assert abs(actual_weight - expected_weight) < 1e-6, \
                f"클래스 {class_id} 가중치 불일치: 예상={expected_weight:.6f}, 실제={actual_weight:.6f}"
        
        # 적은 샘플 클래스가 높은 가중치를 가져야 함
        assert class_weights[3] > class_weights[0]  # 클래스 3(5개) > 클래스 0(100개)
    
    def test_memory_usage_monitoring(self, realistic_test_environment):
        """메모리 사용량 모니터링 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        # 메모리 사용량 측정
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 큰 데이터셋 생성
        large_paths = env['image_paths'] * 5  # 5배 확장
        large_labels = env['class_labels'] * 5
        large_edi_codes = env['edi_codes'] * 5
        
        dataset = SinglePillDatasetHandler(
            image_paths=large_paths,
            class_labels=large_labels,
            edi_codes=large_edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        
        # 여러 아이템 로드
        for i in range(min(50, len(dataset))):
            item = dataset[i]
            del item  # 명시적 삭제
        
        # 가비지 컬렉션
        gc.collect()
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # 메모리 증가가 합리적인 범위에 있는지 확인 (500MB 미만)
        assert memory_increase < 500, f"메모리 사용량 과다: {memory_increase:.1f}MB"
    
    def test_edi_code_lookup_functionality(self, realistic_test_environment):
        """EDI 코드 조회 기능 테스트"""
        env = realistic_test_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=env['image_paths'],
            class_labels=env['class_labels'],
            edi_codes=env['edi_codes'],
            preprocessor=preprocessor,
            enable_data_validation=True
        )
        
        # 각 EDI 코드별 샘플 조회
        for class_id, expected_count in env['class_distributions'].items():
            edi_code = f"EDI{class_id:03d}"
            sample_indices = dataset.get_sample_by_edi_code(edi_code)
            
            assert len(sample_indices) == expected_count, \
                f"EDI 코드 {edi_code} 샘플 수 불일치: 예상={expected_count}, 실제={len(sample_indices)}"
            
            # 인덱스가 유효한 범위에 있는지 확인
            for idx in sample_indices:
                assert 0 <= idx < len(dataset)
                assert dataset.edi_codes[idx] == edi_code
        
        # 존재하지 않는 EDI 코드
        non_existent_samples = dataset.get_sample_by_edi_code("NONEXISTENT")
        assert len(non_existent_samples) == 0


class TestCombinationPillDatasetHandlerStrictValidation:
    """Combination Pill 데이터셋 핸들러 엄격 검증"""
    
    @pytest.fixture
    def realistic_yolo_environment(self):
        """현실적인 YOLO 환경 구성"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # 이미지와 어노테이션 생성
        images_dir = temp_path / "images"
        labels_dir = temp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        image_paths = []
        annotation_paths = []
        
        # 다양한 시나리오의 어노테이션 생성
        annotation_scenarios = [
            # (이미지명, 검출 객체들)
            ("combo_001.jpg", [
                (0, 0.5, 0.5, 0.3, 0.4),  # 중앙 1개
            ]),
            ("combo_002.jpg", [
                (0, 0.3, 0.3, 0.2, 0.2),  # 좌상단
                (1, 0.7, 0.7, 0.2, 0.2),  # 우하단
            ]),
            ("combo_003.jpg", [
                (0, 0.2, 0.2, 0.15, 0.15),
                (1, 0.5, 0.2, 0.15, 0.15),
                (2, 0.8, 0.2, 0.15, 0.15),
                (0, 0.2, 0.5, 0.15, 0.15),
                (1, 0.5, 0.5, 0.15, 0.15),
            ]),
            ("combo_empty.jpg", []),  # 빈 어노테이션
        ]
        
        for image_name, detections in annotation_scenarios:
            # 이미지 생성
            image_path = images_dir / image_name
            width, height = np.random.randint(400, 1200), np.random.randint(400, 1200)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            image = Image.new('RGB', (width, height), color=color)
            image.save(image_path)
            
            # 어노테이션 생성
            annotation_path = labels_dir / (Path(image_name).stem + ".txt")
            with open(annotation_path, 'w') as f:
                for class_id, x_center, y_center, width, height in detections:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
            
            image_paths.append(image_path)
            annotation_paths.append(annotation_path)
        
        # 손상된 어노테이션 파일 생성
        corrupted_annotation = labels_dir / "corrupted.txt"
        with open(corrupted_annotation, 'w') as f:
            f.write("invalid format line\\n")
            f.write("1 0.5 0.5 0.3\\n")  # 값 부족
            f.write("not_number 0.5 0.5 0.3 0.4\\n")  # 숫자가 아님
            f.write("1 -0.1 0.5 0.3 0.4\\n")  # 범위 초과
        
        yield {
            'temp_path': temp_path,
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'image_paths': image_paths,
            'annotation_paths': annotation_paths,
            'scenarios': annotation_scenarios,
            'corrupted_annotation': corrupted_annotation
        }
        
        # 클린업
        shutil.rmtree(temp_dir)
    
    def test_yolo_annotation_loading_strictness(self, realistic_yolo_environment):
        """YOLO 어노테이션 로딩 엄격성 테스트"""
        env = realistic_yolo_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        # 정상 어노테이션 로딩
        dataset = CombinationPillDatasetHandler(
            image_paths=env['image_paths'],
            annotation_paths=env['annotation_paths'],
            preprocessor=preprocessor,
            class_id_mapping={0: 0, 1: 1, 2: 2},
            enable_data_validation=True
        )
        
        # 어노테이션 캐시 검증
        assert len(dataset.annotation_cache) == len(env['image_paths'])
        
        # 각 시나리오별 검증
        for i, (image_name, expected_detections) in enumerate(env['scenarios']):
            cached_annotations = dataset.annotation_cache[i]
            assert len(cached_annotations) == len(expected_detections), \
                f"{image_name}: 검출 수 불일치 - 예상={len(expected_detections)}, 실제={len(cached_annotations)}"
            
            for j, (exp_class, exp_x, exp_y, exp_w, exp_h) in enumerate(expected_detections):
                actual = cached_annotations[j]
                assert actual['class_id'] == exp_class
                assert abs(actual['x_center'] - exp_x) < 1e-6
                assert abs(actual['y_center'] - exp_y) < 1e-6
                assert abs(actual['width'] - exp_w) < 1e-6
                assert abs(actual['height'] - exp_h) < 1e-6
    
    def test_annotation_format_validation_thoroughness(self, realistic_yolo_environment):
        """어노테이션 형식 검증 철저함 테스트"""
        env = realistic_yolo_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        # 손상된 어노테이션이 있는 데이터셋
        corrupted_image_path = env['images_dir'] / "corrupted.jpg"
        Image.new('RGB', (640, 480), color='black').save(corrupted_image_path)
        
        corrupted_paths = [corrupted_image_path]
        corrupted_annotations = [env['corrupted_annotation']]
        
        # 검증 활성화시 실패해야 함
        with pytest.raises(DataIntegrityValidationError, match="어노테이션 파일 형식 오류"):
            CombinationPillDatasetHandler(
                image_paths=corrupted_paths,
                annotation_paths=corrupted_annotations,
                preprocessor=preprocessor,
                class_id_mapping={0: 0, 1: 1},
                enable_data_validation=True
            )
    
    def test_max_detections_limit_enforcement(self, realistic_yolo_environment):
        """최대 검출 수 제한 강제 테스트"""
        env = realistic_yolo_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        # 많은 검출이 있는 어노테이션 생성
        many_detections_annotation = env['labels_dir'] / "many_detections.txt"
        many_detections_image = env['images_dir'] / "many_detections.jpg"
        
        # 100개 검출 생성
        with open(many_detections_annotation, 'w') as f:
            for i in range(100):
                x = 0.1 + (i % 10) * 0.08
                y = 0.1 + (i // 10) * 0.08
                f.write(f"0 {x:.6f} {y:.6f} 0.05 0.05\\n")
        
        Image.new('RGB', (640, 480), color='red').save(many_detections_image)
        
        # 최대 검출 수 제한 설정
        max_detections = 20
        dataset = CombinationPillDatasetHandler(
            image_paths=[many_detections_image],
            annotation_paths=[many_detections_annotation],
            preprocessor=preprocessor,
            class_id_mapping={0: 0},
            enable_data_validation=False,  # 다른 검증 스킵
            max_detections_per_image=max_detections
        )
        
        # 제한된 수만 로드되어야 함
        cached_annotations = dataset.annotation_cache[0]
        assert len(cached_annotations) == max_detections, \
            f"최대 검출 수 제한 미적용: 예상={max_detections}, 실제={len(cached_annotations)}"
    
    def test_dataset_getitem_batch_compatibility(self, realistic_yolo_environment):
        """데이터셋 __getitem__ 배치 호환성 테스트"""
        env = realistic_yolo_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = CombinationPillDatasetHandler(
            image_paths=env['image_paths'],
            annotation_paths=env['annotation_paths'],
            preprocessor=preprocessor,
            class_id_mapping={0: 0, 1: 1, 2: 2},
            enable_data_validation=True
        )
        
        # 모든 아이템을 로드하여 일관성 확인
        for i in range(len(dataset)):
            item = dataset[i]
            
            # 필수 키 존재 확인
            required_keys = ['image', 'targets', 'image_path', 'annotation_path', 
                           'preprocessing_info', 'num_detections']
            for key in required_keys:
                assert key in item, f"필수 키 누락: {key}"
            
            # 이미지 형태 확인
            assert item['image'].shape == (3, 640, 640), \
                f"이미지 형태 오류: {item['image'].shape}"
            
            # 타겟 형태 확인
            targets = item['targets']
            assert targets.ndim == 2, f"타겟 차원 오류: {targets.ndim}"
            if len(targets) > 0:
                assert targets.shape[1] == 5, f"타겟 특성 수 오류: {targets.shape[1]}"
                
                # 좌표 범위 확인
                for target in targets:
                    class_id, x_center, y_center, width, height = target.tolist()
                    assert isinstance(class_id, (int, float))
                    assert 0 <= x_center <= 1
                    assert 0 <= y_center <= 1  
                    assert 0 < width <= 1
                    assert 0 < height <= 1
            
            # 검출 수 일치 확인
            expected_detections = item['num_detections']
            actual_detections = len(targets)
            assert actual_detections == expected_detections, \
                f"검출 수 불일치: 예상={expected_detections}, 실제={actual_detections}"
    
    def test_dataset_statistics_computation_accuracy(self, realistic_yolo_environment):
        """데이터셋 통계 계산 정확성 테스트"""
        env = realistic_yolo_environment
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = CombinationPillDatasetHandler(
            image_paths=env['image_paths'],
            annotation_paths=env['annotation_paths'],
            preprocessor=preprocessor,
            class_id_mapping={0: 0, 1: 1, 2: 2},
            enable_data_validation=True
        )
        
        stats = dataset.get_dataset_statistics()
        
        # 수동으로 계산한 통계와 비교
        manual_total_detections = 0
        manual_class_distribution = {0: 0, 1: 0, 2: 0}
        manual_detections_per_image = []
        
        for _, detections in env['scenarios']:
            manual_detections_per_image.append(len(detections))
            manual_total_detections += len(detections)
            
            for class_id, _, _, _, _ in detections:
                manual_class_distribution[class_id] += 1
        
        # 통계 검증
        assert stats['total_samples'] == len(env['image_paths'])
        assert stats['total_detections'] == manual_total_detections
        assert stats['class_distribution'] == manual_class_distribution
        assert stats['max_detections_per_image'] == max(manual_detections_per_image)
        assert stats['min_detections_per_image'] == min(manual_detections_per_image)
        
        expected_avg = manual_total_detections / len(env['image_paths'])
        assert abs(stats['avg_detections_per_image'] - expected_avg) < 1e-6


class TestBalancedClassSamplerStrictValidation:
    """균형 클래스 샘플러 엄격 검증"""
    
    def test_balanced_sampling_mathematical_accuracy(self):
        """균형 샘플링 수학적 정확성 테스트"""
        # 불균형 데이터셋 생성
        image_paths = [Path(f"/fake/image_{i}.jpg") for i in range(200)]
        class_labels = ([0] * 100 +    # 클래스 0: 100개
                       [1] * 50 +     # 클래스 1: 50개  
                       [2] * 30 +     # 클래스 2: 30개
                       [3] * 20)      # 클래스 3: 20개
        edi_codes = [f"EDI{label:03d}" for label in class_labels]
        
        preprocessor = image_preprocessing_factory("stage1")
        
        # 검증 비활성화하여 가짜 경로 허용
        dataset = SinglePillDatasetHandler(
            image_paths=image_paths,
            class_labels=class_labels,
            edi_codes=edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False
        )
        
        # 균형 샘플러 생성 (가장 적은 클래스 기준)
        sampler = BalancedClassSampler(dataset)
        
        # 샘플러 기본 속성 확인
        assert sampler.num_classes == 4
        assert sampler.samples_per_class == 20  # 가장 적은 클래스 (클래스 3)
        assert sampler.total_samples == 4 * 20  # 80개
        
        # 여러 번 샘플링하여 분포 확인
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        num_iterations = 10
        
        for _ in range(num_iterations):
            sampled_indices = list(sampler)
            
            # 길이 확인
            assert len(sampled_indices) == sampler.total_samples
            
            # 클래스별 샘플 수 집계
            iteration_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for idx in sampled_indices:
                class_label = dataset.class_labels[idx]
                iteration_counts[class_label] += 1
            
            # 각 반복에서 클래스별 샘플 수가 동일해야 함
            for class_id in range(4):
                assert iteration_counts[class_id] == sampler.samples_per_class, \
                    f"클래스 {class_id} 샘플 수 불일치: 예상={sampler.samples_per_class}, 실제={iteration_counts[class_id]}"
                
                class_counts[class_id] += iteration_counts[class_id]
        
        # 전체 반복에서 클래스 균형 확인
        total_samples_per_class = num_iterations * sampler.samples_per_class
        for class_id in range(4):
            assert class_counts[class_id] == total_samples_per_class, \
                f"전체 클래스 {class_id} 샘플 수 불일치: 예상={total_samples_per_class}, 실제={class_counts[class_id]}"
    
    def test_sampler_edge_cases_handling(self):
        """샘플러 경계 사례 처리 테스트"""
        preprocessor = image_preprocessing_factory("stage1")
        
        # 1. 단일 클래스 데이터셋
        single_class_paths = [Path(f"/fake/single_{i}.jpg") for i in range(10)]
        single_class_labels = [0] * 10
        single_class_edi_codes = ["EDI000"] * 10
        
        single_class_dataset = SinglePillDatasetHandler(
            image_paths=single_class_paths,
            class_labels=single_class_labels,
            edi_codes=single_class_edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False
        )
        
        single_sampler = BalancedClassSampler(single_class_dataset)
        assert single_sampler.num_classes == 1
        assert single_sampler.samples_per_class == 10
        
        # 2. 클래스별 매우 적은 샘플
        minimal_paths = [Path(f"/fake/minimal_{i}.jpg") for i in range(6)]
        minimal_labels = [0, 0, 1, 1, 2, 2]  # 각 클래스 2개씩
        minimal_edi_codes = [f"EDI{label:03d}" for label in minimal_labels]
        
        minimal_dataset = SinglePillDatasetHandler(
            image_paths=minimal_paths,
            class_labels=minimal_labels,
            edi_codes=minimal_edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False
        )
        
        minimal_sampler = BalancedClassSampler(minimal_dataset)
        assert minimal_sampler.num_classes == 3
        assert minimal_sampler.samples_per_class == 2
        assert minimal_sampler.total_samples == 6
        
        # 샘플링 수행하여 오류 없음 확인
        sampled_indices = list(minimal_sampler)
        assert len(sampled_indices) == 6
    
    def test_sampler_deterministic_behavior_with_seed(self):
        """샘플러 시드를 통한 결정적 동작 테스트"""
        image_paths = [Path(f"/fake/seed_{i}.jpg") for i in range(100)]
        class_labels = [i % 4 for i in range(100)]  # 4개 클래스 균등 분포
        edi_codes = [f"EDI{label:03d}" for label in class_labels]
        
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=image_paths,
            class_labels=class_labels,
            edi_codes=edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False
        )
        
        # 동일한 시드로 두 번 샘플링
        import random
        
        random.seed(42)
        sampler1 = BalancedClassSampler(dataset, samples_per_class=10)
        samples1 = list(sampler1)
        
        random.seed(42)
        sampler2 = BalancedClassSampler(dataset, samples_per_class=10)
        samples2 = list(sampler2)
        
        # 결과가 동일해야 함
        assert samples1 == samples2, "동일한 시드로 다른 샘플링 결과"


class TestBatchDataProcessingManagerStrictValidation:
    """배치 데이터 처리 관리자 엄격 검증"""
    
    def test_dataloader_creation_parameter_validation(self):
        """데이터로더 생성 매개변수 검증"""
        config = create_dataloader_configuration_for_stage1()
        manager = BatchDataProcessingManager(config)
        
        # 가짜 데이터셋 생성
        fake_paths = [Path(f"/fake/test_{i}.jpg") for i in range(50)]
        fake_labels = list(range(50))
        fake_edi_codes = [f"EDI{i:03d}" for i in range(50)]
        preprocessor = image_preprocessing_factory("stage1")
        
        dataset = SinglePillDatasetHandler(
            image_paths=fake_paths,
            class_labels=fake_labels,
            edi_codes=fake_edi_codes,
            preprocessor=preprocessor,
            enable_data_validation=False
        )
        
        # Single Pill 데이터로더 생성
        dataloader = manager.create_single_pill_dataloader(dataset)
        
        # 데이터로더 설정 확인
        assert dataloader.batch_size == config.batch_size
        assert dataloader.num_workers == config.num_workers
        assert dataloader.pin_memory == config.pin_memory
        assert dataloader.persistent_workers == config.persistent_workers
        assert dataloader.prefetch_factor == config.prefetch_factor
        
        # 배치 수 계산 확인
        expected_batches = (len(dataset) + config.batch_size - 1) // config.batch_size
        if config.drop_last_batch and len(dataset) % config.batch_size != 0:
            expected_batches -= 1
        
        assert len(dataloader) == expected_batches
    
    def test_single_pill_collate_function_accuracy(self):
        """Single Pill 콜레이트 함수 정확성 테스트"""
        config = create_dataloader_configuration_for_stage1()
        manager = BatchDataProcessingManager(config)
        
        # 실제 이미지로 테스트
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # 테스트 이미지들 생성
            image_paths = []
            for i in range(8):  # 작은 배치
                img_path = temp_path / f"test_{i}.jpg"
                img = Image.new('RGB', (400, 300), color=(i*30, 100, 150))
                img.save(img_path)
                image_paths.append(img_path)
            
            class_labels = list(range(8))
            edi_codes = [f"EDI{i:03d}" for i in range(8)]
            preprocessor = image_preprocessing_factory("stage1")
            
            dataset = SinglePillDatasetHandler(
                image_paths=image_paths,
                class_labels=class_labels,
                edi_codes=edi_codes,
                preprocessor=preprocessor,
                enable_data_validation=True
            )
            
            dataloader = manager.create_single_pill_dataloader(dataset)
            
            # 첫 번째 배치 가져오기
            batch = next(iter(dataloader))
            
            # 배치 구조 검증
            assert 'images' in batch
            assert 'class_labels' in batch
            assert 'edi_codes' in batch
            assert 'image_paths' in batch
            assert 'preprocessing_infos' in batch
            assert 'batch_size' in batch
            
            # 배치 크기 일관성 확인
            batch_size = min(config.batch_size, len(dataset))
            assert batch['batch_size'] == batch_size
            assert batch['images'].shape[0] == batch_size
            assert batch['class_labels'].shape[0] == batch_size
            assert len(batch['edi_codes']) == batch_size
            assert len(batch['image_paths']) == batch_size
            assert len(batch['preprocessing_infos']) == batch_size
            
            # 이미지 형태 확인
            assert batch['images'].shape == (batch_size, 3, 384, 384)
            assert batch['images'].dtype == torch.float32
            
            # 클래스 라벨 확인
            assert batch['class_labels'].dtype == torch.long
            
            # 메모리 포맷 확인 (channels_last)
            if config.memory_format == "channels_last":
                # 4D 텐서에서는 channels_last 확인 가능
                assert batch['images'].is_contiguous(memory_format=torch.channels_last)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_combination_pill_collate_function_accuracy(self):
        """Combination Pill 콜레이트 함수 정확성 테스트"""
        config = create_dataloader_configuration_for_stage1()
        manager = BatchDataProcessingManager(config)
        
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # 이미지와 어노테이션 생성
            images_dir = temp_path / "images"
            labels_dir = temp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            
            image_paths = []
            annotation_paths = []
            
            # 다양한 검출 수를 가진 이미지들
            detection_configs = [
                [  # 이미지 0: 2개 검출
                    (0, 0.3, 0.3, 0.2, 0.2),
                    (1, 0.7, 0.7, 0.2, 0.2)
                ],
                [  # 이미지 1: 1개 검출
                    (1, 0.5, 0.5, 0.3, 0.3)
                ],
                [  # 이미지 2: 3개 검출
                    (0, 0.2, 0.2, 0.15, 0.15),
                    (1, 0.5, 0.2, 0.15, 0.15),
                    (2, 0.8, 0.2, 0.15, 0.15)
                ]
            ]
            
            for i, detections in enumerate(detection_configs):
                # 이미지 생성
                img_path = images_dir / f"combo_{i}.jpg"
                img = Image.new('RGB', (640, 480), color=(i*80, 120, 200))
                img.save(img_path)
                image_paths.append(img_path)
                
                # 어노테이션 생성
                anno_path = labels_dir / f"combo_{i}.txt"
                with open(anno_path, 'w') as f:
                    for class_id, x, y, w, h in detections:
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\\n")
                annotation_paths.append(anno_path)
            
            preprocessor = image_preprocessing_factory("stage1")
            
            dataset = CombinationPillDatasetHandler(
                image_paths=image_paths,
                annotation_paths=annotation_paths,
                preprocessor=preprocessor,
                class_id_mapping={0: 0, 1: 1, 2: 2},
                enable_data_validation=True
            )
            
            dataloader = manager.create_combination_pill_dataloader(dataset)
            
            # 배치 가져오기
            batch = next(iter(dataloader))
            
            # 배치 구조 검증
            required_keys = ['images', 'targets', 'image_paths', 'annotation_paths', 
                           'preprocessing_infos', 'num_detections', 'batch_size']
            for key in required_keys:
                assert key in batch, f"필수 키 누락: {key}"
            
            # 이미지 배치 확인
            assert batch['images'].shape == (3, 3, 640, 640)  # (batch_size, channels, height, width)
            
            # 타겟 형태 확인 - [total_detections_in_batch, 6]
            # 6 = [batch_idx, class_id, x_center, y_center, width, height]
            targets = batch['targets']
            expected_total_detections = sum(len(detections) for detections in detection_configs)
            assert targets.shape == (expected_total_detections, 6)
            
            # 배치 인덱스 확인
            batch_indices = targets[:, 0].long().tolist()
            expected_batch_indices = []
            for i, detections in enumerate(detection_configs):
                expected_batch_indices.extend([i] * len(detections))
            
            assert batch_indices == expected_batch_indices
            
            # 검출 수 리스트 확인
            assert batch['num_detections'] == [2, 1, 3]
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_dataloader_multiprocessing_stability(self):
        """데이터로더 멀티프로세싱 안정성 테스트"""
        config = create_dataloader_configuration_for_stage1()
        config.num_workers = 2  # 멀티프로세싱 활성화
        manager = BatchDataProcessingManager(config)
        
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # 충분한 수의 이미지 생성
            image_paths = []
            for i in range(50):
                img_path = temp_path / f"mp_test_{i}.jpg"
                img = Image.new('RGB', (300, 300), color=(i*5, 100, 150))
                img.save(img_path)
                image_paths.append(img_path)
            
            class_labels = [i % 5 for i in range(50)]  # 5개 클래스
            edi_codes = [f"EDI{label:03d}" for label in class_labels]
            preprocessor = image_preprocessing_factory("stage1")
            
            dataset = SinglePillDatasetHandler(
                image_paths=image_paths,
                class_labels=class_labels,
                edi_codes=edi_codes,
                preprocessor=preprocessor,
                enable_data_validation=True
            )
            
            dataloader = manager.create_single_pill_dataloader(dataset)
            
            # 전체 데이터로더 반복하여 안정성 확인
            total_samples = 0
            batch_count = 0
            
            start_time = time.time()
            
            for batch in dataloader:
                batch_count += 1
                total_samples += batch['batch_size']
                
                # 배치 유효성 간단 확인
                assert batch['images'].shape[0] == batch['batch_size']
                assert batch['class_labels'].shape[0] == batch['batch_size']
                assert len(batch['edi_codes']) == batch['batch_size']
            
            end_time = time.time()
            
            # 전체 샘플 수 확인
            assert total_samples == len(dataset)
            
            # 처리 시간이 합리적인지 확인 (30초 미만)
            processing_time = end_time - start_time
            assert processing_time < 30, f"처리 시간 과다: {processing_time:.1f}초"
            
            print(f"멀티프로세싱 데이터로더 테스트 완료: {batch_count}개 배치, {processing_time:.1f}초")
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])