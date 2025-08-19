"""
COCO → YOLO 포맷 변환기 단위 테스트

Two-Stage Conditional Pipeline Combination Pills 검출용:
- COCO 바운딩 박스 → YOLO 상대좌표 변환 정확성
- K-code 추출 및 클래스 ID 매핑 검증
- 어노테이션 유효성 검증
- 파일 I/O 및 데이터셋 구조 검증
"""

import pytest
import tempfile
import json
import shutil
from pathlib import Path
from PIL import Image
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.format_converter import (
    BoundingBox,
    YOLOAnnotation,
    COCOToYOLOConverter,
    create_coco_to_yolo_converter_for_stage1
)


class TestBoundingBox:
    """바운딩 박스 클래스 테스트"""
    
    def test_coco_to_yolo_coordinate_conversion(self):
        """COCO → YOLO 좌표 변환 테스트"""
        # 테스트 케이스: 640x480 이미지에서 [100, 50, 200, 150] COCO 박스
        img_width, img_height = 640, 480
        coco_bbox = [100, 50, 200, 150]  # [x, y, width, height] 절대좌표
        
        bbox = BoundingBox.from_coco_bbox(coco_bbox, img_width, img_height)
        
        # 예상 YOLO 좌표 계산
        # COCO: x=100, y=50, w=200, h=150
        # YOLO 중심점: x_center = (100 + 200/2) / 640 = 200/640 = 0.3125
        #           y_center = (50 + 150/2) / 480 = 125/480 = 0.260417
        #           width = 200/640 = 0.3125
        #           height = 150/480 = 0.3125
        
        assert abs(bbox.x_center - 0.3125) < 1e-6
        assert abs(bbox.y_center - 125/480) < 1e-6
        assert abs(bbox.width - 0.3125) < 1e-6
        assert abs(bbox.height - 0.3125) < 1e-6
    
    def test_yolo_to_absolute_coordinate_conversion(self):
        """YOLO → 절대좌표 변환 테스트"""
        bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.4, height=0.3)
        img_width, img_height = 800, 600
        
        x1, y1, x2, y2 = bbox.to_absolute_coords(img_width, img_height)
        
        # 예상 절대좌표 계산
        # x_center=0.5 * 800 = 400, y_center=0.5 * 600 = 300
        # width=0.4 * 800 = 320, height=0.3 * 600 = 180
        # x1 = 400 - 320/2 = 240, y1 = 300 - 180/2 = 210
        # x2 = 400 + 320/2 = 560, y2 = 300 + 180/2 = 390
        
        assert x1 == 240
        assert y1 == 210
        assert x2 == 560
        assert y2 == 390
    
    def test_bounding_box_validation(self):
        """바운딩 박스 유효성 검증 테스트"""
        # 유효한 바운딩 박스
        valid_bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4)
        assert valid_bbox.is_valid() == True
        
        # 유효하지 않은 바운딩 박스들
        invalid_cases = [
            BoundingBox(x_center=-0.1, y_center=0.5, width=0.3, height=0.4),  # x_center < 0
            BoundingBox(x_center=1.1, y_center=0.5, width=0.3, height=0.4),   # x_center > 1
            BoundingBox(x_center=0.5, y_center=-0.1, width=0.3, height=0.4),  # y_center < 0
            BoundingBox(x_center=0.5, y_center=1.1, width=0.3, height=0.4),   # y_center > 1
            BoundingBox(x_center=0.5, y_center=0.5, width=0, height=0.4),     # width = 0
            BoundingBox(x_center=0.5, y_center=0.5, width=1.1, height=0.4),   # width > 1
            BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0),     # height = 0
            BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=1.1),   # height > 1
        ]
        
        for bbox in invalid_cases:
            assert bbox.is_valid() == False
    
    def test_yolo_format_string_generation(self):
        """YOLO 포맷 문자열 생성 테스트"""
        bbox = BoundingBox(x_center=0.5, y_center=0.3, width=0.4, height=0.6)
        class_id = 2
        
        yolo_string = bbox.to_yolo_string(class_id)
        expected = "2 0.500000 0.300000 0.400000 0.600000"
        
        assert yolo_string == expected
    
    def test_edge_cases(self):
        """경계 케이스 테스트"""
        # 이미지 가장자리의 바운딩 박스
        edge_cases = [
            ([0, 0, 100, 100], 100, 100),       # 좌상단 모서리
            ([50, 50, 50, 50], 100, 100),       # 우하단 모서리
            ([0, 25, 100, 50], 100, 100),       # 전체 폭
            ([25, 0, 50, 100], 100, 100),       # 전체 높이
        ]
        
        for coco_bbox, img_w, img_h in edge_cases:
            bbox = BoundingBox.from_coco_bbox(coco_bbox, img_w, img_h)
            
            # 변환된 좌표가 유효한 범위에 있는지 확인
            assert 0 <= bbox.x_center <= 1
            assert 0 <= bbox.y_center <= 1
            assert 0 < bbox.width <= 1
            assert 0 < bbox.height <= 1
            assert bbox.is_valid()


class TestYOLOAnnotation:
    """YOLO 어노테이션 클래스 테스트"""
    
    @pytest.fixture
    def sample_image_file(self):
        """테스트용 이미지 파일 생성"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_image = Image.new('RGB', (640, 480), color='blue')
            test_image.save(f.name)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # 클린업
        temp_path.unlink()
    
    def test_yolo_annotation_creation(self, sample_image_file):
        """YOLO 어노테이션 생성 테스트"""
        bboxes = [
            BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4),
            BoundingBox(x_center=0.2, y_center=0.8, width=0.2, height=0.3)
        ]
        class_ids = [0, 1]
        
        annotation = YOLOAnnotation(
            image_path=sample_image_file,
            image_width=640,
            image_height=480,
            bboxes=bboxes,
            class_ids=class_ids
        )
        
        assert annotation.image_path == sample_image_file
        assert annotation.image_width == 640
        assert annotation.image_height == 480
        assert len(annotation.bboxes) == 2
        assert len(annotation.class_ids) == 2
    
    def test_yolo_format_output(self, sample_image_file):
        """YOLO 포맷 출력 테스트"""
        bboxes = [
            BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4),
            BoundingBox(x_center=0.2, y_center=0.8, width=0.2, height=0.3)
        ]
        class_ids = [0, 1]
        
        annotation = YOLOAnnotation(
            image_path=sample_image_file,
            image_width=640,
            image_height=480,
            bboxes=bboxes,
            class_ids=class_ids
        )
        
        yolo_text = annotation.to_yolo_format()
        lines = yolo_text.strip().split('\n')
        
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.500000 0.300000 0.400000"
        assert lines[1] == "1 0.200000 0.800000 0.200000 0.300000"
    
    def test_annotation_filename_generation(self, sample_image_file):
        """어노테이션 파일명 생성 테스트"""
        annotation = YOLOAnnotation(
            image_path=sample_image_file,
            image_width=640,
            image_height=480,
            bboxes=[],
            class_ids=[]
        )
        
        filename = annotation.get_annotation_filename()
        expected_filename = sample_image_file.stem + '.txt'
        
        assert filename == expected_filename
    
    def test_annotation_validation(self, sample_image_file):
        """어노테이션 유효성 검증 테스트"""
        # 유효한 어노테이션
        valid_annotation = YOLOAnnotation(
            image_path=sample_image_file,
            image_width=640,
            image_height=480,
            bboxes=[BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4)],
            class_ids=[0]
        )
        
        validation_result = valid_annotation.validate()
        assert validation_result['is_valid'] == True
        assert len(validation_result['errors']) == 0
        assert validation_result['stats']['total_boxes'] == 1
        assert validation_result['stats']['valid_boxes'] == 1
        assert validation_result['stats']['invalid_boxes'] == 0
    
    def test_annotation_validation_failures(self):
        """어노테이션 유효성 검증 실패 케이스"""
        fake_path = Path("/nonexistent/fake.jpg")
        
        # 존재하지 않는 이미지 파일
        invalid_annotation = YOLOAnnotation(
            image_path=fake_path,
            image_width=640,
            image_height=480,
            bboxes=[BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4)],
            class_ids=[0]
        )
        
        validation_result = invalid_annotation.validate()
        assert validation_result['is_valid'] == False
        assert any("이미지 파일 없음" in error for error in validation_result['errors'])
    
    def test_bbox_class_count_mismatch(self, sample_image_file):
        """바운딩 박스와 클래스 ID 개수 불일치 테스트"""
        annotation = YOLOAnnotation(
            image_path=sample_image_file,
            image_width=640,
            image_height=480,
            bboxes=[
                BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4),
                BoundingBox(x_center=0.2, y_center=0.8, width=0.2, height=0.3)
            ],
            class_ids=[0]  # 바운딩 박스는 2개, 클래스 ID는 1개
        )
        
        validation_result = annotation.validate()
        assert validation_result['is_valid'] == False
        assert any("개수 불일치" in error for error in validation_result['errors'])


class TestCOCOToYOLOConverter:
    """COCO → YOLO 변환기 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        yield temp_path
        
        # 클린업
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """임시 출력 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        yield temp_path
        
        # 클린업
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_coco_data(self, temp_data_dir):
        """샘플 COCO 데이터 생성"""
        # 테스트용 이미지 생성
        images_dir = temp_data_dir / "images"
        images_dir.mkdir()
        
        image1_path = images_dir / "pill1.jpg"
        image2_path = images_dir / "pill2.jpg"
        
        test_image1 = Image.new('RGB', (640, 480), color='red')
        test_image1.save(image1_path)
        
        test_image2 = Image.new('RGB', (800, 600), color='green')
        test_image2.save(image2_path)
        
        # COCO 어노테이션 데이터
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "pill1.jpg",
                    "width": 640,
                    "height": 480
                },
                {
                    "id": 2,
                    "file_name": "pill2.jpg",
                    "width": 800,
                    "height": 600
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 50, 200, 150],  # [x, y, width, height]
                    "area": 30000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 150, 100],
                    "area": 15000,
                    "iscrowd": 0
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [200, 100, 300, 200],
                    "area": 60000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "pill_K-029066",
                    "supercategory": "pill"
                },
                {
                    "id": 2,
                    "name": "drug_K-016551",
                    "supercategory": "pill"
                }
            ]
        }
        
        # COCO 어노테이션 파일 저장
        annotations_file = temp_data_dir / "annotations.json"
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)
        
        return coco_data, annotations_file
    
    def test_converter_initialization(self, temp_data_dir, temp_output_dir):
        """변환기 초기화 테스트"""
        k_code_mapping = {"K-029066": 0, "K-016551": 1}
        
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            k_code_to_class_mapping=k_code_mapping
        )
        
        assert converter.data_root == temp_data_dir
        assert converter.output_dir == temp_output_dir
        assert converter.k_code_to_class_mapping == k_code_mapping
        assert converter.default_class_id == 0
    
    def test_k_code_extraction(self, temp_data_dir, temp_output_dir):
        """K-code 추출 테스트"""
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir)
        )
        
        # 다양한 카테고리명에서 K-code 추출 테스트
        test_cases = [
            ("pill_K-029066", "K-029066"),
            ("drug_K-016551", "K-016551"),
            ("medication_K123456", "K-123456"),
            ("K-045678_tablet", "K-045678"),
            ("capsule_no_code", None),
            ("invalid_format", None),
        ]
        
        for category_name, expected_k_code in test_cases:
            extracted_k_code = converter.extract_k_code_from_category(category_name)
            assert extracted_k_code == expected_k_code
    
    def test_category_mapping_building(self, temp_data_dir, temp_output_dir, sample_coco_data):
        """카테고리 매핑 구축 테스트"""
        coco_data, _ = sample_coco_data
        
        k_code_mapping = {"K-029066": 0, "K-016551": 1}
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            k_code_to_class_mapping=k_code_mapping
        )
        
        category_mapping = converter.build_category_mapping(coco_data)
        
        # 예상 매핑: COCO category_id 1 → K-029066 → YOLO class 0
        #           COCO category_id 2 → K-016551 → YOLO class 1
        assert category_mapping[1] == 0
        assert category_mapping[2] == 1
    
    def test_output_directory_setup(self, temp_data_dir, temp_output_dir):
        """출력 디렉토리 구조 생성 테스트"""
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir)
        )
        
        converter.setup_output_directory()
        
        # YOLO 표준 디렉토리 구조 확인
        expected_dirs = [
            temp_output_dir / 'images' / 'train',
            temp_output_dir / 'images' / 'val',
            temp_output_dir / 'labels' / 'train',
            temp_output_dir / 'labels' / 'val'
        ]
        
        for dir_path in expected_dirs:
            assert dir_path.exists()
            assert dir_path.is_dir()
    
    def test_coco_annotation_loading(self, temp_data_dir, temp_output_dir, sample_coco_data):
        """COCO 어노테이션 로딩 테스트"""
        coco_data, annotations_file = sample_coco_data
        
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir)
        )
        
        loaded_data = converter.load_coco_annotation(annotations_file)
        
        assert loaded_data == coco_data
        assert len(loaded_data['images']) == 2
        assert len(loaded_data['annotations']) == 3
        assert len(loaded_data['categories']) == 2
    
    def test_single_annotation_conversion(self, temp_data_dir, temp_output_dir, sample_coco_data):
        """단일 어노테이션 변환 테스트"""
        coco_data, _ = sample_coco_data
        
        k_code_mapping = {"K-029066": 0, "K-016551": 1}
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir),
            k_code_to_class_mapping=k_code_mapping
        )
        
        category_mapping = converter.build_category_mapping(coco_data)
        yolo_annotations = converter.convert_single_annotation(coco_data, category_mapping)
        
        # 결과 검증
        assert len(yolo_annotations) == 2  # 2개 이미지
        
        # 첫 번째 이미지 (pill1.jpg)
        anno1 = yolo_annotations[0]
        assert anno1.image_path.name == "pill1.jpg"
        assert anno1.image_width == 640
        assert anno1.image_height == 480
        assert len(anno1.bboxes) == 2  # 2개 바운딩 박스
        assert len(anno1.class_ids) == 2
        
        # 두 번째 이미지 (pill2.jpg)
        anno2 = yolo_annotations[1]
        assert anno2.image_path.name == "pill2.jpg"
        assert anno2.image_width == 800
        assert anno2.image_height == 600
        assert len(anno2.bboxes) == 1  # 1개 바운딩 박스
        assert len(anno2.class_ids) == 1
    
    def test_dataset_yaml_creation(self, temp_data_dir, temp_output_dir):
        """데이터셋 YAML 파일 생성 테스트"""
        converter = COCOToYOLOConverter(
            data_root=str(temp_data_dir),
            output_dir=str(temp_output_dir)
        )
        
        converter.setup_output_directory()
        
        class_names = ["pill", "tablet", "capsule"]
        yaml_path = converter.create_dataset_yaml(class_names)
        
        assert yaml_path.exists()
        assert yaml_path.name == "dataset.yaml"
        
        # YAML 내용 확인
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        assert f"nc: {len(class_names)}" in yaml_content
        assert str(class_names) in yaml_content
        assert "train: images/train" in yaml_content
        assert "val: images/val" in yaml_content


class TestCOCOToYOLOConverterIntegration:
    """COCO → YOLO 변환기 통합 테스트"""
    
    @pytest.fixture
    def complete_test_environment(self):
        """완전한 테스트 환경 설정"""
        # 임시 디렉토리들
        data_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        # 이미지 디렉토리 생성
        images_dir = data_path / "combination"
        images_dir.mkdir()
        
        # 테스트 이미지들 생성
        for i, (name, size, color) in enumerate([
            ("combo1.jpg", (640, 480), "red"),
            ("combo2.jpg", (800, 600), "green"),
            ("combo3.jpg", (1024, 768), "blue")
        ]):
            image_path = images_dir / name
            test_image = Image.new('RGB', size, color=color)
            test_image.save(image_path)
        
        # COCO 어노테이션 생성
        coco_data = {
            "images": [
                {"id": 1, "file_name": "combo1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "combo2.jpg", "width": 800, "height": 600},
                {"id": 3, "file_name": "combo3.jpg", "width": 1024, "height": 768}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 100, 100], "area": 10000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 150, 120, 80], "area": 9600, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0},
                {"id": 4, "image_id": 3, "category_id": 3, "bbox": [300, 200, 150, 200], "area": 30000, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "pill_K-029066", "supercategory": "pill"},
                {"id": 2, "name": "tablet_K-016551", "supercategory": "pill"},
                {"id": 3, "name": "capsule_K-010913", "supercategory": "pill"}
            ]
        }
        
        annotations_file = data_path / "combo_annotations.json"
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)
        
        yield data_path, output_path, coco_data
        
        # 클린업
        shutil.rmtree(data_dir)
        shutil.rmtree(output_dir)
    
    def test_end_to_end_conversion(self, complete_test_environment):
        """End-to-End 변환 테스트"""
        data_path, output_path, coco_data = complete_test_environment
        
        # K-code 매핑
        k_code_mapping = {
            "K-029066": 0,
            "K-016551": 1,
            "K-010913": 2
        }
        
        converter = COCOToYOLOConverter(
            data_root=str(data_path),
            output_dir=str(output_path),
            k_code_to_class_mapping=k_code_mapping
        )
        
        # 전체 변환 실행
        results = converter.convert_all_annotations(
            train_val_split=0.8,
            class_names=["pill_class_0", "pill_class_1", "pill_class_2"]
        )
        
        # 결과 검증
        assert results['conversion_stats']['processed_images'] == 3
        assert results['conversion_stats']['successful_conversions'] == 3
        assert results['conversion_stats']['failed_conversions'] == 0
        assert results['conversion_stats']['total_annotations'] == 4
        assert results['conversion_stats']['valid_bboxes'] == 4
        
        # 출력 디렉토리 구조 확인
        assert (output_path / 'images' / 'train').exists()
        assert (output_path / 'images' / 'val').exists()
        assert (output_path / 'labels' / 'train').exists()
        assert (output_path / 'labels' / 'val').exists()
        assert (output_path / 'dataset.yaml').exists()
        
        # 이미지와 라벨 파일 개수 확인
        train_images = list((output_path / 'images' / 'train').glob('*.jpg'))
        val_images = list((output_path / 'images' / 'val').glob('*.jpg'))
        train_labels = list((output_path / 'labels' / 'train').glob('*.txt'))
        val_labels = list((output_path / 'labels' / 'val').glob('*.txt'))
        
        total_images = len(train_images) + len(val_images)
        total_labels = len(train_labels) + len(val_labels)
        
        assert total_images == 3  # 총 3개 이미지
        assert total_labels == 3  # 총 3개 라벨 파일
        assert len(train_images) == len(train_labels)  # 훈련 이미지와 라벨 수 일치
        assert len(val_images) == len(val_labels)      # 검증 이미지와 라벨 수 일치
    
    def test_yolo_label_content_accuracy(self, complete_test_environment):
        """YOLO 라벨 내용 정확성 테스트"""
        data_path, output_path, coco_data = complete_test_environment
        
        k_code_mapping = {"K-029066": 0, "K-016551": 1, "K-010913": 2}
        converter = COCOToYOLOConverter(
            data_root=str(data_path),
            output_dir=str(output_path),
            k_code_to_class_mapping=k_code_mapping
        )
        
        # 변환 실행
        converter.convert_all_annotations(train_val_split=1.0)  # 모두 train으로
        
        # 생성된 라벨 파일 확인
        label_files = list((output_path / 'labels' / 'train').glob('*.txt'))
        assert len(label_files) == 3
        
        # 첫 번째 라벨 파일 내용 확인 (combo1.jpg)
        combo1_label = output_path / 'labels' / 'train' / 'combo1.txt'
        if combo1_label.exists():
            with open(combo1_label, 'r') as f:
                lines = f.read().strip().split('\n')
            
            # combo1.jpg는 2개의 바운딩 박스를 가져야 함
            assert len(lines) == 2
            
            # 각 라인이 올바른 YOLO 포맷인지 확인
            for line in lines:
                parts = line.split()
                assert len(parts) == 5  # class_id x_center y_center width height
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # 클래스 ID가 유효한 범위에 있는지 확인
                assert class_id in [0, 1, 2]
                
                # 좌표가 유효한 범위에 있는지 확인
                assert 0 <= x_center <= 1
                assert 0 <= y_center <= 1
                assert 0 < width <= 1
                assert 0 < height <= 1
    
    def test_error_handling_robustness(self):
        """에러 처리 견고성 테스트"""
        # 존재하지 않는 데이터 디렉토리
        with tempfile.TemporaryDirectory() as temp_output:
            converter = COCOToYOLOConverter(
                data_root="/nonexistent/path",
                output_dir=temp_output
            )
            
            # 어노테이션 파일을 찾을 수 없어야 함
            annotation_files = converter.find_coco_annotations()
            assert len(annotation_files) == 0
    
    def test_factory_function(self):
        """팩토리 함수 테스트"""
        with tempfile.TemporaryDirectory() as temp_data, tempfile.TemporaryDirectory() as temp_output:
            k_code_mapping = {"K-029066": 0}
            
            converter = create_coco_to_yolo_converter_for_stage1(
                data_root=temp_data,
                output_dir=temp_output,
                k_code_to_class_mapping=k_code_mapping
            )
            
            assert isinstance(converter, COCOToYOLOConverter)
            assert str(converter.data_root) == temp_data
            assert str(converter.output_dir) == temp_output
            assert converter.k_code_to_class_mapping == k_code_mapping


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])