#!/usr/bin/env python3
"""
COCO to YOLO Converter 테스트

COCO JSON 형식을 YOLO TXT 형식으로 변환하는 기능을 검증:
- bbox 좌표 변환 정확성
- 배치 처리 성능
- 정규화된 좌표 범위 검증
- 멀티프로세싱 안정성
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.converters.coco_to_yolo import COCOToYOLOConverter


class TestCOCOToYOLOConverter:
    """COCO→YOLO 변환기 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_coco_data(self):
        """샘플 COCO JSON 데이터"""
        return {
            "images": [
                {
                    "file_name": "test_image.png",
                    "width": 640,
                    "height": 480,
                    "id": 1
                }
            ],
            "type": "instances",
            "annotations": [
                {
                    "area": 10000,
                    "iscrowd": 0,
                    "bbox": [100, 100, 200, 150],  # [x, y, width, height]
                    "category_id": 1,
                    "ignore": 0,
                    "segmentation": [],
                    "id": 1,
                    "image_id": 1
                }
            ],
            "categories": [
                {
                    "supercategory": "pill",
                    "id": 1,
                    "name": "Drug"
                }
            ]
        }
    
    @pytest.fixture
    def converter(self, temp_dir):
        """COCO→YOLO 변환기 인스턴스"""
        coco_dir = temp_dir / "coco"
        yolo_dir = temp_dir / "yolo"
        
        coco_dir.mkdir(parents=True, exist_ok=True)
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        return COCOToYOLOConverter(str(coco_dir), str(yolo_dir))
    
    def test_bbox_conversion_accuracy(self, converter):
        """bbox 좌표 변환 정확성 테스트"""
        # 테스트 케이스: [x, y, width, height] → [center_x, center_y, width, height]
        test_cases = [
            # (coco_bbox, img_size, expected_yolo_bbox)
            ([100, 100, 200, 150], (640, 480), [0.3125, 0.3646, 0.3125, 0.3125]),  # 중앙 박스
            ([0, 0, 100, 100], (200, 200), [0.25, 0.25, 0.5, 0.5]),  # 왼쪽 위 박스
            ([150, 150, 50, 50], (200, 200), [0.875, 0.875, 0.25, 0.25]),  # 오른쪽 아래 박스
        ]
        
        for coco_bbox, (img_w, img_h), expected_yolo in test_cases:
            result = converter.convert_bbox_coco_to_yolo(coco_bbox, img_w, img_h)
            
            # 정확도 검증 (소수점 4자리까지)
            for i, (actual, expected) in enumerate(zip(result, expected_yolo)):
                assert abs(actual - expected) < 0.0001, \
                    f"bbox 변환 오류 - 인덱스 {i}: {actual} != {expected}"
    
    def test_normalized_coordinates(self, converter):
        """정규화된 좌표 범위 검증 (0~1)"""
        # 극단적인 케이스들
        extreme_cases = [
            ([0, 0, 1000, 1000], (100, 100)),    # 이미지보다 큰 박스
            ([-50, -50, 200, 200], (640, 480)),  # 음수 좌표
            ([600, 400, 100, 100], (640, 480)),  # 경계 넘는 박스
        ]
        
        for coco_bbox, (img_w, img_h) in extreme_cases:
            result = converter.convert_bbox_coco_to_yolo(coco_bbox, img_w, img_h)
            
            # 모든 좌표가 0~1 범위 내에 있는지 검증
            for i, coord in enumerate(result):
                assert 0.0 <= coord <= 1.0, \
                    f"좌표 범위 초과 - 인덱스 {i}: {coord} (범위: 0~1)"
    
    def test_convert_annotation(self, converter, sample_coco_data, temp_dir):
        """어노테이션 변환 기능 테스트"""
        # 샘플 COCO JSON 파일 생성
        coco_file = temp_dir / "coco" / "test.json"
        with open(coco_file, 'w') as f:
            json.dump(sample_coco_data, f)
        
        # 어노테이션 변환
        result = converter.convert_annotation(coco_file)
        
        # 결과 검증
        assert result is not None, "변환 결과가 None"
        
        img_filename, yolo_lines = result
        assert img_filename == "test_image.png", f"이미지 파일명 오류: {img_filename}"
        assert len(yolo_lines) == 1, f"YOLO 라인 수 오류: {len(yolo_lines)}"
        
        # YOLO 라인 형식 검증
        yolo_line = yolo_lines[0]
        parts = yolo_line.split()
        assert len(parts) == 5, f"YOLO 라인 형식 오류: {parts}"
        
        # 클래스 ID 검증
        class_id = int(parts[0])
        assert class_id == 0, f"클래스 ID 오류: {class_id}"
        
        # 좌표 범위 검증
        coords = [float(x) for x in parts[1:5]]
        for coord in coords:
            assert 0.0 <= coord <= 1.0, f"좌표 범위 초과: {coord}"
    
    def test_multiple_annotations(self, converter, temp_dir):
        """다중 어노테이션 처리 테스트"""
        # 다중 어노테이션 COCO 데이터
        multi_coco_data = {
            "images": [
                {
                    "file_name": "multi_pills.png",
                    "width": 800,
                    "height": 600,
                    "id": 1
                }
            ],
            "type": "instances",
            "annotations": [
                {
                    "area": 5000,
                    "iscrowd": 0,
                    "bbox": [100, 100, 100, 100],  # 첫 번째 약품
                    "category_id": 1,
                    "ignore": 0,
                    "segmentation": [],
                    "id": 1,
                    "image_id": 1
                },
                {
                    "area": 6000,
                    "iscrowd": 0,
                    "bbox": [300, 200, 120, 80],   # 두 번째 약품
                    "category_id": 1,
                    "ignore": 0,
                    "segmentation": [],
                    "id": 2,
                    "image_id": 1
                }
            ],
            "categories": [
                {
                    "supercategory": "pill",
                    "id": 1,
                    "name": "Drug"
                }
            ]
        }
        
        # COCO 파일 생성
        coco_file = temp_dir / "coco" / "multi.json"
        with open(coco_file, 'w') as f:
            json.dump(multi_coco_data, f)
        
        # 변환 실행
        result = converter.convert_annotation(coco_file)
        assert result is not None, "다중 어노테이션 변환 실패"
        
        img_filename, yolo_lines = result
        assert len(yolo_lines) == 2, f"다중 어노테이션 수 오류: {len(yolo_lines)}"
        
        # 각 라인 검증
        for i, line in enumerate(yolo_lines):
            parts = line.split()
            assert len(parts) == 5, f"라인 {i} 형식 오류: {parts}"
            
            # 모든 좌표가 정규화 범위 내
            coords = [float(x) for x in parts[1:5]]
            for coord in coords:
                assert 0.0 <= coord <= 1.0, f"라인 {i} 좌표 범위 초과: {coord}"
    
    def test_batch_conversion_small(self, converter, temp_dir):
        """소규모 배치 변환 테스트"""
        # 여러 COCO 파일 생성
        test_files = []
        for i in range(5):
            coco_data = {
                "images": [
                    {
                        "file_name": f"image_{i:03d}.png",
                        "width": 640,
                        "height": 480,
                        "id": 1
                    }
                ],
                "type": "instances",
                "annotations": [
                    {
                        "area": 1000 + i * 100,
                        "iscrowd": 0,
                        "bbox": [50 + i * 10, 50 + i * 10, 80, 60],
                        "category_id": 1,
                        "ignore": 0,
                        "segmentation": [],
                        "id": 1,
                        "image_id": 1
                    }
                ],
                "categories": [
                    {
                        "supercategory": "pill",
                        "id": 1,
                        "name": "Drug"
                    }
                ]
            }
            
            coco_file = temp_dir / "coco" / f"test_{i:03d}.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f)
            test_files.append(coco_file)
        
        # 배치 변환 실행
        stats = converter.convert_batch(max_files=5, num_workers=2)
        
        # 통계 검증
        assert stats['total_files'] == 5, f"총 파일 수 오류: {stats['total_files']}"
        assert stats['converted_files'] == 5, f"변환 성공 수 오류: {stats['converted_files']}"
        assert stats['failed_files'] == 0, f"변환 실패 수 오류: {stats['failed_files']}"
        assert stats['total_annotations'] == 5, f"총 어노테이션 수 오류: {stats['total_annotations']}"
        
        # 출력 파일 확인
        yolo_files = list(converter.yolo_dir.glob('*.txt'))
        assert len(yolo_files) == 5, f"YOLO 파일 수 오류: {len(yolo_files)}"
    
    def test_conversion_verification(self, converter, sample_coco_data, temp_dir):
        """변환 결과 검증 기능 테스트"""
        # COCO 파일 생성
        coco_file = temp_dir / "coco" / "verify_test.json"
        with open(coco_file, 'w') as f:
            json.dump(sample_coco_data, f)
        
        # 변환 실행
        stats = converter.convert_batch(num_workers=1)
        assert stats['converted_files'] > 0, "변환된 파일이 없음"
        
        # 검증 실행
        verification_result = converter.verify_conversion(num_samples=1)
        assert verification_result is True, "변환 결과 검증 실패"
    
    def test_edge_case_empty_annotations(self, converter, temp_dir):
        """어노테이션 없는 경우 처리 테스트"""
        # 어노테이션 없는 COCO 데이터
        empty_coco_data = {
            "images": [
                {
                    "file_name": "empty_image.png",
                    "width": 640,
                    "height": 480,
                    "id": 1
                }
            ],
            "type": "instances",
            "annotations": [],  # 비어있음
            "categories": [
                {
                    "supercategory": "pill",
                    "id": 1,
                    "name": "Drug"
                }
            ]
        }
        
        # COCO 파일 생성
        coco_file = temp_dir / "coco" / "empty.json"
        with open(coco_file, 'w') as f:
            json.dump(empty_coco_data, f)
        
        # 변환 시도
        result = converter.convert_annotation(coco_file)
        assert result is not None, "빈 어노테이션 처리 실패"
        
        img_filename, yolo_lines = result
        assert img_filename == "empty_image.png", "이미지 파일명 처리 오류"
        assert len(yolo_lines) == 0, "빈 어노테이션이 YOLO 라인 생성함"
    
    def test_invalid_coco_format(self, converter, temp_dir):
        """잘못된 COCO 형식 처리 테스트"""
        # 잘못된 형식들
        invalid_formats = [
            {},  # 빈 JSON
            {"images": []},  # 이미지 정보 없음
            {"images": [{"width": 640}], "annotations": []},  # 불완전한 이미지 정보
        ]
        
        for i, invalid_data in enumerate(invalid_formats):
            # 잘못된 COCO 파일 생성
            coco_file = temp_dir / "coco" / f"invalid_{i}.json"
            with open(coco_file, 'w') as f:
                json.dump(invalid_data, f)
            
            # 변환 시도 (실패 예상)
            result = converter.convert_annotation(coco_file)
            
            # None 반환 예상 (오류 처리)
            if result is None:
                continue  # 정상적인 오류 처리
            else:
                # 일부는 성공할 수도 있음 (빈 어노테이션)
                img_filename, yolo_lines = result
                assert isinstance(yolo_lines, list), "YOLO 라인이 리스트가 아님"
    
    def test_class_mapping_custom(self, temp_dir):
        """커스텀 클래스 매핑 테스트"""
        # 디렉토리 생성
        coco_dir = temp_dir / "coco"
        yolo_dir = temp_dir / "yolo"
        coco_dir.mkdir(parents=True, exist_ok=True)
        yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # 커스텀 클래스 매핑
        custom_mapping = {"Tablet": 0, "Capsule": 1, "Drug": 2}
        
        converter = COCOToYOLOConverter(
            coco_dir=str(coco_dir),
            yolo_dir=str(yolo_dir),
            class_mapping=custom_mapping
        )
        
        # 다중 클래스 COCO 데이터
        multi_class_data = {
            "images": [
                {
                    "file_name": "multi_class.png",
                    "width": 640,
                    "height": 480,
                    "id": 1
                }
            ],
            "type": "instances",
            "annotations": [
                {
                    "area": 1000,
                    "iscrowd": 0,
                    "bbox": [100, 100, 50, 50],
                    "category_id": 1,
                    "ignore": 0,
                    "segmentation": [],
                    "id": 1,
                    "image_id": 1
                },
                {
                    "area": 1200,
                    "iscrowd": 0,
                    "bbox": [200, 200, 60, 40],
                    "category_id": 2,
                    "ignore": 0,
                    "segmentation": [],
                    "id": 2,
                    "image_id": 1
                }
            ],
            "categories": [
                {
                    "supercategory": "pill",
                    "id": 1,
                    "name": "Tablet"
                },
                {
                    "supercategory": "pill",
                    "id": 2,
                    "name": "Capsule"
                }
            ]
        }
        
        # COCO 파일 생성
        coco_file = coco_dir / "multi_class.json"
        with open(coco_file, 'w') as f:
            json.dump(multi_class_data, f)
        
        # 변환 실행
        result = converter.convert_annotation(coco_file)
        assert result is not None, "다중 클래스 변환 실패"
        
        img_filename, yolo_lines = result
        assert len(yolo_lines) == 2, "다중 클래스 어노테이션 수 오류"
        
        # 클래스 ID 검증
        class_ids = [int(line.split()[0]) for line in yolo_lines]
        assert 0 in class_ids, "Tablet 클래스 ID 누락"
        assert 1 in class_ids, "Capsule 클래스 ID 누락"
    
    def test_performance_metrics(self, converter, temp_dir):
        """성능 메트릭 테스트"""
        import time
        
        # 성능 테스트용 파일들 생성 (100개)
        start_time = time.time()
        
        for i in range(100):
            coco_data = {
                "images": [{"file_name": f"perf_{i:03d}.png", "width": 640, "height": 480, "id": 1}],
                "type": "instances",
                "annotations": [
                    {
                        "area": 1000,
                        "iscrowd": 0,
                        "bbox": [100, 100, 50, 50],
                        "category_id": 1,
                        "ignore": 0,
                        "segmentation": [],
                        "id": 1,
                        "image_id": 1
                    }
                ],
                "categories": [{"supercategory": "pill", "id": 1, "name": "Drug"}]
            }
            
            coco_file = temp_dir / "coco" / f"perf_{i:03d}.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f)
        
        creation_time = time.time() - start_time
        
        # 배치 변환 성능 측정
        start_time = time.time()
        stats = converter.convert_batch(num_workers=4)
        conversion_time = time.time() - start_time
        
        # 성능 검증
        assert stats['converted_files'] == 100, "성능 테스트 변환 실패"
        assert conversion_time < 10.0, f"변환 시간 과다: {conversion_time:.2f}초"
        
        # 처리량 계산
        throughput = stats['converted_files'] / conversion_time
        assert throughput > 10, f"처리량 부족: {throughput:.1f} files/sec"
        
        print(f"성능 테스트 결과:")
        print(f"  파일 생성: {creation_time:.2f}초")
        print(f"  변환 시간: {conversion_time:.2f}초") 
        print(f"  처리량: {throughput:.1f} files/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])