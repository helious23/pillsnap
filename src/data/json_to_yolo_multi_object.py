"""
Multi-object JSON to YOLO format converter for PillSnap Detection
Combination 이미지의 4개 K-code별 annotation을 하나의 YOLO 파일로 통합
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiObjectJSONToYOLOConverter:
    """Multi-object JSON (COCO format) to YOLO format converter"""
    
    def __init__(self, json_root: str, output_root: str):
        """
        Args:
            json_root: JSON annotation 파일들이 있는 루트 디렉토리
            output_root: YOLO 형식 .txt 파일을 저장할 루트 디렉토리
        """
        self.json_root = Path(json_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        COCO bbox [x, y, width, height] -> YOLO [x_center, y_center, width, height] (normalized)
        """
        x, y, w, h = bbox
        
        # Center coordinates
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        
        # Normalized width and height
        norm_width = w / img_width
        norm_height = h / img_height
        
        return x_center, y_center, norm_width, norm_height
    
    def group_json_files_by_image(self) -> Dict[str, List[Path]]:
        """
        이미지별로 JSON 파일들을 그룹화
        예: K-006835-026993-038972-043233_0_2_0_2_75_000_200.json -> 4개 K-code별 JSON 파일들
        
        Returns:
            Dict[image_name, List[json_paths]]
        """
        json_files = list(self.json_root.rglob("*.json"))
        grouped = defaultdict(list)
        
        for json_file in json_files:
            # 파일명에서 이미지 이름 추출
            image_name = json_file.stem  # .json 확장자 제거
            grouped[image_name].append(json_file)
            
        return grouped
    
    def convert_multi_object_json(self, image_name: str, json_paths: List[Path]) -> bool:
        """
        하나의 이미지에 대한 여러 JSON 파일을 YOLO 형식으로 변환
        
        Args:
            image_name: 이미지 파일명 (확장자 없음)
            json_paths: 해당 이미지의 모든 K-code별 JSON 파일 경로들
            
        Returns:
            bool: 변환 성공 여부
        """
        try:
            all_annotations = []
            img_width = None
            img_height = None
            
            # 모든 JSON 파일에서 annotation 수집
            for json_path in json_paths:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 첫 번째 파일에서 이미지 크기 정보 가져오기
                if img_width is None and 'images' in data and len(data['images']) > 0:
                    image_info = data['images'][0]
                    img_width = image_info['width']
                    img_height = image_info['height']
                
                # Annotations 수집
                if 'annotations' in data:
                    for ann in data['annotations']:
                        bbox = ann['bbox']
                        # YOLO 형식으로 변환
                        x_center, y_center, width, height = self.convert_bbox_to_yolo(
                            bbox, img_width, img_height
                        )
                        
                        # 모든 pill을 class 0으로 통일
                        class_id = 0
                        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        all_annotations.append(yolo_line)
            
            # 출력 파일 경로 생성
            output_path = self.output_root / f"{image_name}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # YOLO 파일 저장
            with open(output_path, 'w') as f:
                f.write('\n'.join(all_annotations))
                
            logger.info(f"Converted: {image_name} -> {len(all_annotations)} objects from {len(json_paths)} JSON files")
            return True
            
        except Exception as e:
            logger.error(f"Error converting {image_name}: {e}")
            return False
    
    def convert_all(self) -> Dict[str, int]:
        """
        모든 JSON 파일을 이미지별로 그룹화하고 YOLO 형식으로 변환
        
        Returns:
            변환 통계 {"success": 성공수, "failed": 실패수}
        """
        # 이미지별로 JSON 파일 그룹화
        grouped_files = self.group_json_files_by_image()
        logger.info(f"Found {len(grouped_files)} unique images with multiple JSON annotations")
        
        stats = {"success": 0, "failed": 0, "total": len(grouped_files)}
        
        for image_name, json_paths in grouped_files.items():
            if self.convert_multi_object_json(image_name, json_paths):
                stats["success"] += 1
            else:
                stats["failed"] += 1
                
        logger.info(f"Multi-object conversion completed: {stats['success']}/{stats['total']} successful")
        return stats


def main():
    """메인 실행 함수"""
    # 경로 설정
    json_root = "/home/max16/pillsnap_data/train/labels/combination"
    output_root = "/home/max16/pillsnap_data/train/labels/combination_yolo_multi"
    
    logger.info("Starting Multi-object JSON to YOLO conversion...")
    logger.info(f"Input:  {json_root}")
    logger.info(f"Output: {output_root}")
    
    # 변환기 초기화 및 실행
    converter = MultiObjectJSONToYOLOConverter(json_root, output_root)
    stats = converter.convert_all()
    
    logger.info("="*50)
    logger.info(f"MULTI-OBJECT CONVERSION SUMMARY:")
    logger.info(f"Total images: {stats['total']}")
    logger.info(f"Successful: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%")


if __name__ == "__main__":
    main()