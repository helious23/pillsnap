"""
JSON to YOLO format converter for PillSnap Detection
Combination 이미지의 multi-object detection을 위한 annotation 변환 도구
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONToYOLOConverter:
    """JSON (COCO format) to YOLO format converter"""
    
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
        
        Args:
            bbox: [x, y, width, height] in pixels
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            (x_center, y_center, width, height) normalized to [0, 1]
        """
        x, y, w, h = bbox
        
        # Center coordinates
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        
        # Normalized width and height
        norm_width = w / img_width
        norm_height = h / img_height
        
        return x_center, y_center, norm_width, norm_height
    
    def convert_single_json(self, json_path: Path) -> bool:
        """
        단일 JSON 파일을 YOLO 형식으로 변환
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            bool: 변환 성공 여부
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이미지 정보 추출
            if 'images' not in data or len(data['images']) == 0:
                logger.warning(f"No images found in {json_path}")
                return False
                
            image_info = data['images'][0]  # 첫 번째 (유일한) 이미지
            img_width = image_info['width']
            img_height = image_info['height']
            
            # 출력 파일 경로 생성
            relative_path = json_path.relative_to(self.json_root)
            output_path = self.output_root / relative_path.with_suffix('.txt')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Annotations 변환
            yolo_lines = []
            if 'annotations' in data:
                for ann in data['annotations']:
                    bbox = ann['bbox']
                    category_id = ann.get('category_id', 1)
                    
                    # YOLO는 0-based class ID 사용
                    class_id = 0  # 모든 pill을 class 0으로 통일
                    
                    # YOLO 형식으로 변환
                    x_center, y_center, width, height = self.convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
            
            # YOLO 파일 저장
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
            logger.info(f"Converted: {json_path.name} -> {output_path.name} ({len(yolo_lines)} objects)")
            return True
            
        except Exception as e:
            logger.error(f"Error converting {json_path}: {e}")
            return False
    
    def convert_all(self) -> Dict[str, int]:
        """
        모든 JSON 파일을 YOLO 형식으로 변환
        
        Returns:
            변환 통계 {"success": 성공수, "failed": 실패수}
        """
        json_files = list(self.json_root.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to convert")
        
        stats = {"success": 0, "failed": 0, "total": len(json_files)}
        
        for json_file in json_files:
            if self.convert_single_json(json_file):
                stats["success"] += 1
            else:
                stats["failed"] += 1
                
        logger.info(f"Conversion completed: {stats['success']}/{stats['total']} successful")
        return stats


def main():
    """메인 실행 함수"""
    # 경로 설정
    json_root = "/home/max16/pillsnap_data/train/labels/combination"
    output_root = "/home/max16/pillsnap_data/train/labels/combination_yolo"
    
    logger.info("Starting JSON to YOLO conversion...")
    logger.info(f"Input:  {json_root}")
    logger.info(f"Output: {output_root}")
    
    # 변환기 초기화 및 실행
    converter = JSONToYOLOConverter(json_root, output_root)
    stats = converter.convert_all()
    
    logger.info("="*50)
    logger.info(f"CONVERSION SUMMARY:")
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Successful: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%")


if __name__ == "__main__":
    main()