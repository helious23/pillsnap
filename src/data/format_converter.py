"""
COCO → YOLO 포맷 변환기

PillSnap ML의 Two-Stage Conditional Pipeline용:
- Combination pills의 COCO 어노테이션을 YOLO 포맷으로 변환
- Bounding box 좌표계 변환 (절대좌표 → 상대좌표)
- YOLOv11x 학습용 데이터셋 생성
- 다중 약품 검출 지원
"""

import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger, load_config


@dataclass
class BoundingBox:
    """바운딩 박스 데이터 클래스"""
    x_center: float  # 중심점 x (상대좌표 0-1)
    y_center: float  # 중심점 y (상대좌표 0-1)
    width: float     # 너비 (상대좌표 0-1)
    height: float    # 높이 (상대좌표 0-1)
    confidence: float = 1.0  # 신뢰도 (어노테이션은 기본 1.0)
    
    def to_yolo_string(self, class_id: int) -> str:
        """YOLO 포맷 문자열로 변환"""
        return f"{class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_absolute_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """절대좌표로 변환 (x1, y1, x2, y2)"""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return x1, y1, x2, y2
    
    @classmethod
    def from_coco_bbox(cls, coco_bbox: List[float], img_width: int, img_height: int) -> 'BoundingBox':
        """COCO 바운딩 박스에서 생성 [x, y, width, height] (절대좌표)"""
        x, y, w, h = coco_bbox
        
        # COCO 좌표 → YOLO 좌표 변환
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return cls(x_center, y_center, width, height)
    
    def is_valid(self) -> bool:
        """바운딩 박스 유효성 검증"""
        return (0 <= self.x_center <= 1 and 
                0 <= self.y_center <= 1 and 
                0 < self.width <= 1 and 
                0 < self.height <= 1)


@dataclass
class YOLOAnnotation:
    """YOLO 어노테이션 데이터"""
    image_path: Path
    image_width: int
    image_height: int
    bboxes: List[BoundingBox]
    class_ids: List[int]
    
    def to_yolo_format(self) -> str:
        """YOLO 포맷 텍스트로 변환"""
        lines = []
        for bbox, class_id in zip(self.bboxes, self.class_ids):
            if bbox.is_valid():
                lines.append(bbox.to_yolo_string(class_id))
        return '\n'.join(lines)
    
    def get_annotation_filename(self) -> str:
        """어노테이션 파일명 생성"""
        return self.image_path.stem + '.txt'
    
    def validate(self) -> Dict[str, Any]:
        """어노테이션 유효성 검증"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_boxes': len(self.bboxes),
                'valid_boxes': 0,
                'invalid_boxes': 0,
                'unique_classes': len(set(self.class_ids))
            }
        }
        
        # 1. 이미지 파일 존재 확인
        if not self.image_path.exists():
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"이미지 파일 없음: {self.image_path}")
            return validation_result
        
        # 2. 바운딩 박스 개수와 클래스 ID 개수 일치 확인
        if len(self.bboxes) != len(self.class_ids):
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"바운딩 박스와 클래스 ID 개수 불일치: {len(self.bboxes)} vs {len(self.class_ids)}")
        
        # 3. 각 바운딩 박스 유효성 확인
        for i, bbox in enumerate(self.bboxes):
            if bbox.is_valid():
                validation_result['stats']['valid_boxes'] += 1
            else:
                validation_result['stats']['invalid_boxes'] += 1
                validation_result['warnings'].append(f"바운딩 박스 {i} 유효하지 않음: {asdict(bbox)}")
        
        # 4. 클래스 ID 유효성 확인
        for i, class_id in enumerate(self.class_ids):
            if not isinstance(class_id, int) or class_id < 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"클래스 ID {i} 유효하지 않음: {class_id}")
        
        # 5. 경고 조건 확인
        if validation_result['stats']['invalid_boxes'] > 0:
            validation_result['warnings'].append(f"{validation_result['stats']['invalid_boxes']}개 유효하지 않은 바운딩 박스")
        
        if validation_result['stats']['total_boxes'] == 0:
            validation_result['warnings'].append("바운딩 박스가 없음")
        
        return validation_result


class COCOToYOLOConverter:
    """COCO → YOLO 포맷 변환기"""
    
    def __init__(self, data_root: str, output_dir: str, k_code_to_class_mapping: Optional[Dict[str, int]] = None):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.logger = PillSnapLogger(__name__)
        
        # K-code → 클래스 ID 매핑 (기본값: 모든 클래스를 0으로)
        self.k_code_to_class_mapping = k_code_to_class_mapping or {}
        self.default_class_id = 0  # 매핑되지 않은 K-code는 0번 클래스
        
        # 변환 통계
        self.conversion_stats = {
            'processed_images': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_annotations': 0,
            'valid_bboxes': 0,
            'invalid_bboxes': 0,
            'unique_k_codes': set(),
            'class_distribution': {}
        }
        
        self.logger.info(f"COCO → YOLO 변환기 초기화")
        self.logger.info(f"  데이터 루트: {self.data_root}")
        self.logger.info(f"  출력 디렉토리: {self.output_dir}")
        self.logger.info(f"  K-code 매핑: {len(self.k_code_to_class_mapping)}개")
    
    def setup_output_directory(self):
        """출력 디렉토리 구조 생성"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # YOLO 표준 디렉토리 구조
            subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
            for subdir in subdirs:
                (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"출력 디렉토리 구조 생성 완료: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"출력 디렉토리 생성 실패: {e}")
            raise
    
    def find_coco_annotations(self) -> List[Path]:
        """COCO 어노테이션 파일 찾기"""
        annotation_files = []
        
        # 일반적인 COCO 어노테이션 파일 패턴
        patterns = [
            '**/*annotations*.json',
            '**/*coco*.json',
            '**/instances_*.json',
            '**/combo_*.json'
        ]
        
        for pattern in patterns:
            found_files = list(self.data_root.rglob(pattern))
            annotation_files.extend(found_files)
        
        # 중복 제거
        annotation_files = list(set(annotation_files))
        
        self.logger.info(f"COCO 어노테이션 파일 {len(annotation_files)}개 발견")
        for file in annotation_files:
            self.logger.info(f"  {file}")
        
        return annotation_files
    
    def load_coco_annotation(self, annotation_path: Path) -> Dict[str, Any]:
        """COCO 어노테이션 파일 로드"""
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            self.logger.info(f"COCO 어노테이션 로드 완료: {annotation_path}")
            self.logger.info(f"  이미지: {len(coco_data.get('images', []))}개")
            self.logger.info(f"  어노테이션: {len(coco_data.get('annotations', []))}개")
            self.logger.info(f"  카테고리: {len(coco_data.get('categories', []))}개")
            
            return coco_data
            
        except Exception as e:
            self.logger.error(f"COCO 어노테이션 로드 실패 {annotation_path}: {e}")
            raise
    
    def extract_k_code_from_category(self, category_name: str) -> Optional[str]:
        """카테고리명에서 K-code 추출"""
        import re
        
        # K-code 패턴 매칭 (K-XXXXXX 형식)
        k_code_pattern = r'K-\d{6}'
        match = re.search(k_code_pattern, category_name)
        
        if match:
            return match.group(0)
        
        # 다른 패턴들 시도
        # 예: "drug_K123456", "pill_K-123456" 등
        alternative_patterns = [
            r'K\d{6}',           # K123456
            r'K-?\d{5}',         # K12345 또는 K-12345
        ]
        
        for pattern in alternative_patterns:
            match = re.search(pattern, category_name)
            if match:
                k_code = match.group(0)
                # 표준 형식으로 변환
                if not k_code.startswith('K-'):
                    k_code = f"K-{k_code[1:]}"
                return k_code
        
        return None
    
    def build_category_mapping(self, coco_data: Dict[str, Any]) -> Dict[int, int]:
        """COCO 카테고리 ID → YOLO 클래스 ID 매핑 구축"""
        coco_to_yolo_mapping = {}
        
        categories = coco_data.get('categories', [])
        
        for category in categories:
            coco_category_id = category['id']
            category_name = category['name']
            
            # 카테고리명에서 K-code 추출
            k_code = self.extract_k_code_from_category(category_name)
            
            if k_code:
                # K-code → YOLO 클래스 ID 매핑
                yolo_class_id = self.k_code_to_class_mapping.get(k_code, self.default_class_id)
                coco_to_yolo_mapping[coco_category_id] = yolo_class_id
                
                # 통계 업데이트
                self.conversion_stats['unique_k_codes'].add(k_code)
                self.conversion_stats['class_distribution'][yolo_class_id] = \
                    self.conversion_stats['class_distribution'].get(yolo_class_id, 0) + 1
                
                self.logger.debug(f"카테고리 매핑: {category_name} (COCO {coco_category_id}) → K-code {k_code} → YOLO {yolo_class_id}")
            else:
                # K-code를 찾을 수 없는 경우 기본 클래스
                coco_to_yolo_mapping[coco_category_id] = self.default_class_id
                self.logger.warning(f"K-code 추출 실패: {category_name} → 기본 클래스 {self.default_class_id}")
        
        self.logger.info(f"카테고리 매핑 구축 완료: {len(coco_to_yolo_mapping)}개")
        return coco_to_yolo_mapping
    
    def convert_single_annotation(
        self, 
        coco_data: Dict[str, Any], 
        category_mapping: Dict[int, int]
    ) -> List[YOLOAnnotation]:
        """단일 COCO 어노테이션을 YOLO 포맷으로 변환"""
        
        yolo_annotations = []
        
        # 이미지별 어노테이션 그룹화
        image_annotations = {}
        for annotation in coco_data.get('annotations', []):
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)
        
        # 이미지 정보 딕셔너리 구축
        images_dict = {img['id']: img for img in coco_data.get('images', [])}
        
        for image_id, annotations in image_annotations.items():
            try:
                # 이미지 정보 가져오기
                if image_id not in images_dict:
                    self.logger.warning(f"이미지 ID {image_id} 정보 없음")
                    continue
                
                image_info = images_dict[image_id]
                image_filename = image_info['file_name']
                img_width = image_info['width']
                img_height = image_info['height']
                
                # 이미지 파일 경로 찾기
                image_path = self.find_image_file(image_filename)
                if not image_path:
                    self.logger.warning(f"이미지 파일 찾을 수 없음: {image_filename}")
                    continue
                
                # 바운딩 박스와 클래스 ID 추출
                bboxes = []
                class_ids = []
                
                for annotation in annotations:
                    coco_category_id = annotation['category_id']
                    coco_bbox = annotation['bbox']  # [x, y, width, height]
                    
                    # 카테고리 매핑
                    if coco_category_id in category_mapping:
                        yolo_class_id = category_mapping[coco_category_id]
                    else:
                        yolo_class_id = self.default_class_id
                        self.logger.warning(f"카테고리 ID {coco_category_id} 매핑 없음 → 기본 클래스 {self.default_class_id}")
                    
                    # COCO → YOLO 바운딩 박스 변환
                    try:
                        bbox = BoundingBox.from_coco_bbox(coco_bbox, img_width, img_height)
                        
                        if bbox.is_valid():
                            bboxes.append(bbox)
                            class_ids.append(yolo_class_id)
                            self.conversion_stats['valid_bboxes'] += 1
                        else:
                            self.logger.warning(f"유효하지 않은 바운딩 박스: {coco_bbox} → {asdict(bbox)}")
                            self.conversion_stats['invalid_bboxes'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"바운딩 박스 변환 실패 {coco_bbox}: {e}")
                        self.conversion_stats['invalid_bboxes'] += 1
                        continue
                
                # YOLO 어노테이션 생성
                if bboxes:
                    yolo_annotation = YOLOAnnotation(
                        image_path=image_path,
                        image_width=img_width,
                        image_height=img_height,
                        bboxes=bboxes,
                        class_ids=class_ids
                    )
                    
                    # 유효성 검증
                    validation_result = yolo_annotation.validate()
                    if validation_result['is_valid']:
                        yolo_annotations.append(yolo_annotation)
                        self.conversion_stats['successful_conversions'] += 1
                    else:
                        self.logger.error(f"YOLO 어노테이션 유효성 검증 실패: {validation_result['errors']}")
                        self.conversion_stats['failed_conversions'] += 1
                else:
                    self.logger.warning(f"유효한 바운딩 박스가 없음: {image_filename}")
                    self.conversion_stats['failed_conversions'] += 1
                
                self.conversion_stats['processed_images'] += 1
                self.conversion_stats['total_annotations'] += len(annotations)
                
            except Exception as e:
                self.logger.error(f"이미지 {image_id} 어노테이션 변환 실패: {e}")
                self.conversion_stats['failed_conversions'] += 1
                continue
        
        return yolo_annotations
    
    def find_image_file(self, filename: str) -> Optional[Path]:
        """이미지 파일 경로 찾기"""
        # 다양한 위치에서 이미지 파일 검색
        search_patterns = [
            f"**/{filename}",
            f"**/images/**/{filename}",
            f"**/combination/**/{filename}",
            f"**/combo/**/{filename}",
        ]
        
        for pattern in search_patterns:
            found_files = list(self.data_root.rglob(pattern))
            if found_files:
                return found_files[0]  # 첫 번째 매치 반환
        
        return None
    
    def save_yolo_annotations(
        self, 
        yolo_annotations: List[YOLOAnnotation], 
        split: str = 'train'
    ) -> Dict[str, Any]:
        """YOLO 어노테이션을 파일로 저장"""
        
        save_stats = {
            'saved_images': 0,
            'saved_labels': 0,
            'copy_errors': 0,
            'label_errors': 0
        }
        
        images_dir = self.output_dir / 'images' / split
        labels_dir = self.output_dir / 'labels' / split
        
        for annotation in yolo_annotations:
            try:
                # 1. 이미지 파일 복사
                target_image_path = images_dir / annotation.image_path.name
                if not target_image_path.exists():
                    shutil.copy2(annotation.image_path, target_image_path)
                    save_stats['saved_images'] += 1
                
                # 2. 라벨 파일 저장
                label_filename = annotation.get_annotation_filename()
                label_path = labels_dir / label_filename
                
                yolo_content = annotation.to_yolo_format()
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(yolo_content)
                
                save_stats['saved_labels'] += 1
                
            except Exception as e:
                if 'image' in str(e).lower():
                    save_stats['copy_errors'] += 1
                    self.logger.error(f"이미지 복사 실패 {annotation.image_path}: {e}")
                else:
                    save_stats['label_errors'] += 1
                    self.logger.error(f"라벨 저장 실패 {annotation.image_path}: {e}")
        
        self.logger.info(f"YOLO 어노테이션 저장 완료 ({split})")
        self.logger.info(f"  이미지: {save_stats['saved_images']}개")
        self.logger.info(f"  라벨: {save_stats['saved_labels']}개")
        if save_stats['copy_errors'] > 0:
            self.logger.warning(f"  이미지 복사 실패: {save_stats['copy_errors']}개")
        if save_stats['label_errors'] > 0:
            self.logger.warning(f"  라벨 저장 실패: {save_stats['label_errors']}개")
        
        return save_stats
    
    def create_dataset_yaml(self, class_names: List[str]):
        """YOLO 데이터셋 YAML 파일 생성"""
        yaml_content = f"""# PillSnap ML YOLO Dataset Configuration
# Generated automatically by COCOToYOLOConverter

path: {self.output_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names

# Dataset info
task: detect  # detection task
source: PillSnap ML Combination Pills
version: 1.0
"""
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        self.logger.info(f"데이터셋 YAML 파일 생성: {yaml_path}")
        return yaml_path
    
    def convert_all_annotations(
        self, 
        train_val_split: float = 0.8,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """모든 COCO 어노테이션을 YOLO 포맷으로 변환"""
        
        # 1. 출력 디렉토리 설정
        self.setup_output_directory()
        
        # 2. COCO 어노테이션 파일 찾기
        annotation_files = self.find_coco_annotations()
        if not annotation_files:
            raise FileNotFoundError(f"COCO 어노테이션 파일을 찾을 수 없음: {self.data_root}")
        
        all_yolo_annotations = []
        
        # 3. 각 COCO 파일 변환
        for annotation_file in annotation_files:
            self.logger.info(f"변환 중: {annotation_file}")
            
            # COCO 데이터 로드
            coco_data = self.load_coco_annotation(annotation_file)
            
            # 카테고리 매핑 구축
            category_mapping = self.build_category_mapping(coco_data)
            
            # YOLO 어노테이션 변환
            yolo_annotations = self.convert_single_annotation(coco_data, category_mapping)
            all_yolo_annotations.extend(yolo_annotations)
        
        # 4. Train/Val 분할
        total_annotations = len(all_yolo_annotations)
        train_count = int(total_annotations * train_val_split)
        
        # 셔플링
        import random
        random.shuffle(all_yolo_annotations)
        
        train_annotations = all_yolo_annotations[:train_count]
        val_annotations = all_yolo_annotations[train_count:]
        
        self.logger.info(f"데이터 분할: Train {len(train_annotations)}개, Val {len(val_annotations)}개")
        
        # 5. YOLO 어노테이션 저장
        train_stats = self.save_yolo_annotations(train_annotations, 'train')
        val_stats = self.save_yolo_annotations(val_annotations, 'val')
        
        # 6. 데이터셋 YAML 생성
        if class_names is None:
            # 자동으로 클래스명 생성
            unique_class_ids = sorted(set(self.conversion_stats['class_distribution'].keys()))
            class_names = [f"class_{i}" for i in unique_class_ids]
        
        yaml_path = self.create_dataset_yaml(class_names)
        
        # 7. 최종 통계
        final_stats = {
            'conversion_stats': self.conversion_stats,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'dataset_yaml': str(yaml_path),
            'output_directory': str(self.output_dir),
            'class_names': class_names
        }
        
        self.logger.info("=== COCO → YOLO 변환 완료 ===")
        self.logger.info(f"처리된 이미지: {self.conversion_stats['processed_images']}개")
        self.logger.info(f"성공한 변환: {self.conversion_stats['successful_conversions']}개")
        self.logger.info(f"실패한 변환: {self.conversion_stats['failed_conversions']}개")
        self.logger.info(f"총 어노테이션: {self.conversion_stats['total_annotations']}개")
        self.logger.info(f"유효한 바운딩 박스: {self.conversion_stats['valid_bboxes']}개")
        self.logger.info(f"유효하지 않은 바운딩 박스: {self.conversion_stats['invalid_bboxes']}개")
        self.logger.info(f"고유 K-code: {len(self.conversion_stats['unique_k_codes'])}개")
        self.logger.info(f"클래스 분포: {self.conversion_stats['class_distribution']}")
        
        return final_stats
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """변환 통계 반환"""
        stats = self.conversion_stats.copy()
        stats['unique_k_codes'] = list(stats['unique_k_codes'])  # set → list 변환
        return stats


def create_coco_to_yolo_converter_for_stage1(
    data_root: str,
    output_dir: str,
    k_code_to_class_mapping: Optional[Dict[str, int]] = None
) -> COCOToYOLOConverter:
    """Stage 1용 COCO → YOLO 변환기 생성"""
    return COCOToYOLOConverter(
        data_root=data_root,
        output_dir=output_dir,
        k_code_to_class_mapping=k_code_to_class_mapping
    )


if __name__ == "__main__":
    # 사용 예제
    config = load_config()
    data_root = config['data']['root']
    output_dir = "/mnt/data/pillsnap_dataset/yolo_format"
    
    # Stage 1용 K-code 매핑 (실제로는 샘플링 결과에서 가져와야 함)
    sample_mapping = {
        "K-029066": 0,
        "K-016551": 1,
        "K-010913": 2,
        # ... 더 많은 매핑
    }
    
    converter = create_coco_to_yolo_converter_for_stage1(
        data_root=data_root,
        output_dir=output_dir,
        k_code_to_class_mapping=sample_mapping
    )
    
    try:
        # 변환 실행
        results = converter.convert_all_annotations(
            train_val_split=0.8,
            class_names=["pill"]  # 단순화된 클래스명
        )
        
        print("변환 완료!")
        print(f"결과 저장 위치: {results['output_directory']}")
        print(f"데이터셋 YAML: {results['dataset_yaml']}")
        
    except Exception as e:
        print(f"변환 실패: {e}")
        import traceback
        traceback.print_exc()