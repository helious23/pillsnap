#!/usr/bin/env python3
"""
COCO to YOLO Format Converter

Combination 데이터의 COCO JSON 어노테이션을 YOLO TXT 형식으로 변환:
- COCO: bbox [x, y, width, height] (픽셀 단위)
- YOLO: bbox [center_x, center_y, width, height] (정규화 0~1)

변환 공식:
- center_x = (x + width/2) / image_width  
- center_y = (y + height/2) / image_height
- norm_width = width / image_width
- norm_height = height / image_height
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.core import PillSnapLogger


class COCOToYOLOConverter:
    """COCO JSON 형식을 YOLO TXT 형식으로 변환하는 클래스"""
    
    def __init__(self, coco_dir: str, yolo_dir: str, class_mapping: Optional[Dict[str, int]] = None):
        self.logger = PillSnapLogger(__name__)
        
        self.coco_dir = Path(coco_dir)
        self.yolo_dir = Path(yolo_dir)
        
        # 기본 클래스 매핑 (Drug → 0)
        self.class_mapping = class_mapping or {"Drug": 0}
        
        # 통계 추적
        self.stats = {
            'total_files': 0,
            'converted_files': 0,
            'failed_files': 0,
            'total_annotations': 0,
            'skipped_annotations': 0
        }
        
        self.logger.info(f"COCO→YOLO 변환기 초기화")
        self.logger.info(f"입력: {self.coco_dir}")
        self.logger.info(f"출력: {self.yolo_dir}")
        self.logger.info(f"클래스 매핑: {self.class_mapping}")
    
    def convert_bbox_coco_to_yolo(self, coco_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        COCO bbox를 YOLO 형식으로 변환
        
        Args:
            coco_bbox: [x, y, width, height] (픽셀 단위)
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            [center_x, center_y, width, height] (정규화 0~1)
        """
        x, y, w, h = coco_bbox
        
        # COCO → YOLO 변환
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # 범위 검증 (0~1)
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))
        
        return [center_x, center_y, norm_width, norm_height]
    
    def _fix_json_syntax(self, text_data: str) -> str:
        """JSON 구문 오류들을 수정"""
        # 일반적인 JSON 오류 패턴들 수정
        import re
        
        # 1. 끝에 콤마 제거 (trailing comma)
        text_data = re.sub(r',(\s*[}\]])', r'\1', text_data)
        
        # 2. 따옴표 누락 수정 (키 이름)
        text_data = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text_data)
        
        # 3. 단일 따옴표를 이중 따옴표로 변경
        text_data = text_data.replace("'", '"')
        
        # 4. 불완전한 배열/객체 닫기
        open_braces = text_data.count('{')
        close_braces = text_data.count('}')
        if open_braces > close_braces:
            text_data += '}' * (open_braces - close_braces)
        
        open_brackets = text_data.count('[')
        close_brackets = text_data.count(']')
        if open_brackets > close_brackets:
            text_data += ']' * (open_brackets - close_brackets)
        
        return text_data
    
    def _extract_filename_from_path(self, coco_json_path: Path) -> str:
        """JSON 파일 경로에서 이미지 파일명 추정"""
        # JSON 파일명에서 확장자를 이미지 확장자로 변경
        base_name = coco_json_path.stem
        
        # 일반적인 이미지 확장자들로 시도
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_filename = base_name + ext
            # 실제 이미지 파일이 존재하는지 확인 (가능한 경우)
            potential_img_path = coco_json_path.parent / img_filename
            if potential_img_path.exists():
                return img_filename
        
        # 기본값 반환 (JPG)
        return base_name + '.jpg'
    
    def _estimate_image_dimensions(self, coco_json_path: Path, coco_data: dict) -> Tuple[Optional[int], Optional[int]]:
        """어노테이션 정보에서 이미지 크기 추정"""
        try:
            annotations = coco_data.get('annotations', [])
            if not annotations:
                return None, None
            
            max_x, max_y = 0, 0
            
            # 모든 어노테이션의 bbox를 분석해서 최대값 찾기
            for ann in annotations:
                if isinstance(ann, dict):
                    bbox = ann.get('bbox')
                    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        try:
                            x, y, w, h = bbox[:4]
                            max_x = max(max_x, float(x) + float(w))
                            max_y = max(max_y, float(y) + float(h))
                        except (ValueError, TypeError):
                            continue
            
            if max_x > 0 and max_y > 0:
                # 여유분 추가 (10% 패딩)
                estimated_width = int(max_x * 1.1)
                estimated_height = int(max_y * 1.1)
                return estimated_width, estimated_height
            
        except Exception:
            pass
        
        # 표준 크기들로 추정
        return self._get_default_dimensions()
    
    def _estimate_image_width(self, coco_data: dict) -> Optional[int]:
        """어노테이션에서 이미지 너비 추정"""
        width, _ = self._estimate_image_dimensions(None, coco_data)
        return width
    
    def _estimate_image_height(self, coco_data: dict) -> Optional[int]:
        """어노테이션에서 이미지 높이 추정"""
        _, height = self._estimate_image_dimensions(None, coco_data)
        return height
    
    def _get_default_dimensions(self) -> Tuple[int, int]:
        """기본 이미지 크기 반환"""
        # PillSnap 데이터셋의 일반적인 크기들
        return 640, 640  # YOLOv11 표준 입력 크기
    
    def convert_annotation(self, coco_json_path: Path) -> Optional[Tuple[str, List[str]]]:
        """
        단일 COCO JSON 파일을 YOLO 형식으로 변환 (개선된 robust 버전)
        
        Args:
            coco_json_path: COCO JSON 파일 경로
            
        Returns:
            (image_filename, yolo_lines) 또는 None (실패시)
        """
        try:
            # 단계별 인코딩 시도 (간단하고 효과적)
            coco_data = None
            
            # 1단계: 표준 UTF-8 시도
            try:
                with open(coco_json_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            
            # 2단계: 한국어 인코딩 시도
            if coco_data is None:
                for encoding in ['cp949', 'euc-kr']:
                    try:
                        with open(coco_json_path, 'r', encoding=encoding) as f:
                            coco_data = json.load(f)
                        break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            
            # 3단계: 바이트 모드로 강제 복구 (errors='replace' 사용)
            if coco_data is None:
                try:
                    with open(coco_json_path, 'rb') as f:
                        raw_data = f.read()
                    
                    # UTF-8로 손상된 문자 대체하여 디코딩
                    text_data = raw_data.decode('utf-8', errors='replace')
                    coco_data = json.loads(text_data)
                    self.logger.info(f"손상 복구 성공: {coco_json_path.name}")
                    
                except Exception:
                    pass
            
            if coco_data is None:
                self.logger.warning(f"파일 읽기 실패: {coco_json_path.name}")
                return None
            
            # 기본 구조 검증
            if not isinstance(coco_data, dict):
                self.logger.warning(f"잘못된 JSON 구조: {coco_json_path.name}")
                return None
            
            # 이미지 정보 추출 (간단한 복구)
            images = coco_data.get('images', [])
            if not images or not isinstance(images, list) or len(images) == 0:
                # 기본값으로 복구
                img_filename = coco_json_path.stem + '.jpg'
                img_width, img_height = 640, 640  # YOLOv11 기본 크기
                self.logger.info(f"이미지 정보 기본값 적용: {img_filename} (640x640)")
            else:
                image_info = images[0]  # 첫 번째 이미지
                if not isinstance(image_info, dict):
                    img_filename = coco_json_path.stem + '.jpg'
                    img_width, img_height = 640, 640
                else:
                    # 필수 정보 추출
                    img_filename = image_info.get('file_name', coco_json_path.stem + '.jpg')
                    img_width = image_info.get('width', 640)
                    img_height = image_info.get('height', 640)
                    
                    # 타입 검증 및 기본값 적용
                    if not isinstance(img_width, (int, float)) or img_width <= 0:
                        img_width = 640
                    if not isinstance(img_height, (int, float)) or img_height <= 0:
                        img_height = 640
            
            # 어노테이션 추출 (안전하게)
            annotations = coco_data.get('annotations', [])
            if not isinstance(annotations, list):
                self.logger.warning(f"어노테이션이 리스트가 아님: {coco_json_path.name}")
                annotations = []
                
            yolo_lines = []
            
            for i, ann in enumerate(annotations):
                try:
                    if not isinstance(ann, dict):
                        self.logger.warning(f"어노테이션 {i} 형식 오류: {coco_json_path.name}")
                        continue
                    
                    # 클래스 ID 변환
                    category_id = ann.get('category_id', 1)
                    if not isinstance(category_id, int):
                        self.logger.warning(f"카테고리 ID 형식 오류: {category_id} in {coco_json_path.name}")
                        continue
                    
                    # 카테고리 이름으로 클래스 ID 찾기
                    categories = coco_data.get('categories', [])
                    class_name = "Drug"  # 기본값
                    
                    if isinstance(categories, list):
                        for cat in categories:
                            if isinstance(cat, dict) and cat.get('id') == category_id:
                                class_name = cat.get('name', 'Drug')
                                break
                    
                    if class_name not in self.class_mapping:
                        self.logger.warning(f"미지의 클래스: {class_name}, 스킵")
                        self.stats['skipped_annotations'] += 1
                        continue
                    
                    yolo_class_id = self.class_mapping[class_name]
                    
                    # bbox 안전 추출 (더 관대한 처리)
                    coco_bbox = ann.get('bbox')
                    if coco_bbox is None or coco_bbox == [] or coco_bbox == []:
                        # bbox가 없으면 스킵 (경고 없이)
                        self.stats['skipped_annotations'] += 1
                        continue
                    
                    # bbox 형식 검증 및 복구 시도
                    if not isinstance(coco_bbox, (list, tuple)):
                        self.stats['skipped_annotations'] += 1
                        continue
                        
                    if len(coco_bbox) < 4:
                        # 부족한 좌표는 0으로 패딩
                        while len(coco_bbox) < 4:
                            coco_bbox.append(0)
                        self.logger.info(f"bbox 패딩 적용: {coco_json_path.name}")
                    elif len(coco_bbox) > 4:
                        # 초과하는 좌표는 제거
                        coco_bbox = coco_bbox[:4]
                    
                    # bbox 값 타입 검증 및 변환 (관대한 처리)
                    try:
                        x, y, w, h = coco_bbox[:4]
                        
                        # 안전한 float 변환
                        try:
                            x, y, w, h = float(x), float(y), float(w), float(h)
                        except (ValueError, TypeError):
                            # 변환 실패시 스킵 (조용히)
                            self.stats['skipped_annotations'] += 1
                            continue
                        
                        # 값 정규화 (극단적인 값들 처리)
                        x = max(0, min(x, img_width))
                        y = max(0, min(y, img_height))
                        w = max(1, min(w, img_width - x))  # 최소 1픽셀
                        h = max(1, min(h, img_height - y))  # 최소 1픽셀
                        
                        # 유효하지 않은 박스는 기본값 적용
                        if w <= 0 or h <= 0:
                            w, h = 50, 50  # 기본 박스 크기
                            
                        if x + w > img_width:
                            w = img_width - x
                        if y + h > img_height:
                            h = img_height - y
                            
                        # 최종 검증 후 기본값 적용
                        if w <= 0 or h <= 0:
                            self.stats['skipped_annotations'] += 1
                            continue
                            
                        normalized_bbox = [x, y, w, h]
                        
                    except Exception:
                        # 모든 예외를 조용히 처리
                        self.stats['skipped_annotations'] += 1
                        continue
                    
                    # YOLO 형식으로 변환
                    try:
                        yolo_bbox = self.convert_bbox_coco_to_yolo(normalized_bbox, img_width, img_height)
                        
                        # 변환 결과 검증
                        if not all(0 <= coord <= 1 for coord in yolo_bbox):
                            self.logger.warning(f"YOLO bbox 범위 오류: {yolo_bbox} in {coco_json_path.name}")
                            self.stats['skipped_annotations'] += 1
                            continue
                        
                        # YOLO 라인 생성
                        yolo_line = f"{yolo_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                        yolo_lines.append(yolo_line)
                        
                        self.stats['total_annotations'] += 1
                        
                    except Exception as e:
                        self.logger.warning(f"YOLO 변환 실패: {normalized_bbox} ({e}) in {coco_json_path.name}")
                        self.stats['skipped_annotations'] += 1
                        continue
                
                except Exception as e:
                    self.logger.warning(f"어노테이션 {i} 처리 실패: {e} in {coco_json_path.name}")
                    self.stats['skipped_annotations'] += 1
                    continue
            
            return img_filename, yolo_lines
            
        except Exception as e:
            self.logger.error(f"변환 실패: {coco_json_path} - {e}")
            return None
    
    def save_yolo_annotation(self, img_filename: str, yolo_lines: List[str]) -> bool:
        """
        YOLO 어노테이션을 TXT 파일로 저장
        
        Args:
            img_filename: 이미지 파일명
            yolo_lines: YOLO 형식 어노테이션 라인들
            
        Returns:
            저장 성공 여부
        """
        try:
            # 이미지 파일명에서 확장자 제거 후 .txt로 변경
            txt_filename = Path(img_filename).stem + '.txt'
            txt_path = self.yolo_dir / txt_filename
            
            # 디렉토리 생성
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # TXT 파일 저장
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines) + '\n')
            
            return True
            
        except Exception as e:
            self.logger.error(f"YOLO 파일 저장 실패: {img_filename} - {e}")
            return False
    
    def convert_single_file(self, coco_json_path: Path) -> bool:
        """
        단일 파일 변환 (멀티프로세싱용)
        
        Args:
            coco_json_path: COCO JSON 파일 경로
            
        Returns:
            변환 성공 여부
        """
        result = self.convert_annotation(coco_json_path)
        
        if result is None:
            return False
        
        img_filename, yolo_lines = result
        
        if not yolo_lines:
            self.logger.warning(f"어노테이션 없음: {img_filename}")
            return False
        
        return self.save_yolo_annotation(img_filename, yolo_lines)
    
    def find_coco_files(self) -> List[Path]:
        """
        COCO JSON 파일들을 찾아 리스트로 반환
        
        Returns:
            COCO JSON 파일 경로 리스트
        """
        coco_files = []
        
        # 재귀적으로 JSON 파일 찾기
        for json_file in self.coco_dir.rglob('*.json'):
            coco_files.append(json_file)
        
        self.logger.info(f"발견된 COCO 파일: {len(coco_files)}개")
        return coco_files
    
    def convert_batch(self, max_files: Optional[int] = None, num_workers: int = 4) -> Dict[str, int]:
        """
        배치 처리로 대용량 변환
        
        Args:
            max_files: 최대 변환 파일 수 (None이면 전체)
            num_workers: 병렬 처리 워커 수
            
        Returns:
            변환 통계 딕셔너리
        """
        self.logger.info("=" * 60)
        self.logger.info("COCO→YOLO 배치 변환 시작")
        self.logger.info("=" * 60)
        
        # YOLO 출력 디렉토리 생성
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO 파일 목록 가져오기
        coco_files = self.find_coco_files()
        
        if max_files:
            coco_files = coco_files[:max_files]
            self.logger.info(f"변환 대상: {len(coco_files)}개 파일 (제한적용)")
        
        self.stats['total_files'] = len(coco_files)
        
        if not coco_files:
            self.logger.warning("변환할 COCO 파일이 없습니다")
            return self.stats
        
        # 멀티프로세싱으로 병렬 변환
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 작업 제출
            future_to_file = {
                executor.submit(self.convert_single_file, coco_file): coco_file 
                for coco_file in coco_files
            }
            
            # 진행률 표시와 함께 결과 수집
            with tqdm(total=len(coco_files), desc="COCO→YOLO 변환") as pbar:
                for future in as_completed(future_to_file):
                    coco_file = future_to_file[future]
                    
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                            self.stats['converted_files'] += 1
                        else:
                            self.stats['failed_files'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"파일 처리 중 예외: {coco_file} - {e}")
                        self.stats['failed_files'] += 1
                    
                    pbar.update(1)
        
        # 결과 리포트
        self.logger.info("=" * 60)
        self.logger.info("COCO→YOLO 변환 완료")
        self.logger.info("=" * 60)
        self.logger.info(f"총 파일: {self.stats['total_files']:,}개")
        self.logger.info(f"성공: {self.stats['converted_files']:,}개")
        self.logger.info(f"실패: {self.stats['failed_files']:,}개")
        self.logger.info(f"총 어노테이션: {self.stats['total_annotations']:,}개")
        self.logger.info(f"스킵된 어노테이션: {self.stats['skipped_annotations']:,}개")
        self.logger.info(f"성공률: {success_count/len(coco_files)*100:.1f}%")
        
        return self.stats
    
    def verify_conversion(self, num_samples: int = 5) -> bool:
        """
        변환 결과 검증 및 시각화
        
        Args:
            num_samples: 검증할 샘플 수
            
        Returns:
            검증 성공 여부
        """
        self.logger.info("변환 결과 검증 시작...")
        
        # YOLO 파일 목록 가져오기
        yolo_files = list(self.yolo_dir.rglob('*.txt'))
        
        if not yolo_files:
            self.logger.error("변환된 YOLO 파일이 없습니다")
            return False
        
        # 샘플 파일들 검증
        sample_files = yolo_files[:min(num_samples, len(yolo_files))]
        
        for yolo_file in sample_files:
            try:
                with open(yolo_file, 'r') as f:
                    lines = f.readlines()
                
                self.logger.info(f"검증: {yolo_file.name}")
                
                for i, line in enumerate(lines[:3]):  # 첫 3개 어노테이션만 표시
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        self.logger.error(f"잘못된 형식: {line.strip()}")
                        return False
                    
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # 범위 검증 (0~1)
                    if not all(0 <= val <= 1 for val in [cx, cy, w, h]):
                        self.logger.error(f"범위 초과: {line.strip()}")
                        return False
                    
                    self.logger.info(f"  어노테이션 {i+1}: class={class_id}, bbox=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})")
                
            except Exception as e:
                self.logger.error(f"검증 실패: {yolo_file} - {e}")
                return False
        
        self.logger.info(f"✅ 변환 검증 완료: {len(sample_files)}개 샘플 정상")
        return True
    
    def generate_conversion_report(self) -> str:
        """
        변환 리포트 생성
        
        Returns:
            리포트 문자열
        """
        report = f"""
COCO→YOLO 변환 리포트
{'='*50}

입력 디렉토리: {self.coco_dir}
출력 디렉토리: {self.yolo_dir}

통계:
- 총 파일: {self.stats['total_files']:,}개
- 변환 성공: {self.stats['converted_files']:,}개
- 변환 실패: {self.stats['failed_files']:,}개
- 총 어노테이션: {self.stats['total_annotations']:,}개
- 스킵된 어노테이션: {self.stats['skipped_annotations']:,}개

성공률: {self.stats['converted_files']/max(self.stats['total_files'], 1)*100:.1f}%

클래스 매핑: {self.class_mapping}
"""
        return report


def main():
    """CLI 실행 함수"""
    parser = argparse.ArgumentParser(description="COCO JSON을 YOLO TXT 형식으로 변환")
    parser.add_argument('--coco-dir', type=str, required=True,
                       help='COCO JSON 파일들이 있는 디렉토리')
    parser.add_argument('--yolo-dir', type=str, required=True,
                       help='YOLO TXT 파일들을 저장할 디렉토리')
    parser.add_argument('--max-files', type=int, default=None,
                       help='최대 변환 파일 수 (테스트용)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='병렬 처리 워커 수')
    parser.add_argument('--verify', action='store_true',
                       help='변환 후 결과 검증 수행')
    
    args = parser.parse_args()
    
    # 변환기 생성 및 실행
    converter = COCOToYOLOConverter(
        coco_dir=args.coco_dir,
        yolo_dir=args.yolo_dir
    )
    
    # 배치 변환 실행
    stats = converter.convert_batch(
        max_files=args.max_files,
        num_workers=args.num_workers
    )
    
    # 검증 수행 (옵션)
    if args.verify:
        converter.verify_conversion()
    
    # 리포트 출력
    report = converter.generate_conversion_report()
    print(report)
    
    # 결과 반환 (성공 시 0, 실패 시 1)
    return 0 if stats['converted_files'] > 0 else 1


if __name__ == "__main__":
    exit(main())