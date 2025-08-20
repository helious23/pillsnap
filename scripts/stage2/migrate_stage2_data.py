#!/usr/bin/env python3
"""
Stage 2 신규 데이터 SSD 이전 스크립트 (점진적 확장)

기존 Stage 1 데이터 유지 + Stage 2 신규 237개 클래스만 추가
"""

import json
import shutil
import os
from pathlib import Path
from typing import Set, List
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.core import PillSnapLogger

class Stage2DataMigrator:
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.source_root = Path("/mnt/data/pillsnap_dataset")
        self.target_root = Path("/home/max16/ssd_pillsnap/dataset")
        
        # 진행 상황 추적
        self.stats = {
            'total_classes': 0,
            'migrated_classes': 0,
            'total_images': 0,
            'migrated_images': 0,
            'skipped_existing': 0,
            'errors': 0
        }
    
    def load_stage_data(self) -> tuple[Set[str], Set[str]]:
        """Stage 1, 2 클래스 정보 로드"""
        stage1_file = Path("artifacts/stage1/sampling/stage1_sample.json")
        stage2_file = Path("artifacts/stage2/sampling/stage2_sample.json")
        
        with open(stage1_file, 'r') as f:
            stage1_data = json.load(f)
        
        with open(stage2_file, 'r') as f:
            stage2_data = json.load(f)
        
        stage1_classes = set(stage1_data['metadata']['selected_classes'])
        stage2_classes = set(stage2_data['metadata']['selected_classes'])
        
        self.logger.info(f"Stage 1 클래스: {len(stage1_classes)}개")
        self.logger.info(f"Stage 2 클래스: {len(stage2_classes)}개")
        
        return stage1_classes, stage2_classes
    
    def get_new_classes(self, stage1_classes: Set[str], stage2_classes: Set[str]) -> Set[str]:
        """Stage 2에서 새로 추가되는 클래스들 반환"""
        overlap = stage1_classes & stage2_classes
        new_classes = stage2_classes - stage1_classes
        
        self.logger.info(f"중복 클래스: {len(overlap)}개 (건너뜀)")
        self.logger.info(f"신규 클래스: {len(new_classes)}개 (이전 대상)")
        
        return new_classes
    
    def find_class_images(self, k_code: str) -> List[Path]:
        """특정 K-코드의 모든 이미지 파일 찾기"""
        images = []
        
        # Single 이미지 검색
        single_pattern = self.source_root / "data/train/images/single/*/{}".format(k_code)
        for ts_dir in self.source_root.glob("data/train/images/single/TS_*"):
            k_code_dir = ts_dir / k_code
            if k_code_dir.exists():
                for img_file in k_code_dir.glob("*.png"):
                    images.append(img_file)
        
        # Combination 이미지 검색
        for ts_dir in self.source_root.glob("data/train/images/combination/TS_*"):
            k_code_dir = ts_dir / k_code
            if k_code_dir.exists():
                for img_file in k_code_dir.glob("*.png"):
                    images.append(img_file)
        
        return images
    
    def copy_class_data(self, k_code: str) -> bool:
        """특정 클래스의 모든 데이터를 SSD로 복사"""
        try:
            images = self.find_class_images(k_code)
            
            if not images:
                self.logger.warning(f"클래스 {k_code}: 이미지를 찾을 수 없음")
                return False
            
            copied_count = 0
            
            for img_path in images:
                # 상대 경로 계산
                rel_path = img_path.relative_to(self.source_root)
                target_path = self.target_root / rel_path
                
                # 대상 디렉토리 생성
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일이 이미 존재하는지 확인
                if target_path.exists():
                    self.stats['skipped_existing'] += 1
                    continue
                
                # 파일 복사
                shutil.copy2(img_path, target_path)
                copied_count += 1
                self.stats['migrated_images'] += 1
            
            self.logger.info(f"클래스 {k_code}: {copied_count}/{len(images)}개 이미지 복사 완료")
            self.stats['migrated_classes'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"클래스 {k_code} 복사 중 오류: {e}")
            self.stats['errors'] += 1
            return False
    
    def check_disk_space(self, new_classes: Set[str]) -> bool:
        """디스크 여유 공간 확인"""
        # 대략적인 크기 추정 (클래스당 ~150MB)
        estimated_size_gb = len(new_classes) * 0.15
        
        # 현재 여유 공간 확인
        stat = shutil.disk_usage(self.target_root)
        free_gb = stat.free / (1024**3)
        
        self.logger.info(f"예상 필요 공간: {estimated_size_gb:.1f}GB")
        self.logger.info(f"SSD 여유 공간: {free_gb:.1f}GB")
        
        if free_gb < estimated_size_gb + 10:  # 10GB 버퍼
            self.logger.error("디스크 여유 공간 부족!")
            return False
        
        return True
    
    def migrate(self) -> bool:
        """Stage 2 신규 데이터 이전 실행"""
        self.logger.info("=== Stage 2 신규 데이터 SSD 이전 시작 ===")
        
        # 1. Stage 데이터 로드
        stage1_classes, stage2_classes = self.load_stage_data()
        new_classes = self.get_new_classes(stage1_classes, stage2_classes)
        
        if not new_classes:
            self.logger.info("이전할 신규 클래스가 없습니다.")
            return True
        
        # 2. 디스크 공간 확인
        if not self.check_disk_space(new_classes):
            return False
        
        # 3. 클래스별 데이터 이전
        self.stats['total_classes'] = len(new_classes)
        
        for i, k_code in enumerate(sorted(new_classes), 1):
            self.logger.info(f"[{i}/{len(new_classes)}] 처리 중: {k_code}")
            self.copy_class_data(k_code)
            
            # 진행률 출력
            if i % 10 == 0 or i == len(new_classes):
                progress = (i / len(new_classes)) * 100
                self.logger.info(f"진행률: {progress:.1f}% ({i}/{len(new_classes)})")
        
        # 4. 결과 요약
        self.logger.info("=== Stage 2 데이터 이전 완료 ===")
        self.logger.info(f"처리된 클래스: {self.stats['migrated_classes']}/{self.stats['total_classes']}")
        self.logger.info(f"이전된 이미지: {self.stats['migrated_images']}개")
        self.logger.info(f"건너뛴 이미지: {self.stats['skipped_existing']}개")
        self.logger.info(f"오류 발생: {self.stats['errors']}개")
        
        return self.stats['errors'] == 0

def main():
    migrator = Stage2DataMigrator()
    success = migrator.migrate()
    
    if success:
        print("✅ Stage 2 데이터 이전 성공!")
    else:
        print("❌ Stage 2 데이터 이전 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main()