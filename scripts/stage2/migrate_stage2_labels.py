#!/usr/bin/env python3
"""
Stage 2 라벨 데이터 SSD 이전 스크립트

기존 Stage 2 이미지 데이터에 대응하는 라벨 데이터를 SSD로 이전합니다.
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

class Stage2LabelMigrator:
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.source_root = Path("/mnt/data/pillsnap_dataset")
        self.target_root = Path("/home/max16/ssd_pillsnap/dataset")
        
        # 진행 상황 추적
        self.stats = {
            'total_labels': 0,
            'migrated_labels': 0,
            'skipped_existing': 0,
            'missing_labels': 0,
            'errors': 0
        }
    
    def get_ssd_k_codes(self) -> Set[str]:
        """SSD에 있는 모든 K-코드 수집"""
        k_codes = set()
        
        ssd_images_root = self.target_root / "data/train/images/single"
        for ts_dir in ssd_images_root.glob("TS_*"):
            for k_dir in ts_dir.glob("K-*"):
                k_codes.add(k_dir.name)
        
        self.logger.info(f"SSD에서 발견한 K-코드: {len(k_codes)}개")
        return k_codes
    
    def find_label_files(self, k_code: str) -> List[Path]:
        """특정 K-코드의 모든 라벨 파일 찾기"""
        labels = []
        
        # Single 라벨 검색
        source_labels_root = self.source_root / "data/train/labels/single"
        for tl_dir in source_labels_root.glob("TL_*"):
            k_label_dir = tl_dir / f"{k_code}_json"
            if k_label_dir.exists():
                for json_file in k_label_dir.glob("*.json"):
                    labels.append(json_file)
        
        # Combination 라벨 검색 (필요시)
        source_combo_labels = self.source_root / "data/train/labels/combination"
        if source_combo_labels.exists():
            for tl_dir in source_combo_labels.glob("TL_*"):
                k_label_dir = tl_dir / f"{k_code}_json"
                if k_label_dir.exists():
                    for json_file in k_label_dir.glob("*.json"):
                        labels.append(json_file)
        
        return labels
    
    def copy_label_files(self, k_code: str) -> bool:
        """특정 K-코드의 라벨 파일들을 SSD로 복사"""
        try:
            labels = self.find_label_files(k_code)
            
            if not labels:
                self.logger.warning(f"K-코드 {k_code}: 라벨 파일을 찾을 수 없음")
                self.stats['missing_labels'] += 1
                return False
            
            copied_count = 0
            
            for label_path in labels:
                # 상대 경로 계산
                rel_path = label_path.relative_to(self.source_root)
                target_path = self.target_root / rel_path
                
                # 대상 디렉토리 생성
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 파일이 이미 존재하는지 확인
                if target_path.exists():
                    self.stats['skipped_existing'] += 1
                    continue
                
                # 파일 복사
                shutil.copy2(label_path, target_path)
                copied_count += 1
                self.stats['migrated_labels'] += 1
                self.stats['total_labels'] += 1
            
            if copied_count > 0:
                self.logger.info(f"K-코드 {k_code}: {copied_count}개 라벨 파일 복사 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"K-코드 {k_code} 라벨 복사 중 오류: {e}")
            self.stats['errors'] += 1
            return False
    
    def check_disk_space(self, k_codes: Set[str]) -> bool:
        """디스크 여유 공간 확인"""
        # 대략적인 크기 추정 (K-코드당 라벨 ~10MB)
        estimated_size_gb = len(k_codes) * 0.01
        
        # 현재 여유 공간 확인
        stat = shutil.disk_usage(self.target_root)
        free_gb = stat.free / (1024**3)
        
        self.logger.info(f"예상 필요 공간: {estimated_size_gb:.1f}GB")
        self.logger.info(f"SSD 여유 공간: {free_gb:.1f}GB")
        
        if free_gb < estimated_size_gb + 5:  # 5GB 버퍼
            self.logger.error("디스크 여유 공간 부족!")
            return False
        
        return True
    
    def migrate_labels(self) -> bool:
        """Stage 2 라벨 데이터 이전 실행"""
        self.logger.info("=== Stage 2 라벨 데이터 SSD 이전 시작 ===")
        
        # 1. SSD에 있는 K-코드 수집
        k_codes = self.get_ssd_k_codes()
        
        if not k_codes:
            self.logger.error("SSD에 K-코드를 찾을 수 없습니다.")
            return False
        
        # 2. 디스크 공간 확인
        if not self.check_disk_space(k_codes):
            return False
        
        # 3. K-코드별 라벨 이전
        for i, k_code in enumerate(sorted(k_codes), 1):
            self.logger.info(f"[{i}/{len(k_codes)}] 라벨 이전 중: {k_code}")
            self.copy_label_files(k_code)
            
            # 진행률 출력
            if i % 50 == 0 or i == len(k_codes):
                progress = (i / len(k_codes)) * 100
                self.logger.info(f"진행률: {progress:.1f}% ({i}/{len(k_codes)})")
        
        # 4. 결과 요약
        self.logger.info("=== Stage 2 라벨 이전 완료 ===")
        self.logger.info(f"처리된 K-코드: {len(k_codes)}개")
        self.logger.info(f"이전된 라벨: {self.stats['migrated_labels']}개")
        self.logger.info(f"건너뛴 라벨: {self.stats['skipped_existing']}개")
        self.logger.info(f"누락된 K-코드: {self.stats['missing_labels']}개")
        self.logger.info(f"오류 발생: {self.stats['errors']}개")
        
        return self.stats['errors'] == 0
    
    def verify_migration(self) -> bool:
        """라벨 이전 결과 검증"""
        self.logger.info("=== 라벨 이전 결과 검증 ===")
        
        # SSD에 라벨 디렉토리가 생성되었는지 확인
        target_labels_root = self.target_root / "data/train/labels"
        if not target_labels_root.exists():
            self.logger.error("라벨 디렉토리가 생성되지 않았습니다.")
            return False
        
        # 일부 K-코드에 대해 이미지-라벨 매칭 확인
        ssd_images_root = self.target_root / "data/train/images/single"
        sample_count = 0
        matched_count = 0
        
        for ts_dir in list(ssd_images_root.glob("TS_*"))[:3]:  # 3개 TS만 샘플 검사
            for k_dir in list(ts_dir.glob("K-*"))[:2]:  # 각 TS에서 2개 K-코드만
                k_code = k_dir.name
                sample_count += 1
                
                # 대응하는 라벨 디렉토리 확인
                ts_label_name = ts_dir.name.replace("TS_", "TL_")
                label_dir = target_labels_root / "single" / ts_label_name / f"{k_code}_json"
                
                if label_dir.exists() and any(label_dir.glob("*.json")):
                    matched_count += 1
                    self.logger.info(f"✅ {k_code}: 이미지-라벨 매칭 확인")
                else:
                    self.logger.warning(f"❌ {k_code}: 라벨 누락")
        
        match_ratio = (matched_count / sample_count) * 100 if sample_count > 0 else 0
        self.logger.info(f"샘플 검증 결과: {matched_count}/{sample_count} ({match_ratio:.1f}%) 매칭")
        
        return match_ratio > 80  # 80% 이상 매칭되면 성공

def main():
    """메인 실행 함수"""
    migrator = Stage2LabelMigrator()
    
    try:
        # 라벨 이전
        success = migrator.migrate_labels()
        
        if not success:
            print("❌ Stage 2 라벨 이전 실패!")
            sys.exit(1)
        
        # 검증
        if migrator.verify_migration():
            print("✅ Stage 2 라벨 이전 및 검증 완료!")
            print(f"📊 이전된 라벨: {migrator.stats['migrated_labels']}개")
            print(f"📁 라벨 디렉토리: {migrator.target_root}/data/train/labels")
        else:
            print("⚠️ Stage 2 라벨 이전은 완료되었으나 검증에서 문제 발견")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Stage 2 라벨 이전 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()