#!/usr/bin/env python3
"""
Stage 2 훈련용 Manifest 생성기

Stage 2 선택 클래스(250개)에서 25,000개 이미지를 균등 샘플링하여
훈련용 manifest 파일을 생성합니다.
"""

import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.core import PillSnapLogger

class Stage2ManifestCreator:
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.ssd_root = Path("/home/max16/ssd_pillsnap/dataset")
        self.stage2_sample_path = Path("artifacts/stage2/sampling/stage2_sample_ssd.json")
        self.output_path = Path("artifacts/stage2/manifest_ssd.csv")
        
        # 샘플링 설정
        self.target_images = 25000
        self.images_per_class = 100  # 25000 ÷ 250 = 100
        self.train_ratio = 0.8
        self.seed = 42
        
        random.seed(self.seed)
        
    def load_stage2_classes(self) -> List[str]:
        """Stage 2 선택 클래스 로드"""
        with open(self.stage2_sample_path, 'r') as f:
            stage2_data = json.load(f)
        
        selected_classes = stage2_data['metadata']['selected_classes']
        self.logger.info(f"Stage 2 선택 클래스: {len(selected_classes)}개")
        return selected_classes
    
    def find_class_images(self, k_code: str) -> List[Tuple[Path, Path]]:
        """특정 K-코드의 이미지와 라벨 파일 쌍 찾기"""
        image_label_pairs = []
        
        # Single 이미지 검색
        single_images_root = self.ssd_root / "data/train/images/single"
        single_labels_root = self.ssd_root / "data/train/labels/single"
        
        for ts_dir in single_images_root.glob("TS_*"):
            k_code_dir = ts_dir / k_code
            if not k_code_dir.exists():
                continue
                
            # 대응하는 라벨 디렉토리 찾기
            ts_label_name = ts_dir.name.replace("TS_", "TL_")
            label_dir = single_labels_root / ts_label_name / f"{k_code}_json"
            
            # 이미지 파일들 순회
            for img_file in k_code_dir.glob("*.png"):
                # 대응하는 JSON 라벨 파일 찾기
                label_file = label_dir / f"{img_file.stem}.json"
                
                if label_file.exists():
                    image_label_pairs.append((img_file, label_file))
        
        return image_label_pairs
    
    def sample_class_images(self, image_label_pairs: List[Tuple[Path, Path]], 
                          target_count: int) -> List[Tuple[Path, Path]]:
        """클래스에서 목표 개수만큼 이미지 샘플링"""
        if len(image_label_pairs) <= target_count:
            return image_label_pairs
        
        return random.sample(image_label_pairs, target_count)
    
    def extract_metadata_from_path(self, img_path: Path, label_path: Path) -> Dict:
        """이미지 경로에서 메타데이터 추출"""
        # 파일명에서 코드 추출 (예: K-000114_0_2_0_0_75_260_200.png)
        code = img_path.stem
        k_code = code.split('_')[0]  # K-000114
        
        # 기본 메타데이터 (실제 JSON에서 읽어올 수도 있음)
        metadata = {
            'image_path': str(img_path),
            'label_path': str(label_path),
            'code': code,
            'is_pair': True,
            'mapping_code': k_code,
            'edi_code': '',  # JSON에서 읽어오거나 기본값
            'json_ok': True,
            'drug_N': k_code,
            'dl_name': '',
            'drug_shape': '',
            'print_front': '',
            'print_back': ''
        }
        
        return metadata
    
    def create_manifest(self) -> pd.DataFrame:
        """Stage 2 manifest 생성"""
        self.logger.info("=== Stage 2 Manifest 생성 시작 ===")
        
        # 1. Stage 2 클래스 로드
        stage2_classes = self.load_stage2_classes()
        
        # 2. 각 클래스별 이미지 수집 및 샘플링
        all_samples = []
        class_stats = {}
        
        for i, k_code in enumerate(stage2_classes, 1):
            self.logger.info(f"[{i}/{len(stage2_classes)}] 처리 중: {k_code}")
            
            # 클래스의 모든 이미지 찾기
            image_label_pairs = self.find_class_images(k_code)
            
            if not image_label_pairs:
                self.logger.warning(f"클래스 {k_code}: 이미지를 찾을 수 없음")
                class_stats[k_code] = 0
                continue
            
            # 샘플링
            sampled_pairs = self.sample_class_images(image_label_pairs, self.images_per_class)
            
            # 메타데이터 생성
            for img_path, label_path in sampled_pairs:
                metadata = self.extract_metadata_from_path(img_path, label_path)
                all_samples.append(metadata)
            
            class_stats[k_code] = len(sampled_pairs)
            
            # 진행률 출력
            if i % 50 == 0 or i == len(stage2_classes):
                progress = (i / len(stage2_classes)) * 100
                self.logger.info(f"진행률: {progress:.1f}% ({i}/{len(stage2_classes)})")
        
        # 3. DataFrame 생성
        df = pd.DataFrame(all_samples)
        
        # 4. 통계 출력
        total_images = len(df)
        valid_classes = sum(1 for count in class_stats.values() if count > 0)
        
        self.logger.info("=== Stage 2 Manifest 생성 완료 ===")
        self.logger.info(f"총 이미지: {total_images:,}개")
        self.logger.info(f"유효 클래스: {valid_classes}/{len(stage2_classes)}개")
        self.logger.info(f"클래스당 평균: {total_images/valid_classes:.1f}개" if valid_classes > 0 else "클래스당 평균: 0개")
        
        # 클래스별 분포 요약
        class_counts = pd.Series(class_stats)
        self.logger.info(f"이미지 분포 - 최소: {class_counts.min()}, 최대: {class_counts.max()}, 평균: {class_counts.mean():.1f}")
        
        return df
    
    def save_manifest(self, df: pd.DataFrame) -> None:
        """Manifest 파일 저장"""
        # 출력 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        df.to_csv(self.output_path, index=False)
        self.logger.info(f"Manifest 저장: {self.output_path}")
        
        # 요약 통계 저장
        stats_path = self.output_path.parent / "stage2_manifest_stats.json"
        stats = {
            'total_samples': len(df),
            'unique_classes': df['mapping_code'].nunique(),
            'target_images': self.target_images,
            'images_per_class': self.images_per_class,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"통계 저장: {stats_path}")

def main():
    """메인 실행 함수"""
    creator = Stage2ManifestCreator()
    
    try:
        # Manifest 생성
        df = creator.create_manifest()
        
        if df.empty:
            print("❌ Manifest 생성 실패: 데이터가 없음")
            sys.exit(1)
        
        # 저장
        creator.save_manifest(df)
        
        print("✅ Stage 2 Manifest 생성 완료!")
        print(f"📁 경로: {creator.output_path}")
        print(f"📊 샘플 수: {len(df):,}개")
        print(f"🏷️ 클래스 수: {df['mapping_code'].nunique()}개")
        
    except Exception as e:
        print(f"❌ Stage 2 Manifest 생성 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()