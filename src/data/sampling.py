"""
Progressive Validation 샘플링 시스템

Stage 1: 526만 이미지 → 5,000개 이미지, 50개 클래스 추출
- 균등 분포 보장: 클래스당 100개 이미지
- 품질 검증: 이미지 무결성, 라벨 정확성
- 재현 가능성: 시드 기반 결정적 샘플링
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
from PIL import Image

from src.utils.core import PillSnapLogger, load_config


@dataclass
class SamplingStats:
    """샘플링 통계 정보"""
    total_images: int
    total_classes: int
    sampled_images: int
    sampled_classes: int
    images_per_class: Dict[str, int]
    single_pill_ratio: float
    combo_pill_ratio: float
    quality_pass_rate: float


@dataclass 
class Stage1SamplingStrategy:
    """Stage 1 샘플링 전략 설정"""
    target_images: int = 5000
    target_classes: int = 50
    images_per_class: int = 100  # 5000 / 50 = 100
    min_images_per_class: int = 80  # 클래스당 최소 이미지 수
    max_images_per_class: int = 120  # 클래스당 최대 이미지 수
    quality_threshold: float = 0.95  # 품질 통과 비율 임계값
    seed: int = 42  # 재현 가능성을 위한 시드
    
    single_combo_ratio: float = 0.7  # Single:Combo = 7:3 비율
    prefer_balanced_distribution: bool = True
    
    def __post_init__(self):
        """설정 유효성 검증"""
        assert self.target_images > 0, "target_images는 양수여야 함"
        assert self.target_classes > 0, "target_classes는 양수여야 함"
        assert self.images_per_class > 0, "images_per_class는 양수여야 함"
        assert 0 < self.quality_threshold <= 1, "quality_threshold는 0~1 사이"
        assert 0 < self.single_combo_ratio < 1, "single_combo_ratio는 0~1 사이"
        
        # 계산된 이미지 수가 목표와 일치하는지 확인
        calculated_images = self.target_classes * self.images_per_class
        if calculated_images != self.target_images:
            self.images_per_class = self.target_images // self.target_classes
            remaining = self.target_images % self.target_classes
            print(f"⚠️  이미지 수 조정: {self.images_per_class}개/클래스 + {remaining}개 추가")


class ProgressiveValidationSampler:
    """Progressive Validation을 위한 샘플러 클래스"""
    
    def __init__(self, data_root: str, strategy: Stage1SamplingStrategy):
        self.data_root = Path(data_root)
        self.strategy = strategy
        self.logger = PillSnapLogger(__name__)
        self.config = load_config()
        
        # 샘플링 결과 저장 경로
        self.artifacts_dir = Path("artifacts/stage1/sampling")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # 시드 설정으로 재현 가능성 보장
        random.seed(strategy.seed)
        
        self.logger.info(f"ProgressiveValidationSampler 초기화")
        self.logger.info(f"데이터 루트: {self.data_root}")
        self.logger.info(f"목표: {strategy.target_images}개 이미지, {strategy.target_classes}개 클래스")
    
    def scan_available_data(self) -> Tuple[Dict[str, List[Path]], Dict[str, int]]:
        """사용 가능한 데이터 스캔"""
        self.logger.info("데이터 스캔 시작...")
        
        single_images = defaultdict(list)
        combo_images = defaultdict(list)
        
        # Single 약품 이미지 스캔
        single_dir = self.data_root / "data/train/images/single"
        if single_dir.exists():
            for ts_dir in single_dir.glob("TS_*"):
                if ts_dir.is_dir():
                    for k_code_dir in ts_dir.iterdir():
                        if k_code_dir.is_dir():
                            k_code = k_code_dir.name
                            images = list(k_code_dir.glob("*.jpg")) + list(k_code_dir.glob("*.png"))
                            if images:
                                single_images[k_code].extend(images)
        
        # Combination 약품 이미지 스캔  
        combo_dir = self.data_root / "data/train/images/combination"
        if combo_dir.exists():
            for ts_dir in combo_dir.glob("TS_*"):
                if ts_dir.is_dir():
                    for k_code_dir in ts_dir.iterdir():
                        if k_code_dir.is_dir():
                            k_code = k_code_dir.name
                            images = list(k_code_dir.glob("*.jpg")) + list(k_code_dir.glob("*.png"))
                            if images:
                                combo_images[k_code].extend(images)
        
        # 통계 계산
        all_k_codes = set(single_images.keys()) | set(combo_images.keys())
        k_code_counts = {}
        for k_code in all_k_codes:
            single_count = len(single_images.get(k_code, []))
            combo_count = len(combo_images.get(k_code, []))
            k_code_counts[k_code] = single_count + combo_count
        
        self.logger.info(f"스캔 완료: {len(all_k_codes)}개 K-코드, "
                        f"{sum(k_code_counts.values())}개 이미지")
        self.logger.info(f"Single: {sum(len(imgs) for imgs in single_images.values())}개")
        self.logger.info(f"Combo: {sum(len(imgs) for imgs in combo_images.values())}개")
        
        return {
            'single': dict(single_images),
            'combo': dict(combo_images),
            'all_k_codes': all_k_codes,
            'k_code_counts': k_code_counts
        }
    
    def select_target_classes(self, k_code_counts: Dict[str, int]) -> List[str]:
        """Stage 1용 50개 클래스 선택"""
        self.logger.info(f"목표 {self.strategy.target_classes}개 클래스 선택...")
        
        # 충분한 이미지가 있는 K-코드만 필터링
        valid_k_codes = [
            k_code for k_code, count in k_code_counts.items()
            if count >= self.strategy.min_images_per_class
        ]
        
        if len(valid_k_codes) < self.strategy.target_classes:
            raise ValueError(
                f"충분한 이미지가 있는 K-코드가 부족합니다. "
                f"필요: {self.strategy.target_classes}, 사용 가능: {len(valid_k_codes)}"
            )
        
        # 이미지 수 기준으로 정렬하여 균등 분포 유지
        sorted_k_codes = sorted(valid_k_codes, key=lambda k: k_code_counts[k], reverse=True)
        
        # 상위 클래스들을 균등하게 선택
        selected_classes = sorted_k_codes[:self.strategy.target_classes]
        
        self.logger.info(f"선택된 {len(selected_classes)}개 클래스:")
        for k_code in selected_classes[:10]:  # 상위 10개만 로깅
            self.logger.info(f"  {k_code}: {k_code_counts[k_code]}개 이미지")
        
        return selected_classes
    
    def validate_image_quality(self, image_path: Path) -> bool:
        """이미지 품질 검증"""
        try:
            with Image.open(image_path) as img:
                # 기본 검증: 파일 열기 가능, 최소 크기
                if img.size[0] < 32 or img.size[1] < 32:
                    return False
                
                # RGB 모드 확인 (변환 가능한지)
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    try:
                        img.convert('RGB')
                    except:
                        return False
                
                return True
        except Exception:
            return False
    
    def sample_images_for_class(
        self, 
        k_code: str, 
        single_images: List[Path], 
        combo_images: List[Path]
    ) -> Tuple[List[Path], List[Path]]:
        """특정 클래스에서 이미지 샘플링"""
        target_count = self.strategy.images_per_class
        
        # Single/Combo 비율 계산
        single_target = int(target_count * self.strategy.single_combo_ratio)
        combo_target = target_count - single_target
        
        # 품질 검증된 이미지만 필터링
        valid_single = [img for img in single_images if self.validate_image_quality(img)]
        valid_combo = [img for img in combo_images if self.validate_image_quality(img)]
        
        # 샘플링
        sampled_single = random.sample(
            valid_single, 
            min(single_target, len(valid_single))
        ) if valid_single else []
        
        sampled_combo = random.sample(
            valid_combo,
            min(combo_target, len(valid_combo))
        ) if valid_combo else []
        
        # 부족한 경우 다른 타입에서 보충
        total_sampled = len(sampled_single) + len(sampled_combo)
        if total_sampled < target_count:
            shortage = target_count - total_sampled
            
            # Single이 부족하면 Combo에서 보충
            if len(sampled_single) < single_target and valid_combo:
                additional_combo = min(shortage, len(valid_combo) - len(sampled_combo))
                if additional_combo > 0:
                    remaining_combo = [img for img in valid_combo if img not in sampled_combo]
                    sampled_combo.extend(random.sample(remaining_combo, additional_combo))
            
            # Combo가 부족하면 Single에서 보충
            elif len(sampled_combo) < combo_target and valid_single:
                additional_single = min(shortage, len(valid_single) - len(sampled_single))
                if additional_single > 0:
                    remaining_single = [img for img in valid_single if img not in sampled_single]
                    sampled_single.extend(random.sample(remaining_single, additional_single))
        
        return sampled_single, sampled_combo
    
    def generate_stage1_sample(self) -> Dict:
        """Stage 1 샘플 생성"""
        self.logger.info("Stage 1 샘플링 시작...")
        
        # 1. 데이터 스캔
        scan_result = self.scan_available_data()
        single_images = scan_result['single']
        combo_images = scan_result['combo']
        k_code_counts = scan_result['k_code_counts']
        
        # 2. 목표 클래스 선택
        selected_classes = self.select_target_classes(k_code_counts)
        
        # 3. 각 클래스별 이미지 샘플링
        stage1_sample = {
            'metadata': {
                'stage': 1,
                'strategy': asdict(self.strategy),
                'timestamp': pd.Timestamp.now().isoformat(),
                'selected_classes': selected_classes
            },
            'samples': {}
        }
        
        total_sampled_images = 0
        quality_pass_count = 0
        total_tested = 0
        
        for k_code in selected_classes:
            single_imgs = single_images.get(k_code, [])
            combo_imgs = combo_images.get(k_code, [])
            
            # 이미지 샘플링
            sampled_single, sampled_combo = self.sample_images_for_class(
                k_code, single_imgs, combo_imgs
            )
            
            # 품질 통계 업데이트
            all_sampled = sampled_single + sampled_combo
            for img_path in all_sampled:
                total_tested += 1
                if self.validate_image_quality(img_path):
                    quality_pass_count += 1
            
            total_sampled_images += len(all_sampled)
            
            # 결과 저장
            stage1_sample['samples'][k_code] = {
                'single_images': [str(p) for p in sampled_single],
                'combo_images': [str(p) for p in sampled_combo],
                'total_images': len(all_sampled),
                'single_count': len(sampled_single),
                'combo_count': len(sampled_combo)
            }
            
            self.logger.info(f"{k_code}: {len(all_sampled)}개 이미지 "
                           f"(Single: {len(sampled_single)}, Combo: {len(sampled_combo)})")
        
        # 통계 계산
        quality_pass_rate = quality_pass_count / total_tested if total_tested > 0 else 0
        single_total = sum(data['single_count'] for data in stage1_sample['samples'].values())
        combo_total = sum(data['combo_count'] for data in stage1_sample['samples'].values())
        
        stats = SamplingStats(
            total_images=sum(k_code_counts.values()),
            total_classes=len(k_code_counts),
            sampled_images=total_sampled_images,
            sampled_classes=len(selected_classes),
            images_per_class={k: data['total_images'] for k, data in stage1_sample['samples'].items()},
            single_pill_ratio=single_total / total_sampled_images if total_sampled_images > 0 else 0,
            combo_pill_ratio=combo_total / total_sampled_images if total_sampled_images > 0 else 0,
            quality_pass_rate=quality_pass_rate
        )
        
        stage1_sample['stats'] = asdict(stats)
        
        self.logger.info(f"Stage 1 샘플링 완료:")
        self.logger.info(f"  총 이미지: {total_sampled_images}개")
        self.logger.info(f"  총 클래스: {len(selected_classes)}개")
        self.logger.info(f"  품질 통과율: {quality_pass_rate:.2%}")
        self.logger.info(f"  Single/Combo 비율: {stats.single_pill_ratio:.1%}/{stats.combo_pill_ratio:.1%}")
        
        return stage1_sample
    
    def save_sample(self, sample_data: Dict, filename: str = "stage1_sample.json"):
        """샘플 데이터 저장"""
        output_path = self.artifacts_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Stage 1 샘플 저장 완료: {output_path}")
        return output_path


def validate_sample_distribution(sample_data: Dict) -> bool:
    """샘플 분포 유효성 검증"""
    logger = PillSnapLogger(__name__)
    
    try:
        stats = sample_data['stats']
        samples = sample_data['samples']
        
        # 기본 검증
        assert stats['sampled_classes'] == 50, f"클래스 수 불일치: {stats['sampled_classes']} != 50"
        assert stats['sampled_images'] == 5000, f"이미지 수 불일치: {stats['sampled_images']} != 5000"
        
        # 클래스별 이미지 수 검증
        for k_code, data in samples.items():
            assert 80 <= data['total_images'] <= 120, \
                f"{k_code}: 이미지 수 범위 벗어남 ({data['total_images']})"
        
        # 분포 균등성 검증
        image_counts = [data['total_images'] for data in samples.values()]
        std_dev = pd.Series(image_counts).std()
        assert std_dev < 10, f"클래스별 이미지 수 편차 과함: {std_dev:.2f}"
        
        # 품질 통과율 검증
        assert stats['quality_pass_rate'] >= 0.95, \
            f"품질 통과율 부족: {stats['quality_pass_rate']:.2%} < 95%"
        
        logger.info("✅ 샘플 분포 검증 통과")
        return True
        
    except AssertionError as e:
        logger.error(f"❌ 샘플 분포 검증 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 검증 중 오류: {e}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    from src.utils.core import load_config
    
    config = load_config()
    data_root = config.data.root
    
    strategy = Stage1SamplingStrategy()
    sampler = ProgressiveValidationSampler(data_root, strategy)
    
    # Stage 1 샘플 생성
    sample_data = sampler.generate_stage1_sample()
    
    # 샘플 저장
    output_path = sampler.save_sample(sample_data)
    
    # 검증
    is_valid = validate_sample_distribution(sample_data)
    print(f"샘플 검증 결과: {'✅ 통과' if is_valid else '❌ 실패'}")