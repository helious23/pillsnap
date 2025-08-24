"""
PillSnap ML 도메인 혼합 샘플러 (1단계 필수)

single:combination = 3:1 비율 유지:
- 별도 Dataset 생성 없이 샘플러/배치 콜레이터에서 비율 유지
- 평가 시 지표는 도메인 분리로 기록
- 경량 구현으로 개발 비용 최소화

RTX 5080 최적화
"""

import random
import math
from typing import Dict, Any, List, Tuple, Optional, Iterator
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler, Dataset
import pandas as pd

from src.utils.core import PillSnapLogger


@dataclass
class DomainMixConfig:
    """도메인 혼합 설정 (1단계 필수)"""
    
    # 도메인 비율 설정
    single_ratio: float = 0.75      # single:combination = 3:1
    combination_ratio: float = 0.25
    
    # 배치 구성 설정
    enforce_batch_ratio: bool = True    # 배치마다 비율 강제
    min_domain_samples: int = 1         # 배치 내 최소 도메인 샘플 수
    
    # 샘플링 전략
    shuffle_within_domain: bool = True  # 도메인 내 셔플
    resample_on_epoch: bool = True      # 에포크마다 리샘플링
    
    # 평가 분리 설정
    separate_domain_metrics: bool = True  # 도메인별 메트릭 분리
    
    # 로깅
    log_domain_stats: bool = True


class DomainMixedSampler(Sampler):
    """도메인 혼합 샘플러 (1단계 필수)"""
    
    def __init__(
        self,
        dataset: Dataset,
        config: DomainMixConfig,
        batch_size: int,
        domain_column: str = "image_type"  # Manifest에서 도메인을 구분하는 컬럼
    ):
        """
        Args:
            dataset: 데이터셋 (ManifestDataset 등)
            config: 도메인 혼합 설정
            batch_size: 배치 크기
            domain_column: 도메인 구분 컬럼명
        """
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size
        self.domain_column = domain_column
        self.logger = PillSnapLogger(__name__)
        
        # 도메인별 인덱스 분리
        self.domain_indices = self._build_domain_indices()
        
        # 배치 구성 계산
        self.single_per_batch = max(1, int(batch_size * config.single_ratio))
        self.combination_per_batch = max(1, batch_size - self.single_per_batch)
        
        # 실제 비율 조정 (배치 크기에 맞춤)
        actual_single_ratio = self.single_per_batch / batch_size
        actual_combination_ratio = self.combination_per_batch / batch_size
        
        self.logger.info(
            f"🎯 도메인 혼합 샘플러 초기화 - "
            f"single: {len(self.domain_indices['single'])}개 ({self.single_per_batch}/배치), "
            f"combination: {len(self.domain_indices['combination'])}개 ({self.combination_per_batch}/배치), "
            f"실제 비율: {actual_single_ratio:.3f}:{actual_combination_ratio:.3f}"
        )
        
        # 샘플링 상태
        self.epoch = 0
        
    def _build_domain_indices(self) -> Dict[str, List[int]]:
        """도메인별 인덱스 구축"""
        domain_indices = defaultdict(list)
        
        # ManifestDataset의 data DataFrame에서 도메인 정보 추출
        if hasattr(self.dataset, 'data') and isinstance(self.dataset.data, pd.DataFrame):
            df = self.dataset.data
            
            for idx, row in df.iterrows():
                domain = row.get(self.domain_column, 'single')  # 기본값은 single
                domain_indices[domain].append(idx)
        else:
            # Fallback: 전체를 single로 처리
            self.logger.warning(f"데이터셋에서 도메인 정보를 찾을 수 없음. 전체를 single로 처리.")
            for idx in range(len(self.dataset)):
                domain_indices['single'].append(idx)
        
        # 빈 도메인 처리
        if len(domain_indices['single']) == 0:
            self.logger.warning("Single 도메인 샘플이 없음")
            domain_indices['single'] = [0]  # 더미 인덱스
        
        if len(domain_indices['combination']) == 0:
            self.logger.warning("Combination 도메인 샘플이 없음")
            domain_indices['combination'] = [0]  # 더미 인덱스
        
        return dict(domain_indices)
    
    def __len__(self) -> int:
        """전체 샘플 수 (배치 개수 * 배치 크기)"""
        total_samples = len(self.domain_indices['single']) + len(self.domain_indices['combination'])
        num_batches = math.ceil(total_samples / self.batch_size)
        return num_batches * self.batch_size
    
    def __iter__(self) -> Iterator[int]:
        """도메인 혼합 샘플링 이터레이터"""
        # 에포크마다 리샘플링
        if self.config.resample_on_epoch:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random.seed(self.epoch)
        
        # 도메인별 인덱스 셔플
        single_indices = self.domain_indices['single'].copy()
        combination_indices = self.domain_indices['combination'].copy()
        
        if self.config.shuffle_within_domain:
            random.shuffle(single_indices)
            random.shuffle(combination_indices)
        
        # 배치 생성
        total_samples = len(self)
        num_batches = total_samples // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_indices = self._create_mixed_batch(
                single_indices, 
                combination_indices, 
                batch_idx
            )
            
            # 배치 내 셔플 (선택적)
            if self.config.shuffle_within_domain:
                random.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield idx
    
    def _create_mixed_batch(
        self, 
        single_indices: List[int], 
        combination_indices: List[int], 
        batch_idx: int
    ) -> List[int]:
        """도메인 혼합 배치 생성"""
        batch_indices = []
        
        # Single 샘플 선택 (순환)
        single_start = (batch_idx * self.single_per_batch) % len(single_indices)
        for i in range(self.single_per_batch):
            idx = (single_start + i) % len(single_indices)
            batch_indices.append(single_indices[idx])
        
        # Combination 샘플 선택 (순환)
        combination_start = (batch_idx * self.combination_per_batch) % len(combination_indices)
        for i in range(self.combination_per_batch):
            idx = (combination_start + i) % len(combination_indices)
            batch_indices.append(combination_indices[idx])
        
        return batch_indices
    
    def set_epoch(self, epoch: int) -> None:
        """에포크 설정 (셔플링용)"""
        self.epoch = epoch
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """도메인 통계 반환"""
        total_samples = sum(len(indices) for indices in self.domain_indices.values())
        
        stats = {
            "total_samples": total_samples,
            "domains": {},
            "batch_composition": {
                "single_per_batch": self.single_per_batch,
                "combination_per_batch": self.combination_per_batch,
                "batch_size": self.batch_size
            },
            "actual_ratios": {}
        }
        
        for domain, indices in self.domain_indices.items():
            domain_count = len(indices)
            domain_ratio = domain_count / total_samples if total_samples > 0 else 0
            
            stats["domains"][domain] = {
                "count": domain_count,
                "ratio": domain_ratio
            }
        
        # 배치 기준 실제 비율
        stats["actual_ratios"] = {
            "single": self.single_per_batch / self.batch_size,
            "combination": self.combination_per_batch / self.batch_size
        }
        
        return stats


class DomainMixedCollator:
    """도메인 혼합 배치 콜레이터 (1단계 필수)"""
    
    def __init__(self, config: DomainMixConfig):
        """
        Args:
            config: 도메인 혼합 설정
        """
        self.config = config
        self.logger = PillSnapLogger(__name__)
        
        # 도메인 통계 추적
        self.batch_count = 0
        self.domain_stats = defaultdict(int)
    
    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        """
        배치 콜레이션 및 도메인 정보 추가
        
        Args:
            batch: [(image, label, domain_info), ...] 형태의 배치
            
        Returns:
            Dict: 콜레이션된 배치 + 도메인 정보
        """
        if not batch:
            return {}
        
        # 기본 콜레이션
        images = []
        labels = []
        domain_labels = []
        
        for item in batch:
            if len(item) == 2:
                # (image, label) 형태
                image, label = item
                images.append(image)
                labels.append(label)
                domain_labels.append("single")  # 기본값
            elif len(item) == 3:
                # (image, label, domain_info) 형태
                image, label, domain_info = item
                images.append(image)
                labels.append(label)
                domain_labels.append(domain_info)
        
        # 텐서 변환
        batch_dict = {
            "images": torch.stack(images) if images else torch.empty(0),
            "labels": torch.tensor(labels) if labels else torch.empty(0, dtype=torch.long),
            "domains": domain_labels
        }
        
        # 도메인 통계 업데이트
        self.batch_count += 1
        for domain in domain_labels:
            self.domain_stats[domain] += 1
        
        # 도메인별 마스크 생성 (평가용)
        if self.config.separate_domain_metrics:
            batch_dict["domain_masks"] = self._create_domain_masks(domain_labels)
        
        # 주기적 통계 로깅
        if self.config.log_domain_stats and self.batch_count % 100 == 0:
            self._log_domain_statistics()
        
        return batch_dict
    
    def _create_domain_masks(self, domain_labels: List[str]) -> Dict[str, torch.Tensor]:
        """도메인별 마스크 생성 (평가시 사용)"""
        masks = {}
        
        unique_domains = set(domain_labels)
        for domain in unique_domains:
            mask = torch.tensor([d == domain for d in domain_labels], dtype=torch.bool)
            masks[domain] = mask
        
        return masks
    
    def _log_domain_statistics(self) -> None:
        """도메인 통계 로깅"""
        total_samples = sum(self.domain_stats.values())
        if total_samples == 0:
            return
        
        domain_ratios = {
            domain: count / total_samples 
            for domain, count in self.domain_stats.items()
        }
        
        self.logger.info(
            f"📊 도메인 통계 (배치 {self.batch_count}): "
            f"single {domain_ratios.get('single', 0):.3f}, "
            f"combination {domain_ratios.get('combination', 0):.3f} "
            f"(총 {total_samples}개 샘플)"
        )
    
    def reset_statistics(self) -> None:
        """통계 리셋"""
        self.batch_count = 0
        self.domain_stats.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """도메인 통계 반환"""
        total_samples = sum(self.domain_stats.values())
        
        return {
            "total_batches": self.batch_count,
            "total_samples": total_samples,
            "domain_counts": dict(self.domain_stats),
            "domain_ratios": {
                domain: count / total_samples if total_samples > 0 else 0
                for domain, count in self.domain_stats.items()
            }
        }


def create_domain_mixed_dataloader(
    dataset: Dataset,
    batch_size: int,
    config: Optional[DomainMixConfig] = None,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    도메인 혼합 데이터로더 생성 함수
    
    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        config: 도메인 혼합 설정
        num_workers: 워커 수
        **kwargs: 추가 DataLoader 인자
        
    Returns:
        DataLoader: 도메인 혼합 설정이 적용된 데이터로더
    """
    if config is None:
        config = DomainMixConfig()
    
    # 도메인 혼합 샘플러 생성
    sampler = DomainMixedSampler(dataset, config, batch_size)
    
    # 도메인 혼합 콜레이터 생성
    collator = DomainMixedCollator(config)
    
    # 데이터로더 생성
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    return dataloader


if __name__ == "__main__":
    print("🧪 도메인 혼합 샘플러 테스트 (1단계 필수)")
    print("=" * 60)
    
    # Mock 데이터셋 생성
    class MockDataset:
        def __init__(self, size: int):
            self.size = size
            # Mock manifest data
            domains = ['single'] * int(size * 0.8) + ['combination'] * int(size * 0.2)
            random.shuffle(domains)
            
            import pandas as pd
            self.data = pd.DataFrame({
                'image_path': [f'/fake/path/img_{i}.jpg' for i in range(size)],
                'image_type': domains,
                'mapping_code': [f'K{i:06d}' for i in range(size)]
            })
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            domain = self.data.iloc[idx]['image_type']
            return torch.randn(3, 224, 224), idx, domain
    
    # 테스트 설정
    config = DomainMixConfig(
        single_ratio=0.75,
        combination_ratio=0.25,
        enforce_batch_ratio=True
    )
    
    dataset = MockDataset(1000)
    sampler = DomainMixedSampler(dataset, config, batch_size=8)
    
    # 도메인 통계 확인
    stats = sampler.get_domain_statistics()
    print(f"✅ 데이터셋 통계: {stats['domains']}")
    print(f"✅ 배치 구성: single {stats['batch_composition']['single_per_batch']}, combination {stats['batch_composition']['combination_per_batch']}")
    
    # 샘플링 테스트
    sample_count = 0
    for batch_indices in sampler:
        sample_count += 1
        if sample_count >= 16:  # 2 배치만 테스트
            break
    
    print(f"✅ 샘플링 테스트 완료: {sample_count}개 샘플")
    
    # 콜레이터 테스트
    collator = DomainMixedCollator(config)
    mock_batch = [dataset[i] for i in range(8)]
    collated = collator(mock_batch)
    
    print(f"✅ 콜레이션 테스트: {collated['images'].shape}, domains {len(collated['domains'])}")
    if 'domain_masks' in collated:
        print(f"✅ 도메인 마스크: {list(collated['domain_masks'].keys())}")
    
    print("🎉 도메인 혼합 샘플러 테스트 완료!")