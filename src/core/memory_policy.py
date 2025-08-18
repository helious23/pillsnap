"""
Memory Policy Stage Locks - Stage-specific Memory Optimization
메모리 정책 단계별 잠금 - Stage별 차별화된 메모리 최적화
"""

import psutil
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """캐시 정책"""
    LABELS_ONLY = "labels_only"      # 라벨만 캐시
    HOTSET = "hotset"                # 핫셋 이미지 캐시
    FULL_CACHE = "full_cache"        # 전체 캐시
    LMDB = "lmdb"                    # LMDB 변환
    NO_CACHE = "no_cache"            # 캐시 없음


@dataclass
class StageMemoryConfig:
    """Stage별 메모리 설정"""
    stage: int
    cache_policy: CachePolicy
    hotset_size: int
    prefetch_factor: int
    num_workers: int
    use_lmdb: bool
    preload_samples: int
    pin_memory: bool
    pin_memory_device: str
    
    # 메모리 할당 계획 (GB)
    reserved_os: float = 8.0
    reserved_cache: float = 0.0
    reserved_prefetch: float = 0.0
    reserved_workers: float = 0.0
    
    def get_total_reserved(self) -> float:
        """총 예약 메모리 계산"""
        return (self.reserved_os + self.reserved_cache + 
                self.reserved_prefetch + self.reserved_workers)


class MemoryPolicyManager:
    """
    Stage별 메모리 정책 관리
    128GB RAM 최적 활용을 위한 단계별 전략
    """
    
    # Stage별 기본 설정
    STAGE_CONFIGS = {
        1: {  # 파이프라인 검증 (5K 샘플)
            "cache_policy": CachePolicy.LABELS_ONLY,
            "hotset_size": 0,
            "prefetch_factor": 4,
            "num_workers": 8,
            "use_lmdb": False,
            "preload_samples": 0
        },
        2: {  # 성능 기준선 (25K 샘플)
            "cache_policy": CachePolicy.HOTSET,
            "hotset_size": 20000,
            "prefetch_factor": 6,
            "num_workers": 12,
            "use_lmdb": False,
            "preload_samples": 1000
        },
        3: {  # 확장성 테스트 (100K 샘플)
            "cache_policy": CachePolicy.HOTSET,
            "hotset_size": 40000,
            "prefetch_factor": 8,
            "num_workers": 16,
            "use_lmdb": True,
            "preload_samples": 5000
        },
        4: {  # 최종 프로덕션 (500K 샘플)
            "cache_policy": CachePolicy.HOTSET,
            "hotset_size": 60000,
            "prefetch_factor": 8,
            "num_workers": 16,
            "use_lmdb": True,
            "preload_samples": 10000
        }
    }
    
    def __init__(self, config: Dict[str, Any], total_ram_gb: int = 128):
        """
        Args:
            config: 설정 딕셔너리
            total_ram_gb: 전체 RAM 크기 (GB)
        """
        self.config = config
        self.total_ram_gb = total_ram_gb
        self.current_stage = config.get("data", {}).get("progressive_validation", {}).get("current_stage", 1)
        self.stage_config = self._build_stage_config()
        self.memory_stats = {}
        
        logger.info(f"MemoryPolicyManager initialized for Stage {self.current_stage}")
        logger.info(f"Total RAM: {total_ram_gb}GB, Policy: {self.stage_config.cache_policy.value}")
    
    def _build_stage_config(self) -> StageMemoryConfig:
        """Stage별 설정 구축"""
        base_config = self.STAGE_CONFIGS.get(self.current_stage, self.STAGE_CONFIGS[1])
        
        # 설정 오버라이드 적용
        dataloader_cfg = self.config.get("dataloader", {})
        ram_opt_cfg = dataloader_cfg.get("ram_optimization", {})
        stage_override = dataloader_cfg.get("stage_overrides", {}).get(str(self.current_stage), {})
        
        # 기본값 병합
        cache_policy = ram_opt_cfg.get("cache_policy", base_config["cache_policy"].value)
        if isinstance(cache_policy, str):
            cache_policy = CachePolicy(cache_policy)
        
        config = StageMemoryConfig(
            stage=self.current_stage,
            cache_policy=cache_policy,
            hotset_size=ram_opt_cfg.get("hotset_size_images", base_config["hotset_size"]),
            prefetch_factor=dataloader_cfg.get("prefetch_factor", base_config["prefetch_factor"]),
            num_workers=dataloader_cfg.get("num_workers", base_config["num_workers"]),
            use_lmdb=ram_opt_cfg.get("use_lmdb", base_config["use_lmdb"]),
            preload_samples=ram_opt_cfg.get("preload_samples", base_config["preload_samples"]),
            pin_memory=dataloader_cfg.get("pin_memory", True),
            pin_memory_device=dataloader_cfg.get("pin_memory_device", "cuda")
        )
        
        # Stage별 오버라이드 적용
        if stage_override:
            if "ram_optimization" in stage_override:
                stage_ram = stage_override["ram_optimization"]
                config.hotset_size = stage_ram.get("hotset_size_images", config.hotset_size)
            if "dataloader" in stage_override:
                stage_dl = stage_override["dataloader"]
                config.num_workers = stage_dl.get("num_workers", config.num_workers)
                config.prefetch_factor = stage_dl.get("prefetch_factor", config.prefetch_factor)
        
        # 메모리 예약 계산
        config = self._calculate_memory_reservations(config)
        
        return config
    
    def _calculate_memory_reservations(self, config: StageMemoryConfig) -> StageMemoryConfig:
        """메모리 예약량 계산"""
        # 기본 OS 예약
        config.reserved_os = 8.0
        
        # 캐시 메모리 계산 (이미지당 평균 크기 가정)
        if config.cache_policy == CachePolicy.HOTSET:
            # 384x384x3 ≈ 0.44MB per image (uint8)
            image_size_mb = 0.44
            config.reserved_cache = (config.hotset_size * image_size_mb) / 1024  # GB
        elif config.cache_policy == CachePolicy.FULL_CACHE:
            # Stage에 따른 전체 캐시 크기 추정
            stage_samples = {1: 5000, 2: 25000, 3: 100000, 4: 500000}
            num_samples = stage_samples.get(config.stage, 5000)
            image_size_mb = 0.44
            config.reserved_cache = min((num_samples * image_size_mb) / 1024, 80.0)  # 최대 80GB
        else:
            config.reserved_cache = 2.0  # 라벨 캐시용
        
        # Prefetch 버퍼 메모리
        # 배치당 2GB 가정, prefetch_factor 개수만큼
        config.reserved_prefetch = min(config.prefetch_factor * 2.0, 16.0)
        
        # 워커 메모리 (워커당 1GB)
        config.reserved_workers = min(config.num_workers * 1.0, 16.0)
        
        return config
    
    def validate_memory_requirements(self) -> Tuple[bool, str]:
        """
        메모리 요구사항 검증
        
        Returns:
            (유효 여부, 메시지)
        """
        required = self.stage_config.get_total_reserved()
        available = self._get_available_memory()
        
        if required > available:
            msg = (f"Memory requirement exceeds available: "
                   f"required={required:.1f}GB, available={available:.1f}GB")
            logger.warning(msg)
            return False, msg
        
        utilization = (required / self.total_ram_gb) * 100
        msg = (f"Memory validation passed: "
               f"required={required:.1f}GB, utilization={utilization:.1f}%")
        logger.info(msg)
        
        return True, msg
    
    def _get_available_memory(self) -> float:
        """사용 가능한 메모리 계산 (GB)"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        return available_gb
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """DataLoader용 설정 반환"""
        return {
            "num_workers": self.stage_config.num_workers,
            "prefetch_factor": self.stage_config.prefetch_factor,
            "pin_memory": self.stage_config.pin_memory,
            "pin_memory_device": self.stage_config.pin_memory_device,
            "persistent_workers": True,
            "drop_last": True
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """캐시 설정 반환"""
        return {
            "cache_policy": self.stage_config.cache_policy.value,
            "hotset_size": self.stage_config.hotset_size,
            "use_lmdb": self.stage_config.use_lmdb,
            "preload_samples": self.stage_config.preload_samples
        }
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 모니터링"""
        mem = psutil.virtual_memory()
        
        stats = {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent,
            "reserved_cache_gb": self.stage_config.reserved_cache,
            "reserved_prefetch_gb": self.stage_config.reserved_prefetch,
            "reserved_workers_gb": self.stage_config.reserved_workers
        }
        
        # GPU 메모리도 모니터링
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            stats["gpu_allocated_gb"] = gpu_allocated
            stats["gpu_reserved_gb"] = gpu_reserved
        
        self.memory_stats = stats
        return stats
    
    def suggest_optimization(self) -> List[str]:
        """최적화 제안"""
        suggestions = []
        stats = self.monitor_memory_usage()
        
        # 메모리 사용률이 높은 경우
        if stats["percent"] > 85:
            suggestions.append("High memory usage detected. Consider reducing cache size or workers.")
            
            if self.stage_config.cache_policy == CachePolicy.HOTSET:
                suggestions.append(f"Reduce hotset_size from {self.stage_config.hotset_size}")
            
            if self.stage_config.num_workers > 8:
                suggestions.append(f"Reduce num_workers from {self.stage_config.num_workers} to 8")
        
        # 메모리 사용률이 낮은 경우
        elif stats["percent"] < 50 and self.current_stage >= 2:
            suggestions.append("Low memory usage. Consider increasing cache for better performance.")
            
            if self.stage_config.cache_policy == CachePolicy.LABELS_ONLY:
                suggestions.append("Enable hotset caching for frequently accessed images")
            
            if not self.stage_config.use_lmdb and self.current_stage >= 3:
                suggestions.append("Enable LMDB for large dataset I/O optimization")
        
        # GPU 메모리 관련
        if "gpu_allocated_gb" in stats and stats["gpu_allocated_gb"] > 14:
            suggestions.append("GPU memory usage high. Consider reducing batch size.")
        
        return suggestions
    
    def apply_stage_transition(self, new_stage: int) -> StageMemoryConfig:
        """
        Stage 전환 시 메모리 정책 업데이트
        
        Args:
            new_stage: 새로운 Stage 번호
            
        Returns:
            새로운 Stage 설정
        """
        logger.info(f"Transitioning from Stage {self.current_stage} to Stage {new_stage}")
        
        # 기존 캐시 정리 (필요시)
        if self.stage_config.cache_policy == CachePolicy.FULL_CACHE:
            logger.info("Clearing previous stage cache")
            # 캐시 정리 로직
        
        # 새 Stage 설정
        self.current_stage = new_stage
        self.stage_config = self._build_stage_config()
        
        # 메모리 요구사항 재검증
        valid, msg = self.validate_memory_requirements()
        if not valid:
            logger.warning(f"Stage {new_stage} memory requirements may be challenging: {msg}")
        
        return self.stage_config
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Stage 설정 요약"""
        return {
            "stage": self.current_stage,
            "cache_policy": self.stage_config.cache_policy.value,
            "hotset_size": self.stage_config.hotset_size,
            "num_workers": self.stage_config.num_workers,
            "prefetch_factor": self.stage_config.prefetch_factor,
            "use_lmdb": self.stage_config.use_lmdb,
            "memory_reserved_gb": self.stage_config.get_total_reserved(),
            "memory_breakdown": {
                "os": self.stage_config.reserved_os,
                "cache": self.stage_config.reserved_cache,
                "prefetch": self.stage_config.reserved_prefetch,
                "workers": self.stage_config.reserved_workers
            }
        }


def create_memory_policy(config: Dict[str, Any]) -> MemoryPolicyManager:
    """
    메모리 정책 매니저 생성 헬퍼
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        MemoryPolicyManager 인스턴스
    """
    # 실제 RAM 크기 감지
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # 128GB 시스템 확인
    if total_ram_gb < 120:  # 여유 마진
        logger.warning(f"System RAM ({total_ram_gb:.1f}GB) is less than expected 128GB")
    
    return MemoryPolicyManager(config, int(total_ram_gb))