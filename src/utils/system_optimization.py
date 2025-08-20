"""
System Optimization Utilities
시스템 최적화 유틸리티

환경별 최적 설정 자동 감지:
- WSL vs Native Linux
- CPU/GPU 리소스 기반 최적화
- DataLoader num_workers 자동 조정
"""

import os
import platform
import multiprocessing as mp
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.core import PillSnapLogger


class SystemOptimizer:
    """시스템 환경 최적화 관리자"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self._setup_thread_limiting()
        self._system_info = self._detect_system()
        
    def _setup_thread_limiting(self):
        """OMP/MKL 스레드 제한 설정"""
        # WSL에서 워커 내부 스레드 과점 방지
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        self.logger.info("🔧 OMP/MKL 스레드 1로 제한 설정")
        
    def _detect_system(self) -> Dict[str, Any]:
        """시스템 환경 감지"""
        info = {
            'platform': platform.system(),
            'cpu_count': mp.cpu_count(),
            'is_wsl': False,
            'has_cuda': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # WSL 감지
        if info['platform'] == 'Linux':
            try:
                with open('/proc/version', 'r') as f:
                    version_info = f.read().lower()
                    if 'microsoft' in version_info or 'wsl' in version_info:
                        info['is_wsl'] = True
            except:
                pass
        
        # 추가 환경 정보
        info['memory_gb'] = self._get_available_memory_gb()
        info['optimal_workers'] = self._calculate_optimal_workers(info)
        
        return info
    
    def _get_available_memory_gb(self) -> float:
        """사용 가능한 메모리 (GB) 계산"""
        try:
            if torch.cuda.is_available():
                # GPU 메모리 우선
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                return gpu_memory / (1024**3)
            else:
                # 시스템 메모리
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            memory_kb = int(line.split()[1])
                            return memory_kb / (1024**2)
        except:
            return 16.0  # 기본값
        
        return 16.0
    
    def _calculate_optimal_workers(self, info: Dict[str, Any]) -> int:
        """최적 worker 수 계산"""
        cpu_count = info['cpu_count']
        
        # WSL 환경에서는 안정성 우선
        if info['is_wsl']:
            # WSL에서 멀티프로세싱 문제로 인해 비활성화
            # batch_size가 클수록 데드락 위험 증가
            return 0
        
        # Native Linux/Windows
        if cpu_count >= 16:
            return 8
        elif cpu_count >= 8:
            return 6
        elif cpu_count >= 4:
            return 4
        else:
            return max(1, cpu_count // 2)  # 최소 1개 보장
    
    def get_dataloader_config(self, stage: int = 1) -> Dict[str, Any]:
        """Stage별 DataLoader 최적 설정"""
        base_workers = self._system_info['optimal_workers']
        
        # Stage별 조정
        stage_multipliers = {
            1: 1.0,   # Stage 1: 기본
            2: 1.2,   # Stage 2: 약간 증가
            3: 1.5,   # Stage 3: 더 증가  
            4: 2.0    # Stage 4: 최대
        }
        
        multiplier = stage_multipliers.get(stage, 1.0)
        workers = min(8, max(1, int(base_workers * multiplier)))
        
        # WSL에서는 최대 4로 제한
        if self._system_info['is_wsl']:
            workers = min(4, workers)
        
        # WSL 환경에서는 안전성 우선 설정
        if self._system_info['is_wsl']:
            config = {
                'num_workers': workers,
                'pin_memory': False,  # WSL에서 안정성 우선 (data_time 증가 트레이드오프)
                'persistent_workers': False,  # WSL에서 문제 발생 가능
                'prefetch_factor': None,  # persistent_workers=False일 때는 None
                'drop_last': True,
                'multiprocessing_context': 'spawn'  # WSL에서 가장 안전
            }
        else:
            # Native 환경에서는 성능 우선
            config = {
                'num_workers': workers,
                'pin_memory': self._system_info['has_cuda'],
                'persistent_workers': workers > 0,
                'prefetch_factor': 2 if workers > 0 else None,
                'drop_last': True
            }
        
        return config
    
    def get_training_config(self, batch_size: int) -> Dict[str, Any]:
        """학습 최적화 설정"""
        config = {
            'mixed_precision': self._system_info['has_cuda'],
            'compile_model': self._system_info['has_cuda'] and not self._system_info['is_wsl'],
            'gradient_clipping': True,
            'channels_last': self._system_info['has_cuda']
        }
        
        return config
    
    def log_system_info(self):
        """시스템 정보 로깅"""
        info = self._system_info
        
        self.logger.info("🖥️  시스템 환경 감지 결과:")
        self.logger.info(f"   플랫폼: {info['platform']}")
        self.logger.info(f"   WSL 환경: {'예' if info['is_wsl'] else '아니오'}")
        self.logger.info(f"   CPU 코어: {info['cpu_count']}개")
        self.logger.info(f"   CUDA 사용: {'예' if info['has_cuda'] else '아니오'}")
        if info['has_cuda']:
            self.logger.info(f"   GPU 개수: {info['gpu_count']}개")
        self.logger.info(f"   메모리: {info['memory_gb']:.1f}GB")
        self.logger.info(f"   최적 Workers: {info['optimal_workers']}개")
        
        # WSL 특별 경고
        if info['is_wsl']:
            self.logger.warning("⚠️  WSL 환경에서 멀티프로세싱 최적화 적용")


# 전역 인스턴스
system_optimizer = SystemOptimizer()


def get_optimal_num_workers(stage: int = 1) -> int:
    """Stage별 최적 num_workers 반환"""
    return system_optimizer.get_dataloader_config(stage)['num_workers']


def get_dataloader_kwargs(stage: int = 1) -> Dict[str, Any]:
    """DataLoader 생성용 최적 kwargs 반환"""
    return system_optimizer.get_dataloader_config(stage)


def log_system_optimization():
    """시스템 최적화 정보 로깅"""
    system_optimizer.log_system_info()