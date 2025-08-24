#!/usr/bin/env python3
"""
PillSnap ML Self-Check System
시작 시 환경 및 설정 검증 시스템
"""

import os
import sys
import torch
import json
import yaml
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.core import config_provider, KST


class SelfCheckSystem:
    """
    학습 시작 전 환경 및 설정 검증
    - GPU 메모리 충분성 확인
    - 의존성 버전 체크
    - Manifest 파일 존재 검증
    - 디스크 공간 확인
    - 설정 일관성 검증
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리 (None이면 ConfigProvider에서 로드)
        """
        if config is None:
            config = config_provider.get_config()
        
        self.config = config
        self.check_results = []
        self.has_errors = False
        self.has_warnings = False
        
    def run_all_checks(self) -> bool:
        """
        모든 체크 실행
        
        Returns:
            bool: 모든 체크 통과 여부
        """
        print("=" * 60)
        print("🔍 PillSnap ML Self-Check System")
        print("=" * 60)
        
        # 1. GPU 체크
        self._check_gpu()
        
        # 2. 의존성 체크
        self._check_dependencies()
        
        # 3. Manifest 파일 체크
        self._check_manifest_files()
        
        # 4. 디스크 공간 체크
        self._check_disk_space()
        
        # 5. 설정 일관성 체크
        self._check_config_consistency()
        
        # 6. 체크포인트 체크
        self._check_checkpoints()
        
        # 결과 출력
        self._print_results()
        
        return not self.has_errors
    
    def _check_gpu(self) -> None:
        """GPU 체크"""
        try:
            if not torch.cuda.is_available():
                self._add_error("GPU를 사용할 수 없습니다")
                return
            
            device_count = torch.cuda.device_count()
            self._add_success(f"GPU {device_count}개 감지됨")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                
                # 현재 사용 중인 메모리
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                self._add_info(f"  GPU {i}: {props.name}")
                self._add_info(f"    - 총 메모리: {total_memory:.1f}GB")
                self._add_info(f"    - 할당된 메모리: {allocated:.1f}GB")
                self._add_info(f"    - 예약된 메모리: {reserved:.1f}GB")
                
                # 메모리 충분성 체크 (배치 크기 기준)
                batch_size = self.config.get('train', {}).get('batch_size', 8)
                required_memory = batch_size * 1.5  # 대략적인 추정 (GB)
                
                if total_memory - reserved < required_memory:
                    self._add_warning(f"GPU {i} 메모리 부족 가능성 (필요: {required_memory:.1f}GB)")
                    
        except Exception as e:
            self._add_error(f"GPU 체크 실패: {e}")
    
    def _check_dependencies(self) -> None:
        """의존성 버전 체크"""
        try:
            import torch
            import torchvision
            import timm
            import ultralytics
            
            self._add_success("핵심 의존성 체크 완료")
            self._add_info(f"  - PyTorch: {torch.__version__}")
            self._add_info(f"  - TorchVision: {torchvision.__version__}")
            self._add_info(f"  - Timm: {timm.__version__}")
            self._add_info(f"  - Ultralytics: {ultralytics.__version__}")
            
            # CUDA 버전 체크
            if torch.cuda.is_available():
                self._add_info(f"  - CUDA: {torch.version.cuda}")
                self._add_info(f"  - cuDNN: {torch.backends.cudnn.version()}")
                
        except ImportError as e:
            self._add_error(f"의존성 누락: {e}")
    
    def _check_manifest_files(self) -> None:
        """Manifest 파일 존재 및 유효성 체크"""
        stage = self.config.get('stage', 3)
        
        manifest_paths = {
            'train': f"/home/max16/pillsnap/artifacts/stage{stage}/manifest_train.csv",
            'val': f"/home/max16/pillsnap/artifacts/stage{stage}/manifest_val.csv"
        }
        
        for split, path in manifest_paths.items():
            path_obj = Path(path)
            
            if not path_obj.exists():
                self._add_error(f"Manifest 파일 없음: {path}")
                continue
            
            # 파일 크기 체크
            size_mb = path_obj.stat().st_size / (1024**2)
            
            # 라인 수 체크 (샘플 수)
            try:
                with open(path, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # 헤더 제외
                
                self._add_success(f"Manifest {split}: {line_count:,}개 샘플 ({size_mb:.1f}MB)")
                
                # Stage별 최소 샘플 수 체크
                min_samples = {1: 4000, 2: 20000, 3: 80000, 4: 400000}
                if stage in min_samples and line_count < min_samples[stage]:
                    self._add_warning(f"Stage {stage} {split} 샘플 수 부족: {line_count:,} < {min_samples[stage]:,}")
                    
            except Exception as e:
                self._add_error(f"Manifest 파일 읽기 실패: {e}")
    
    def _check_disk_space(self) -> None:
        """디스크 공간 체크"""
        try:
            import shutil
            
            paths_to_check = [
                ("/home/max16/pillsnap", "코드베이스"),
                ("/home/max16/pillsnap_data", "데이터셋"),
                ("/tmp", "임시 파일")
            ]
            
            for path, name in paths_to_check:
                if os.path.exists(path):
                    total, used, free = shutil.disk_usage(path)
                    free_gb = free / (1024**3)
                    used_percent = (used / total) * 100
                    
                    self._add_info(f"  {name} ({path}):")
                    self._add_info(f"    - 여유 공간: {free_gb:.1f}GB ({100-used_percent:.1f}%)")
                    
                    # 경고 임계값
                    if free_gb < 10:
                        self._add_warning(f"{name} 디스크 공간 부족: {free_gb:.1f}GB")
                    elif free_gb < 50:
                        self._add_info(f"    ⚠️  디스크 공간 주의 필요")
                        
        except Exception as e:
            self._add_warning(f"디스크 공간 체크 실패: {e}")
    
    def _check_config_consistency(self) -> None:
        """설정 일관성 체크"""
        try:
            # num_classes 일관성
            num_classes_config = self.config.get('num_classes', 5000)
            num_classes_model = self.config.get('models', {}).get('classifier', {}).get('num_classes', 5000)
            
            if num_classes_config != num_classes_model:
                self._add_error(f"num_classes 불일치: config={num_classes_config}, model={num_classes_model}")
            else:
                self._add_success(f"num_classes 일관성 확인: {num_classes_config}")
            
            # 배치 크기 vs GPU 메모리
            batch_size = self.config.get('train', {}).get('batch_size', 8)
            if batch_size > 16:
                self._add_warning(f"배치 크기가 큽니다: {batch_size} (OOM 위험)")
            
            # Learning rate 범위 체크
            lr = self.config.get('train', {}).get('lr', 1e-4)
            if lr > 1e-2:
                self._add_warning(f"Learning rate가 너무 높습니다: {lr}")
            elif lr < 1e-6:
                self._add_warning(f"Learning rate가 너무 낮습니다: {lr}")
                
        except Exception as e:
            self._add_error(f"설정 일관성 체크 실패: {e}")
    
    def _check_checkpoints(self) -> None:
        """체크포인트 파일 체크"""
        checkpoint_dir = Path("/home/max16/pillsnap_data/exp/exp01/checkpoints")
        
        if not checkpoint_dir.exists():
            self._add_warning("체크포인트 디렉토리가 없습니다")
            return
        
        # 체크포인트 파일 목록
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            self._add_info("  체크포인트 파일 없음 (새로운 학습)")
        else:
            self._add_success(f"체크포인트 {len(checkpoints)}개 발견")
            
            # 최신 체크포인트 정보
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            size_mb = latest.stat().st_size / (1024**2)
            
            self._add_info(f"  최신: {latest.name} ({size_mb:.1f}MB)")
            
            # 체크포인트 크기 체크 (너무 작으면 문제)
            if size_mb < 100:
                self._add_warning(f"체크포인트 크기가 작습니다: {size_mb:.1f}MB (손상 가능성)")
    
    def _add_success(self, message: str) -> None:
        """성공 메시지 추가"""
        self.check_results.append(("SUCCESS", message))
    
    def _add_info(self, message: str) -> None:
        """정보 메시지 추가"""
        self.check_results.append(("INFO", message))
    
    def _add_warning(self, message: str) -> None:
        """경고 메시지 추가"""
        self.check_results.append(("WARNING", message))
        self.has_warnings = True
    
    def _add_error(self, message: str) -> None:
        """에러 메시지 추가"""
        self.check_results.append(("ERROR", message))
        self.has_errors = True
    
    def _print_results(self) -> None:
        """결과 출력"""
        print("\n" + "=" * 60)
        print("📋 Self-Check 결과")
        print("=" * 60)
        
        # 색상 코드
        colors = {
            "SUCCESS": "\033[92m✅",
            "INFO": "\033[94mℹ️ ",
            "WARNING": "\033[93m⚠️ ",
            "ERROR": "\033[91m❌"
        }
        reset = "\033[0m"
        
        for level, message in self.check_results:
            prefix = colors.get(level, "")
            print(f"{prefix} {message}{reset}")
        
        print("=" * 60)
        
        # 최종 상태
        if self.has_errors:
            print(f"{colors['ERROR']} Self-Check 실패: 에러를 해결하세요{reset}")
        elif self.has_warnings:
            print(f"{colors['WARNING']} Self-Check 완료: 경고 사항 확인 필요{reset}")
        else:
            print(f"{colors['SUCCESS']} Self-Check 완료: 모든 체크 통과{reset}")
        
        print("=" * 60)


def run_self_check(config: Optional[Dict] = None) -> bool:
    """
    Self-check 실행 헬퍼 함수
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        bool: 체크 통과 여부
    """
    checker = SelfCheckSystem(config)
    return checker.run_all_checks()


if __name__ == "__main__":
    # 독립 실행 테스트
    import argparse
    
    parser = argparse.ArgumentParser(description="PillSnap ML Self-Check System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--stage", type=int, default=3, help="Stage number (1-4)")
    args = parser.parse_args()
    
    # ConfigProvider 초기화
    config_provider.load(args.config)
    
    # Stage 설정
    if args.stage:
        config_provider.set("stage", args.stage)
    
    # Self-check 실행
    success = run_self_check()
    
    # 종료 코드
    sys.exit(0 if success else 1)