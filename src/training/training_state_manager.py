"""
Training State Manager
학습 상태 관리 시스템

상업용 수준의 학습 상태 관리:
- 체크포인트 저장/로드
- 학습 재개 (Resume Training)
- 학습 히스토리 추적
- 배포용 모델 패키징
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.utils.core import PillSnapLogger


@dataclass
class TrainingState:
    """학습 상태 데이터 클래스"""
    epoch: int
    best_metric: float
    best_epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    scaler_state_dict: Optional[Dict[str, Any]]
    training_config: Dict[str, Any]
    timestamp: str


@dataclass
class ModelMetadata:
    """모델 메타데이터"""
    model_name: str
    stage: int
    num_classes: int
    input_size: tuple
    best_metric_value: float
    best_metric_name: str
    training_samples: int
    validation_samples: int
    training_time_hours: float
    created_timestamp: str


class TrainingStateManager:
    """학습 상태 관리자"""
    
    def __init__(
        self,
        experiment_name: str,
        stage: int = 1,
        checkpoint_dir: str = "artifacts/checkpoints",
        save_every_n_epochs: int = 5,
        keep_last_n_checkpoints: int = 3
    ):
        self.experiment_name = experiment_name
        self.stage = stage
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.logger = PillSnapLogger(__name__)
        
        # 실험별 디렉토리 생성
        self.experiment_dir = self.checkpoint_dir / f"stage{stage}" / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 상태 추적
        self.training_start_time = None
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.training_history = []
        
        self.logger.info(f"TrainingStateManager 초기화: {experiment_name} (Stage {stage})")
        self.logger.info(f"체크포인트 디렉토리: {self.experiment_dir}")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[Any] = None,
        metric_value: float = 0.0,
        is_best: bool = False,
        training_config: Optional[Dict] = None
    ) -> str:
        """체크포인트 저장"""
        
        try:
            # 최고 성능 업데이트 (상태 생성 전에 수행)
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.best_epoch = epoch
                is_best = True
            
            # 상태 생성
            state = TrainingState(
                epoch=epoch,
                best_metric=self.best_metric,
                best_epoch=self.best_epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                scaler_state_dict=scaler.state_dict() if scaler else None,
                training_config=training_config or {},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # 파일명 생성
            checkpoint_file = self.experiment_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            best_file = self.experiment_dir / "best_model.pt"
            latest_file = self.experiment_dir / "latest_checkpoint.pt"
            
            # 체크포인트 저장
            torch.save(asdict(state), checkpoint_file)
            torch.save(asdict(state), latest_file)
            
            # 최고 성능 모델 저장
            if is_best:
                torch.save(asdict(state), best_file)
                self.logger.success(f"🏆 새로운 최고 성능 모델 저장: {metric_value:.4f}")
            
            # 메타데이터 저장
            self._save_metadata(epoch, metric_value)
            
            # 오래된 체크포인트 정리
            self._cleanup_old_checkpoints()
            
            self.logger.info(f"체크포인트 저장: Epoch {epoch}")
            return str(checkpoint_file)
            
        except Exception as e:
            self.logger.error(f"체크포인트 저장 실패: {e}")
            raise
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """체크포인트 로드"""
        
        try:
            # 체크포인트 파일 결정
            if checkpoint_path:
                ckpt_file = Path(checkpoint_path)
            elif load_best:
                ckpt_file = self.experiment_dir / "best_model.pt"
            else:
                ckpt_file = self.experiment_dir / "latest_checkpoint.pt"
            
            if not ckpt_file.exists():
                raise FileNotFoundError(f"체크포인트 파일 없음: {ckpt_file}")
            
            # 체크포인트 로드
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            
            # 모델 상태 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 옵티마이저 상태 로드
            if optimizer and checkpoint.get('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 스케줄러 상태 로드
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 스케일러 상태 로드
            if scaler and checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 상태 복원
            self.best_metric = checkpoint.get('best_metric', -float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            
            resume_info = {
                'epoch': checkpoint['epoch'],
                'best_metric': checkpoint.get('best_metric', -float('inf')),  # 체크포인트에서 직접 가져오기
                'best_epoch': checkpoint.get('best_epoch', 0),
                'training_config': checkpoint.get('training_config', {}),
                'timestamp': checkpoint.get('timestamp', 'Unknown')
            }
            
            self.logger.success(f"체크포인트 로드 완료: {ckpt_file}")
            self.logger.info(f"  에포크: {resume_info['epoch']}")
            self.logger.info(f"  최고 성능: {self.best_metric:.4f}")
            
            return resume_info
            
        except Exception as e:
            self.logger.error(f"체크포인트 로드 실패: {e}")
            raise
    
    def save_final_model_for_deployment(
        self,
        model: nn.Module,
        model_metadata: ModelMetadata,
        include_training_artifacts: bool = False
    ) -> str:
        """배포용 최종 모델 저장"""
        
        try:
            deploy_dir = self.experiment_dir / "deployment"
            deploy_dir.mkdir(exist_ok=True)
            
            # 모델만 저장 (추론용)
            model_file = deploy_dir / "model_inference_only.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': asdict(model_metadata),
                'inference_only': True
            }, model_file)
            
            # 메타데이터 JSON 저장
            metadata_file = deploy_dir / "model_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(model_metadata), f, ensure_ascii=False, indent=2)
            
            # 학습 아티팩트 포함 버전
            if include_training_artifacts:
                full_file = deploy_dir / "model_with_training_artifacts.pt"
                best_model_path = self.experiment_dir / "best_model.pt"
                if best_model_path.exists():
                    best_checkpoint = torch.load(best_model_path)
                    torch.save(best_checkpoint, full_file)
                else:
                    self.logger.warning("best_model.pt가 없어 학습 아티팩트 포함 버전 생성을 건너뜁니다")
            
            # 배포 가이드 생성
            self._create_deployment_guide(deploy_dir, model_metadata)
            
            self.logger.success(f"배포용 모델 저장 완료: {deploy_dir}")
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"배포용 모델 저장 실패: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        
        try:
            # 메타데이터 파일 확인
            metadata_file = self.experiment_dir / "training_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # 체크포인트 파일들 확인
            checkpoint_files = list(self.experiment_dir.glob("checkpoint_*.pt"))
            
            summary = {
                'experiment_name': self.experiment_name,
                'stage': self.stage,
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
                'total_checkpoints': len(checkpoint_files),
                'metadata': metadata,
                'has_best_model': (self.experiment_dir / "best_model.pt").exists(),
                'has_deployment_ready': (self.experiment_dir / "deployment").exists(),
                'experiment_dir': str(self.experiment_dir)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"학습 요약 생성 실패: {e}")
            return {}
    
    def _save_metadata(self, epoch: int, metric_value: float) -> None:
        """메타데이터 저장"""
        try:
            metadata = {
                'experiment_name': self.experiment_name,
                'stage': self.stage,
                'current_epoch': epoch,
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
                'current_metric': metric_value,
                'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_training_time_hours': self._get_training_time_hours()
            }
            
            metadata_file = self.experiment_dir / "training_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.warning(f"메타데이터 저장 실패: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """오래된 체크포인트 정리"""
        try:
            checkpoint_files = sorted(
                self.experiment_dir.glob("checkpoint_*.pt"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # 최신 N개만 유지
            for old_file in checkpoint_files[self.keep_last_n_checkpoints:]:
                old_file.unlink()
                self.logger.debug(f"오래된 체크포인트 삭제: {old_file.name}")
                
        except Exception as e:
            self.logger.warning(f"체크포인트 정리 실패: {e}")
    
    def _get_training_time_hours(self) -> float:
        """학습 시간 계산"""
        if self.training_start_time:
            return (time.time() - self.training_start_time) / 3600
        return 0.0
    
    def _create_deployment_guide(self, deploy_dir: Path, metadata: ModelMetadata) -> None:
        """배포 가이드 생성"""
        try:
            guide_content = f"""# 모델 배포 가이드

## 모델 정보
- 모델명: {metadata.model_name}
- Stage: {metadata.stage}
- 클래스 수: {metadata.num_classes}
- 입력 크기: {metadata.input_size}
- 최고 성능: {metadata.best_metric_value:.4f}

## 파일 설명
- `model_inference_only.pt`: 추론 전용 모델 (권장)
- `model_metadata.json`: 모델 메타데이터
- `model_with_training_artifacts.pt`: 학습 정보 포함 (디버깅용)

## 사용 예시
```python
import torch

# 모델 로드
checkpoint = torch.load('model_inference_only.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 추론
with torch.no_grad():
    output = model(input_tensor)
```

## 요구사항
- PyTorch >= 2.0
- CUDA 지원 (RTX 5080 최적화)
- 메모리: 최소 2GB, 권장 4GB

생성 시간: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            guide_file = deploy_dir / "deployment_guide.md"
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide_content)
                
        except Exception as e:
            self.logger.warning(f"배포 가이드 생성 실패: {e}")
    
    def start_training_timer(self) -> None:
        """학습 시작 시간 기록"""
        self.training_start_time = time.time()
        self.logger.info("학습 타이머 시작")


def main():
    """TrainingStateManager 테스트"""
    print("🔧 Training State Manager Test")
    print("=" * 50)
    
    try:
        # 더미 모델 및 옵티마이저
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 상태 관리자 생성
        state_manager = TrainingStateManager(
            experiment_name="test_experiment",
            stage=1,
            checkpoint_dir="test_checkpoints"
        )
        
        state_manager.start_training_timer()
        
        # 체크포인트 저장 테스트
        checkpoint_path = state_manager.save_checkpoint(
            epoch=10,
            model=model,
            optimizer=optimizer,
            metric_value=0.85,
            is_best=True,
            training_config={'lr': 0.001, 'batch_size': 32}
        )
        
        print(f"✅ 체크포인트 저장: {checkpoint_path}")
        
        # 체크포인트 로드 테스트
        resume_info = state_manager.load_checkpoint(model, optimizer, load_best=True)
        print(f"✅ 체크포인트 로드: Epoch {resume_info['epoch']}")
        
        # 배포용 모델 저장
        metadata = ModelMetadata(
            model_name="test_model",
            stage=1,
            num_classes=5,
            input_size=(10,),
            best_metric_value=0.85,
            best_metric_name="accuracy",
            training_samples=1000,
            validation_samples=200,
            training_time_hours=0.1,
            created_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        deploy_path = state_manager.save_final_model_for_deployment(model, metadata)
        print(f"✅ 배포용 모델 저장: {deploy_path}")
        
        # 학습 요약
        summary = state_manager.get_training_summary()
        print(f"✅ 학습 요약: {summary['total_checkpoints']}개 체크포인트")
        
        # 정리
        shutil.rmtree("test_checkpoints", ignore_errors=True)
        print("✅ 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()