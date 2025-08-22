"""
Stage 3용 EfficientNetV2-L 분류기

PillSnap ML Stage 3 (100K 이미지, 1000 클래스) 전용:
- EfficientNetV2-L 백본 (S보다 4배 큰 모델)
- 1000개 클래스 분류 최적화
- RTX 5080 16GB 메모리 최적화
- 그래디언트 체크포인팅 및 효율적 메모리 관리
- 85% 목표 정확도 달성 위한 고급 정규화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import timm

from ..utils.core import PillSnapLogger, load_config
from .classifier_efficientnetv2 import ClassificationResult, ClassifierConfig


@dataclass
class Stage3ClassifierConfig(ClassifierConfig):
    """Stage 3 전용 분류기 설정"""
    model_name: str = "tf_efficientnetv2_l"  # L 모델 사용
    num_classes: int = 1000  # Stage 3 클래스 수
    
    # 정규화 강화 (1000 클래스 대응)
    dropout_rate: float = 0.2  # 0.1 → 0.2
    drop_path_rate: float = 0.3  # 0.1 → 0.3
    label_smoothing: float = 0.15  # 0.1 → 0.15
    
    # 메모리 최적화 (RTX 5080 16GB)
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    
    # 고급 정규화
    weight_decay: float = 0.05
    ema_decay: float = 0.9999  # Exponential Moving Average
    
    # Stage 3 목표
    target_accuracy: float = 0.85  # 85% 목표


class PillSnapStage3Classifier(nn.Module):
    """Stage 3용 고성능 EfficientNetV2-L 분류기"""
    
    def __init__(self, config: Optional[Stage3ClassifierConfig] = None):
        super().__init__()
        self.config = config or Stage3ClassifierConfig()
        self.logger = PillSnapLogger(__name__)
        
        # 백본 모델 생성
        self._create_backbone()
        
        # Stage 3 전용 최적화
        self._apply_stage3_optimizations()
        
        # EMA 모델 초기화 (선택적)
        self.ema_model = None
        if self.config.ema_decay > 0:
            self._init_ema()
        
        self.logger.info(f"PillSnapStage3Classifier 초기화 완료")
        self.logger.info(f"  백본: {self.config.model_name} (Large)")
        self.logger.info(f"  클래스: {self.config.num_classes}개 (Stage 3)")
        self.logger.info(f"  목표 정확도: {self.config.target_accuracy:.1%}")
        self.logger.info(f"  메모리 최적화: {'활성화' if self.config.gradient_checkpointing else '비활성화'}")
    
    def _create_backbone(self):
        """EfficientNetV2-L 백본 생성"""
        try:
            # timm 모델 생성 (Large 모델)
            self.backbone = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=self.config.num_classes,
                drop_rate=self.config.dropout_rate,
                drop_path_rate=self.config.drop_path_rate
            )
            
            # 그래디언트 체크포인팅 적용
            if self.config.gradient_checkpointing:
                self._enable_gradient_checkpointing()
            
            # 디바이스 이동
            self.backbone = self.backbone.to(self.config.device)
            
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in self.backbone.parameters())
            self.logger.info(f"EfficientNetV2-L 백본 생성 성공")
            self.logger.info(f"  전체 파라미터: {total_params:,}개")
            self.logger.info(f"  예상 모델 크기: {total_params * 4 / 1024**2:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"백본 모델 생성 실패: {e}")
            raise
    
    def _enable_gradient_checkpointing(self):
        """그래디언트 체크포인팅 활성화"""
        try:
            # EfficientNet의 블록들에 체크포인팅 적용
            if hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks:
                    if hasattr(block, 'forward'):
                        # 원본 forward 저장
                        original_forward = block.forward
                        
                        # 체크포인팅이 적용된 forward로 교체
                        def checkpointed_forward(x):
                            return checkpoint.checkpoint(original_forward, x, use_reentrant=False)
                        
                        block.forward = checkpointed_forward
            
            self.logger.info("그래디언트 체크포인팅 활성화 완료")
            
        except Exception as e:
            self.logger.warning(f"그래디언트 체크포인팅 설정 실패: {e}")
    
    def _apply_stage3_optimizations(self):
        """Stage 3 전용 최적화"""
        try:
            # channels_last 메모리 포맷
            if self.config.channels_last:
                self.backbone = self.backbone.to(memory_format=torch.channels_last)
                self.logger.info("channels_last 메모리 포맷 적용")
            
            # Mixed Precision 설정 확인
            if self.config.mixed_precision:
                self.logger.info("Mixed Precision 학습 준비됨")
            
            # torch.compile 준비
            if self.config.torch_compile:
                self.logger.info("torch.compile 최적화 준비됨")
            
            self.logger.info("Stage 3 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"Stage 3 최적화 적용 중 일부 실패: {e}")
    
    def _init_ema(self):
        """Exponential Moving Average 모델 초기화"""
        try:
            from copy import deepcopy
            self.ema_model = deepcopy(self.backbone)
            self.ema_model.eval()
            
            # EMA 파라미터 업데이트 비활성화
            for param in self.ema_model.parameters():
                param.requires_grad = False
            
            self.logger.info(f"EMA 모델 초기화 완료 (decay={self.config.ema_decay})")
            
        except Exception as e:
            self.logger.warning(f"EMA 모델 초기화 실패: {e}")
            self.ema_model = None
    
    def update_ema(self):
        """EMA 모델 업데이트"""
        if self.ema_model is None:
            return
        
        try:
            with torch.no_grad():
                decay = self.config.ema_decay
                for ema_param, model_param in zip(self.ema_model.parameters(), self.backbone.parameters()):
                    ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
        except Exception as e:
            self.logger.warning(f"EMA 업데이트 실패: {e}")
    
    def forward(self, x: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        """순전파"""
        if self.config.channels_last:
            x = x.to(memory_format=torch.channels_last)
        
        # EMA 모델 사용 여부 결정
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.backbone
        
        if self.training and self.config.gradient_checkpointing:
            # 훈련 시 그래디언트 체크포인팅 사용
            logits = model(x)
        else:
            # 추론 시 일반 forward
            logits = model(x)
        
        return logits
    
    def predict_with_tta(
        self, 
        x: torch.Tensor,
        tta_transforms: Optional[List] = None,
        temperature: float = 1.0
    ) -> ClassificationResult:
        """Test Time Augmentation을 사용한 예측"""
        
        self.eval()
        
        if tta_transforms is None:
            # 기본 TTA: 원본 + 수평 뒤집기
            tta_transforms = [
                lambda img: img,  # 원본
                lambda img: torch.flip(img, dims=[-1]),  # 수평 뒤집기
            ]
        
        all_logits = []
        
        with torch.no_grad():
            for transform in tta_transforms:
                transformed_x = transform(x)
                logits = self.forward(transformed_x, use_ema=True)
                
                # 온도 스케일링 적용
                if temperature != 1.0:
                    logits = logits / temperature
                
                all_logits.append(logits)
        
        # TTA 결과 평균
        avg_logits = torch.stack(all_logits).mean(dim=0)
        probabilities = F.softmax(avg_logits, dim=1)
        confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
        
        return ClassificationResult(
            logits=avg_logits,
            probabilities=probabilities,
            predicted_classes=predicted_classes,
            confidence_scores=confidence_scores
        )
    
    def compute_class_attention(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """특정 클래스에 대한 어텐션 맵 계산 (Grad-CAM)"""
        self.eval()
        
        # 특징 추출을 위한 훅 설정
        feature_maps = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            feature_maps['features'] = output
        
        def backward_hook(module, grad_input, grad_output):
            gradients['gradients'] = grad_output[0]
        
        # 마지막 convolutional layer에 훅 등록
        target_layer = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            raise ValueError("Convolutional layer를 찾을 수 없음")
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            x.requires_grad_(True)
            logits = self.forward(x)
            
            # Backward pass
            score = logits[0, target_class]
            score.backward()
            
            # Grad-CAM 계산
            features = feature_maps['features'][0]  # [C, H, W]
            grads = gradients['gradients'][0]  # [C, H, W]
            
            # 채널별 가중치 계산
            weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            
            # 가중 합으로 어텐션 맵 생성
            attention_map = (weights * features).sum(dim=0)  # [H, W]
            attention_map = F.relu(attention_map)
            
            # 정규화
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            return attention_map
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Stage 3 모델 정보 반환"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            return {
                'stage': 3,
                'model_name': self.config.model_name,
                'num_classes': self.config.num_classes,
                'target_accuracy': self.config.target_accuracy,
                'input_size': self.config.input_size,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / 1024**2,
                'device': str(next(self.parameters()).device),
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'ema_enabled': self.ema_model is not None,
                'dropout_rate': self.config.dropout_rate,
                'drop_path_rate': self.config.drop_path_rate,
                'label_smoothing': self.config.label_smoothing,
                'memory_optimizations': {
                    'channels_last': self.config.channels_last,
                    'mixed_precision': self.config.mixed_precision,
                    'torch_compile': self.config.torch_compile,
                    'gradient_checkpointing': self.config.gradient_checkpointing
                }
            }
        except Exception as e:
            self.logger.error(f"모델 정보 수집 실패: {e}")
            return {}


def create_stage3_classifier(
    num_classes: int = 1000,
    device: str = "cuda",
    pretrained: bool = True,
    target_accuracy: float = 0.85
) -> PillSnapStage3Classifier:
    """Stage 3 분류기 생성 헬퍼 함수"""
    
    config = Stage3ClassifierConfig(
        num_classes=num_classes,
        device=device,
        pretrained=pretrained,
        target_accuracy=target_accuracy
    )
    
    return PillSnapStage3Classifier(config=config)


if __name__ == "__main__":
    # Stage 3 분류기 테스트
    logger = PillSnapLogger(__name__)
    
    try:
        # Stage 3 분류기 생성
        classifier = create_stage3_classifier(num_classes=1000)
        
        # 모델 정보 출력
        model_info = classifier.get_model_info()
        logger.info("Stage 3 모델 정보:")
        for key, value in model_info.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 메모리 사용량 체크
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            
            # 더미 입력으로 테스트
            device = classifier.config.device
            dummy_input = torch.randn(4, 3, 384, 384, device=device)
            if classifier.config.channels_last:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            logger.info("더미 입력 메모리 테스트 수행...")
            
            # 추론 테스트
            result = classifier.predict_with_tta(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            current_memory = torch.cuda.memory_allocated() / 1024**2
            
            logger.info(f"메모리 사용량:")
            logger.info(f"  초기: {initial_memory:.1f}MB")
            logger.info(f"  현재: {current_memory:.1f}MB")
            logger.info(f"  최대: {peak_memory:.1f}MB")
            
            logger.info(f"Stage 3 분류 결과:")
            logger.info(f"  배치 크기: {len(result)}")
            logger.info(f"  예측 클래스: {result.predicted_classes[:2]}")
            logger.info(f"  신뢰도: {result.confidence_scores[:2]}")
        
        logger.info("✅ Stage 3 EfficientNetV2-L 분류기 테스트 성공")
        
    except Exception as e:
        logger.error(f"❌ Stage 3 테스트 실패: {e}")
        import traceback
        traceback.print_exc()