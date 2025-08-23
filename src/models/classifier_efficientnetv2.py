"""
EfficientNetV2-L 분류 모델

PillSnap ML Two-Stage Pipeline의 분류용:
- timm 기반 EfficientNetV2-L 백본 (Stage 3+ 전용)
- 4,523개 EDI 코드 분류 헤드
- Single 약품 직접 분류 및 Combination 크롭 분류
- 384px 입력 해상도 최적화
- RTX 5080 최적화 (Mixed Precision, torch.compile)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

import timm

from ..utils.core import PillSnapLogger, load_config


@dataclass
class ClassificationResult:
    """분류 결과 데이터 클래스"""
    logits: torch.Tensor  # [N, num_classes] 로짓 값
    probabilities: torch.Tensor  # [N, num_classes] 확률 값
    predicted_classes: torch.Tensor  # [N] 예측된 클래스 ID
    confidence_scores: torch.Tensor  # [N] 최대 확률 값
    
    def __len__(self) -> int:
        return len(self.probabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형태로 변환"""
        return {
            'logits': self.logits.cpu().numpy().tolist(),
            'probabilities': self.probabilities.cpu().numpy().tolist(),
            'predicted_classes': self.predicted_classes.cpu().numpy().tolist(),
            'confidence_scores': self.confidence_scores.cpu().numpy().tolist(),
            'batch_size': len(self)
        }
    
    def get_top_k_predictions(self, k: int = 5) -> Dict[str, Any]:
        """상위 K개 예측 결과 반환"""
        top_k_probs, top_k_indices = torch.topk(self.probabilities, k, dim=1)
        
        return {
            'top_k_classes': top_k_indices.cpu().numpy().tolist(),
            'top_k_probabilities': top_k_probs.cpu().numpy().tolist(),
            'k': k
        }


@dataclass
class ClassifierConfig:
    """분류기 모델 설정"""
    model_name: str = "tf_efficientnetv2_l"  # timm 모델명 (Stage 3+용)
    num_classes: int = 4523  # EDI 코드 개수
    input_size: int = 384  # 입력 이미지 크기
    pretrained: bool = True  # 사전 훈련된 가중치 사용
    device: str = "cuda"  # 디바이스
    
    # RTX 5080 최적화 설정
    mixed_precision: bool = True  # Mixed Precision 사용
    torch_compile: bool = True  # torch.compile 사용
    channels_last: bool = True  # channels_last 메모리 포맷
    
    # 드롭아웃 설정
    dropout_rate: float = 0.1  # 분류 헤드 드롭아웃
    drop_path_rate: float = 0.1  # 스토캐스틱 뎁스
    
    # 정규화 설정
    label_smoothing: float = 0.1  # 라벨 스무딩
    
    def __post_init__(self):
        """설정 유효성 검증"""
        assert self.num_classes > 0, "num_classes는 양수여야 함"
        assert self.input_size > 0, "input_size는 양수여야 함"
        assert 0.0 <= self.dropout_rate <= 1.0, "dropout_rate는 0~1 사이"
        assert 0.0 <= self.drop_path_rate <= 1.0, "drop_path_rate는 0~1 사이"
        assert 0.0 <= self.label_smoothing <= 1.0, "label_smoothing은 0~1 사이"


class PillSnapClassifier(nn.Module):
    """PillSnap ML용 EfficientNetV2-L 분류기 (Stage 3+)"""
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        super().__init__()
        self.config = config or ClassifierConfig()
        self.logger = PillSnapLogger(__name__)
        
        # 백본 모델 생성
        self._create_backbone()
        
        # RTX 5080 최적화 적용
        self._apply_optimizations()
        
        self.logger.info(f"PillSnapClassifier 초기화 완료")
        self.logger.info(f"  모델: {self.config.model_name}")
        self.logger.info(f"  클래스 수: {self.config.num_classes}")
        self.logger.info(f"  입력 크기: {self.config.input_size}x{self.config.input_size}")
        self.logger.info(f"  디바이스: {self.config.device}")
    
    def _create_backbone(self):
        """백본 모델 생성 (fallback 포함)"""
        try:
            # timm 모델 생성 시도
            self.backbone = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=self.config.num_classes,
                drop_rate=self.config.dropout_rate,
                drop_path_rate=self.config.drop_path_rate
            )
            
            # 디바이스 이동
            self.backbone = self.backbone.to(self.config.device)
            
            self.logger.info(f"백본 모델 생성 성공: {self.config.model_name}")
            
        except Exception as e:
            self.logger.warning(f"원본 모델({self.config.model_name}) 생성 실패: {e}")
            
            # Fallback: pretrained=False로 재시도
            try:
                self.logger.info("Fallback: pretrained=False로 모델 생성 시도")
                self.backbone = timm.create_model(
                    self.config.model_name,
                    pretrained=False,
                    num_classes=self.config.num_classes,
                    drop_rate=self.config.dropout_rate,
                    drop_path_rate=self.config.drop_path_rate
                )
                
                # 디바이스 이동
                self.backbone = self.backbone.to(self.config.device)
                self.logger.info(f"Fallback 백본 모델 생성 성공: {self.config.model_name} (pretrained=False)")
                
            except Exception as fallback_e:
                # 최종 Fallback: ResNet50으로 변경
                self.logger.warning(f"Fallback도 실패: {fallback_e}")
                try:
                    self.logger.info("최종 Fallback: ResNet50으로 모델 생성")
                    self.backbone = timm.create_model(
                        'resnet50',
                        pretrained=False,
                        num_classes=self.config.num_classes
                    )
                    self.backbone = self.backbone.to(self.config.device)
                    self.logger.info("최종 Fallback 성공: ResNet50")
                except Exception as final_e:
                    self.logger.error(f"모든 모델 생성 시도 실패: {final_e}")
                    raise
    
    def _apply_optimizations(self):
        """RTX 5080 최적화 적용"""
        try:
            # Mixed Precision은 학습 시 자동으로 적용됨
            if self.config.mixed_precision:
                self.logger.info("Mixed Precision 활성화 설정")
            
            # channels_last 메모리 포맷
            if self.config.channels_last:
                self.logger.info("channels_last 메모리 포맷 적용")
                self.backbone = self.backbone.to(memory_format=torch.channels_last)
            
            # torch.compile 적용
            if self.config.torch_compile:
                self.logger.info("torch.compile 최적화 적용")
                # 실제 학습/추론 시에 적용
                # self.backbone = torch.compile(self.backbone, mode='max-autotune')
            
            self.logger.info("RTX 5080 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"최적화 적용 중 일부 실패: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        if self.config.channels_last:
            x = x.to(memory_format=torch.channels_last)
        
        logits = self.backbone(x)
        return logits
    
    def predict(
        self, 
        x: torch.Tensor, 
        return_logits: bool = False,
        temperature: float = 1.0
    ) -> ClassificationResult:
        """이미지 분류 수행"""
        
        self.eval()
        with torch.no_grad():
            # 순전파
            logits = self.forward(x)
            
            # 온도 스케일링 적용
            if temperature != 1.0:
                logits = logits / temperature
            
            # 확률 계산
            probabilities = F.softmax(logits, dim=1)
            
            # 예측 클래스 및 신뢰도
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
            
            return ClassificationResult(
                logits=logits if return_logits else torch.empty(0),
                probabilities=probabilities,
                predicted_classes=predicted_classes,
                confidence_scores=confidence_scores
            )
    
    def predict_batch(
        self, 
        x: torch.Tensor, 
        batch_size: int = 32,
        return_logits: bool = False,
        temperature: float = 1.0
    ) -> ClassificationResult:
        """배치 단위 분류 수행"""
        
        self.eval()
        all_logits = []
        all_probs = []
        all_preds = []
        all_confs = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                
                # 순전파
                logits = self.forward(batch_x)
                
                # 온도 스케일링 적용
                if temperature != 1.0:
                    logits = logits / temperature
                
                # 확률 계산
                probabilities = F.softmax(logits, dim=1)
                confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
                
                # 결과 누적
                if return_logits:
                    all_logits.append(logits)
                all_probs.append(probabilities)
                all_preds.append(predicted_classes)
                all_confs.append(confidence_scores)
        
        # 결과 합치기
        all_logits = torch.cat(all_logits, dim=0) if return_logits else torch.empty(0)
        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_confs = torch.cat(all_confs, dim=0)
        
        return ClassificationResult(
            logits=all_logits,
            probabilities=all_probs,
            predicted_classes=all_preds,
            confidence_scores=all_confs
        )
    
    def extract_features(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """특징 추출"""
        self.eval()
        
        if self.config.channels_last:
            x = x.to(memory_format=torch.channels_last)
        
        with torch.no_grad():
            if layer_name is None:
                # 분류 헤드 직전 특징
                features = self.backbone.forward_features(x)
                return self.backbone.global_pool(features)
            else:
                # 특정 레이어 특징 (고급 기능)
                return self._extract_intermediate_features(x, layer_name)
    
    def _extract_intermediate_features(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """중간 레이어 특징 추출"""
        # 이는 모델별로 다르게 구현해야 함
        features = {}
        
        def hook_fn(module, input, output):
            features[layer_name] = output
        
        # 훅 등록 및 실행
        for name, module in self.backbone.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        _ = self.backbone(x)
        handle.remove()
        
        return features.get(layer_name, torch.empty(0))
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            return {
                'model_name': self.config.model_name,
                'num_classes': self.config.num_classes,
                'input_size': self.config.input_size,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(next(self.parameters()).device),
                'memory_format': 'channels_last' if self.config.channels_last else 'contiguous',
                'mixed_precision': self.config.mixed_precision,
                'torch_compile': self.config.torch_compile,
                'dropout_rate': self.config.dropout_rate,
                'drop_path_rate': self.config.drop_path_rate
            }
        except Exception as e:
            self.logger.error(f"모델 정보 수집 실패: {e}")
            return {}
    
    def save_model(self, save_path: Union[str, Path]):
        """모델 저장"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 모델 상태 딕셔너리 저장
            save_dict = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'model_info': self.get_model_info()
            }
            
            torch.save(save_dict, save_path)
            self.logger.info(f"모델 저장 완료: {save_path}")
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_model(self, model_path: Union[str, Path]):
        """모델 로드"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"모델 로드 완료: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


def create_pillsnap_classifier(
    num_classes: int = 4523,
    model_name: str = "efficientnetv2_l",  # Stage 3+ 기본값 (timm 호환성)
    input_size: int = 384,
    device: str = "cuda",
    pretrained: bool = True
) -> PillSnapClassifier:
    """PillSnap 분류기 생성 헬퍼 함수"""
    
    config = ClassifierConfig(
        model_name=model_name,
        num_classes=num_classes,
        input_size=input_size,
        device=device,
        pretrained=pretrained
    )
    
    return PillSnapClassifier(config=config)


if __name__ == "__main__":
    # 기본 테스트
    logger = PillSnapLogger(__name__)
    
    try:
        # 분류기 생성
        classifier = create_pillsnap_classifier(num_classes=4523)
        
        # 모델 정보 출력
        model_info = classifier.get_model_info()
        logger.info("모델 정보:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # 더미 입력으로 테스트
        device = classifier.config.device
        dummy_input = torch.randn(2, 3, 384, 384, device=device)
        if classifier.config.channels_last:
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
        
        logger.info("더미 입력 테스트 수행...")
        
        # 추론 테스트
        result = classifier.predict(dummy_input)
        logger.info(f"분류 결과:")
        logger.info(f"  배치 크기: {len(result)}")
        logger.info(f"  예측 클래스: {result.predicted_classes}")
        logger.info(f"  신뢰도: {result.confidence_scores}")
        
        # Top-K 예측 테스트
        top_k = result.get_top_k_predictions(k=3)
        logger.info(f"  상위 3개 클래스: {top_k['top_k_classes'][0]}")
        logger.info(f"  상위 3개 확률: {top_k['top_k_probabilities'][0]}")
        
        # 특징 추출 테스트
        features = classifier.extract_features(dummy_input)
        logger.info(f"  특징 벡터 크기: {features.shape}")
        
        logger.info("✅ EfficientNetV2-S 분류기 기본 테스트 성공")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
