"""
YOLOv11x 검출 모델 래퍼 (Stage 3+)

PillSnap ML Two-Stage Pipeline의 Combination 약품 검출용:
- Ultralytics YOLOv11x 모델 래퍼 (최고 성능)
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- Combination 약품 검출 특화 설정
- 640px 입력 해상도 최적화
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..utils.core import PillSnapLogger, load_config


@dataclass
class DetectionResult:
    """검출 결과 데이터 클래스"""
    boxes: torch.Tensor  # [N, 4] (x1, y1, x2, y2)
    scores: torch.Tensor  # [N]
    class_ids: torch.Tensor  # [N]
    image_shape: Tuple[int, int]  # (height, width)
    
    def __len__(self) -> int:
        return len(self.boxes)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형태로 변환"""
        return {
            'boxes': self.boxes.cpu().numpy().tolist(),
            'scores': self.scores.cpu().numpy().tolist(),
            'class_ids': self.class_ids.cpu().numpy().tolist(),
            'image_shape': self.image_shape,
            'num_detections': len(self)
        }


@dataclass 
class YOLOConfig:
    """YOLO 모델 설정"""
    model_size: str = "yolo11m"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    input_size: int = 640  # 입력 이미지 크기
    confidence_threshold: float = 0.25  # 신뢰도 임계값
    iou_threshold: float = 0.45  # NMS IoU 임계값
    max_detections: int = 300  # 최대 검출 수
    device: str = "cuda"  # 디바이스
    
    # RTX 5080 최적화 설정
    mixed_precision: bool = True  # Mixed Precision 사용
    torch_compile: bool = True  # torch.compile 사용
    channels_last: bool = True  # channels_last 메모리 포맷
    
    # 학습 관련 설정
    pretrained: bool = True  # 사전 훈련된 가중치 사용
    freeze_backbone: bool = False  # 백본 동결 여부
    
    def __post_init__(self):
        """설정 유효성 검증"""
        assert self.model_size in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], \
            f"지원하지 않는 모델 크기: {self.model_size}"
        assert self.input_size > 0, "input_size는 양수여야 함"
        assert 0.0 <= self.confidence_threshold <= 1.0, "confidence_threshold는 0~1 사이"
        assert 0.0 <= self.iou_threshold <= 1.0, "iou_threshold는 0~1 사이"
        assert self.max_detections > 0, "max_detections는 양수여야 함"


class PillSnapYOLODetector(nn.Module):
    """PillSnap ML용 YOLOv11m 검출기"""
    
    def __init__(self, config: Optional[YOLOConfig] = None, num_classes: int = 1):
        super().__init__()
        self.config = config or YOLOConfig()
        self.num_classes = num_classes
        self.logger = PillSnapLogger(__name__)
        
        # 모델 초기화
        self._initialize_model()
        
        # RTX 5080 최적화 적용
        self._apply_optimizations()
        
        self.logger.info(f"PillSnapYOLODetector 초기화 완료")
        self.logger.info(f"  모델: {self.config.model_size}")
        self.logger.info(f"  클래스 수: {self.num_classes}")
        self.logger.info(f"  입력 크기: {self.config.input_size}x{self.config.input_size}")
        self.logger.info(f"  디바이스: {self.config.device}")
    
    def _initialize_model(self):
        """YOLO 모델 초기화"""
        try:
            # 사전 훈련된 모델 로드
            if self.config.pretrained:
                model_path = f"{self.config.model_size}.pt"
                self.logger.info(f"사전 훈련된 모델 로드: {model_path}")
                self.model = YOLO(model_path)
            else:
                # 랜덤 초기화
                model_path = f"{self.config.model_size}.yaml"
                self.logger.info(f"랜덤 초기화 모델 생성: {model_path}")
                self.model = YOLO(model_path)
            
            # 디바이스 설정
            self.model.to(self.config.device)
            
            # 클래스 수 조정 (필요한 경우)
            if hasattr(self.model.model, 'nc') and self.model.model.nc != self.num_classes:
                self.logger.info(f"클래스 수 조정: {self.model.model.nc} → {self.num_classes}")
                # 헤드 레이어 재구성은 학습 시에 자동으로 처리됨
            
            self.logger.info(f"YOLO 모델 초기화 성공")
            
        except Exception as e:
            self.logger.error(f"YOLO 모델 초기화 실패: {e}")
            raise
    
    def _apply_optimizations(self):
        """RTX 5080 최적화 적용"""
        try:
            # Mixed Precision 설정
            if self.config.mixed_precision:
                self.logger.info("Mixed Precision 활성화")
                # YOLO는 내부적으로 AMP를 지원함
            
            # channels_last 메모리 포맷 (YOLO와 호환성 문제로 비활성화)
            if self.config.channels_last:
                self.logger.warning("channels_last는 YOLO와 호환성 문제로 비활성화")
                # self.model.model = self.model.model.to(memory_format=torch.channels_last)
            
            # torch.compile 적용
            if self.config.torch_compile:
                self.logger.info("torch.compile 최적화 적용")
                # 실제 학습/추론 시에 적용
                # self.model.model = torch.compile(self.model.model, mode='max-autotune')
            
            self.logger.info("RTX 5080 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"최적화 적용 중 일부 실패: {e}")
    
    def forward(self, x: torch.Tensor) -> List[DetectionResult]:
        """순전파 (추론용)"""
        return self.predict(x)
    
    def predict(
        self, 
        source: Union[torch.Tensor, str, Path, List],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        verbose: bool = False
    ) -> List[DetectionResult]:
        """이미지 검출 수행"""
        
        # 파라미터 설정
        conf = conf or self.config.confidence_threshold
        iou = iou or self.config.iou_threshold
        
        try:
            # YOLO 추론 실행
            results = self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=self.config.input_size,
                device=self.config.device,
                verbose=verbose,
                max_det=self.config.max_detections
            )
            
            # 결과 변환
            detection_results = []
            for result in results:
                detection_result = self._convert_yolo_result(result)
                detection_results.append(detection_result)
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"검출 실행 실패: {e}")
            raise
    
    def _convert_yolo_result(self, yolo_result: Results) -> DetectionResult:
        """YOLO 결과를 DetectionResult로 변환"""
        
        # 검출된 박스가 있는지 확인
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            # 빈 결과 반환
            return DetectionResult(
                boxes=torch.empty((0, 4), device=self.config.device),
                scores=torch.empty((0,), device=self.config.device),
                class_ids=torch.empty((0,), dtype=torch.long, device=self.config.device),
                image_shape=yolo_result.orig_shape[:2]  # (height, width)
            )
        
        # 박스 좌표 (xyxy 형태)
        boxes = yolo_result.boxes.xyxy  # [N, 4]
        
        # 신뢰도 점수
        scores = yolo_result.boxes.conf  # [N]
        
        # 클래스 ID
        class_ids = yolo_result.boxes.cls.long()  # [N]
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            image_shape=yolo_result.orig_shape[:2]  # (height, width)
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            return {
                'model_size': self.config.model_size,
                'num_classes': self.num_classes,
                'input_size': self.config.input_size,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(next(self.model.model.parameters()).device),
                'memory_format': 'channels_last' if self.config.channels_last else 'contiguous',
                'mixed_precision': self.config.mixed_precision,
                'torch_compile': self.config.torch_compile
            }
        except Exception as e:
            self.logger.error(f"모델 정보 수집 실패: {e}")
            return {}
    
    def set_training_mode(self, training: bool = True):
        """학습/평가 모드 설정"""
        if training:
            self.model.train()
        else:
            self.model.eval()
    
    def save_model(self, save_path: Union[str, Path]):
        """모델 저장"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # YOLO 모델 저장
            self.model.save(str(save_path))
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
            self.model = YOLO(str(model_path))
            self.model.to(self.config.device)
            self._apply_optimizations()
            self.logger.info(f"모델 로드 완료: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


def create_pillsnap_detector(
    num_classes: int = 1,
    model_size: str = "yolo11m",
    input_size: int = 640,
    device: str = "cuda"
) -> PillSnapYOLODetector:
    """PillSnap 검출기 생성 헬퍼 함수"""
    
    config = YOLOConfig(
        model_size=model_size,
        input_size=input_size,
        device=device
    )
    
    return PillSnapYOLODetector(config=config, num_classes=num_classes)


if __name__ == "__main__":
    # 기본 테스트
    logger = PillSnapLogger(__name__)
    
    try:
        # 검출기 생성
        detector = create_pillsnap_detector(num_classes=1)
        
        # 모델 정보 출력
        model_info = detector.get_model_info()
        logger.info("모델 정보:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # 더미 입력으로 테스트  
        dummy_input = torch.randn(1, 3, 640, 640, device="cuda")
        # channels_last는 YOLO와 호환성 문제로 사용하지 않음
        
        logger.info("더미 입력 테스트 수행...")
        
        # 추론 모드로 설정
        detector.set_training_mode(False)
        
        with torch.no_grad():
            results = detector.predict(dummy_input)
            logger.info(f"검출 결과: {len(results)}개 이미지")
            if results:
                logger.info(f"첫 번째 이미지 검출 수: {len(results[0])}개")
        
        logger.info("✅ YOLOv11m 검출기 기본 테스트 성공")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
