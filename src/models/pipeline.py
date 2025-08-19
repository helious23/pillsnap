"""
PillSnap ML Two-Stage 파이프라인

사용자 제어 기반 조건부 파이프라인:
- Single 모드 (기본): 직접 EfficientNetV2-S 분류 (384px)
- Combo 모드 (명시적 선택): YOLOv11m 검출 (640px) → 크롭 → 분류 (384px)
- 명확한 사용자 선택 기반, 자동 판단 로직 제거
- RTX 5080 최적화 및 배치 처리 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Literal
from dataclasses import dataclass
import time
from PIL import Image
import numpy as np

from .detector import PillSnapYOLODetector, YOLOConfig, DetectionResult
from .classifier import PillSnapClassifier, ClassifierConfig, ClassificationResult
from ..utils.core import PillSnapLogger, load_config


@dataclass
class PipelineConfig:
    """Two-Stage 파이프라인 설정"""
    # 모드 설정
    default_mode: Literal["single", "combo"] = "single"  # 기본 모드
    
    # 검출기 설정
    detection_config: Optional[YOLOConfig] = None
    detection_confidence_threshold: float = 0.3
    detection_iou_threshold: float = 0.45
    
    # 분류기 설정
    classification_config: Optional[ClassifierConfig] = None
    classification_temperature: float = 1.0
    
    # 성능 설정
    device: str = "cuda"
    batch_size: int = 16
    enable_optimization: bool = True
    
    # 크롭 설정 (combo 모드용)
    crop_padding: float = 0.1  # 크롭 시 여백 비율
    min_crop_size: int = 64   # 최소 크롭 크기
    max_crop_size: int = 512  # 최대 크롭 크기
    
    def __post_init__(self):
        """설정 유효성 검증"""
        assert self.default_mode in ["single", "combo"], "default_mode는 'single' 또는 'combo'"
        assert 0.0 <= self.detection_confidence_threshold <= 1.0, "confidence_threshold는 0~1 사이"
        assert 0.0 <= self.detection_iou_threshold <= 1.0, "iou_threshold는 0~1 사이"
        assert 0.0 <= self.crop_padding <= 1.0, "crop_padding은 0~1 사이"
        assert self.min_crop_size > 0, "min_crop_size는 양수"
        assert self.max_crop_size >= self.min_crop_size, "max_crop_size >= min_crop_size"


@dataclass
class PipelineResult:
    """파이프라인 결과 데이터 클래스"""
    mode_used: Literal["single", "combo"]  # 실제 사용된 모드
    mode_requested: Literal["single", "combo"]  # 요청된 모드
    
    # Single 모드 결과
    classification_result: Optional[ClassificationResult] = None
    
    # Combo 모드 결과
    detection_result: Optional[DetectionResult] = None
    cropped_classifications: Optional[List[ClassificationResult]] = None
    
    # 성능 메트릭
    timing: Dict[str, float] = None  # 실행 시간 (ms)
    input_shape: Tuple[int, ...] = None
    
    def __post_init__(self):
        if self.timing is None:
            self.timing = {}
    
    def __len__(self) -> int:
        """결과 개수 반환"""
        if self.mode_used == "single":
            return len(self.classification_result) if self.classification_result else 0
        else:
            return len(self.cropped_classifications) if self.cropped_classifications else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 형태로 변환"""
        result = {
            'mode_used': self.mode_used,
            'mode_requested': self.mode_requested,
            'timing': self.timing,
            'input_shape': list(self.input_shape) if self.input_shape else None,
            'count': len(self)
        }
        
        if self.mode_used == "single":
            result['classification'] = (
                self.classification_result.to_dict() if self.classification_result else None
            )
        else:
            result['detection'] = (
                self.detection_result.to_dict() if self.detection_result else None
            )
            result['cropped_classifications'] = [
                cls_result.to_dict() for cls_result in (self.cropped_classifications or [])
            ]
        
        return result
    
    def get_predictions(self) -> List[Dict[str, Any]]:
        """통합된 예측 결과 반환"""
        predictions = []
        
        if self.mode_used == "single" and self.classification_result:
            # Single 모드: 직접 분류 결과
            for i in range(len(self.classification_result)):
                predictions.append({
                    'class_id': self.classification_result.predicted_classes[i].item(),
                    'confidence': self.classification_result.confidence_scores[i].item(),
                    'bbox': None,  # Single 모드에서는 bbox 없음
                    'mode': 'single'
                })
        
        elif self.mode_used == "combo" and self.detection_result and self.cropped_classifications:
            # Combo 모드: 검출 + 분류 결과
            for i, (bbox, cls_result) in enumerate(
                zip(self.detection_result.boxes, self.cropped_classifications)
            ):
                if len(cls_result) > 0:
                    predictions.append({
                        'class_id': cls_result.predicted_classes[0].item(),
                        'confidence': cls_result.confidence_scores[0].item(),
                        'bbox': bbox.tolist(),
                        'detection_confidence': self.detection_result.scores[i].item(),
                        'mode': 'combo'
                    })
        
        return predictions


class PillSnapPipeline(nn.Module):
    """PillSnap ML Two-Stage 파이프라인"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__()
        self.config = config or PipelineConfig()
        self.logger = PillSnapLogger(__name__)
        
        # 모델 초기화
        self._initialize_models()
        
        self.logger.info(f"PillSnapPipeline 초기화 완료")
        self.logger.info(f"  기본 모드: {self.config.default_mode}")
        self.logger.info(f"  디바이스: {self.config.device}")
    
    def _initialize_models(self):
        """검출기 및 분류기 초기화"""
        try:
            # 분류기 초기화 (항상 필요)
            classifier_config = self.config.classification_config or ClassifierConfig(
                device=self.config.device
            )
            self.classifier = PillSnapClassifier(config=classifier_config)
            self.logger.info("분류기 초기화 완료")
            
            # 검출기 초기화 (Combo 모드용)
            detector_config = self.config.detection_config or YOLOConfig(
                device=self.config.device
            )
            self.detector = PillSnapYOLODetector(config=detector_config)
            self.logger.info("검출기 초기화 완료")
            
            # 최적화 적용
            if self.config.enable_optimization:
                self._apply_optimizations()
        
        except Exception as e:
            self.logger.error(f"모델 초기화 실패: {e}")
            raise
    
    def _apply_optimizations(self):
        """RTX 5080 최적화 적용"""
        try:
            self.logger.info("RTX 5080 최적화 적용 중...")
            
            # torch.compile 최적화 (실제 추론 시 적용)
            if hasattr(self.classifier.config, 'torch_compile') and self.classifier.config.torch_compile:
                self.logger.info("분류기 torch.compile 최적화 준비")
            
            if hasattr(self.detector.config, 'torch_compile') and self.detector.config.torch_compile:
                self.logger.info("검출기 torch.compile 최적화 준비")
            
            self.logger.info("최적화 적용 완료")
        
        except Exception as e:
            self.logger.warning(f"최적화 적용 중 일부 실패: {e}")
    
    def predict(
        self,
        x: torch.Tensor,
        mode: Literal["single", "combo"] = None,
        confidence_threshold: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> PipelineResult:
        """이미지 예측 수행"""
        
        # 모드 결정 (사용자 선택 > 기본값)
        requested_mode = mode or self.config.default_mode
        actual_mode = requested_mode
        
        # 임계값 설정
        conf_threshold = confidence_threshold or self.config.detection_confidence_threshold
        temp = temperature or self.config.classification_temperature
        
        # 입력 형태 기록
        input_shape = tuple(x.shape)
        
        # 타이밍 시작
        start_time = time.time()
        timing = {}
        
        self.logger.debug(f"파이프라인 예측 시작: mode={actual_mode}, shape={input_shape}")
        
        try:
            if actual_mode == "single":
                result = self._predict_single_mode(x, temp, timing)
            else:  # combo
                result = self._predict_combo_mode(x, conf_threshold, temp, timing)
            
            # 전체 실행 시간 기록
            total_time = (time.time() - start_time) * 1000  # ms
            timing['total'] = total_time
            
            # 결과 생성
            pipeline_result = PipelineResult(
                mode_used=actual_mode,
                mode_requested=requested_mode,
                timing=timing,
                input_shape=input_shape,
                **result
            )
            
            self.logger.debug(f"파이프라인 예측 완료: {total_time:.2f}ms, count={len(pipeline_result)}")
            return pipeline_result
        
        except Exception as e:
            self.logger.error(f"파이프라인 예측 실패: {e}")
            raise
    
    def _predict_single_mode(
        self, 
        x: torch.Tensor, 
        temperature: float, 
        timing: Dict[str, float]
    ) -> Dict[str, Any]:
        """Single 모드 예측 (직접 분류)"""
        
        # 분류 수행
        cls_start = time.time()
        classification_result = self.classifier.predict(
            x, 
            temperature=temperature,
            return_logits=False
        )
        timing['classification'] = (time.time() - cls_start) * 1000
        
        return {
            'classification_result': classification_result,
            'detection_result': None,
            'cropped_classifications': None
        }
    
    def _predict_combo_mode(
        self, 
        x: torch.Tensor, 
        confidence_threshold: float,
        temperature: float,
        timing: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combo 모드 예측 (검출 → 크롭 → 분류)"""
        
        # 1. 검출 수행
        det_start = time.time()
        detection_results = self.detector.predict(
            x,
            conf=confidence_threshold,
            iou=self.config.detection_iou_threshold
        )
        timing['detection'] = (time.time() - det_start) * 1000
        
        # 배치의 첫 번째 결과 사용 (현재는 단일 이미지 처리)
        detection_result = detection_results[0] if detection_results else None
        
        # 2. 검출된 객체가 없으면 빈 결과 반환
        if detection_result is None or len(detection_result) == 0:
            self.logger.warning("Combo 모드에서 검출된 객체 없음")
            timing['classification'] = 0.0
            # 빈 결과 생성
            from .detector import DetectionResult
            empty_detection = DetectionResult(
                boxes=torch.empty(0, 4),
                scores=torch.empty(0),
                class_ids=torch.empty(0, dtype=torch.long),
                image_shape=(640, 640)
            )
            return {
                'classification_result': None,
                'detection_result': empty_detection,
                'cropped_classifications': []
            }
        
        # 3. 크롭 및 분류 수행
        cls_start = time.time()
        cropped_classifications = self._classify_detected_objects(
            x, detection_result, temperature
        )
        timing['classification'] = (time.time() - cls_start) * 1000
        
        return {
            'classification_result': None,
            'detection_result': detection_result,
            'cropped_classifications': cropped_classifications
        }
    
    def _classify_detected_objects(
        self,
        x: torch.Tensor,
        detection_result: DetectionResult,
        temperature: float
    ) -> List[ClassificationResult]:
        """검출된 객체들을 크롭하여 분류"""
        
        cropped_classifications = []
        
        # 각 검출된 객체에 대해 크롭 및 분류
        for i, bbox in enumerate(detection_result.boxes):
            try:
                # 크롭 수행
                cropped_tensor = self._crop_detection(
                    x[i:i+1] if len(x.shape) == 4 else x.unsqueeze(0),
                    bbox
                )
                
                # 분류 수행
                if cropped_tensor.numel() > 0:
                    cls_result = self.classifier.predict(
                        cropped_tensor,
                        temperature=temperature,
                        return_logits=False
                    )
                    cropped_classifications.append(cls_result)
                else:
                    # 빈 크롭인 경우 빈 결과
                    empty_result = ClassificationResult(
                        logits=torch.empty(0),
                        probabilities=torch.empty(0, self.classifier.config.num_classes),
                        predicted_classes=torch.empty(0, dtype=torch.long),
                        confidence_scores=torch.empty(0)
                    )
                    cropped_classifications.append(empty_result)
            
            except Exception as e:
                self.logger.warning(f"검출 객체 {i} 분류 실패: {e}")
                # 오류 시 빈 결과 추가
                empty_result = ClassificationResult(
                    logits=torch.empty(0),
                    probabilities=torch.empty(0, self.classifier.config.num_classes),
                    predicted_classes=torch.empty(0, dtype=torch.long),
                    confidence_scores=torch.empty(0)
                )
                cropped_classifications.append(empty_result)
        
        return cropped_classifications
    
    def _crop_detection(self, x: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """검출 결과를 이용해 이미지 크롭"""
        
        # bbox는 [x1, y1, x2, y2] 형태 (normalized 0~1)
        if x.dim() != 4:
            raise ValueError(f"입력 텐서는 4차원이어야 함: {x.shape}")
        
        batch_size, channels, height, width = x.shape
        
        # bbox를 픽셀 좌표로 변환
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # 패딩 적용
        padding = self.config.crop_padding
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        pad_x = int(crop_width * padding)
        pad_y = int(crop_height * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        
        # 최소/최대 크기 제한 확인
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        if crop_w < self.config.min_crop_size or crop_h < self.config.min_crop_size:
            self.logger.warning(f"크롭 크기가 너무 작음: {crop_w}x{crop_h}")
            return torch.empty(0, channels, 0, 0, device=x.device)
        
        # 크롭 수행
        cropped = x[:, :, y1:y2, x1:x2]
        
        # 분류기 입력 크기로 리사이즈
        target_size = self.classifier.config.input_size
        cropped_resized = F.interpolate(
            cropped,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        
        return cropped_resized
    
    def predict_batch(
        self,
        x: torch.Tensor,
        mode: Literal["single", "combo"] = None,
        batch_size: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> List[PipelineResult]:
        """배치 단위 예측"""
        
        batch_size = batch_size or self.config.batch_size
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            
            # 각 이미지별로 예측 (현재는 개별 처리)
            for j in range(len(batch_x)):
                single_x = batch_x[j:j+1]
                result = self.predict(
                    single_x,
                    mode=mode,
                    confidence_threshold=confidence_threshold,
                    temperature=temperature
                )
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """파이프라인 모델 정보 반환"""
        try:
            detector_info = self.detector.get_model_info()
            classifier_info = self.classifier.get_model_info()
            
            return {
                'pipeline_config': {
                    'default_mode': self.config.default_mode,
                    'device': self.config.device,
                    'batch_size': self.config.batch_size,
                    'enable_optimization': self.config.enable_optimization
                },
                'detector': detector_info,
                'classifier': classifier_info,
                'total_parameters': (
                    detector_info.get('total_parameters', 0) + 
                    classifier_info.get('total_parameters', 0)
                )
            }
        except Exception as e:
            self.logger.error(f"모델 정보 수집 실패: {e}")
            return {}
    
    def save_models(self, save_dir: Union[str, Path]):
        """모델들 저장"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 검출기 저장
            detector_path = save_dir / "detector.pth"
            self.detector.save_model(detector_path)
            
            # 분류기 저장
            classifier_path = save_dir / "classifier.pth"
            self.classifier.save_model(classifier_path)
            
            # 파이프라인 설정 저장
            config_path = save_dir / "pipeline_config.pth"
            torch.save(self.config, config_path)
            
            self.logger.info(f"파이프라인 모델 저장 완료: {save_dir}")
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def load_models(self, load_dir: Union[str, Path]):
        """모델들 로드"""
        load_dir = Path(load_dir)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없음: {load_dir}")
        
        try:
            # 검출기 로드
            detector_path = load_dir / "detector.pth"
            if detector_path.exists():
                self.detector.load_model(detector_path)
            
            # 분류기 로드
            classifier_path = load_dir / "classifier.pth"
            if classifier_path.exists():
                self.classifier.load_model(classifier_path)
            
            self.logger.info(f"파이프라인 모델 로드 완료: {load_dir}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise


def create_pillsnap_pipeline(
    default_mode: Literal["single", "combo"] = "single",
    device: str = "cuda",
    num_classes: int = 4523,
    detector_input_size: int = 640,
    classifier_input_size: int = 384
) -> PillSnapPipeline:
    """PillSnap 파이프라인 생성 헬퍼 함수"""
    
    # 검출기 설정
    detector_config = YOLOConfig(
        model_size="yolo11m",
        input_size=detector_input_size,
        device=device
    )
    
    # 분류기 설정
    classifier_config = ClassifierConfig(
        model_name="tf_efficientnetv2_s",
        num_classes=num_classes,
        input_size=classifier_input_size,
        device=device
    )
    
    # 파이프라인 설정
    pipeline_config = PipelineConfig(
        default_mode=default_mode,
        detection_config=detector_config,
        classification_config=classifier_config,
        device=device
    )
    
    return PillSnapPipeline(config=pipeline_config)


if __name__ == "__main__":
    # 기본 테스트
    logger = PillSnapLogger(__name__)
    
    try:
        # 파이프라인 생성
        pipeline = create_pillsnap_pipeline(
            default_mode="single",
            num_classes=100,  # 테스트용
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 모델 정보 출력
        model_info = pipeline.get_model_info()
        logger.info("파이프라인 정보:")
        logger.info(f"  기본 모드: {model_info['pipeline_config']['default_mode']}")
        logger.info(f"  총 파라미터: {model_info['total_parameters']:,}")
        logger.info(f"  검출기 파라미터: {model_info['detector']['total_parameters']:,}")
        logger.info(f"  분류기 파라미터: {model_info['classifier']['total_parameters']:,}")
        
        # 더미 입력으로 테스트
        device = pipeline.config.device
        dummy_input = torch.randn(2, 3, 640, 640, device=device)
        
        logger.info("Single 모드 테스트...")
        single_result = pipeline.predict(dummy_input, mode="single")
        logger.info(f"  Single 결과: {len(single_result)}개, {single_result.timing['total']:.2f}ms")
        
        logger.info("Combo 모드 테스트...")
        combo_result = pipeline.predict(dummy_input, mode="combo")
        logger.info(f"  Combo 결과: {len(combo_result)}개, {combo_result.timing['total']:.2f}ms")
        
        # 예측 결과 확인
        single_predictions = single_result.get_predictions()
        combo_predictions = combo_result.get_predictions()
        
        logger.info(f"  Single 예측: {len(single_predictions)}개")
        logger.info(f"  Combo 예측: {len(combo_predictions)}개")
        
        logger.info("✅ Two-Stage 파이프라인 기본 테스트 성공")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()