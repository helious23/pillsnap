"""
EfficientNetV2-L 분류 헤드 모델 팩토리
우선순위: timm -> torchvision 폴백
"""

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def create_efficientnetv2_l(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNetV2-L 모델 생성

    Args:
        num_classes: 출력 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부

    Returns:
        nn.Module: EfficientNetV2-L 모델
    """
    print(
        f"🔧 Creating EfficientNetV2-L model for {num_classes} classes (pretrained={pretrained})"
    )

    # 1) timm 우선 사용
    try:
        import timm

        model = timm.create_model(
            "efficientnetv2_l", pretrained=pretrained, num_classes=num_classes
        )
        print("✅ EfficientNetV2-L created via timm")
        return model
    except ImportError:
        logger.warning("timm not available, trying torchvision")
    except Exception as e:
        logger.warning(f"timm model creation failed: {e}, trying torchvision")

    # 2) torchvision 폴백
    try:
        from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

        # 가중치 설정
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None

        model = efficientnet_v2_l(weights=weights)

        # 분류 헤드 교체
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes, bias=True)

        print("✅ EfficientNetV2-L created via torchvision")
        return model

    except ImportError as e:
        raise RuntimeError(
            "EfficientNetV2-L 생성 실패: timm 또는 torchvision이 필요합니다.\n"
            "설치 방법: pip install timm torchvision\n"
            f"Import error: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"torchvision으로 EfficientNetV2-L 생성 실패: {e}\n"
            "timm 설치를 권장합니다: pip install timm"
        )


def get_model_info(model: nn.Module) -> dict:
    """모델 정보 반환"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,  # float32 기준
        "architecture": model.__class__.__name__,
    }
