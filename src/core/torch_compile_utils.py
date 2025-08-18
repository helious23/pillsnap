"""
torch.compile Safe Compilation with Fallback
PyTorch 2.5 torch.compile 안전 실행 및 폴백 관리
"""

import torch
import logging
from typing import Optional, Literal, Any
from enum import Enum

logger = logging.getLogger(__name__)

CompileMode = Literal["default", "reduce-overhead", "max-autotune", None]


class CompileStage(Enum):
    """컴파일 단계별 안전성 우선순위"""
    TRAINING = "training"        # 학습: 안정성 우선
    INFERENCE = "inference"      # 추론: 성능 우선
    EXPORT = "export"           # 내보내기: 호환성 우선


def safe_torch_compile_for_stage(
    model: torch.nn.Module, 
    stage: CompileStage = CompileStage.TRAINING,
    force_mode: Optional[CompileMode] = None
) -> torch.nn.Module:
    """
    단계별 안전한 torch.compile
    
    Args:
        model: PyTorch 모델
        stage: 컴파일 단계 (training/inference/export)
        force_mode: 강제 모드 지정 (None이면 단계별 기본값)
        
    Returns:
        컴파일된 모델 또는 원본 모델 (폴백)
    """
    # PyTorch 버전 체크
    torch_version = torch.__version__
    if not torch_version.startswith("2."):
        logger.warning(f"torch.compile requires PyTorch 2.x, got {torch_version}")
        return model
    
    # 강제 모드가 지정된 경우
    if force_mode is not None:
        return _try_compile_with_mode(model, force_mode, f"forced mode {force_mode}")
    
    # 단계별 모드 우선순위
    if stage == CompileStage.TRAINING:
        # 학습: 안정성 우선 (공식 문서 권장)
        compile_modes = ["reduce-overhead", "default", None]
        reason = "training stability"
    elif stage == CompileStage.INFERENCE:
        # 추론: 성능 우선
        compile_modes = ["max-autotune", "reduce-overhead", "default", None]
        reason = "inference performance"
    else:  # EXPORT
        # 내보내기: 호환성 우선
        compile_modes = ["default", None]
        reason = "export compatibility"
    
    logger.info(f"torch.compile for {stage.value} ({reason})")
    
    for mode in compile_modes:
        compiled_model = _try_compile_with_mode(model, mode, reason)
        if compiled_model is not None:
            return compiled_model
    
    # 모든 시도 실패
    logger.error("All torch.compile attempts failed, using vanilla model")
    return model


def _try_compile_with_mode(
    model: torch.nn.Module, 
    mode: Optional[CompileMode], 
    reason: str
) -> Optional[torch.nn.Module]:
    """지정된 모드로 컴파일 시도"""
    
    try:
        if mode is None:
            logger.info(f"torch.compile disabled ({reason}), using vanilla model")
            return model
        
        logger.info(f"Trying torch.compile mode='{mode}' for {reason}")
        compiled_model = torch.compile(model, mode=mode)
        
        # 컴파일 검증 (더미 입력으로 테스트)
        if torch.cuda.is_available():
            test_input = torch.randn(1, 3, 224, 224).cuda()
            model_device = next(model.parameters()).device
            test_input = test_input.to(model_device)
        else:
            test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            _ = compiled_model(test_input)
        
        logger.info(f"✓ torch.compile successful with mode='{mode}'")
        return compiled_model
        
    except Exception as e:
        logger.warning(f"✗ torch.compile mode='{mode}' failed: {e}")
        return None


def get_compilation_info() -> dict:
    """현재 torch.compile 환경 정보"""
    info = {
        "torch_version": torch.__version__,
        "compile_available": hasattr(torch, 'compile'),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0)
        })
    
    # PyTorch 2.5 특정 기능 체크
    try:
        # 2.5에서 추가된 기능들 체크
        info["has_inductor"] = hasattr(torch._inductor, 'compile')
        info["has_cuda_graphs"] = hasattr(torch.cuda, 'CUDAGraph')
    except:
        info["has_inductor"] = False
        info["has_cuda_graphs"] = False
    
    return info


def safe_torch_compile_for_training(model: torch.nn.Module) -> torch.nn.Module:
    """학습용 안전한 torch.compile (PART_D 호환)"""
    return safe_torch_compile_for_stage(model, CompileStage.TRAINING)


def safe_torch_compile_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """추론용 안전한 torch.compile (PART_F API 호환)"""
    return safe_torch_compile_for_stage(model, CompileStage.INFERENCE)


def safe_torch_compile_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """ONNX 내보내기용 안전한 torch.compile (PART_E 호환)"""
    return safe_torch_compile_for_stage(model, CompileStage.EXPORT)


def validate_torch_compile_environment() -> bool:
    """torch.compile 환경 검증"""
    info = get_compilation_info()
    
    if not info["compile_available"]:
        logger.error("torch.compile not available")
        return False
    
    if not info["torch_version"].startswith("2."):
        logger.error(f"torch.compile requires PyTorch 2.x, got {info['torch_version']}")
        return False
    
    if info["cuda_available"]:
        logger.info(f"CUDA environment: {info['gpu_name']}, CUDA {info['cuda_version']}")
    else:
        logger.warning("CUDA not available, torch.compile will use CPU backend")
    
    logger.info(f"torch.compile environment validated: PyTorch {info['torch_version']}")
    return True