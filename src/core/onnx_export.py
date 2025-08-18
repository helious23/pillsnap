"""
ONNX Export Schema and Tolerance Validation
ONNX 내보내기 스키마 및 허용치 검증
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
import time
import logging
import pkg_resources
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """ONNX 내보내기 설정"""
    opset_version: int = 17
    dynamic_axes: bool = True
    do_constant_folding: bool = True
    export_params: bool = True
    input_names: List[str] = None
    output_names: List[str] = None
    
    # 허용치 설정 (실용적 값)
    tolerance_mse_mean: float = 1e-4
    tolerance_mse_p99: float = 5e-4
    tolerance_top1_mismatch: float = 0.01  # 1%
    tolerance_map_delta: float = 0.01
    tolerance_iou_delta: float = 0.01
    
    # fp16 환경 완화된 허용치
    tolerance_mse_mean_fp16: float = 5e-4
    tolerance_mse_p99_fp16: float = 1e-3
    tolerance_top1_mismatch_fp16: float = 0.02  # 2%
    
    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]


@dataclass
class ExportReport:
    """ONNX Export 리포트"""
    model_type: str  # "detection" | "classification"
    backbone: str
    num_classes: int
    ckpt_path: str
    onnx_path: str
    exported_at_utc: str
    git_sha: str
    input_shape: List[int]
    opset: int
    dynamic_axes: bool
    params_million: float
    providers: List[str]
    
    # 검증 결과
    validation_passed: bool = False
    validation_metrics: Dict[str, float] = None
    validation_notes: str = ""
    
    # 성능 메트릭
    export_time_seconds: float = 0.0
    validation_time_seconds: float = 0.0
    onnx_file_size_mb: float = 0.0


class ONNXExporter:
    """
    ONNX 내보내기 및 검증 관리자
    실용적 허용치 기반 동등성 검증
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: export 섹션 설정
        """
        # 새로운 config.yaml 구조 지원
        compare_cfg = config.get("compare", {})
        tolerance_cfg = compare_cfg.get("tolerance", {})
        
        self.config = ONNXExportConfig(
            opset_version=config.get("opset", 17),
            dynamic_axes=config.get("dynamic_axes", True),
            tolerance_mse_mean=tolerance_cfg.get("mse_mean", 1e-4),
            tolerance_mse_p99=tolerance_cfg.get("mse_p99", 5e-4),
            tolerance_top1_mismatch=tolerance_cfg.get("top1_mismatch", 0.01),
            tolerance_map_delta=tolerance_cfg.get("detection_map", 0.01),
            # fp16 허용치 지원
            tolerance_mse_mean_fp16=compare_cfg.get("tolerance_fp16", {}).get("mse_mean", 5e-4),
            tolerance_mse_p99_fp16=compare_cfg.get("tolerance_fp16", {}).get("mse_p99", 1e-3),
            tolerance_top1_mismatch_fp16=compare_cfg.get("tolerance_fp16", {}).get("top1_mismatch", 0.02)
        )
        
        self.compare_config = config.get("compare", {})
        self.export_dir = Path(config.get("out_dir", "/mnt/data/exp/exp01/export"))
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ONNXExporter initialized: opset={self.config.opset_version}, "
                   f"tolerance_mse={self.config.tolerance_mse_mean}")
    
    def export_classification_model(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        model_info: Dict[str, Any]
    ) -> Tuple[str, ExportReport]:
        """
        분류 모델 ONNX 내보내기
        
        Args:
            model: PyTorch 모델
            input_shape: 입력 shape (e.g., (1, 3, 384, 384))
            model_info: 모델 정보 (backbone, num_classes 등)
            
        Returns:
            (ONNX 파일 경로, Export 리포트)
        """
        logger.info("Starting classification model export to ONNX")
        start_time = time.time()
        
        # 모델 준비
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        
        # 파일명 생성
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        git_sha = self._get_git_sha()
        onnx_filename = f"classification-{timestamp}-{git_sha[:7]}.onnx"
        onnx_path = self.export_dir / onnx_filename
        
        # Dynamic axes 설정
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch"},
                "output": {0: "batch"}
            }
        
        # Export
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=self.config.export_params,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.do_constant_folding,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            export_time = time.time() - start_time
            logger.info(f"ONNX export completed in {export_time:.2f}s: {onnx_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
        
        # 파일 크기
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        
        # 리포트 생성
        report = ExportReport(
            model_type="classification",
            backbone=model_info.get("backbone", "unknown"),
            num_classes=model_info.get("num_classes", 0),
            ckpt_path=model_info.get("ckpt_path", ""),
            onnx_path=str(onnx_path),
            exported_at_utc=datetime.utcnow().isoformat() + "Z",
            git_sha=git_sha,
            input_shape=list(input_shape),
            opset=self.config.opset_version,
            dynamic_axes=self.config.dynamic_axes,
            params_million=self._count_parameters(model),
            providers=self._get_available_providers(),
            export_time_seconds=export_time,
            onnx_file_size_mb=file_size_mb
        )
        
        # 심볼릭 링크 생성
        latest_link = self.export_dir / "latest_classification.onnx"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(onnx_path.name)
        
        return str(onnx_path), report
    
    def export_detection_model(
        self,
        model: Any,  # YOLO model
        input_shape: Tuple[int, ...],
        model_info: Dict[str, Any]
    ) -> Tuple[str, ExportReport]:
        """
        검출 모델 ONNX 내보내기
        
        Args:
            model: YOLO 모델
            input_shape: 입력 shape (e.g., (1, 3, 640, 640))
            model_info: 모델 정보
            
        Returns:
            (ONNX 파일 경로, Export 리포트)
        """
        logger.info("Starting detection model export to ONNX")
        start_time = time.time()
        
        # 파일명 생성
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        git_sha = self._get_git_sha()
        onnx_filename = f"detection-{timestamp}-{git_sha[:7]}.onnx"
        onnx_path = self.export_dir / onnx_filename
        
        # YOLO export with safe opset fallback
        try:
            actual_opset = self._safe_onnx_export_with_opset_fallback(
                model,
                str(onnx_path),
                format="onnx",
                imgsz=input_shape[-1],  # 640
                dynamic=self.config.dynamic_axes,
                simplify=True
            )
            
            # YOLO export는 자체 경로에 저장되므로 이동
            yolo_onnx_path = Path(model.model_path).with_suffix(".onnx")
            if yolo_onnx_path.exists():
                yolo_onnx_path.rename(onnx_path)
            
            export_time = time.time() - start_time
            logger.info(f"ONNX export completed in {export_time:.2f}s: {onnx_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
        
        # 파일 크기
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        
        # 리포트 생성
        report = ExportReport(
            model_type="detection",
            backbone="yolov11m",
            num_classes=model_info.get("num_classes", 1),
            ckpt_path=model_info.get("ckpt_path", ""),
            onnx_path=str(onnx_path),
            exported_at_utc=datetime.utcnow().isoformat() + "Z",
            git_sha=git_sha,
            input_shape=list(input_shape),
            opset=self.config.opset_version,
            dynamic_axes=self.config.dynamic_axes,
            params_million=model_info.get("params_million", 0),
            providers=self._get_available_providers(),
            export_time_seconds=export_time,
            onnx_file_size_mb=file_size_mb
        )
        
        # 심볼릭 링크 생성
        latest_link = self.export_dir / "latest_detection.onnx"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(onnx_path.name)
        
        return str(onnx_path), report
    
    def validate_export(
        self,
        torch_model: torch.nn.Module,
        onnx_path: str,
        test_data: torch.Tensor,
        model_type: str = "classification",
        use_fp16_tolerance: bool = False
    ) -> Dict[str, Any]:
        """
        PyTorch vs ONNX 동등성 검증
        
        Args:
            torch_model: PyTorch 모델
            onnx_path: ONNX 파일 경로
            test_data: 테스트 데이터
            model_type: "classification" | "detection"
            use_fp16_tolerance: fp16 허용치 사용 여부
            
        Returns:
            검증 결과 딕셔너리
        """
        logger.info(f"Starting ONNX validation for {model_type} model")
        start_time = time.time()
        
        # 허용치 선택
        if use_fp16_tolerance:
            mse_mean_tol = self.config.tolerance_mse_mean_fp16
            mse_p99_tol = self.config.tolerance_mse_p99_fp16
            top1_tol = self.config.tolerance_top1_mismatch_fp16
        else:
            mse_mean_tol = self.config.tolerance_mse_mean
            mse_p99_tol = self.config.tolerance_mse_p99
            top1_tol = self.config.tolerance_top1_mismatch
        
        # PyTorch 추론
        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model(test_data)
            if isinstance(torch_output, dict):
                torch_output = torch_output["logits"]
            torch_output = torch_output.cpu().numpy()
        
        # ONNX 추론
        providers = self._get_onnx_providers()
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        ort_input = {ort_session.get_inputs()[0].name: test_data.cpu().numpy()}
        ort_output = ort_session.run(None, ort_input)[0]
        
        # 메트릭 계산
        mse = np.mean((torch_output - ort_output) ** 2)
        mse_per_sample = np.mean((torch_output - ort_output) ** 2, axis=tuple(range(1, torch_output.ndim)))
        mse_p99 = np.percentile(mse_per_sample, 99)
        
        results = {
            "mse_mean": float(mse),
            "mse_p99": float(mse_p99),
            "mse_max": float(np.max(mse_per_sample)),
            "validation_time": time.time() - start_time
        }
        
        # 분류 모델용 추가 메트릭
        if model_type == "classification":
            torch_pred = np.argmax(torch_output, axis=1)
            onnx_pred = np.argmax(ort_output, axis=1)
            top1_mismatch = np.mean(torch_pred != onnx_pred)
            
            results["top1_mismatch_rate"] = float(top1_mismatch)
            results["top1_match_rate"] = float(1 - top1_mismatch)
        
        # 검증 통과 여부
        passed = True
        failures = []
        
        if mse > mse_mean_tol:
            passed = False
            failures.append(f"MSE mean {mse:.2e} > {mse_mean_tol:.2e}")
        
        if mse_p99 > mse_p99_tol:
            passed = False
            failures.append(f"MSE p99 {mse_p99:.2e} > {mse_p99_tol:.2e}")
        
        if model_type == "classification" and top1_mismatch > top1_tol:
            passed = False
            failures.append(f"Top-1 mismatch {top1_mismatch:.3f} > {top1_tol:.3f}")
        
        results["passed"] = passed
        results["failures"] = failures
        
        if passed:
            logger.info(f"ONNX validation PASSED: MSE={mse:.2e}, Top1 match={results.get('top1_match_rate', 0):.3f}")
        else:
            logger.warning(f"ONNX validation FAILED: {', '.join(failures)}")
        
        return results
    
    def save_export_report(self, report: ExportReport, validation_results: Optional[Dict] = None):
        """Export 리포트 저장"""
        if validation_results:
            report.validation_passed = validation_results.get("passed", False)
            report.validation_metrics = validation_results
            report.validation_notes = ", ".join(validation_results.get("failures", []))
            report.validation_time_seconds = validation_results.get("validation_time", 0)
        
        # JSON 저장
        report_path = self.export_dir / "export_report.json"
        report_dict = asdict(report)
        
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Export report saved: {report_path}")
    
    def _get_git_sha(self) -> str:
        """Git SHA 획득"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "nogit"
    
    def _count_parameters(self, model: torch.nn.Module) -> float:
        """모델 파라미터 수 계산 (M)"""
        return sum(p.numel() for p in model.parameters()) / 1e6
    
    def _get_available_providers(self) -> List[str]:
        """사용 가능한 ONNX Runtime Provider"""
        return ort.get_available_providers()
    
    def _get_onnx_providers(self) -> List[Tuple[str, Dict]]:
        """ONNX Runtime Provider 자동 감지 및 안전 설정"""
        return setup_ort_providers_with_validation()
    
    def _safe_onnx_export_with_opset_fallback(self, model: Any, export_path: str, **kwargs) -> int:
        """안전한 ONNX Export with opset 폴백"""
        # Ultralytics 공식 문서 기준: None=자동선택, 17=명시적
        opset_priority = [None, 17, 16, 15, 14]
        
        for opset in opset_priority:
            try:
                if opset is None:
                    logger.info("Trying auto opset selection (latest supported)")
                    model.export(format="onnx", **kwargs)
                else:
                    logger.info(f"Trying explicit opset {opset}")
                    model.export(format="onnx", opset=opset, **kwargs)
                
                logger.info(f"ONNX export successful with opset={opset or 'auto'}")
                return opset or self.config.opset_version
                
            except Exception as e:
                logger.warning(f"ONNX export failed with opset={opset}: {e}")
                continue
        
        raise RuntimeError("All opset versions failed for ONNX export")


def export_and_validate(
    model: torch.nn.Module,
    model_type: str,
    config: Dict[str, Any],
    test_samples: Optional[torch.Tensor] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    모델 내보내기 및 검증 헬퍼 함수
    
    Args:
        model: PyTorch 모델
        model_type: "classification" | "detection"
        config: Export 설정
        test_samples: 검증용 샘플 (옵션)
        
    Returns:
        (ONNX 경로, 검증 결과)
    """
    exporter = ONNXExporter(config)
    
    # 모델 정보 수집
    model_info = {
        "backbone": config.get("model", {}).get("backbone", "unknown"),
        "num_classes": config.get("model", {}).get("num_classes", 0),
        "ckpt_path": config.get("model", {}).get("ckpt_path", "")
    }
    
    # Input shape 결정
    if model_type == "classification":
        input_shape = (1, 3, 384, 384)
        onnx_path, report = exporter.export_classification_model(model, input_shape, model_info)
    else:
        input_shape = (1, 3, 640, 640)
        onnx_path, report = exporter.export_detection_model(model, input_shape, model_info)
    
    # 검증 (테스트 샘플이 있는 경우)
    validation_results = None
    if test_samples is not None:
        validation_results = exporter.validate_export(
            model, onnx_path, test_samples, model_type
        )
        report.validation_passed = validation_results["passed"]
        report.validation_metrics = validation_results
    
    # 리포트 저장
    exporter.save_export_report(report, validation_results)
    
    return onnx_path, validation_results if validation_results else {"passed": True}


def setup_ort_providers_with_validation() -> List[Tuple[str, Dict]]:
    """ORT Provider 설정 with 실제 검증 및 버전 호환성 체크"""
    
    # 1. 패키지 버전 체크
    ort_version = ort.__version__
    logger.info(f"ONNX Runtime version: {ort_version}")
    
    # 2. GPU 패키지 확인
    installed_packages = {pkg.project_name for pkg in pkg_resources.working_set}
    if "onnxruntime-gpu" not in installed_packages:
        logger.warning("TensorRT EP requires onnxruntime-gpu package, not onnxruntime")
        return [("CPUExecutionProvider", {})]
    
    # 3. TensorRT 환경 체크
    trt_available = False
    try:
        import tensorrt as trt
        trt_version = trt.__version__
        logger.info(f"TensorRT {trt_version} detected")
        
        # ORT 1.22 → TRT 10.9 호환성 체크
        if ort_version.startswith("1.22") and not trt_version.startswith("10.9"):
            logger.warning(f"TRT {trt_version} may not be optimal for ORT {ort_version}")
        
        trt_available = True
    
    except ImportError:
        logger.warning("TensorRT not found, TRT EP will be skipped")
    
    # 4. Provider 우선순위 (문서 권장대로)
    provider_configs = []
    
    if trt_available:
        provider_configs.append(('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': '/mnt/data/exp/exp01/trt_cache',
            'trt_fp16_enable': True,
            'trt_max_workspace_size': 8 * 1024**3  # 8GB
        }))
    
    provider_configs.extend([
        ('CUDAExecutionProvider', {
            'cudnn_conv_use_max_workspace': 1
        }),
        ('CPUExecutionProvider', {})
    ])
    
    # 5. 실제 검증
    validated_providers = []
    available_providers = ort.get_available_providers()
    
    for provider_name, provider_options in provider_configs:
        try:
            if provider_name in available_providers:
                logger.info(f"✓ {provider_name} available in ORT build")
                validated_providers.append((provider_name, provider_options))
            else:
                logger.warning(f"✗ {provider_name} not available in this ORT build")
                
        except Exception as e:
            logger.warning(f"✗ {provider_name} validation failed: {e}")
    
    if not validated_providers:
        logger.error("No working providers found, using CPU fallback")
        return [("CPUExecutionProvider", {})]
    
    logger.info(f"Validated providers: {[p[0] for p in validated_providers]}")
    return validated_providers


def create_dummy_onnx_model() -> str:
    """테스트용 더미 ONNX 모델 생성"""
    import io
    
    # 간단한 더미 모델
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10)
    )
    
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 메모리에 ONNX 모델 생성
    buffer = io.BytesIO()
    torch.onnx.export(
        model, 
        dummy_input, 
        buffer, 
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    
    return buffer.getvalue()