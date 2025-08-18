"""
Compatibility Validator - 전체 시스템 호환성 검증
PyTorch 2.5, ONNX Runtime, TensorRT 등 모든 컴포넌트 호환성 체크
"""

import sys
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pkg_resources

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """호환성 검사 결과"""
    component: str
    version: str
    required_version: str
    compatible: bool
    message: str
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CompatibilityValidator:
    """시스템 호환성 검증기"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: compatibility 섹션 설정
        """
        self.config = config
        self.results = []
        self.critical_failures = []
        self.warnings = []
        
    def validate_all(self) -> Tuple[bool, List[CompatibilityResult]]:
        """모든 컴포넌트 호환성 검증"""
        logger.info("Starting comprehensive compatibility validation...")
        
        # 1. PyTorch 스택 검증
        self._validate_pytorch_stack()
        
        # 2. ONNX Runtime 검증  
        self._validate_onnxruntime()
        
        # 3. CUDA 환경 검증
        if self.config.get("validation", {}).get("check_cuda", True):
            self._validate_cuda_environment()
        
        # 4. TensorRT 검증 (선택적)
        if self.config.get("validation", {}).get("check_tensorrt", False):
            self._validate_tensorrt()
        
        # 5. Git 환경 검증
        if self.config.get("validation", {}).get("check_git", True):
            self._validate_git_environment()
        
        # 6. 파일 시스템 검증
        self._validate_filesystem()
        
        # 결과 종합
        all_compatible = len(self.critical_failures) == 0
        
        if all_compatible:
            logger.info("✓ All compatibility checks passed")
        else:
            logger.error(f"✗ {len(self.critical_failures)} critical compatibility issues found")
            for failure in self.critical_failures:
                logger.error(f"  - {failure}")
        
        if self.warnings:
            logger.warning(f"Compatibility warnings: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        return all_compatible, self.results
    
    def _validate_pytorch_stack(self):
        """PyTorch 스택 호환성 검증"""
        pytorch_config = self.config.get("pytorch", {})
        min_version = pytorch_config.get("min_version", "2.5.0")
        preferred_version = pytorch_config.get("preferred_version", "2.5.1")
        
        # PyTorch 버전 체크
        try:
            import torch
            torch_version = torch.__version__
            
            result = CompatibilityResult(
                component="PyTorch",
                version=torch_version,
                required_version=f">={min_version}",
                compatible=self._version_compare(torch_version, min_version) >= 0,
                message=f"PyTorch {torch_version}"
            )
            
            if not result.compatible:
                result.message = f"PyTorch {torch_version} < required {min_version}"
                self.critical_failures.append(f"PyTorch version too old: {torch_version}")
            elif torch_version != preferred_version:
                result.warnings.append(f"Preferred version is {preferred_version}")
            
            self.results.append(result)
            
        except ImportError:
            self.critical_failures.append("PyTorch not installed")
            return
        
        # TorchVision 호환성 체크
        try:
            import torchvision
            tv_version = torchvision.__version__
            
            # PyTorch와 TorchVision 버전 페어링 체크
            expected_tv_version = self._get_expected_torchvision_version(torch_version)
            
            result = CompatibilityResult(
                component="TorchVision",
                version=tv_version,
                required_version=expected_tv_version,
                compatible=tv_version.startswith(expected_tv_version[:4]),  # 메이저.마이너 일치
                message=f"TorchVision {tv_version}"
            )
            
            if not result.compatible:
                result.message = f"TorchVision {tv_version} may not be compatible with PyTorch {torch_version}"
                self.warnings.append(f"TorchVision version mismatch: got {tv_version}, expected {expected_tv_version}")
            
            self.results.append(result)
            
        except ImportError:
            self.warnings.append("TorchVision not installed")
        
        # torch.compile 검증
        if pytorch_config.get("torch_compile_required", True):
            if hasattr(torch, 'compile'):
                self.results.append(CompatibilityResult(
                    component="torch.compile",
                    version="available",
                    required_version="required",
                    compatible=True,
                    message="torch.compile available"
                ))
            else:
                self.critical_failures.append("torch.compile not available (PyTorch 2.x required)")
    
    def _validate_onnxruntime(self):
        """ONNX Runtime 호환성 검증"""
        ort_config = self.config.get("onnxruntime", {})
        min_version = ort_config.get("min_version", "1.20.0")
        preferred_version = ort_config.get("preferred_version", "1.22.0")
        require_gpu = ort_config.get("require_gpu_build", True)
        
        try:
            import onnxruntime as ort
            ort_version = ort.__version__
            
            result = CompatibilityResult(
                component="ONNX Runtime",
                version=ort_version,
                required_version=f">={min_version}",
                compatible=self._version_compare(ort_version, min_version) >= 0,
                message=f"ONNX Runtime {ort_version}"
            )
            
            if not result.compatible:
                result.message = f"ONNX Runtime {ort_version} < required {min_version}"
                self.critical_failures.append(f"ONNX Runtime version too old: {ort_version}")
            
            # GPU 빌드 확인
            if require_gpu:
                installed_packages = {pkg.project_name for pkg in pkg_resources.working_set}
                if "onnxruntime-gpu" not in installed_packages:
                    result.warnings.append("CPU-only ONNX Runtime detected, GPU build recommended")
                    self.warnings.append("Consider installing onnxruntime-gpu for better performance")
            
            # Provider 검증
            if ort_config.get("validate_providers", True):
                available_providers = ort.get_available_providers()
                
                if "CUDAExecutionProvider" in available_providers:
                    result.warnings.append("CUDA EP available")
                else:
                    result.warnings.append("CUDA EP not available")
                
                if "TensorrtExecutionProvider" in available_providers:
                    result.warnings.append("TensorRT EP available")
                else:
                    result.warnings.append("TensorRT EP not available (optional)")
            
            self.results.append(result)
            
        except ImportError:
            self.critical_failures.append("ONNX Runtime not installed")
    
    def _validate_cuda_environment(self):
        """CUDA 환경 검증"""
        try:
            import torch
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
                
                result = CompatibilityResult(
                    component="CUDA",
                    version=cuda_version,
                    required_version="12.x recommended",
                    compatible=True,
                    message=f"CUDA {cuda_version}, {gpu_count} GPU(s), {gpu_name}"
                )
                
                # CUDA 12.x 권장 체크
                if not cuda_version.startswith("12."):
                    result.warnings.append(f"CUDA {cuda_version} detected, 12.x recommended for PyTorch 2.5")
                
                self.results.append(result)
            else:
                self.warnings.append("CUDA not available - CPU-only mode")
                
        except Exception as e:
            self.warnings.append(f"CUDA validation failed: {e}")
    
    def _validate_tensorrt(self):
        """TensorRT 환경 검증"""
        try:
            import tensorrt as trt
            trt_version = trt.__version__
            
            result = CompatibilityResult(
                component="TensorRT",
                version=trt_version,
                required_version="10.9 recommended",
                compatible=True,
                message=f"TensorRT {trt_version}"
            )
            
            # ORT 1.22와 TRT 10.9 호환성 체크
            if not trt_version.startswith("10.9"):
                result.warnings.append("TensorRT 10.9 recommended for ONNX Runtime 1.22")
            
            self.results.append(result)
            
        except ImportError:
            self.warnings.append("TensorRT not installed (optional)")
    
    def _validate_git_environment(self):
        """Git 환경 검증"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Git SHA 획득
            sha_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            
            git_sha = sha_result.stdout.strip()
            
            self.results.append(CompatibilityResult(
                component="Git",
                version=git_sha,
                required_version="any",
                compatible=True,
                message=f"Git repository, SHA: {git_sha}"
            ))
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.warnings.append("Git not available or not in repository")
    
    def _validate_filesystem(self):
        """파일 시스템 검증"""
        # WSL 경로 검증
        required_paths = [
            "/mnt/data",
            "/home",
            "/tmp"
        ]
        
        for path in required_paths:
            if Path(path).exists():
                self.results.append(CompatibilityResult(
                    component=f"Filesystem",
                    version=path,
                    required_version="exists",
                    compatible=True,
                    message=f"Path {path} accessible"
                ))
            else:
                self.warnings.append(f"Path {path} not accessible")
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """버전 비교 (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
        
        try:
            v1_tuple = version_tuple(version1.split('+')[0])  # +cu121 등 제거
            v2_tuple = version_tuple(version2.split('+')[0])
            
            if v1_tuple < v2_tuple:
                return -1
            elif v1_tuple > v2_tuple:
                return 1
            else:
                return 0
        except:
            return 0  # 비교 실패 시 동일한 것으로 간주
    
    def _get_expected_torchvision_version(self, torch_version: str) -> str:
        """PyTorch 버전에 대응하는 TorchVision 버전 반환"""
        version_mapping = {
            "2.5.1": "0.20.1",
            "2.5.0": "0.20.0",
            "2.4.1": "0.19.1",
            "2.4.0": "0.19.0"
        }
        
        torch_base = torch_version.split('+')[0]  # +cu121 등 제거
        return version_mapping.get(torch_base, "0.20.1")  # 기본값
    
    def get_summary(self) -> str:
        """호환성 검사 요약 반환"""
        compatible_count = sum(1 for r in self.results if r.compatible)
        total_count = len(self.results)
        
        summary = f"Compatibility Summary: {compatible_count}/{total_count} compatible\n"
        summary += f"Critical failures: {len(self.critical_failures)}\n"
        summary += f"Warnings: {len(self.warnings)}\n\n"
        
        for result in self.results:
            status = "✓" if result.compatible else "✗"
            summary += f"{status} {result.component}: {result.message}\n"
            for warning in result.warnings:
                summary += f"  ⚠ {warning}\n"
        
        if self.critical_failures:
            summary += f"\nCritical Issues:\n"
            for failure in self.critical_failures:
                summary += f"  ✗ {failure}\n"
        
        return summary


def validate_system_compatibility(config: Dict[str, Any]) -> bool:
    """시스템 호환성 검증 헬퍼 함수"""
    compatibility_config = config.get("compatibility", {})
    validator = CompatibilityValidator(compatibility_config)
    
    is_compatible, results = validator.validate_all()
    
    # 결과 출력
    print(validator.get_summary())
    
    # 경고 전용 모드 체크
    warn_only = compatibility_config.get("validation", {}).get("warn_only", False)
    
    if not is_compatible and not warn_only:
        logger.error("System compatibility validation failed")
        return False
    elif not is_compatible and warn_only:
        logger.warning("System compatibility issues detected but proceeding (warn_only=true)")
    
    return True