"""
Detector Manager - Lazy Load/Unload with TTL
검출 모델 지연 로딩 및 자동 언로드 관리
"""

import torch
import time
import threading
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DetectorState:
    """검출기 상태 관리"""
    loaded: bool = False
    last_used: Optional[float] = None
    load_count: int = 0
    total_inferences: int = 0
    last_load_time: Optional[float] = None
    last_unload_time: Optional[float] = None
    

class DetectorManager:
    """
    YOLOv11m 검출 모델 지연 로딩/언로드 관리
    - Load Once Guard: 첫 combo 요청 시만 로드
    - Idle TTL Reaper: 10분 유휴 시 자동 언로드
    - Hysteresis: 로드/언로드 사이 2분 최소 체류
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        idle_ttl_minutes: int = 10,
        hysteresis_minutes: int = 2,
        device: str = "cuda",
        lazy_load: bool = True
    ):
        """
        Args:
            model_path: 검출 모델 경로
            idle_ttl_minutes: 유휴 시간 후 언로드 (분)
            hysteresis_minutes: 로드/언로드 간 최소 시간 (분)
            device: 실행 디바이스
            lazy_load: 지연 로딩 활성화
        """
        self.model_path = model_path
        self.idle_ttl = timedelta(minutes=idle_ttl_minutes)
        self.hysteresis = timedelta(minutes=hysteresis_minutes)
        self.device = device
        self.lazy_load = lazy_load
        
        # 상태 관리
        self.state = DetectorState()
        self.model: Optional[Any] = None  # YOLO 모델
        self.lock = threading.Lock()
        self.reaper_thread: Optional[threading.Thread] = None
        self.stop_reaper = threading.Event()
        
        # 설정 로깅
        logger.info(f"DetectorManager initialized: lazy_load={lazy_load}, idle_ttl={idle_ttl_minutes}min")
        
        # 즉시 로드 모드
        if not lazy_load and model_path:
            self._load_model()
    
    def _load_model(self) -> bool:
        """모델 로드 (내부 메서드)"""
        if self.state.loaded:
            logger.debug("Model already loaded")
            return True
        
        if not self.model_path or not Path(self.model_path).exists():
            logger.error(f"Model path invalid: {self.model_path}")
            return False
        
        try:
            logger.info(f"Loading detection model from {self.model_path}")
            start_time = time.time()
            
            # YOLOv11m 로드 (ultralytics 가정)
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)
            
            # 워밍업
            dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
            _ = self.model(dummy_input, verbose=False)
            
            load_time = time.time() - start_time
            
            # 상태 업데이트
            self.state.loaded = True
            self.state.last_load_time = time.time()
            self.state.load_count += 1
            
            logger.info(f"Detection model loaded in {load_time:.2f}s (load_count={self.state.load_count})")
            
            # Reaper 스레드 시작
            if self.lazy_load:
                self._start_reaper()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            return False
    
    def _unload_model(self) -> bool:
        """모델 언로드 (내부 메서드)"""
        if not self.state.loaded:
            logger.debug("Model already unloaded")
            return True
        
        # Hysteresis 체크
        if self.state.last_load_time:
            time_since_load = time.time() - self.state.last_load_time
            if time_since_load < self.hysteresis.total_seconds():
                logger.debug(f"Hysteresis active: {time_since_load:.1f}s < {self.hysteresis.total_seconds()}s")
                return False
        
        try:
            logger.info("Unloading detection model")
            
            # 모델 정리
            if self.model is not None:
                del self.model
                self.model = None
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 상태 업데이트
            self.state.loaded = False
            self.state.last_unload_time = time.time()
            
            logger.info(f"Detection model unloaded (total_inferences={self.state.total_inferences})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload detection model: {e}")
            return False
    
    def get_model(self) -> Optional[Any]:
        """
        모델 획득 (필요 시 자동 로드)
        
        Returns:
            로드된 모델 또는 None
        """
        with self.lock:
            # 미로드 상태면 로드
            if not self.state.loaded:
                if not self._load_model():
                    return None
            
            # 사용 시간 업데이트
            self.state.last_used = time.time()
            
            return self.model
    
    def predict(self, image: Union[torch.Tensor, Any], **kwargs) -> Optional[Any]:
        """
        검출 수행
        
        Args:
            image: 입력 이미지
            **kwargs: YOLO 추론 파라미터
            
        Returns:
            검출 결과 또는 None
        """
        model = self.get_model()
        if model is None:
            logger.error("Failed to get detection model")
            return None
        
        try:
            # 추론 수행
            results = model(image, **kwargs)
            
            # 통계 업데이트
            with self.lock:
                self.state.total_inferences += 1
                self.state.last_used = time.time()
            
            return results
            
        except Exception as e:
            logger.error(f"Detection inference failed: {e}")
            return None
    
    def _start_reaper(self):
        """Idle TTL Reaper 스레드 시작"""
        if self.reaper_thread and self.reaper_thread.is_alive():
            return
        
        self.stop_reaper.clear()
        self.reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True)
        self.reaper_thread.start()
        logger.debug("Reaper thread started")
    
    def _reaper_loop(self):
        """Reaper 루프 (백그라운드 스레드)"""
        check_interval = 60  # 1분마다 체크
        
        while not self.stop_reaper.is_set():
            time.sleep(check_interval)
            
            with self.lock:
                if not self.state.loaded:
                    continue
                
                if self.state.last_used is None:
                    continue
                
                # Idle 시간 체크
                idle_time = time.time() - self.state.last_used
                if idle_time > self.idle_ttl.total_seconds():
                    logger.info(f"Model idle for {idle_time/60:.1f} minutes, unloading")
                    self._unload_model()
    
    def force_load(self) -> bool:
        """강제 로드"""
        with self.lock:
            return self._load_model()
    
    def force_unload(self) -> bool:
        """강제 언로드 (hysteresis 무시)"""
        with self.lock:
            # Hysteresis 임시 비활성화
            original_last_load = self.state.last_load_time
            self.state.last_load_time = None
            
            result = self._unload_model()
            
            # 복원
            if not result:
                self.state.last_load_time = original_last_load
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        with self.lock:
            idle_time = None
            if self.state.loaded and self.state.last_used:
                idle_time = time.time() - self.state.last_used
            
            return {
                "loaded": self.state.loaded,
                "load_count": self.state.load_count,
                "total_inferences": self.state.total_inferences,
                "idle_time_seconds": idle_time,
                "model_path": self.model_path,
                "lazy_load": self.lazy_load,
                "idle_ttl_minutes": self.idle_ttl.total_seconds() / 60,
                "hysteresis_minutes": self.hysteresis.total_seconds() / 60
            }
    
    def update_model_path(self, new_path: str) -> bool:
        """모델 경로 업데이트 (리로드 필요)"""
        with self.lock:
            # 기존 모델 언로드
            if self.state.loaded:
                self._unload_model()
            
            # 새 경로 설정
            self.model_path = new_path
            
            # 즉시 로드 모드면 로드
            if not self.lazy_load:
                return self._load_model()
            
            return True
    
    def shutdown(self):
        """매니저 종료"""
        logger.info("Shutting down DetectorManager")
        
        # Reaper 스레드 중지
        self.stop_reaper.set()
        if self.reaper_thread:
            self.reaper_thread.join(timeout=5)
        
        # 모델 언로드
        with self.lock:
            if self.state.loaded:
                self._unload_model()
        
        logger.info("DetectorManager shutdown complete")


# 전역 싱글톤 인스턴스
_detector_manager: Optional[DetectorManager] = None


def get_detector_manager(
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None
) -> DetectorManager:
    """
    DetectorManager 싱글톤 반환
    
    Args:
        config: 설정 딕셔너리
        model_path: 모델 경로 (오버라이드)
        
    Returns:
        DetectorManager 인스턴스
    """
    global _detector_manager
    
    if _detector_manager is None:
        if config is None:
            config = {}
        
        # 설정에서 값 추출
        lazy_load = config.get("inference", {}).get("lazy_load_detector", True)
        idle_ttl = config.get("inference", {}).get("detector_idle_ttl_minutes", 10)
        hysteresis = config.get("inference", {}).get("detector_hysteresis_minutes", 2)
        
        # 모델 경로 결정
        if model_path is None:
            model_path = config.get("detection", {}).get("model_path")
        
        _detector_manager = DetectorManager(
            model_path=model_path,
            idle_ttl_minutes=idle_ttl,
            hysteresis_minutes=hysteresis,
            lazy_load=lazy_load
        )
    
    return _detector_manager


def reset_detector_manager():
    """매니저 리셋 (테스트용)"""
    global _detector_manager
    if _detector_manager:
        _detector_manager.shutdown()
    _detector_manager = None