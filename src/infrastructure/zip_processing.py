"""
Large-scale ZIP Processing with Integrity Validation
대용량 ZIP 처리를 위한 무결성 검증 및 샤딩 스트리밍
"""

import zipfile
import random
import time
import queue
import threading
import psutil
import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ZipIntegrityLevel(Enum):
    """ZIP 무결성 검증 레벨"""
    SKIP = "skip"           # 무결성 검사 생략 (최고 속도)
    QUICK = "quick"         # 헤더 + 샘플링 검증 (권장)
    FULL = "full"          # 전체 CRC32 검증 (최고 안전성)


@dataclass
class ZipProcessingConfig:
    """ZIP 처리 설정"""
    integrity_level: ZipIntegrityLevel = ZipIntegrityLevel.QUICK
    shard_size_gb: int = 8
    max_concurrent_extracts: int = 2
    stream_loading: bool = True
    sample_ratio: float = 0.1  # quick 모드 샘플링 비율
    max_memory_gb: int = 64    # 최대 메모리 사용량


class SafeZipLoader:
    """안전한 ZIP 로더 with 옵션화된 무결성 검증"""
    
    def __init__(self, config: ZipProcessingConfig):
        self.config = config
        self.corruption_count = 0
        self.processed_files = []
        self.extraction_stats = {}
        
    def load_zip_safe(self, zip_path: Path, extract_to: Path) -> bool:
        """무결성 레벨에 따른 안전한 ZIP 로딩"""
        
        if not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            return False
        
        logger.info(f"Loading {zip_path} with integrity level: {self.config.integrity_level.value}")
        start_time = time.time()
        
        try:
            if self.config.integrity_level == ZipIntegrityLevel.SKIP:
                success = self._fast_extract(zip_path, extract_to)
            elif self.config.integrity_level == ZipIntegrityLevel.QUICK:
                success = self._quick_verify_extract(zip_path, extract_to)
            else:  # FULL
                success = self._full_verify_extract(zip_path, extract_to)
            
            # 통계 기록
            extract_time = time.time() - start_time
            file_size_mb = zip_path.stat().st_size / (1024 * 1024)
            
            self.extraction_stats[str(zip_path)] = {
                "success": success,
                "time_seconds": extract_time,
                "size_mb": file_size_mb,
                "speed_mbps": file_size_mb / extract_time if extract_time > 0 else 0,
                "integrity_level": self.config.integrity_level.value
            }
            
            if success:
                logger.info(f"✓ {zip_path.name} extracted in {extract_time:.1f}s "
                           f"({file_size_mb/extract_time:.1f} MB/s)")
            else:
                logger.error(f"✗ {zip_path.name} extraction failed")
                self.corruption_count += 1
            
            return success
            
        except Exception as e:
            logger.error(f"ZIP loading exception for {zip_path}: {e}")
            return False
    
    def _fast_extract(self, zip_path: Path, extract_to: Path) -> bool:
        """고속 추출 (무결성 검사 생략)"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
                return True
        except Exception as e:
            logger.error(f"Fast extract failed for {zip_path}: {e}")
            return False
    
    def _quick_verify_extract(self, zip_path: Path, extract_to: Path) -> bool:
        """빠른 검증: 헤더 + 샘플링"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 1. ZIP 헤더 무결성 체크
                bad_entries = zf.testzip()
                if bad_entries:
                    logger.error(f"Corrupted entries in {zip_path}: {bad_entries}")
                    return False
                
                # 2. 샘플링 검증
                all_files = zf.namelist()
                sample_size = max(1, int(len(all_files) * self.config.sample_ratio))
                sample_files = random.sample(all_files, sample_size)
                
                logger.debug(f"Sampling {sample_size}/{len(all_files)} files for verification")
                
                for filename in sample_files:
                    try:
                        zf.read(filename)  # CRC 자동 검증
                    except zipfile.BadZipFile:
                        logger.error(f"Corrupted file {filename} in {zip_path}")
                        return False
                
                # 3. 안전한 추출
                zf.extractall(extract_to)
                return True
                
        except Exception as e:
            logger.error(f"Quick verify extract failed for {zip_path}: {e}")
            return False
    
    def _full_verify_extract(self, zip_path: Path, extract_to: Path) -> bool:
        """전체 검증: 모든 파일 CRC32 체크"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 1. ZIP 헤더 무결성 체크
                bad_entries = zf.testzip()
                if bad_entries:
                    logger.error(f"Corrupted entries in {zip_path}: {bad_entries}")
                    return False
                
                # 2. 모든 파일 CRC 검증
                all_files = zf.namelist()
                logger.info(f"Full verification of {len(all_files)} files")
                
                for i, filename in enumerate(all_files):
                    if i % 1000 == 0:
                        logger.debug(f"Verifying {i}/{len(all_files)} files...")
                    
                    try:
                        zf.read(filename)  # CRC 자동 검증
                    except zipfile.BadZipFile:
                        logger.error(f"Corrupted file {filename} in {zip_path}")
                        return False
                
                # 3. 안전한 추출
                zf.extractall(extract_to)
                return True
                
        except Exception as e:
            logger.error(f"Full verify extract failed for {zip_path}: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        if not self.extraction_stats:
            return {"total_files": 0, "corruption_count": 0}
        
        total_time = sum(s["time_seconds"] for s in self.extraction_stats.values())
        total_size = sum(s["size_mb"] for s in self.extraction_stats.values())
        successful_files = sum(1 for s in self.extraction_stats.values() if s["success"])
        
        return {
            "total_files": len(self.extraction_stats),
            "successful_files": successful_files,
            "corruption_count": self.corruption_count,
            "total_time_seconds": total_time,
            "total_size_mb": total_size,
            "average_speed_mbps": total_size / total_time if total_time > 0 else 0,
            "integrity_level": self.config.integrity_level.value
        }


class ShardedDataLoader:
    """대용량 데이터셋을 샤드 단위로 스트리밍 로딩"""
    
    def __init__(self, config: ZipProcessingConfig):
        self.config = config
        self.shard_size = config.shard_size_gb * 1024**3  # bytes
        self.max_memory = config.max_memory_gb * 1024**3
        self.active_shards = {}
        self.shard_queue = queue.Queue(maxsize=3)  # 3개 샤드 미리 로드
        
    def stream_dataset(self, zip_files: List[Path]) -> Iterator[Dict[str, Any]]:
        """ZIP 파일들을 샤드 단위로 스트리밍"""
        
        if not self.config.stream_loading:
            # 스트리밍 비활성화 시 기본 로딩
            yield from self._traditional_loading(zip_files)
            return
        
        # 1. 샤드 계획 수립
        shards = self._plan_shards(zip_files)
        logger.info(f"Planned {len(shards)} shards for streaming")
        
        # 2. 백그라운드 로더 시작
        loader_thread = threading.Thread(
            target=self._background_loader, 
            args=(shards,), 
            daemon=True
        )
        loader_thread.start()
        
        # 3. 샤드 단위 순차 처리
        for shard_id in range(len(shards)):
            try:
                shard_data = self.shard_queue.get(timeout=600)  # 10분 타임아웃
                
                logger.info(f"Processing shard {shard_id+1}/{len(shards)}")
                yield from self._process_shard(shard_data)
                
            except queue.Empty:
                logger.error(f"Shard {shard_id} loading timeout")
                break
            finally:
                # 메모리 정리
                self._cleanup_shard(shard_data.get('id', shard_id))
    
    def _plan_shards(self, zip_files: List[Path]) -> List[Dict]:
        """파일 크기 기반 샤드 계획"""
        shards = []
        current_shard = []
        current_size = 0
        
        # 파일 크기순 정렬 (큰 파일부터)
        sorted_files = sorted(zip_files, key=lambda x: x.stat().st_size, reverse=True)
        
        for zip_file in sorted_files:
            file_size = zip_file.stat().st_size
            
            if current_size + file_size > self.shard_size and current_shard:
                # 현재 샤드 완성
                shards.append({
                    'id': len(shards),
                    'files': current_shard.copy(),
                    'size': current_size
                })
                current_shard = []
                current_size = 0
            
            current_shard.append(zip_file)
            current_size += file_size
        
        # 마지막 샤드
        if current_shard:
            shards.append({
                'id': len(shards),
                'files': current_shard,
                'size': current_size
            })
        
        avg_size_gb = sum(s['size'] for s in shards) / len(shards) / (1024**3)
        logger.info(f"Shard planning complete: {len(shards)} shards, "
                   f"avg size: {avg_size_gb:.1f}GB")
        
        return shards
    
    def _background_loader(self, shards: List[Dict]):
        """백그라운드에서 샤드 미리 로딩"""
        zip_loader = SafeZipLoader(self.config)
        
        for shard in shards:
            # 메모리 사용량 체크
            memory_usage = psutil.virtual_memory()
            if memory_usage.available < self.max_memory * 0.3:
                logger.warning("Low memory, waiting for cleanup...")
                time.sleep(10)
            
            shard_data = self._load_shard(shard, zip_loader)
            
            try:
                self.shard_queue.put(shard_data, timeout=60)
                logger.debug(f"Shard {shard['id']} loaded and queued")
            except queue.Full:
                logger.error(f"Shard queue full, dropping shard {shard['id']}")
    
    def _load_shard(self, shard: Dict, zip_loader: SafeZipLoader) -> Dict[str, Any]:
        """개별 샤드 로딩"""
        shard_id = shard['id']
        logger.info(f"Loading shard {shard_id}: {len(shard['files'])} files, "
                   f"{shard['size']/1024**3:.1f}GB")
        
        shard_data = {
            'id': shard_id,
            'files': [],
            'load_time': time.time(),
            'success': True
        }
        
        for zip_file in shard['files']:
            extract_to = Path(f"/tmp/shard_{shard_id}") / zip_file.stem
            extract_to.mkdir(parents=True, exist_ok=True)
            
            success = zip_loader.load_zip_safe(zip_file, extract_to)
            
            shard_data['files'].append({
                'zip_path': zip_file,
                'extract_path': extract_to,
                'success': success
            })
            
            if not success:
                shard_data['success'] = False
        
        return shard_data
    
    def _process_shard(self, shard_data: Dict) -> Iterator[Dict[str, Any]]:
        """샤드 처리 및 데이터 yield"""
        if not shard_data['success']:
            logger.warning(f"Skipping failed shard {shard_data['id']}")
            return
        
        for file_info in shard_data['files']:
            if file_info['success']:
                yield {
                    'shard_id': shard_data['id'],
                    'extract_path': file_info['extract_path'],
                    'original_zip': file_info['zip_path']
                }
    
    def _cleanup_shard(self, shard_id: int):
        """샤드 메모리 정리"""
        shard_temp_dir = Path(f"/tmp/shard_{shard_id}")
        if shard_temp_dir.exists():
            import shutil
            shutil.rmtree(shard_temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up shard {shard_id} temporary files")
    
    def _traditional_loading(self, zip_files: List[Path]) -> Iterator[Dict[str, Any]]:
        """전통적인 순차 로딩 (스트리밍 비활성화 시)"""
        zip_loader = SafeZipLoader(self.config)
        
        for zip_file in zip_files:
            extract_to = Path("/tmp/traditional") / zip_file.stem
            extract_to.mkdir(parents=True, exist_ok=True)
            
            success = zip_loader.load_zip_safe(zip_file, extract_to)
            
            if success:
                yield {
                    'shard_id': 0,
                    'extract_path': extract_to,
                    'original_zip': zip_file
                }


def create_zip_processor(config: Dict[str, Any]) -> ShardedDataLoader:
    """ZIP 처리기 생성 헬퍼"""
    zip_config = ZipProcessingConfig(
        integrity_level=ZipIntegrityLevel(
            config.get("zip_processing", {}).get("integrity_level", "quick")
        ),
        shard_size_gb=config.get("zip_processing", {}).get("shard_size_gb", 8),
        max_concurrent_extracts=config.get("zip_processing", {}).get("max_concurrent_extracts", 2),
        stream_loading=config.get("zip_processing", {}).get("stream_loading", True),
        max_memory_gb=config.get("zip_processing", {}).get("max_memory_gb", 64)
    )
    
    return ShardedDataLoader(zip_config)