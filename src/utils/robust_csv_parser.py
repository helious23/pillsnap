#!/usr/bin/env python3
"""
견고한 CSV 파싱 유틸리티
YOLO results.csv 파일을 안전하게 읽기 위한 재시도 로직 포함
"""

import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
import math
from src.utils.safe_float import safe_float, sanitize_metrics


class RobustCSVParser:
    """견고한 CSV 파서"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 컬럼 매핑 우선순위 (YOLO 버전별 호환성)
        self.column_mappings = {
            'map50': ['metrics/mAP50(B)', 'metrics/mAP_0.5', 'metrics/mAP50', 'mAP50'],
            'precision': ['metrics/precision(B)', 'metrics/precision', 'precision'],
            'recall': ['metrics/recall(B)', 'metrics/recall', 'recall'],
            'box_loss': ['train/box_loss', 'box_loss', 'loss_box'],
            'cls_loss': ['train/cls_loss', 'cls_loss', 'loss_cls'],
            'dfl_loss': ['train/dfl_loss', 'dfl_loss', 'loss_dfl']
        }
    
    def parse_results_csv(
        self,
        csv_path: Path,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Union[float, bool]]:
        """
        YOLO results.csv 파일 파싱 (재시도 로직 포함)
        
        Args:
            csv_path: CSV 파일 경로
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
        
        Returns:
            파싱된 메트릭 딕셔너리
        """
        raw_metrics = {
            'map50': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'dfl_loss': 0.0
        }
        
        if not csv_path.exists():
            self.logger.warning(f"CSV 파일 없음: {csv_path}")
            return sanitize_metrics(raw_metrics)
        
        # 재시도 로직
        for attempt in range(max_retries):
            try:
                # CSV 읽기
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    self.logger.warning(f"CSV 파일이 비어있음: {csv_path}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return sanitize_metrics(raw_metrics)
                
                # 마지막 행 가져오기
                last_row = df.iloc[-1]
                
                # 컬럼 매핑으로 값 추출
                for metric_key, column_candidates in self.column_mappings.items():
                    value = None
                    
                    for col_name in column_candidates:
                        if col_name in df.columns:
                            try:
                                raw_value = last_row[col_name]
                                # 안전한 float 변환 사용
                                value = safe_float(raw_value, None)
                                if value is not None:
                                    break
                            except Exception:
                                continue
                    
                    if value is not None:
                        raw_metrics[metric_key] = value
                    else:
                        self.logger.debug(f"메트릭 {metric_key}를 찾을 수 없음")
                        # 기본값 0.0 유지
                
                # 성공적으로 파싱
                self.logger.info(f"CSV 파싱 성공 (시도 {attempt + 1}/{max_retries})")
                return sanitize_metrics(raw_metrics)
                
            except pd.errors.EmptyDataError:
                self.logger.warning(f"CSV 읽기 실패 - 빈 데이터 (시도 {attempt + 1}/{max_retries})")
                
            except pd.errors.ParserError as e:
                self.logger.warning(f"CSV 파싱 오류: {e} (시도 {attempt + 1}/{max_retries})")
                
            except Exception as e:
                self.logger.error(f"예상치 못한 오류: {e} (시도 {attempt + 1}/{max_retries})")
            
            # 재시도 전 대기 (백오프)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # 지수 백오프
                self.logger.info(f"{wait_time:.1f}초 후 재시도...")
                time.sleep(wait_time)
        
        self.logger.error(f"CSV 파싱 실패 (모든 시도 소진): {csv_path}")
        return sanitize_metrics(raw_metrics)
    
    def validate_metrics(self, metrics: Dict[str, Union[float, bool]]) -> bool:
        """
        메트릭 유효성 검사
        이미 정규화된 메트릭을 받으므로 None 체크 불필요
        """
        # valid 플래그 확인
        if not metrics.get('valid', False):
            self.logger.info("메트릭이 치환되었음 (valid=False)")
        
        # mAP, precision, recall은 0-1 범위
        for key in ['map50', 'precision', 'recall']:
            value = safe_float(metrics.get(key, 0), 0.0)
            if not (0 <= value <= 1):
                self.logger.warning(f"비정상 메트릭 값: {key}={value}")
                return False
        
        # loss는 0 이상
        for key in ['box_loss', 'cls_loss', 'dfl_loss']:
            value = safe_float(metrics.get(key, 0), 0.0)
            if value < 0:
                self.logger.warning(f"비정상 손실 값: {key}={value}")
                return False
        
        return True
    
    def get_all_metrics_history(self, csv_path: Path) -> pd.DataFrame:
        """전체 메트릭 히스토리 가져오기"""
        try:
            if not csv_path.exists():
                return pd.DataFrame()
            
            df = pd.read_csv(csv_path)
            
            # 컬럼 이름 정규화
            normalized_df = pd.DataFrame()
            
            for metric_key, column_candidates in self.column_mappings.items():
                for col_name in column_candidates:
                    if col_name in df.columns:
                        normalized_df[metric_key] = df[col_name]
                        break
            
            # epoch 컬럼 추가 (인덱스 기반)
            normalized_df['epoch'] = range(1, len(normalized_df) + 1)
            
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"히스토리 로드 실패: {e}")
            return pd.DataFrame()