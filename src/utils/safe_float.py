#!/usr/bin/env python3
"""
안전한 float 변환 유틸리티
NoneType 비교 에러 방지를 위한 강제 정규화 함수들
"""

import math
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    모든 입력을 안전하게 float로 변환
    
    Args:
        value: 변환할 값
        default: 변환 실패 시 반환할 기본값
    
    Returns:
        float 값 (변환 실패 시 default)
    """
    # None 체크
    if value is None:
        return default
    
    # 이미 float인 경우
    if isinstance(value, float):
        # NaN, inf 체크
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    
    # int인 경우
    if isinstance(value, int):
        return float(value)
    
    # 문자열인 경우
    if isinstance(value, str):
        try:
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except (ValueError, TypeError):
            return default
    
    # 기타 타입
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Union[float, bool]]:
    """
    메트릭 딕셔너리를 안전하게 정규화
    
    Args:
        metrics: 원본 메트릭 딕셔너리
    
    Returns:
        정규화된 메트릭 딕셔너리 (valid 플래그 포함)
    """
    # 필수 키 정의
    required_keys = ['map50', 'precision', 'recall', 'box_loss', 'cls_loss', 'dfl_loss']
    
    # 결과 딕셔너리 초기화
    result = {}
    replaced_count = 0
    
    # 각 필수 키에 대해 안전한 변환
    for key in required_keys:
        original = metrics.get(key)
        converted = safe_float(original, 0.0)
        result[key] = converted
        
        # 치환 발생 체크
        if original is None:
            replaced_count += 1
        elif isinstance(original, (int, float)):
            if math.isnan(original) or math.isinf(original) or original != converted:
                replaced_count += 1
        elif str(original) != str(converted):
            replaced_count += 1
    
    # valid 플래그 설정
    result['valid'] = (replaced_count == 0)
    
    # 치환 발생 시 로깅
    if replaced_count > 0:
        logger.info(f"CSV_SANITIZE | replaced={replaced_count} | valid=False")
    
    return result


def safe_compare(a: Any, b: Any, operator: str = '>') -> bool:
    """
    안전한 비교 연산
    
    Args:
        a: 첫 번째 값
        b: 두 번째 값
        operator: 비교 연산자 ('>', '<', '>=', '<=', '==', '!=')
    
    Returns:
        비교 결과 (에러 시 False)
    """
    a_safe = safe_float(a, 0.0)
    b_safe = safe_float(b, 0.0)
    
    try:
        if operator == '>':
            return a_safe > b_safe
        elif operator == '<':
            return a_safe < b_safe
        elif operator == '>=':
            return a_safe >= b_safe
        elif operator == '<=':
            return a_safe <= b_safe
        elif operator == '==':
            return a_safe == b_safe
        elif operator == '!=':
            return a_safe != b_safe
        else:
            logger.warning(f"알 수 없는 연산자: {operator}")
            return False
    except Exception as e:
        logger.error(f"비교 연산 실패: {e}")
        return False