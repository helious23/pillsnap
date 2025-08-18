"""
Stage 1: 데이터 파이프라인 (스캔→전처리→검증)

목적: 대용량 의약품 이미지 데이터셋의 안전한 처리 및 품질 보증
입력: 원시 이미지/라벨 파일 (2.6M+ 파일)  
출력: 검증된 CSV 매니페스트, 품질 리포트
검증 포인트: 파일 존재성, 이미지-라벨 쌍 매칭, 중복 제거, 각도 규칙
"""

from .verify import main as verify_main
from .run import main as run_main

__all__ = ["verify_main", "run_main"]