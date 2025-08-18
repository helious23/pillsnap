"""
Stage 1 빠른 검증 (스모크 테스트)

목적: 현재 환경에서 1~2분 내 완료되는 빠른 데이터 파이프라인 검증
입력: PILLSNAP_DATA_ROOT 환경변수로 지정된 데이터 루트
출력: 콘솔 요약 (데이터 루트, 샘플 수, 쌍 매칭률, 검증 통과 여부)
검증 포인트: 
- 환경 설정 (data root, 가상환경)
- 데이터 접근성 (이미지/라벨 파일 존재)
- 파이프라인 동작 (스캔→전처리→검증)
- 품질 게이트 (PASSED/FAILED)
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("⚠️  rich 패키지가 필요합니다: pip install rich")
    sys.exit(1)

# 프로젝트 모듈 (상대 경로 방지)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from dataset.scan import scan_dataset
from dataset.preprocess import preprocess
from dataset.validate import validate_manifest

console = Console()
logger = logging.getLogger(__name__)


def main(max_seconds: int = 60, sample_limit: int = 200) -> int:
    """
    빠른 스모크 테스트 실행
    
    Args:
        max_seconds: 최대 실행 시간 (초)
        sample_limit: 샘플링 제한 (빠른 테스트용)
    
    Returns:
        int: 성공 시 0, 실패 시 1
    """
    start_time = time.time()
    
    try:
        # 설정 로드 (환경변수 PILLSNAP_DATA_ROOT 우선 적용)
        console.print(Panel.fit("🔍 Stage 1 빠른 검증 시작", style="bold blue"))
        
        cfg = config.load_config()
        data_root = cfg.data.root
        
        console.print(f"📁 데이터 루트: {data_root}")
        console.print(f"🔧 이미지 확장자: {cfg.data.image_exts}")
        console.print(f"📄 라벨 확장자: {cfg.data.label_ext}")
        console.print(f"🎯 샘플 제한: {sample_limit}")
        
        # 데이터 루트 존재 확인
        if not Path(data_root).exists():
            console.print(f"❌ 데이터 루트가 존재하지 않습니다: {data_root}", style="red")
            console.print("💡 해결방법:")
            console.print("  export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset/data")
            return 1
        
        # 1단계: 데이터셋 스캔
        console.print("\n🔍 1단계: 데이터셋 스캔...")
        df, stats = scan_dataset(
            data_root, 
            cfg.data.image_exts, 
            cfg.data.label_ext
        )
        
        # 샘플링 (스모크 테스트용)
        if sample_limit > 0 and len(df) > sample_limit:
            df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
            console.print(f"📦 {sample_limit}개 샘플로 제한")
        
        if len(df) == 0:
            console.print("❌ 스캔 결과가 비어있습니다", style="red")
            return 1
        
        # 2단계: 전처리
        console.print("🔧 2단계: 전처리...")
        df_processed = preprocess(
            df, 
            cfg.preprocess, 
            artifacts_dir="artifacts"
        )
        
        # 3단계: 검증
        console.print("✅ 3단계: 검증...")
        report = validate_manifest(
            df_processed,
            cfg.validation,
            require_files_exist=True,
            min_pair_rate=None  # 스모크 테스트에서는 관대하게
        )
        
        # 결과 테이블 생성
        table = Table(title="📊 검증 결과 요약", show_header=True)
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")
        
        table.add_row("총 이미지 파일", f"{stats.get('total_images', 0):,}")
        table.add_row("총 라벨 파일", f"{stats.get('total_labels', 0):,}")
        table.add_row("처리된 쌍", f"{len(df_processed):,}")
        table.add_row("쌍 매칭률", f"{report.stats.get('pair_rate', 0):.1%}")
        table.add_row("파일 존재율", f"{report.stats.get('file_exists_rate', 0):.1%}")
        table.add_row("중복 코드", f"{report.stats.get('duplicate_codes', 0)}")
        
        console.print(table)
        
        # 최종 결과
        elapsed = time.time() - start_time
        if report.passed:
            console.print(Panel.fit(
                f"✅ 검증 통과 (소요시간: {elapsed:.1f}초)", 
                style="bold green"
            ))
            return 0
        else:
            console.print(Panel.fit(
                f"❌ 검증 실패 (소요시간: {elapsed:.1f}초)", 
                style="bold red"
            ))
            if report.errors:
                console.print("🔍 오류 목록:")
                for error in report.errors[:3]:  # 최대 3개만 표시
                    console.print(f"  • {error}")
            return 1
    
    except KeyboardInterrupt:
        console.print("\n⏹️  사용자 중단", style="yellow")
        return 1
    except Exception as e:
        console.print(f"\n💥 예상치 못한 오류: {e}", style="red")
        console.print("💡 해결방법:")
        console.print("  1. 가상환경 활성화: source $HOME/pillsnap/.venv/bin/activate")
        console.print("  2. 환경변수 설정: export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset/data")
        console.print("  3. 경로 확인: ls $PILLSNAP_DATA_ROOT")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1 빠른 검증")
    parser.add_argument("--max-seconds", type=int, default=60,
                       help="최대 실행 시간 (초)")
    parser.add_argument("--sample-limit", type=int, default=200,
                       help="샘플링 제한 (빠른 테스트용)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)  # 스모크 테스트에서는 경고만
    
    exit_code = main(args.max_seconds, args.sample_limit)
    sys.exit(exit_code)