"""
Stage 1 전체 파이프라인 실행

목적: 데이터 스캔→전처리→검증→리포팅을 한 번에 실행하는 완전한 파이프라인
입력: PILLSNAP_DATA_ROOT 환경변수, 옵션으로 limit/manifest 경로
출력:
- artifacts/manifest_stage1.csv (검증된 매니페스트)
- artifacts/stage1_stats.json (스캔 통계)
- artifacts/stage1_validation_report.json (검증 결과)
- artifacts/step6_report.md (사람이 읽기 쉬운 리포트, 기존 파일 있으면 건너뜀)
검증 포인트:
- 전체 파이프라인 무결성
- 산출물 스키마 검증
- 품질 게이트 통과
"""

import sys
import json
import time
import logging
from pathlib import Path
import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:
    print("⚠️  rich 패키지가 필요합니다: pip install rich")
    sys.exit(1)

# 프로젝트 모듈
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from dataset.scan import scan_dataset
from dataset.preprocess import preprocess
from dataset.validate import validate_manifest
from pillsnap.stage1.utils import build_edi_classes

console = Console()
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def main(limit: int = 400, manifest: str = "artifacts/manifest_stage1.csv") -> int:
    """
    Stage 1 전체 파이프라인 실행

    Args:
        limit: 스캔 제한 (0=전체, 기본 400 안전 권장)
        manifest: 출력 매니페스트 경로

    Returns:
        int: 성공 시 0, 실패 시 1
    """
    start_time = time.time()

    try:
        console.print(Panel.fit("🚀 Stage 1 전체 파이프라인 시작", style="bold cyan"))

        # 설정 로드
        cfg = config.load_config()
        data_root = cfg.data.root

        console.print(f"📁 데이터 루트: {data_root}")
        console.print(f"🎯 스캔 제한: {limit if limit > 0 else '제한 없음'}")
        console.print(f"📄 출력 매니페스트: {manifest}")

        # 출력 디렉토리 생성
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # 데이터 루트 존재 확인
        if not Path(data_root).exists():
            console.print(
                f"❌ 데이터 루트가 존재하지 않습니다: {data_root}", style="red"
            )
            console.print("💡 해결방법:")
            console.print("  export PILLSNAP_DATA_ROOT=/mnt/data/pillsnap_dataset/data")
            return 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # 1단계: 데이터셋 스캔
            task1 = progress.add_task("🔍 데이터셋 스캔 중...", total=None)
            df, stats = scan_dataset(data_root, cfg.data.image_exts, cfg.data.label_ext)

            # 제한 적용 (안전을 위해)
            if limit > 0 and len(df) > limit:
                df = df.sample(n=limit, random_state=42).reset_index(drop=True)
                # 통계도 업데이트
                stats["limited"] = True
                stats["original_count"] = len(df)
            progress.update(task1, description="✅ 스캔 완료")

            if len(df) == 0:
                console.print("❌ 스캔 결과가 비어있습니다", style="red")
                return 1

            # 2단계: 전처리
            task2 = progress.add_task("🔧 전처리 중...", total=None)
            df_processed = preprocess(df, cfg.preprocess, artifacts_dir="artifacts")

            # 매니페스트 수동 저장
            Path(manifest).parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(manifest, index=False)
            progress.update(task2, description="✅ 전처리 완료")

            # 3단계: 검증
            task3 = progress.add_task("✅ 검증 중...", total=None)
            report = validate_manifest(
                df_processed,
                cfg.validation,
                require_files_exist=True,
                min_pair_rate=0.8,  # 80% 이상 쌍 매칭 요구
            )
            progress.update(task3, description="✅ 검증 완료")

            # 4단계: 리포팅
            task4 = progress.add_task("📊 리포팅 중...", total=None)

            # 통계 저장 (NumPy 타입 변환)
            stats_path = artifacts_dir / "stage1_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(convert_numpy_types(stats), f, indent=2, ensure_ascii=False)

            # 검증 리포트 저장
            report_path = artifacts_dir / "stage1_validation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    convert_numpy_types(
                        {
                            "passed": report.passed,
                            "errors": report.errors,
                            "warnings": report.warnings,
                            "stats": report.stats,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    ),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Markdown 리포트 (기존 파일 있으면 건너뜀)
            md_report_path = artifacts_dir / "step6_report.md"
            if not md_report_path.exists():
                try:
                    _generate_markdown_report(
                        md_report_path, stats, report, df_processed
                    )
                except Exception as e:
                    console.print(f"⚠️  Markdown 리포트 생성 실패: {e}", style="yellow")
            else:
                console.print(f"ℹ️  기존 리포트 존재, 건너뜀: {md_report_path}")

            progress.update(task4, description="✅ 리포팅 완료")

        # 결과 요약 테이블
        table = Table(title="📋 파이프라인 실행 결과", show_header=True)
        table.add_column("구분", style="cyan")
        table.add_column("결과", style="green")

        elapsed = time.time() - start_time
        table.add_row("소요 시간", f"{elapsed:.1f}초")
        table.add_row(
            "스캔된 파일",
            f"{stats.get('total_images', 0):,} 이미지, {stats.get('total_labels', 0):,} 라벨",
        )
        table.add_row("처리된 쌍", f"{len(df_processed):,}")
        table.add_row("쌍 매칭률", f"{report.stats.get('pair_rate', 0):.1%}")
        table.add_row("검증 결과", "✅ 통과" if report.passed else "❌ 실패")

        # 산출물 정보
        table.add_row("매니페스트", manifest)
        table.add_row("통계", str(stats_path))
        table.add_row("검증 리포트", str(report_path))
        if md_report_path.exists():
            table.add_row("Markdown 리포트", str(md_report_path))

        console.print(table)

        # 최종 결과
        if report.passed:
            console.print(
                Panel.fit("🎉 Stage 1 파이프라인 성공적으로 완료!", style="bold green")
            )

            # 5단계: 클래스 맵 생성 (EDI → class_id)
            console.print("\n🔧 EDI 클래스 맵 생성 중...")
            try:
                class_map = build_edi_classes(
                    manifest_csv=manifest, outfile="artifacts/classes_step11.json"
                )
                console.print(f"  ✅ 클래스 맵 생성 완료: {len(class_map)} EDI codes")
            except Exception as e:
                console.print(f"  ⚠️  클래스 맵 생성 실패: {e}", style="yellow")
                # 실패해도 파이프라인은 계속 진행

            # 다음 단계 안내
            console.print("\n🔗 다음 단계:")
            console.print(
                "  python -m src.train --mode single --epochs 10 --batch-size 32"
            )
            console.print("  python tests/test_pipeline.py  # 스모크 테스트")

            return 0
        else:
            console.print(Panel.fit("❌ Stage 1 파이프라인 실패", style="bold red"))

            if report.errors:
                console.print("\n🔍 주요 오류:")
                for error in report.errors[:5]:  # 최대 5개
                    console.print(f"  • {error}")

            return 1

    except KeyboardInterrupt:
        console.print("\n⏹️  사용자 중단", style="yellow")
        return 1
    except Exception as e:
        console.print(f"\n💥 예상치 못한 오류: {e}", style="red")
        logger.exception("Stage 1 파이프라인 오류")
        return 1


def _generate_markdown_report(output_path: Path, stats: dict, report, df):
    """간단한 Markdown 리포트 생성"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Stage 1 파이프라인 실행 리포트\n\n")
        f.write(f"**생성일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 📊 통계 요약\n\n")
        f.write(f"- 총 이미지: {stats.get('total_images', 0):,}\n")
        f.write(f"- 총 라벨: {stats.get('total_labels', 0):,}\n")
        f.write(f"- 처리된 쌍: {len(df):,}\n")
        f.write(f"- 쌍 매칭률: {report.stats.get('pair_rate', 0):.1%}\n\n")

        f.write("## ✅ 검증 결과\n\n")
        f.write(f"**상태**: {'✅ 통과' if report.passed else '❌ 실패'}\n\n")

        if report.errors:
            f.write("### 오류\n\n")
            for error in report.errors:
                f.write(f"- {error}\n")
            f.write("\n")

        if report.warnings:
            f.write("### 경고\n\n")
            for warning in report.warnings:
                f.write(f"- {warning}\n")
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1 전체 파이프라인 실행")
    parser.add_argument(
        "--limit", type=int, default=400, help="스캔 제한 (0=전체, 기본 400 안전 권장)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/manifest_stage1.csv",
        help="출력 매니페스트 경로",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    exit_code = main(args.limit, args.manifest)
    sys.exit(exit_code)
