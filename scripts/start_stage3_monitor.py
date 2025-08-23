#!/usr/bin/env python3
"""
Stage 3 실시간 모니터링 시작 스크립트

사용법:
  python scripts/start_stage3_monitor.py                    # 기본 모니터링
  python scripts/start_stage3_monitor.py --log-file /path/to/train.log  # 로그 파일 모니터링
  python scripts/start_stage3_monitor.py --log-cmd "tail -f /var/log/training.log"  # 명령어 모니터링
  python scripts/start_stage3_monitor.py --port 9999        # 다른 포트 사용
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.stage3_realtime_monitor import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 실시간 모니터링 서버 시작",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  %(prog)s                                    # 기본 모니터링
  %(prog)s --port 9999                        # 포트 9999에서 실행
  %(prog)s --log-file logs/train.log          # 특정 로그 파일 모니터링
  %(prog)s --log-cmd "tail -f /tmp/train.log" # 명령어로 로그 스트리밍
  %(prog)s --log-cmd "python train.py"        # 훈련 실행과 동시에 모니터링
        """
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="서버 호스트 (기본값: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8888, 
        help="서버 포트 (기본값: 8888)"
    )
    
    parser.add_argument(
        "--log-file", 
        type=Path,
        help="모니터링할 로그 파일 경로"
    )
    
    parser.add_argument(
        "--log-cmd", 
        help="로그 스트리밍을 위한 명령어"
    )
    
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="모니터링 자동 시작 안함 (수동으로 API 호출 필요)"
    )
    
    args = parser.parse_args()
    
    # 로그 소스 결정
    log_source = None
    if args.log_file:
        if not args.log_file.exists():
            print(f"❌ 로그 파일이 존재하지 않습니다: {args.log_file}")
            sys.exit(1)
        log_source = str(args.log_file)
        print(f"📁 로그 파일 모니터링: {log_source}")
    elif args.log_cmd:
        log_source = args.log_cmd
        print(f"⚡ 명령어 모니터링: {log_source}")
    else:
        print("🔍 자동 로그 감지 모드")
    
    # 서버 실행
    try:
        if args.no_auto_start:
            print("⏸️  자동 시작 비활성화됨. /api/start를 호출하여 모니터링을 시작하세요.")
            log_source = None
        
        run_server(
            host=args.host,
            port=args.port,
            log_source=log_source
        )
        
    except KeyboardInterrupt:
        print("\n👋 모니터링 서버가 종료되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()