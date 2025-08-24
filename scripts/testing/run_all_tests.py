#!/usr/bin/env python3
"""
통합 테스트 실행 도구
모든 테스트를 카테고리별로 실행하고 결과를 요약합니다.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """테스트 실행 관리자"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path("/home/max16/pillsnap")
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        
    def run_category(self, category: str) -> Tuple[bool, str]:
        """특정 카테고리의 테스트 실행"""
        
        category_dir = self.tests_dir / category
        if not category_dir.exists():
            return False, f"디렉토리 없음: {category_dir}"
        
        # pytest 명령어 구성
        cmd = [
            sys.executable, "-m", "pytest",
            str(category_dir),
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--no-header",
            "-ra"  # 실패한 테스트 요약
        ]
        
        print(f"\n{'='*60}")
        print(f"🧪 {category.upper()} TESTS 실행 중...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            elapsed = time.time() - start_time
            
            # 결과 파싱
            output = result.stdout
            if "passed" in output:
                # 성공 개수 추출
                import re
                match = re.search(r'(\d+) passed', output)
                passed = int(match.group(1)) if match else 0
                
                # 실패 개수 추출
                match = re.search(r'(\d+) failed', output)
                failed = int(match.group(1)) if match else 0
                
                # 스킵 개수 추출
                match = re.search(r'(\d+) skipped', output)
                skipped = int(match.group(1)) if match else 0
                
                summary = f"✅ Passed: {passed}, ❌ Failed: {failed}, ⏭️ Skipped: {skipped} ({elapsed:.1f}s)"
                success = (failed == 0)
            else:
                summary = f"❌ 테스트 실행 실패 ({elapsed:.1f}s)"
                success = False
                
            if self.verbose:
                print(output)
                
            return success, summary
            
        except subprocess.TimeoutExpired:
            return False, f"⏱️ 타임아웃 (5분 초과)"
        except Exception as e:
            return False, f"❌ 오류: {str(e)}"
    
    def run_all(self, categories: List[str] = None):
        """모든 카테고리 테스트 실행"""
        
        if categories is None:
            categories = ["unit", "integration", "smoke", "performance", "scripts"]
        
        print(f"\n{'='*60}")
        print(f"🚀 PillSnap ML 테스트 실행")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        total_start = time.time()
        
        # 각 카테고리 실행
        for category in categories:
            success, summary = self.run_category(category)
            self.results[category] = {
                "success": success,
                "summary": summary
            }
        
        total_elapsed = time.time() - total_start
        
        # 최종 요약
        self.print_summary(total_elapsed)
    
    def print_summary(self, total_time: float):
        """테스트 결과 요약 출력"""
        
        print(f"\n{'='*60}")
        print(f"📊 테스트 결과 요약")
        print(f"{'='*60}")
        
        all_passed = True
        
        for category, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {category.upper():15} {result['summary']}")
            if not result["success"]:
                all_passed = False
        
        print(f"\n총 실행 시간: {total_time:.1f}초")
        
        if all_passed:
            print("\n🎉 모든 테스트 통과!")
        else:
            print("\n⚠️ 일부 테스트 실패 - 확인 필요")
            
    def run_specific_test(self, test_path: str):
        """특정 테스트 파일 실행"""
        
        test_file = self.project_root / test_path
        
        if not test_file.exists():
            print(f"❌ 테스트 파일 없음: {test_file}")
            return False
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short"
        ]
        
        print(f"\n🧪 테스트 실행: {test_path}")
        
        try:
            result = subprocess.run(cmd, timeout=60)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("⏱️ 타임아웃")
            return False


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="PillSnap ML 테스트 실행 도구")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "smoke", "performance", "scripts", "all"],
        default="all",
        help="실행할 테스트 카테고리"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="특정 테스트 파일 경로 (예: tests/unit/test_classifier.py)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    if args.test:
        # 특정 테스트 실행
        success = runner.run_specific_test(args.test)
        sys.exit(0 if success else 1)
    elif args.category == "all":
        # 모든 테스트 실행
        runner.run_all()
    else:
        # 특정 카테고리만 실행
        success, summary = runner.run_category(args.category)
        print(f"\n{summary}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()