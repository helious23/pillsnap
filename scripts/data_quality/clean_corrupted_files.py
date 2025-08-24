#!/usr/bin/env python3
"""
손상된 파일 정리 유틸리티
- Truncated/불러오기 실패 파일 검출 및 제거
- 블랙리스트 생성 및 유지
"""

import argparse
import concurrent.futures
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

from base import DataQualityBase


class CorruptedFileCleaner(DataQualityBase):
    """손상된 파일 검출 및 정리"""
    
    def __init__(self, args):
        super().__init__(args)
        self.corrupted_files: Set[str] = set()
        self.blacklist_path = self.project_root / "artifacts" / "data_quality_reports" / "blacklist.txt"
        self.load_existing_blacklist()
        
    def load_existing_blacklist(self):
        """기존 블랙리스트 로드"""
        if self.blacklist_path.exists():
            with open(self.blacklist_path, 'r') as f:
                existing = set(line.strip() for line in f if line.strip())
                self.corrupted_files.update(existing)
                if existing:
                    self.logger.info(f"Loaded {len(existing)} files from existing blacklist")
    
    def check_image_file(self, image_path: str) -> Tuple[str, bool, str]:
        """이미지 파일 검증"""
        path = Path(image_path)
        
        # 절대 경로로 변환
        if not path.is_absolute():
            # manifest의 상대 경로를 절대 경로로 변환
            if str(path).startswith("train/") or str(path).startswith("val/"):
                path = self.data_root / path
            else:
                path = self.data_root / "train" / path
        
        try:
            # 1. 파일 존재 확인
            if not path.exists():
                return str(path), False, "File not found"
            
            # 2. 파일 크기 확인
            if path.stat().st_size == 0:
                return str(path), False, "Empty file"
            
            # 3. 이미지 열기 시도
            with Image.open(path) as img:
                # 4. 실제 디코딩 시도 (작은 썸네일로)
                img.verify()  # 무결성 검증
                
            # 다시 열어서 실제 로드 테스트
            with Image.open(path) as img:
                img.thumbnail((64, 64))  # 작은 크기로 실제 디코딩
                
            return str(path), True, "OK"
            
        except FileNotFoundError:
            return str(path), False, "File not found"
        except (IOError, OSError) as e:
            if "truncated" in str(e).lower():
                return str(path), False, "Truncated file"
            return str(path), False, f"IO Error: {e}"
        except Exception as e:
            return str(path), False, f"Unknown error: {e}"
    
    def check_files_parallel(self, file_paths: List[str], max_workers: int = 8) -> Dict[str, Tuple[bool, str]]:
        """병렬로 파일 검증"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_path = {
                executor.submit(self.check_image_file, path): path 
                for path in file_paths
            }
            
            # 진행 상황 표시
            with tqdm(total=len(file_paths), desc="Checking files") as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    path, is_valid, reason = future.result()
                    results[path] = (is_valid, reason)
                    pbar.update(1)
                    
                    if not is_valid:
                        self.corrupted_files.add(path)
                        
        return results
    
    def clean_manifest(self, df: pd.DataFrame, check_results: Dict[str, Tuple[bool, str]]) -> pd.DataFrame:
        """Manifest에서 손상된 파일 제거"""
        initial_count = len(df)
        
        # 손상된 파일 마스크 생성
        def is_valid_row(row):
            image_path = row['image_path']
            # 절대 경로 변환
            if not Path(image_path).is_absolute():
                if image_path.startswith("train/") or image_path.startswith("val/"):
                    full_path = str(self.data_root / image_path)
                else:
                    full_path = str(self.data_root / "train" / image_path)
            else:
                full_path = image_path
                
            if full_path in check_results:
                return check_results[full_path][0]
            return True  # 체크하지 않은 파일은 유효한 것으로 간주
        
        # 유효한 행만 필터링
        df_clean = df[df.apply(is_valid_row, axis=1)].copy()
        
        removed_count = initial_count - len(df_clean)
        self.logger.info(f"Removed {removed_count} rows with corrupted files")
        
        return df_clean
    
    def update_blacklist(self):
        """블랙리스트 파일 업데이트"""
        if self.args.dry_run:
            self.logger.info(f"[DRY RUN] Would update blacklist with {len(self.corrupted_files)} files")
            return
            
        with open(self.blacklist_path, 'w') as f:
            for file_path in sorted(self.corrupted_files):
                f.write(f"{file_path}\n")
                
        self.logger.info(f"Updated blacklist: {self.blacklist_path}")
    
    def clean_yolo_links(self):
        """YOLO 심볼릭 링크 정리"""
        yolo_dataset_path = self.data_root / "yolo_configs" / "yolo_dataset"
        if not yolo_dataset_path.exists():
            return
            
        removed_links = 0
        for link_path in yolo_dataset_path.glob("**/*"):
            if link_path.is_symlink():
                target = link_path.resolve()
                if str(target) in self.corrupted_files:
                    if not self.args.dry_run:
                        link_path.unlink()
                    removed_links += 1
                    
        self.logger.info(f"{'[DRY RUN] Would remove' if self.args.dry_run else 'Removed'} {removed_links} YOLO symlinks")
    
    def run(self):
        """메인 실행"""
        self.console.print("[bold cyan]🔍 Corrupted File Cleaner[/bold cyan]")
        self.console.print(f"Mode: {'DRY RUN' if self.args.dry_run else 'ACTUAL EXECUTION'}")
        
        # Manifest 로드
        train_df, val_df = self.load_manifests()
        
        # 모든 이미지 경로 수집
        all_image_paths = set()
        all_image_paths.update(train_df['image_path'].unique())
        all_image_paths.update(val_df['image_path'].unique())
        
        self.logger.info(f"Total unique images to check: {len(all_image_paths):,}")
        
        # 파일 검증
        check_results = self.check_files_parallel(list(all_image_paths))
        
        # 결과 집계
        corrupted_count = sum(1 for is_valid, _ in check_results.values() if not is_valid)
        error_reasons = {}
        for path, (is_valid, reason) in check_results.items():
            if not is_valid:
                error_reasons[reason] = error_reasons.get(reason, 0) + 1
        
        # 통계 출력
        before_stats = {
            "Train samples": len(train_df),
            "Val samples": len(val_df),
            "Unique images": len(all_image_paths),
            "Corrupted files": 0
        }
        
        # Manifest 정리
        train_df_clean = self.clean_manifest(train_df, check_results)
        val_df_clean = self.clean_manifest(val_df, check_results)
        
        after_stats = {
            "Train samples": len(train_df_clean),
            "Val samples": len(val_df_clean),
            "Unique images": len(all_image_paths) - corrupted_count,
            "Corrupted files": corrupted_count
        }
        
        # 전후 비교 테이블
        self.print_before_after_table(before_stats, after_stats, "File Cleaning Results")
        
        # 에러 원인 분석
        if error_reasons:
            self.print_summary_table("Error Reasons", error_reasons)
        
        # 특정 손상 파일 리스트 (K-001900 관련)
        k001900_corrupted = [p for p in self.corrupted_files if "K-001900" in p]
        if k001900_corrupted:
            self.console.print(f"\n[yellow]Found {len(k001900_corrupted)} corrupted K-001900 files (known issue)[/yellow]")
        
        # Manifest 저장
        if not self.args.dry_run:
            self.save_manifest(train_df_clean, self.train_manifest_path)
            self.save_manifest(val_df_clean, self.val_manifest_path)
        
        # 블랙리스트 업데이트
        self.update_blacklist()
        
        # YOLO 링크 정리
        self.clean_yolo_links()
        
        # 리포트 저장
        report_data = {
            "before": before_stats,
            "after": after_stats,
            "corrupted_files": list(self.corrupted_files),
            "error_reasons": error_reasons,
            "k001900_affected": len(k001900_corrupted)
        }
        self.save_report(report_data, "corrupted_file_cleaning")
        
        # 최종 결론
        if corrupted_count == 0:
            self.print_conclusion("PASS", "No corrupted files found!")
        else:
            message = f"Found and {'removed' if not self.args.dry_run else 'would remove'} {corrupted_count} corrupted files"
            self.print_conclusion("PASS" if not self.args.dry_run else "READY", message)
        
        return 0 if corrupted_count == 0 or not self.args.dry_run else 1


def main():
    parser = argparse.ArgumentParser(
        description="Clean corrupted files from manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    DataQualityBase.add_common_arguments(parser)
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Number of parallel workers for file checking (default: 8)'
    )
    
    # 사용 예시 추가
    parser.epilog = """
Examples:
  # Dry run (default)
  python clean_corrupted_files.py
  
  # Actual execution
  python clean_corrupted_files.py --no-dry-run
  
  # Custom manifest paths
  python clean_corrupted_files.py --train-manifest /path/to/train.csv --val-manifest /path/to/val.csv
  
  # With more workers for faster checking
  python clean_corrupted_files.py --max-workers 16
"""
    
    args = parser.parse_args()
    cleaner = CorruptedFileCleaner(args)
    return cleaner.run()


if __name__ == "__main__":
    exit(main())