#!/usr/bin/env python3
"""
데이터 품질 유틸리티 공통 베이스 클래스
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

# Rich console for beautiful output
console = Console()


class DataQualityBase:
    """데이터 품질 점검/수정 유틸리티 베이스 클래스"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = console
        
        # 기본 경로 설정
        self.project_root = Path("/home/max16/pillsnap")
        self.data_root = Path("/home/max16/pillsnap_data")
        self.report_dir = self.project_root / "artifacts" / "data_quality_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # Manifest 경로 자동 감지
        self.setup_manifest_paths()
        
        # 실행 정보 기록
        self.execution_info = {
            "timestamp": self.timestamp,
            "command": " ".join(sys.argv),
            "dry_run": self.args.dry_run,
            "parameters": vars(self.args)
        }
        
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.report_dir / f"{self.__class__.__name__}_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO if not hasattr(self.args, 'verbose') or not self.args.verbose else logging.DEBUG,
            format="%(message)s",
            handlers=[
                RichHandler(console=self.console, rich_tracebacks=True),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def setup_manifest_paths(self):
        """Manifest 파일 경로 설정"""
        # 기본 Stage 3 manifest 경로
        default_train = self.project_root / "artifacts" / "stage3" / "manifest_train.csv"
        default_val = self.project_root / "artifacts" / "stage3" / "manifest_val.csv"
        
        # 사용자 지정 또는 기본값
        train_manifest = getattr(self.args, 'train_manifest', None)
        val_manifest = getattr(self.args, 'val_manifest', None)
        
        self.train_manifest_path = Path(train_manifest) if train_manifest else default_train
        self.val_manifest_path = Path(val_manifest) if val_manifest else default_val
        
        # 존재 여부 확인
        if not self.train_manifest_path.exists():
            self.logger.error(f"Train manifest not found: {self.train_manifest_path}")
            sys.exit(1)
        if not self.val_manifest_path.exists():
            self.logger.error(f"Val manifest not found: {self.val_manifest_path}")
            sys.exit(1)
            
    def load_manifests(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Manifest 파일 로드"""
        self.logger.info(f"Loading train manifest: {self.train_manifest_path}")
        train_df = pd.read_csv(self.train_manifest_path)
        
        self.logger.info(f"Loading val manifest: {self.val_manifest_path}")
        val_df = pd.read_csv(self.val_manifest_path)
        
        # 필수 컬럼 확인
        required_columns = ['mapping_code', 'image_type', 'image_path']
        for df, name in [(train_df, 'train'), (val_df, 'val')]:
            missing = set(required_columns) - set(df.columns)
            if missing:
                self.logger.error(f"Missing columns in {name} manifest: {missing}")
                sys.exit(1)
                
        return train_df, val_df
    
    def backup_file(self, file_path: Path) -> Path:
        """파일 백업 생성"""
        if not self.args.backup:
            return None
            
        backup_path = file_path.parent / f"{file_path.stem}_backup_{self.timestamp}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def save_manifest(self, df: pd.DataFrame, original_path: Path, suffix: str = None) -> Path:
        """수정된 manifest 저장"""
        if self.args.dry_run:
            self.logger.info("[DRY RUN] Would save manifest but dry_run is enabled")
            return None
            
        # 백업 생성
        if self.args.backup:
            self.backup_file(original_path)
        
        # 새 파일 경로
        suffix = suffix or self.args.out_suffix
        new_path = original_path.parent / f"{original_path.stem}{suffix}{original_path.suffix}"
        
        # 저장
        df.to_csv(new_path, index=False)
        self.logger.info(f"Saved modified manifest: {new_path}")
        return new_path
    
    def print_summary_table(self, title: str, data: Dict[str, any]):
        """요약 테이블 출력"""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            table.add_row(key, str(value))
            
        self.console.print(table)
    
    def print_before_after_table(self, before: Dict, after: Dict, title: str = "Before/After Comparison"):
        """전후 비교 테이블 출력"""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Before", style="yellow")
        table.add_column("After", style="green")
        table.add_column("Change", style="bold")
        
        for key in before.keys():
            before_val = before[key]
            after_val = after.get(key, before_val)
            
            if isinstance(before_val, (int, float)):
                change = after_val - before_val
                if isinstance(before_val, float):
                    change_str = f"{change:+.2f}"
                else:
                    change_str = f"{change:+,}"
            else:
                change_str = "→"
                
            table.add_row(
                key,
                str(before_val),
                str(after_val),
                change_str
            )
            
        self.console.print(table)
    
    def save_report(self, report_data: Dict, report_name: str = None):
        """리포트 저장"""
        report_name = report_name or f"{self.__class__.__name__}_report"
        report_path = self.report_dir / f"{report_name}_{self.timestamp}.json"
        
        # 실행 정보 추가
        report_data['execution_info'] = self.execution_info
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
        self.logger.info(f"Report saved: {report_path}")
        return report_path
    
    def print_conclusion(self, status: str = "PASS", message: str = None):
        """최종 결론 출력"""
        color = "green" if status == "PASS" else "red"
        emoji = "✅" if status == "PASS" else "❌"
        
        panel = Panel(
            f"{emoji} Status: [bold {color}]{status}[/bold {color}]\n\n{message or ''}",
            title="Conclusion",
            border_style=color
        )
        self.console.print(panel)
    
    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser):
        """공통 인자 추가"""
        parser.add_argument(
            '--train-manifest',
            type=str,
            help='Path to train manifest CSV'
        )
        parser.add_argument(
            '--val-manifest', 
            type=str,
            help='Path to validation manifest CSV'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            default=True,
            help='Dry run mode (default: True)'
        )
        parser.add_argument(
            '--no-dry-run',
            dest='dry_run',
            action='store_false',
            help='Disable dry run mode'
        )
        parser.add_argument(
            '--backup',
            action='store_true',
            default=True,
            help='Create backup before modification (default: True)'
        )
        parser.add_argument(
            '--no-backup',
            dest='backup',
            action='store_false',
            help='Disable backup creation'
        )
        parser.add_argument(
            '--out-suffix',
            type=str,
            default='.cleaned',
            help='Output file suffix (default: .cleaned)'
        )
        parser.add_argument(
            '--report-path',
            type=str,
            help='Custom report file path'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )
        
    def run(self):
        """메인 실행 메서드 (서브클래스에서 구현)"""
        raise NotImplementedError("Subclass must implement run() method")