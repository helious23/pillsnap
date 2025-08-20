#!/usr/bin/env python3
"""
Scripts 폴더 재구성 스크립트
기능별 + Stage별로 직관적인 구조로 정리
"""

import os
import shutil
from pathlib import Path

def reorganize_scripts():
    scripts_dir = Path("scripts")
    
    # 새로운 구조 정의
    new_structure = {
        "core/": [
            "python_safe.sh",
            "setup_aliases.sh", 
            "setup_venv.sh",
            "update_docs.sh"
        ],
        "stage1/": [
            "migrate_stage1_images_only.sh",
            "migrate_stage1_to_ssd.sh"
        ],
        "stage2/": [
            "run_stage2_sampling.py",
            "migrate_stage2_data.py",
            "monitor_stage2_migration.sh",
            "quick_status.sh",
            "check_stage_overlap.py"
        ],
        "monitoring/": [
            "monitor_deadlock.sh",
            "monitor_simple.sh", 
            "monitor_training.sh",
            "simple_monitor.sh",
            "simple_watch.sh",
            "live_log.sh",
            "watch_training.sh"
        ],
        "training/": [
            "train_and_monitor.sh",
            "train_with_monitor.sh"
        ]
    }
    
    print("🔄 Scripts 폴더 재구성 시작...")
    
    # 백업 생성
    backup_dir = scripts_dir / "_backup_old_structure"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    # 루트 레벨 파일들을 백업으로 복사
    backup_dir.mkdir()
    for file in scripts_dir.glob("*.sh"):
        shutil.copy2(file, backup_dir)
    for file in scripts_dir.glob("*.py"):
        shutil.copy2(file, backup_dir)
    
    print(f"📦 기존 파일들을 {backup_dir}에 백업했습니다.")
    
    # 새 디렉토리 구조 생성 및 파일 이동
    for new_dir, files in new_structure.items():
        target_dir = scripts_dir / new_dir
        target_dir.mkdir(exist_ok=True)
        
        print(f"📁 {new_dir} 디렉토리 생성...")
        
        for filename in files:
            source_file = scripts_dir / filename
            target_file = target_dir / filename
            
            if source_file.exists():
                shutil.move(str(source_file), str(target_file))
                print(f"  ✅ {filename} → {new_dir}")
            else:
                print(f"  ⚠️  {filename} 파일을 찾을 수 없음")
    
    # README.md 업데이트
    readme_content = """# Scripts 디렉토리 구조

## 📁 구조 개요

```
scripts/
├── core/                   # 핵심 유틸리티
│   ├── python_safe.sh     # 안전한 Python 실행
│   ├── setup_aliases.sh   # 편의 별칭 설정
│   ├── setup_venv.sh      # 가상환경 설정
│   └── update_docs.sh     # 문서 업데이트
│
├── stage1/                 # Stage 1 관련
│   ├── migrate_stage1_images_only.sh
│   └── migrate_stage1_to_ssd.sh
│
├── stage2/                 # Stage 2 관련
│   ├── run_stage2_sampling.py         # Stage 2 샘플링
│   ├── migrate_stage2_data.py         # Stage 2 데이터 이전
│   ├── monitor_stage2_migration.sh    # 실시간 모니터링
│   ├── quick_status.sh               # 빠른 상태 확인
│   └── check_stage_overlap.py        # Stage 중복 확인
│
├── monitoring/             # 모니터링 도구
│   ├── monitor_deadlock.sh
│   ├── monitor_simple.sh
│   ├── monitor_training.sh
│   ├── simple_monitor.sh
│   ├── simple_watch.sh
│   ├── live_log.sh
│   └── watch_training.sh
│
├── training/               # 학습 관련
│   ├── train_and_monitor.sh
│   └── train_with_monitor.sh
│
├── data/                   # 데이터 처리 (기존 유지)
├── deployment/             # 배포 관련 (기존 유지)
└── testing/               # 테스트 관련 (기존 유지)
```

## 🚀 빠른 사용법

### Stage 2 작업
```bash
# Stage 2 샘플링
./scripts/stage2/run_stage2_sampling.py

# Stage 2 데이터 이전
./scripts/stage2/migrate_stage2_data.py

# 진행 상황 모니터링
./scripts/stage2/quick_status.sh
./scripts/stage2/monitor_stage2_migration.sh
```

### 모니터링
```bash
# 학습 모니터링
./scripts/monitoring/monitor_training.sh

# 데드락 모니터링  
./scripts/monitoring/monitor_deadlock.sh
```

### 핵심 도구
```bash
# 안전한 Python 실행
./scripts/core/python_safe.sh [명령어]

# 환경 설정
./scripts/core/setup_venv.sh
```
"""
    
    with open(scripts_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📝 README.md 업데이트 완료")
    print("\n✅ Scripts 폴더 재구성 완료!")
    print(f"📦 기존 파일들은 {backup_dir}에 백업되었습니다.")
    
    # 새로운 구조 출력
    print("\n📁 새로운 구조:")
    for root, dirs, files in os.walk(scripts_dir):
        level = root.replace(str(scripts_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and file != "reorganize_scripts.py":
                print(f"{subindent}{file}")

if __name__ == "__main__":
    reorganize_scripts()