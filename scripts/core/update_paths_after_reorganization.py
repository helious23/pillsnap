#!/usr/bin/env python3
"""
Scripts 폴더 재구성 후 경로 참조 업데이트 스크립트
"""
import os
import re
from pathlib import Path

def update_path_references():
    """모든 파일에서 스크립트 경로 참조 업데이트"""
    
    # 업데이트할 경로 매핑
    path_mappings = {
        # Python 안전 실행 스크립트
        r'(\./)scripts/python_safe\.sh': r'\1scripts/core/python_safe.sh',
        
        # 환경 설정 스크립트
        r'(source |bash )scripts/setup_aliases\.sh': r'\1scripts/core/setup_aliases.sh',
        r'(source |bash )scripts/setup_venv\.sh': r'\1scripts/core/setup_venv.sh',
        r'(source |bash )scripts/update_docs\.sh': r'\1scripts/core/update_docs.sh',
        
        # Stage 2 스크립트
        r'(\./)scripts/run_stage2_sampling\.py': r'\1scripts/stage2/run_stage2_sampling.py',
        r'(\./)scripts/migrate_stage2_data\.py': r'\1scripts/stage2/migrate_stage2_data.py',
        r'(\./)scripts/check_stage_overlap\.py': r'\1scripts/stage2/check_stage_overlap.py',
        r'(\./)scripts/monitor_stage2_migration\.sh': r'\1scripts/stage2/monitor_stage2_migration.sh',
        r'(\./)scripts/quick_status\.sh': r'\1scripts/stage2/quick_status.sh',
        
        # Stage 1 스크립트
        r'(\./)scripts/migrate_stage1_to_ssd\.sh': r'\1scripts/stage1/migrate_stage1_to_ssd.sh',
        r'(\./)scripts/migrate_stage1_images_only\.sh': r'\1scripts/stage1/migrate_stage1_images_only.sh',
        
        # 모니터링 스크립트
        r'(\./)scripts/monitor_training\.sh': r'\1scripts/monitoring/monitor_training.sh',
        r'(\./)scripts/monitor_deadlock\.sh': r'\1scripts/monitoring/monitor_deadlock.sh',
        r'(\./)scripts/live_log\.sh': r'\1scripts/monitoring/live_log.sh',
        r'(\./)scripts/watch_training\.sh': r'\1scripts/monitoring/watch_training.sh',
        
        # 학습 스크립트
        r'(\./)scripts/train_and_monitor\.sh': r'\1scripts/training/train_and_monitor.sh',
        r'(\./)scripts/train_with_monitor\.sh': r'\1scripts/training/train_with_monitor.sh',
        
        # 일반적인 scripts/ 참조 (각주나 설명에서)
        r'bash scripts/bootstrap_venv\.sh': r'bash scripts/core/setup_venv.sh',
        r'bash scripts/run_api\.sh': r'bash scripts/deployment/run_api.sh',
        r'bash scripts/export_onnx\.sh': r'bash scripts/deployment/export_onnx.sh',
        r'bash scripts/maintenance\.sh': r'bash scripts/deployment/maintenance.sh',
    }
    
    # 업데이트할 파일 패턴
    file_patterns = [
        '**/*.md',
        '**/*.sh', 
        '**/*.py',
        '.claude/**/*.md'
    ]
    
    root = Path('/home/max16/pillsnap')
    updated_files = []
    
    print("🔄 Scripts 경로 참조 업데이트 시작...")
    
    # 각 파일 패턴에 대해
    for pattern in file_patterns:
        for file_path in root.glob(pattern):
            # 백업 폴더와 git 폴더는 제외
            if '_backup_old_structure' in str(file_path) or '.git' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 모든 경로 매핑 적용
                for old_pattern, new_pattern in path_mappings.items():
                    content = re.sub(old_pattern, new_pattern, content)
                
                # 변경사항이 있으면 파일 업데이트
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updated_files.append(str(file_path))
                    print(f"  ✅ {file_path.relative_to(root)}")
                    
            except Exception as e:
                print(f"  ⚠️  {file_path.relative_to(root)}: {e}")
    
    print(f"\n✅ 업데이트 완료! {len(updated_files)}개 파일 수정됨")
    
    # 중요한 파일들 개별 확인
    critical_files = [
        'README.md',
        'CLAUDE.md', 
        '.claude/commands/venv.md',
        '.claude/commands/initial-prompt.md',
        'scripts/core/setup_venv.sh',
        'scripts/core/setup_aliases.sh'
    ]
    
    print("\n🎯 중요 파일 업데이트 확인:")
    for file_name in critical_files:
        file_path = root / file_name
        if file_path.exists() and str(file_path) in updated_files:
            print(f"  ✅ {file_name}")
        elif file_path.exists():
            print(f"  ➖ {file_name} (변경사항 없음)")
        else:
            print(f"  ❌ {file_name} (파일 없음)")

if __name__ == "__main__":
    update_path_references()