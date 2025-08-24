#!/usr/bin/env python3
"""
Resume 기능 테스트 스크립트
"""

import sys
import os
sys.path.append('/home/max16/pillsnap')

from pathlib import Path
import tempfile
import shutil

def test_resume_help():
    """Help message에 resume 옵션이 포함되어 있는지 확인"""
    try:
        os.system("cd /home/max16/pillsnap && python -m src.training.train_stage3_two_stage --help > /tmp/help_output.txt 2>&1")
        
        with open('/tmp/help_output.txt', 'r') as f:
            help_text = f.read()
            
        print("🔍 Help message 테스트:")
        
        if '--resume' in help_text:
            print("✅ --resume 옵션 확인됨")
        else:
            print("❌ --resume 옵션이 help에 없음")
            
        if '--lr-classifier' in help_text:
            print("✅ --lr-classifier 옵션 확인됨")
        else:
            print("❌ --lr-classifier 옵션이 help에 없음")
            
        if '--lr-detector' in help_text:
            print("✅ --lr-detector 옵션 확인됨")
        else:
            print("❌ --lr-detector 옵션이 help에 없음")
            
        print("\nHelp output excerpt:")
        lines = help_text.split('\n')
        for line in lines:
            if 'resume' in line or 'lr-classifier' in line or 'lr-detector' in line:
                print(f"  {line}")
                
    except Exception as e:
        print(f"❌ Help test 실패: {e}")

def test_checkpoint_files_exist():
    """현재 학습에서 생성된 체크포인트 파일들 확인"""
    print("\n🔍 체크포인트 파일 확인:")
    
    checkpoint_dir = Path("/home/max16/pillsnap_data/exp/exp01/checkpoints")
    
    if checkpoint_dir.exists():
        print(f"✅ Checkpoint 디렉토리 존재: {checkpoint_dir}")
        
        for checkpoint_file in checkpoint_dir.glob("stage3_*.pt"):
            file_size = checkpoint_file.stat().st_size / (1024*1024)  # MB
            print(f"  📁 {checkpoint_file.name}: {file_size:.1f} MB")
            
    else:
        print(f"❌ Checkpoint 디렉토리가 없음: {checkpoint_dir}")

def test_checkpoint_loading():
    """체크포인트 로딩 기능 테스트"""
    print("\n🔍 체크포인트 로딩 테스트:")
    
    try:
        from src.training.train_stage3_two_stage import Stage3TwoStageTrainer
        
        trainer = Stage3TwoStageTrainer(
            config_path="config.yaml",
            manifest_train="artifacts/stage3/manifest_train.csv", 
            manifest_val="artifacts/stage3/manifest_val.csv",
            device="cuda"
        )
        
        # Best checkpoint 로딩 테스트
        best_checkpoint = Path("/home/max16/pillsnap_data/exp/exp01/checkpoints/stage3_classification_best.pt")
        
        if best_checkpoint.exists():
            epoch, accuracy = trainer.load_checkpoint(str(best_checkpoint))
            print(f"✅ Best checkpoint 로딩 성공")
            print(f"  📊 Epoch: {epoch}, Accuracy: {accuracy:.3f}")
        else:
            print(f"❌ Best checkpoint 파일이 없음: {best_checkpoint}")
            
    except Exception as e:
        print(f"❌ 체크포인트 로딩 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 Resume 기능 테스트 시작\n")
    
    test_resume_help()
    test_checkpoint_files_exist()
    test_checkpoint_loading()
    
    print("\n✅ 모든 테스트 완료!")