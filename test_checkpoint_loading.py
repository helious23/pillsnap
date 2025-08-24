#!/usr/bin/env python3
"""체크포인트 로딩 테스트"""

import sys
import torch
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.train_stage3_two_stage import TwoStageTrainingConfig
from src.models.classifier_efficientnetv2 import create_pillsnap_classifier

def test_checkpoint_loading():
    """체크포인트 로딩 테스트"""
    
    # 1. 체크포인트 로드
    checkpoint_path = "artifacts/stage3/checkpoints/stage3_classification_best.pt"
    print(f"1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    print(f"   - Epoch: {checkpoint.get('epoch')}")
    print(f"   - Accuracy: {checkpoint.get('accuracy', 0):.4f}")
    print(f"   - Top5 Accuracy: {checkpoint.get('top5_accuracy', 0):.4f}")
    
    # 2. 모델 생성
    print("\n2. Creating model...")
    model = create_pillsnap_classifier(
        num_classes=4020,  # Stage 3 클래스 수
        model_name="efficientnetv2_l",
        pretrained=False,
        device="cuda"
    )
    
    # 모델 키 확인
    model_keys = set(model.state_dict().keys())
    print(f"   - Model has {len(model_keys)} keys")
    print(f"   - First 5 keys: {list(model_keys)[:5]}")
    
    # 3. 체크포인트 state_dict 확인
    print("\n3. Checking checkpoint state_dict...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"   - Found model_state_dict with {len(state_dict)} keys")
        print(f"   - First 5 keys: {list(state_dict.keys())[:5]}")
        
        # _orig_mod prefix 제거
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['_orig_mod.', 'module.', 'ema.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned_state_dict[new_key] = v
        
        print(f"   - After cleaning: {len(cleaned_state_dict)} keys")
        print(f"   - First 5 cleaned keys: {list(cleaned_state_dict.keys())[:5]}")
        
        # 4. 키 매칭 확인
        print("\n4. Key matching...")
        loaded_keys = set(cleaned_state_dict.keys())
        matched = model_keys & loaded_keys
        missing = model_keys - loaded_keys
        unexpected = loaded_keys - model_keys
        
        print(f"   - Matched: {len(matched)}/{len(model_keys)}")
        print(f"   - Missing: {len(missing)}")
        print(f"   - Unexpected: {len(unexpected)}")
        
        if missing:
            print(f"   - First 5 missing: {list(missing)[:5]}")
        if unexpected:
            print(f"   - First 5 unexpected: {list(unexpected)[:5]}")
        
        # 5. 모델 로드 시도
        print("\n5. Loading state_dict...")
        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
            print("   ✅ Strict loading successful!")
        except RuntimeError as e:
            print(f"   ⚠️ Strict loading failed: {str(e)[:200]}")
            print("   Trying non-strict loading...")
            model.load_state_dict(cleaned_state_dict, strict=False)
            print("   ✅ Non-strict loading successful!")
        
        # 6. 간단한 추론 테스트
        print("\n6. Simple inference test...")
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384).cuda()
            output = model(dummy_input)
            print(f"   - Output shape: {output.shape}")
            print(f"   - Output mean: {output.mean().item():.4f}")
            print(f"   - Output std: {output.std().item():.4f}")
            
            # Softmax 확인
            probs = torch.softmax(output, dim=1)
            max_prob, max_idx = probs.max(1)
            print(f"   - Max probability: {max_prob.item():.4f}")
            print(f"   - Predicted class: {max_idx.item()}")
            
    else:
        print("   ❌ No model_state_dict found in checkpoint!")
        print(f"   Available keys: {list(checkpoint.keys())}")

if __name__ == "__main__":
    test_checkpoint_loading()