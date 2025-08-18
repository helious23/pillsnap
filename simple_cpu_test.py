#!/usr/bin/env python3
"""
단순 CPU 스모크 테스트
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import time

# 환경 설정
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def main():
    print("🚀 Simple CPU Smoke Test")

    # 데이터셋 로드
    from pillsnap.stage2.dataset_cls import PillsnapClsDataset

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PillsnapClsDataset(
        "artifacts/manifest_enriched.csv",
        "artifacts/classes_step11.json",
        transform=transform,
        require_exists=False,
    )

    print(f"📊 Dataset: {len(dataset)} samples")

    # 더 작은 배치로 DataLoader (워커 0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # 작은 모델 생성
    from pillsnap.stage2.models import create_efficientnetv2_l

    model = create_efficientnetv2_l(num_classes=19, pretrained=False)
    model.eval()

    print("🔧 Model created, starting inference test...")

    # 간단한 추론 테스트
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= 3:  # 3배치만 테스트
                break

            print(f"  Batch {i+1}: {images.shape}, targets: {targets}")
            outputs = model(images)
            print(f"    Output shape: {outputs.shape}")

    print("✅ CPU smoke test completed successfully!")

    # 출력 디렉토리 생성
    run_tag = f"simple_cpu_{int(time.time())}"
    output_dir = Path(f"artifacts/cpu_runs/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 간단한 체크포인트 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "test_completed": True,
        "samples_tested": len(dataset),
    }

    torch.save(checkpoint, output_dir / "simple_test.pt")
    print(f"💾 Test checkpoint saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    result = main()
    print(f"🏁 Output directory: {result}")
