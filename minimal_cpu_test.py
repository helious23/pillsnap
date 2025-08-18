#!/usr/bin/env python3
"""
최소 CPU 테스트 - 작은 모델 사용
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import time
import json

# 환경 설정
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class SimpleModel(nn.Module):
    """매우 작은 테스트용 모델"""

    def __init__(self, num_classes=19):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    print("🚀 Minimal CPU Test")
    start_time = time.time()

    # 출력 디렉토리 생성
    run_tag = f"minimal_cpu_{int(time.time())}"
    output_dir = Path(f"artifacts/cpu_runs/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")

    # 데이터셋 로드
    from pillsnap.stage2.dataset_cls import PillsnapClsDataset

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # 매우 작은 크기
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

    # 최소 데이터로더
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 작은 모델 생성
    model = SimpleModel(num_classes=19)
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 옵티마이저 및 손실 함수
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("🏃 Starting minimal training...")

    # 매우 짧은 학습
    model.train()
    losses = []

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= 3:  # 3개 샘플만 처리
            break

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(
            f"  Sample {batch_idx+1}: loss={loss.item():.4f}, target={targets.item()}"
        )

    avg_loss = sum(losses) / len(losses)
    print(f"✅ Training completed: avg_loss={avg_loss:.4f}")

    # 체크포인트 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_loss,
        "num_classes": 19,
        "model_type": "SimpleModel",
    }

    torch.save(checkpoint, output_dir / "minimal_checkpoint.pt")

    # 메트릭 저장
    metrics = {
        "training_completed": True,
        "device": "cpu",
        "samples_processed": len(losses),
        "train_loss": avg_loss,
        "elapsed_seconds": time.time() - start_time,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "run_tag": run_tag,
        "model_type": "SimpleModel",
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"🏁 Minimal test completed in {elapsed:.1f}s")
    print(f"📊 Results saved to: {output_dir}")

    return metrics, output_dir


if __name__ == "__main__":
    result, outdir = main()
    print(f"✅ SUCCESS: {result['run_tag']}")
    print(f"📁 Directory: {outdir}")
