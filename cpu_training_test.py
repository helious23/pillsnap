#!/usr/bin/env python3
"""
CPU 학습 스모크 테스트
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
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"


def main():
    print("🚀 CPU Training Smoke Test")
    start_time = time.time()

    # 출력 디렉토리 생성
    run_tag = f"cpu_train_{int(time.time())}"
    output_dir = Path(f"artifacts/cpu_runs/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")

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

    # 간단한 train/val 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # 데이터로더 (워커 0, 작은 배치)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    print(f"📊 Split: train={len(train_dataset)}, val={len(val_dataset)}")

    # 모델 생성
    from pillsnap.stage2.models import create_efficientnetv2_l

    model = create_efficientnetv2_l(num_classes=19, pretrained=False)

    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print("🏃 Starting training...")

    # 짧은 학습 루프 (1 epoch)
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= 5:  # 최대 5배치만 학습
            break

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        print(f"  Batch {batch_idx+1}: loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, num_batches)
    print(f"✅ Training completed: avg_loss={avg_loss:.4f}")

    # 간단한 검증
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 3:  # 최대 3배치만 검증
                break

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            print(f"  Val batch {batch_idx+1}: loss={loss.item():.4f}")

    val_acc = correct / max(1, total)
    print(f"✅ Validation: loss={val_loss/3:.4f}, acc={val_acc:.3f}")

    # 체크포인트 저장
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_loss,
        "val_loss": val_loss / 3,
        "val_accuracy": val_acc,
        "epoch": 1,
        "num_classes": 19,
    }

    torch.save(checkpoint, output_dir / "checkpoint.pt")

    # 메트릭 저장
    metrics = {
        "training_completed": True,
        "device": "cpu",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": avg_loss,
        "val_loss": val_loss / 3,
        "val_accuracy": val_acc,
        "elapsed_seconds": time.time() - start_time,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "run_tag": run_tag,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"🏁 CPU training test completed in {elapsed:.1f}s")
    print(f"📊 Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    result = main()
    print(f"✅ SUCCESS: {result['run_tag']}")
