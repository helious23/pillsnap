"""
목적:
  - artifacts/manifest_enriched.csv + classes_step11.json 기반 분류 학습 실행
입력:
  - --manifest, --classes, --epochs, --batch-size, --limit, --amp, --outdir
출력:
  - checkpoints/best.pt, logs/train_metrics.json 등
검증 포인트:
  - limit로 빠른 스모크 실행 가능
  - AMP/CPU 모두 안전 동작
  - 클래스 수 = classes_step11.json 길이
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .dataset_cls import PillsnapClsDataset
from .models import create_efficientnetv2_l, get_model_info

logger = logging.getLogger(__name__)


def build_transforms(
    input_size: int = 448,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """학습 및 검증용 이미지 변환 생성"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def build_loader(
    manifest: str,
    classes: str,
    batch_size: int,
    limit: int = 0,
    workers: int = 6,
    val_split: float = 0.2,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, int]:
    """데이터로더 생성"""
    train_transform, val_transform = build_transforms()

    # 전체 데이터셋 로드
    full_dataset = PillsnapClsDataset(
        manifest_csv=manifest,
        classes_json=classes,
        transform=train_transform,
        require_exists=True,
    )

    # limit 적용
    if limit > 0 and len(full_dataset) > limit:
        indices = list(range(limit))
        full_dataset = Subset(full_dataset, indices)
        print(f"📊 Dataset limited to {limit} samples")

    # train/val 분할
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 검증 데이터셋 transform 변경
    if hasattr(val_dataset.dataset, "transform"):
        val_dataset.dataset.transform = val_transform

    print(f"📊 Split: train={len(train_dataset)}, val={len(val_dataset)}")

    # 데이터로더 생성 (CPU 최적화)
    pin_memory = device == "cuda"  # CPU에서는 pin_memory=False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,  # WSL CPU 안정성
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # CPU에서는 동일한 배치 크기 사용
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    # 클래스 수 반환
    with open(classes, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    num_classes = len(class_map)

    return train_loader, val_loader, num_classes


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float]:
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 유효한 타겟만 사용 (-1 제외)
        valid_mask = targets >= 0
        if not valid_mask.any():
            continue

        images = images[valid_mask]
        targets = targets[valid_mask]

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Tuple[float, float]:
    """한 에포크 학습"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 유효한 타겟만 사용 (-1 제외)
        valid_mask = targets >= 0
        if not valid_mask.any():
            continue

        images = images[valid_mask]
        targets = targets[valid_mask]

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        # 주기적 로그
        if batch_idx % max(1, len(loader) // 10) == 0:
            current_acc = correct / total if total > 0 else 0.0
            print(
                f"  Batch {batch_idx:3d}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.3f}"
            )

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_acc: float,
    output_dir: Path,
    is_best: bool = False,
):
    """체크포인트 저장"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_accuracy": best_acc,
    }

    # 항상 last.pt 저장
    torch.save(checkpoint, output_dir / "last.pt")

    # 최고 성능일 때 best.pt 저장
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")
        print(f"💾 New best model saved (accuracy: {best_acc:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="PillSnap Stage 2 Classification Training"
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/manifest_enriched.csv",
        help="Enriched manifest CSV file",
    )
    parser.add_argument(
        "--classes",
        default="artifacts/classes_step11.json",
        help="EDI classes JSON file",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="Limit samples for quick smoke test (0 for all)",
    )
    parser.add_argument(
        "--outdir",
        default="artifacts/cpu_runs/default",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training (forced CPU for smoke test)",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        default=False,
        help="Disable automatic mixed precision (default: disabled)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of data loading workers (WSL optimized)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()

    # CPU 전용 최적화 설정
    if args.device == "cpu":
        torch.set_num_threads(8)  # WSL CPU 스레드 최적화
        print("🔧 CPU optimization: torch.set_num_threads(8)")

    # 출력 디렉토리 생성
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 설정 출력
    print("🚀 PillSnap Stage 2 Classification Training (CPU-ONLY SMOKE)")
    print(f"   Manifest: {args.manifest}")
    print(f"   Classes:  {args.classes}")
    print(f"   Epochs:   {args.epochs}")
    print(f"   Batch:    {args.batch_size}")
    print(f"   Limit:    {args.limit}")
    print(f"   Output:   {args.outdir}")
    print(f"   AMP:      {args.amp}")

    # 디바이스 설정 (강제 CPU 또는 사용자 지정)
    device = torch.device(args.device)
    print(f"   Device:   {device} (forced via --device)")

    # 데이터로더 생성
    print("\n📊 Building data loaders...")
    train_loader, val_loader, num_classes = build_loader(
        args.manifest,
        args.classes,
        args.batch_size,
        args.limit,
        args.workers,
        device=args.device,
    )

    # 모델 생성 (CPU 스모크는 pretrained=False로 가벼운 시작)
    print(f"\n🔧 Creating model for {num_classes} classes...")
    pretrained = args.device == "cuda"  # CPU에서는 pretrained=False
    model = create_efficientnetv2_l(num_classes=num_classes, pretrained=pretrained)

    if not pretrained:
        print("   Using random weights for CPU smoke test")
    model = model.to(device)

    # 모델 정보 출력
    model_info = get_model_info(model)
    print(
        f"   Parameters: {model_info['total_parameters']:,} total, {model_info['trainable_parameters']:,} trainable"
    )
    print(f"   Model size: {model_info['model_size_mb']:.1f} MB")

    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # AMP는 CUDA에서만 활성화
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)  # 폴백
    else:
        scaler = None

    if device.type == "cpu" and args.amp:
        print("⚠️  AMP disabled on CPU device")

    print(f"   Mixed Precision: {use_amp}")

    # 학습 루프
    print(f"\n🏃 Training for {args.epochs} epochs...")
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n📈 Epoch {epoch}/{args.epochs}")

        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )

        # 검증
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)

        # 결과 출력
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.3f}")

        # 체크포인트 저장
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc

        save_checkpoint(model, optimizer, epoch, best_accuracy, output_dir, is_best)

    # 학습 완료
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed:.1f}s")
    print(f"   Best validation accuracy: {best_accuracy:.3f}")
    print(f"   Checkpoints saved to: {output_dir}")
    print("   - last.pt: latest model")
    print("   - best.pt: best validation model")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
