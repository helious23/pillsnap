#!/usr/bin/env python3
"""
GPU-B 스모크 테스트: 실데이터 + PillsnapClsDataset (RTX 5080)
목적: 실제 데이터셋으로 GPU 학습 파이프라인 검증
"""
import time, json, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# RTX 5080 최적화 설정
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def gpu_safe_collate(batch):
    """GPU 최적화된 collate 함수"""
    xs, ys = [], []
    for x, y in batch:
        # PIL 이미지를 Tensor로 변환
        if not isinstance(x, torch.Tensor):
            try:
                import numpy as np
                if hasattr(x, "size"):  # PIL
                    x = np.array(x)
                    x = torch.from_numpy(x)
                elif isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
            except Exception:
                raise RuntimeError("Input is not a Tensor and cannot be coerced safely.")
        
        if x.dtype != torch.float32: 
            x = x.float()
        
        # [H,W,C] → [C,H,W] 변환
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        
        # GPU 메모리 활용을 위해 더 큰 해상도 유지 (224x224)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            import torch.nn.functional as F
            x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # 0~1 정규화
        if x.max() > 1.5: 
            x = x / 255.0
        
        # ImageNet 정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - mean) / std
            
        xs.append(x)
        ys.append(torch.as_tensor(y, dtype=torch.long))
    
    return torch.stack(xs, 0), torch.stack(ys, 0)

class GPUEfficientHead(nn.Module):
    """GPU 최적화 EfficientNet 스타일 헤드"""
    def __init__(self, num_classes=19):
        super().__init__()
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            
            # Block 1: 32 -> 64
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            
            # Block 2: 64 -> 128  
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    print("🚀 GPU-B Smoke Test (Real Data + RTX 5080)")
    
    # GPU 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"🔧 Device: {gpu_name} (sm_{capability[0]}{capability[1]})")
    
    start_time = time.time()
    run_tag = f"GPU_B_real_{int(time.time())}"
    output_dir = Path(f"artifacts/gpu_runs/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PillsnapClsDataset 로드
    from pillsnap.stage2.dataset_cls import PillsnapClsDataset
    
    dataset = PillsnapClsDataset(
        manifest_csv="artifacts/manifest_enriched.csv",
        classes_json="artifacts/classes_step11.json",
        transform=None,  # GPU collate에서 처리
        require_exists=False
    )
    
    print(f"📊 Dataset: {len(dataset)} samples")
    num_classes = len(dataset.clsmap)
    
    # 작은 샘플 선택 (GPU 메모리 고려)
    if len(dataset) > 16:
        indices = list(range(16))  # 첫 16개 샘플
        dataset = Subset(dataset, indices)
    
    # GPU 최적화 DataLoader
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        num_workers=2, pin_memory=True,  # GPU 전송 최적화
        persistent_workers=True,
        collate_fn=gpu_safe_collate
    )
    
    print(f"📋 Training: {len(dataset)} samples, {num_classes} classes")
    
    # 모델
    model = GPUEfficientHead(num_classes).to(device)
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision 학습
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    losses = []
    
    print("🏃 GPU real data training started...")
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        # 예측 정확도 계산
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
        
        print(f"  Batch {i+1}/{len(loader)}: loss={loss.item():.4f}, acc={acc:.3f}, GPU_mem={torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        if i >= 3:  # 4배치 학습
            break
    
    avg_loss = sum(losses) / len(losses)
    
    # 간단한 검증
    model.eval()
    val_losses = []
    val_accs = []
    
    print("🔍 Validation...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= 2:  # 2배치 검증
                break
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                val_loss = criterion(outputs, targets)
            
            preds = outputs.argmax(dim=1)
            val_acc = (preds == targets).float().mean().item()
            
            val_losses.append(val_loss.item())
            val_accs.append(val_acc)
            
            print(f"  Val batch {i+1}: loss={val_loss.item():.4f}, acc={val_acc:.3f}")
    
    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    avg_val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
    
    # GPU 메모리 정보
    memory_stats = {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "max_reserved": torch.cuda.max_memory_reserved() / 1e9,
    }
    
    # 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "train_loss": avg_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_acc,
        "device": str(device),
        "gpu_name": gpu_name,
        "capability": capability,
        "memory_stats": memory_stats,
        "num_classes": num_classes
    }, output_dir / "gpu_real_checkpoint.pt")
    
    metrics = {
        "success": True,
        "device": str(device),
        "gpu_name": gpu_name,
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_loss": avg_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_acc,
        "num_classes": num_classes,
        "batches_processed": len(losses),
        "memory_peak_gb": memory_stats["max_allocated"],
        "elapsed_seconds": time.time() - start_time,
        "run_tag": run_tag,
        "dataset_type": "PillsnapClsDataset"
    }
    
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    elapsed = time.time() - start_time
    print(f"✅ GPU-B test completed in {elapsed:.1f}s")
    print(f"📊 Train loss: {avg_loss:.4f}")
    print(f"📊 Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.3f}")
    print(f"🎯 Peak GPU memory: {memory_stats['max_allocated']:.1f}GB")
    print(f"📁 Results: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    result = main()
    print(f"🎉 SUCCESS: {result['run_tag']}")