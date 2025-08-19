#!/usr/bin/env python3
"""
GPU-A 스모크 테스트: 순수 PyTorch 합성 데이터 (RTX 5080)
목적: PyTorch 2.7.0+cu128에서 기본 GPU 연산 및 sm_120 최적화 검증
"""
import time, json, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# RTX 5080 최적화 설정
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class GPUSynthDataset(Dataset):
    def __init__(self, n=64, num_classes=19, img_size=224):
        self.n, self.C, self.size = n, num_classes, img_size
    
    def __len__(self): 
        return self.n
    
    def __getitem__(self, i):
        # RTX 5080 메모리 활용을 위해 더 큰 이미지
        x = torch.rand(3, self.size, self.size)
        y = torch.randint(0, self.C, ())
        return x, y

class GPUNet(nn.Module):
    """RTX 5080 최적화 모델 (EfficientNet-B0 크기)"""
    def __init__(self, num_classes=19):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
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
    print("🚀 GPU-A Smoke Test (RTX 5080 + PyTorch 2.7.0)")
    
    # GPU 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"🔧 Device: {gpu_name} (sm_{capability[0]}{capability[1]})")
    
    start_time = time.time()
    run_tag = f"GPU_A_synth_{int(time.time())}"
    output_dir = Path(f"artifacts/gpu_runs/{run_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 (RTX 5080 메모리 활용)
    dataset = GPUSynthDataset(n=128, num_classes=19, img_size=224)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, 
                       num_workers=4, pin_memory=True)
    
    # 모델
    model = GPUNet(19).to(device)
    
    # torch.compile 최적화 (PyTorch 2.7.0) - 개발환경에서 헤더 파일 이슈로 비활성화
    # try:
    #     model = torch.compile(model, mode='max-autotune')
    #     print("✅ torch.compile enabled")
    # except Exception as e:
    #     print(f"⚠️ torch.compile failed: {e}")
    print("ℹ️ torch.compile disabled (dev environment)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision 학습
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    losses = []
    
    print("🏃 GPU training started...")
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        print(f"  Batch {i+1}/{len(loader)}: loss={loss.item():.4f}, GPU_mem={torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        if i >= 4:  # 5배치 학습
            break
    
    avg_loss = sum(losses) / len(losses)
    
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
        "avg_loss": avg_loss,
        "device": str(device),
        "gpu_name": gpu_name,
        "capability": capability,
        "memory_stats": memory_stats
    }, output_dir / "gpu_checkpoint.pt")
    
    metrics = {
        "success": True,
        "device": str(device),
        "gpu_name": gpu_name,
        "compute_capability": f"sm_{capability[0]}{capability[1]}",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "avg_loss": avg_loss,
        "batches_processed": len(losses),
        "memory_peak_gb": memory_stats["max_allocated"],
        "elapsed_seconds": time.time() - start_time,
        "run_tag": run_tag,
        "torch_compile": "enabled" if hasattr(model, '_compiled') else "disabled"
    }
    
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    elapsed = time.time() - start_time
    print(f"✅ GPU-A test completed in {elapsed:.1f}s")
    print(f"📊 Average loss: {avg_loss:.4f}")
    print(f"🎯 Peak GPU memory: {memory_stats['max_allocated']:.1f}GB")
    print(f"📁 Results: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    result = main()
    print(f"🎉 SUCCESS: {result['run_tag']}")