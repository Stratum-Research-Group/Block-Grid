import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from blockgrid.config import BlockGridConfig
from blockgrid.model import BlockGridModel
from baselines.perceiver import PerceiverBaseline
from experiments.common import set_seed, count_parameters

def patchify_cifar(images: torch.Tensor, P: int = 4, Q: int = 4) -> torch.Tensor:
    """
    Transform 32x32 CIFAR images [B, C, 32, 32] into an 8x8 grid of 4x4 patches:
    Returns: [B, H=8, W=8, P=4, Q=4, F=3]
    """
    B, C, H_orig, W_orig = images.shape
    H, W = H_orig // P, W_orig // Q # 8, 8
    # Unfold along height then width
    # [B, C, H, P, W, Q]
    x = images.view(B, C, H, P, W, Q)
    # Permute to [B, H, W, P, Q, C]
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x

class PerceiverClassifier(nn.Module):
    def __init__(self, baseline: PerceiverBaseline, num_classes: int = 10):
        super().__init__()
        self.baseline = baseline
        self.head = nn.Linear(baseline.F_dim, num_classes)

    def forward(self, x: torch.Tensor):
        out = self.baseline(x) # [B, H, W, P, Q, F]
        pooled = out.mean(dim=(1, 2, 3, 4)) # [B, F]
        return self.head(pooled)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        x = patchify_cifar(imgs)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = patchify_cifar(imgs)
            logits = model(x)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def run_cifar_pilot(config_path: str = "configs/pilot_cifar.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    print(f"=== Stage 2: CIFAR-10 Cell Grid Smoke Test ===")
    print(f"Device: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    data_dir = cfg.get("data_dir", "./data")
    print("Loading CIFAR-10 dataset...")
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=2)
    
    H, W, P, Q, F_dim = 8, 8, 4, 4, 3
    epochs = cfg.get("epochs", 5)
    criterion = nn.CrossEntropyLoss()
    
    # 1. BlockGrid Model (4 layers, H=W=8, P=Q=4, F=3)
    bg_cfg = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=0,
        K=32, d=64,
        slot_mix="planar_uncrossed",
        inner_op=True,
        tied_readback=True,
        layernorm_cells=True
    )
    blockgrid_model = BlockGridModel(bg_cfg, num_layers=4, num_classes=10).to(device)
    bg_params = count_parameters(blockgrid_model)
    print(f"BlockGrid Classifier Parameters: {bg_params:,}")
    
    bg_optimizer = torch.optim.AdamW(blockgrid_model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 0.01))
    
    print("\nTraining BlockGrid Model on CIFAR-10 Grid...")
    t0 = time.time()
    for ep in range(epochs):
        train_loss, train_acc = train_epoch(blockgrid_model, train_loader, bg_optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(blockgrid_model, test_loader, criterion, device)
        print(f"Epoch {ep+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    print(f"BlockGrid training completed in {time.time() - t0:.2f}s")
    
    # 2. Baseline Perceiver Model
    b2_base = PerceiverBaseline(H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=0, K=32, num_layers=4)
    perceiver_model = PerceiverClassifier(b2_base, num_classes=10).to(device)
    p_params = count_parameters(perceiver_model)
    print(f"\nPerceiver Classifier Parameters: {p_params:,}")
    
    p_optimizer = torch.optim.AdamW(perceiver_model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 0.01))
    
    print("\nTraining Perceiver Baseline on CIFAR-10...")
    t0 = time.time()
    for ep in range(epochs):
        train_loss, train_acc = train_epoch(perceiver_model, train_loader, p_optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(perceiver_model, test_loader, criterion, device)
        print(f"Epoch {ep+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    print(f"Perceiver training completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pilot_cifar.yaml")
    args = parser.parse_args()
    run_cifar_pilot(args.config)
