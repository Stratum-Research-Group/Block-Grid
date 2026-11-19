import argparse
import yaml
import torch
import torch.nn as nn
from blockvga.config import BlockGridConfig
from blockvga.model import BlockGridModel
from experiments.common import set_seed, memory_preflight, count_parameters
from experiments.dataset import SyntheticBlockTransportDataset

def run_pilot(config_path: str):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    set_seed(cfg_dict.get("seed", 42))
    device = cfg_dict.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    print(f"=== Synthetic Transport Pilot Experiment ===")
    print(f"Device: {device}, Config: {config_path}")
    
    H, W, P, Q, F_dim, R = (
        cfg_dict["H"], cfg_dict["W"], cfg_dict["P"], cfg_dict["Q"], cfg_dict["F"], cfg_dict["R"]
    )
    
    cfg = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=R,
        K=cfg_dict.get("K", 32),
        d=cfg_dict.get("d", 128),
        tied_readback=False # Untied readback as recommended for asymmetric transport
    )
    
    model = BlockGridModel(cfg, num_layers=2)
    model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Section 4 Memory preflight
    peak_mb = memory_preflight(
        model, batch_size=cfg_dict["batch_size"],
        H=H, W=W, P=P, Q=Q, F_dim=F_dim, R=R, device=device
    )
    print(f"Memory preflight passed. Peak VRAM: {peak_mb:.2f} MB")
    
    # Synthetic dataset
    train_dataset = SyntheticBlockTransportDataset(
        num_episodes=cfg_dict.get("batch_size") * 10,
        task=cfg_dict.get("task", "1a"),
        H=H, W=W, P=P, Q=Q, F_dim=F_dim, seed=42
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_dict["lr"]),
        weight_decay=float(cfg_dict["weight_decay"])
    )
    criterion = nn.MSELoss()
    
    print("Running initial smoke training steps...")
    model.train()
    batch_size = cfg_dict["batch_size"]
    
    for step in range(5):
        optimizer.zero_grad()
        X_list, rf_list, dst_list, target_list = [], [], [], []
        for i in range(batch_size):
            X, rf, (di, dj), target, _ = train_dataset[step * batch_size + i]
            X_list.append(X)
            rf_list.append(rf)
            dst_list.append((di, dj))
            target_list.append(target)
            
        X_b = torch.stack(X_list).to(device)
        rf_b = torch.stack(rf_list).to(device)
        target_b = torch.stack(target_list).to(device)
        
        out = model(X_b, routing_features=rf_b)
        
        # Loss at destination cells only
        dst_preds = torch.stack([out[b, dst_list[b][0], dst_list[b][1]] for b in range(batch_size)])
        loss = criterion(dst_preds, target_b)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_dict.get("grad_clip", 1.0))
        optimizer.step()
        
        print(f"Step {step+1}/5 - Loss (MSE): {loss.item():.4f}")
        
    print("Pilot transport run successfully initialized and verified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pilot_transport.yaml")
    args = parser.parse_args()
    run_pilot(args.config)
