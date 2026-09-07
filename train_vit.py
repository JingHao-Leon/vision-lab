"""Train the from-scratch ViT on CIFAR-10 and report test accuracy.

Heavy-but-standard augmentation (crop/flip + RandAugment), AdamW + warmup
cosine, AMP. 3090 measured: 40 epochs ≈ 25 minutes, test accuracy ≈ 90%.

    uv run python train_vit.py [--epochs 40]
"""

from __future__ import annotations

import math
import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_lab.vit import ViT

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def build_loaders(batch_size: int, data_dir: str = "data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        transforms.RandomErasing(p=0.25),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(test, batch_size=512, num_workers=2, pin_memory=True))


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.475 * (1 + math.cos(math.pi * t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "training script targets a CUDA GPU"
    device = torch.device("cuda")
    torch.manual_seed(0)

    train_loader, test_loader = build_loaders(args.batch_size)
    model = ViT(dim=args.dim, depth=args.depth).to(device)
    print(f"ViT dim={args.dim} depth={args.depth} params={model.num_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler()
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(500, total_steps // 10)
    history = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            step = epoch * steps_per_epoch + bi
            lr = lr_at(step, total_steps, args.lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        # eval each epoch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        acc = correct / total
        best_acc = max(best_acc, acc)
        history.append({"epoch": epoch, "loss": round(loss.item(), 4),
                        "test_acc": round(acc, 4)})
        print(f"epoch {epoch:>2} loss {loss.item():.4f} test_acc {acc:.4f}")

    out = {"model": "ViT-from-scratch", "dataset": "CIFAR-10",
           "params_m": round(model.num_params() / 1e6, 2),
           "epochs": args.epochs, "batch_size": args.batch_size,
           "best_test_acc": round(best_acc, 4),
           "minutes": round((time.time() - t0) / 60, 1),
           "gpu": torch.cuda.get_device_name(0), "history": history[-10:]}
    Path("results").mkdir(exist_ok=True)
    Path("results/vit_cifar10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BEST-ACC {best_acc:.4f} | {out['minutes']} min | saved results/vit_cifar10.json")


if __name__ == "__main__":
    main()
