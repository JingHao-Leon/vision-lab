"""Train the from-scratch DDPM on MNIST, then sample a grid of digits.

3090 measured: 12 epochs ≈ 12 minutes; ancestral sampling (400 steps) of 32
digits ≈ 1 minute. Output: results/ddpm_samples.png + ddpm_mnist.json.

    uv run python train_ddpm.py [--epochs 12]
"""

from __future__ import annotations

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
from torchvision.utils import save_image

from vision_lab.ddpm import DDPM, UNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=400)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "training script targets a CUDA GPU"
    device = torch.device("cuda")
    torch.manual_seed(0)

    tf = transforms.Compose([
        transforms.Pad(2),  # 28→32 keeps the UNet's downsample math uniform
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST("data", train=True, download=True, transform=tf)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UNet(in_ch=1, base=64).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params={n_params/1e6:.2f}M timesteps={args.timesteps}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device)
            xt, noise = ddpm.q_sample(x, t)
            loss = F.mse_loss(model(xt, t), noise)  # ε-prediction objective
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"epoch {epoch:>2} eps-mse {loss.item():.4f}")

    # sample 32 digits → 4x8 grid, denormalized for the png
    samples = ddpm.sample(model, n=32, img_size=32, in_ch=1)
    Path("results").mkdir(exist_ok=True)
    save_image((samples + 1) / 2, "results/ddpm_samples.png", nrow=8)

    out = {"model": "DDPM-from-scratch", "dataset": "MNIST",
           "params_m": round(n_params / 1e6, 2), "timesteps": args.timesteps,
           "epochs": args.epochs, "final_eps_mse": round(loss.item(), 5),
           "minutes": round((time.time() - t0) / 60, 1),
           "gpu": torch.cuda.get_device_name(0)}
    Path("results/ddpm_mnist.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE {out['minutes']} min | saved results/ddpm_samples.png + ddpm_mnist.json")


if __name__ == "__main__":
    main()
