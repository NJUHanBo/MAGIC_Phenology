"""
黄海农场 STSC 推理骨架。

需要：
  - data/processed/stsc/YYYY_reflectance.npy
  - trained checkpoint under saved/stsc/models/PILA_STSC_A/...

用法：
  python tests/infer_huanghai_stsc.py --year 2015
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def find_ckpt():
    root = os.path.join(PROJECT_ROOT, "saved/stsc/models/PILA_STSC_A")
    if not os.path.isdir(root):
        return None
    runs = sorted(os.listdir(root))
    for r in reversed(runs):
        p = os.path.join(root, r, "model_best.pth")
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="无真实数据时用合成 STSC 冒烟")
    args = parser.parse_args()

    cfg_path = os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_STSC_A.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    refl_path = os.path.join(PROJECT_ROOT, f"data/processed/stsc/{args.year}_reflectance.npy")
    if args.demo or not os.path.exists(refl_path):
        print("Using demo reflectance (run qc_modis_stsc.py --demo for real path)")
        from scripts.qc_modis_stsc import temporal_linear_interpolate
        rng = np.random.default_rng(args.year)
        H = W = 30
        arr = rng.uniform(0.05, 0.4, size=(H * W, 7, 46)).astype(np.float32)
        rowcol = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
    else:
        arr = np.load(refl_path)
        rowcol = np.load(os.path.join(PROJECT_ROOT, f"data/processed/stsc/{args.year}_rowcol.npy"))
        H = int(rowcol[:, 0].max()) + 1
        W = int(rowcol[:, 1].max()) + 1

    x_mean = np.load(os.path.join(PROJECT_ROOT, "data/processed/stsc/x_mean.npy"))
    x_scale = np.load(os.path.join(PROJECT_ROOT, "data/processed/stsc/x_scale.npy"))
    flat = arr.reshape(len(arr), -1)
    x_std = torch.tensor((flat - x_mean) / x_scale, dtype=torch.float32)

    from model.model_phys_smpl import PHYS_VAE_SMPL
    model = PHYS_VAE_SMPL(cfg)
    ckpt = args.ckpt or find_ckpt()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
        print("loaded", ckpt)
    else:
        print("WARNING: no STSC checkpoint — random encoder (pipeline smoke only)")

    model.eval()
    fracs = {k: [] for k in ["wheat_frac", "rice_frac", "maize_frac", "soy_frac"]}
    bs = 256
    with torch.no_grad():
        for i in range(0, len(x_std), bs):
            xb = x_std[i:i + bs]
            z_phy_stat, _ = model.encode(xb)
            z_phy = torch.sigmoid(z_phy_stat["mean"])
            pred = model.physics_model.rescale(z_phy)
            for k in fracs:
                fracs[k].append(pred[k].cpu().numpy())

    for k in fracs:
        fracs[k] = np.concatenate(fracs[k])

    # 热力图
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, k in zip(axes.ravel(), fracs):
        grid = np.full((H, W), np.nan)
        for (r, c), v in zip(rowcol, fracs[k]):
            grid[int(r), int(c)] = v
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="YlGn")
        ax.set_title(k)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Huanghai STSC inferred fractions {args.year}")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"huanghai_stsc_fractions_{args.year}.png")
    fig.savefig(out, dpi=150)
    print("saved", out)

    # 农场级均值（像元等权；正式版应面积加权）
    summary = {k: float(np.nanmean(v)) for k, v in fracs.items()}
    print("farm-mean fractions:", summary)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, f"reports/huanghai_stsc_summary_{args.year}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
