"""
在同一套合成参数上对比 STSC 版 vs EVI 版的参数恢复精度。

用法（先有测试集与可选 checkpoint）：
  python scripts/compare_stsc_vs_evi.py --tiny

输出 figures/compare_stsc_vs_evi_*.png 与 reports/compare_stsc_vs_evi.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


FRAC_KEYS_STSC = ["wheat_frac", "rice_frac", "maize_frac", "soy_frac"]


def recover_params_from_encoder(model, x_std, physics):
    """Encode standardized x → rescaled physical params dict of tensors."""
    model.eval()
    with torch.no_grad():
        z_phy_stat, _z_aux = model.encode(x_std)
        z_phy = torch.sigmoid(z_phy_stat["mean"])
        return physics.rescale(z_phy)


def load_pila(config_path, ckpt_path=None):
    from model.model_phys_smpl import PHYS_VAE_SMPL
    with open(config_path) as f:
        cfg = json.load(f)
    # ConfigParser-like: PHYS_VAE_SMPL expects nested config with trainer etc.
    # Wrap minimal structure
    class C(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)
    model = PHYS_VAE_SMPL(cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        key = "state_dict" if "state_dict" in state else None
        model.load_state_dict(state[key] if key else state, strict=False)
        print("loaded", ckpt_path)
    else:
        print("WARNING: no checkpoint — using randomly initialized encoder (diagnostic only)")
    return model, cfg


def eval_stsc(tiny=False):
    stsc_dir = os.path.join(PROJECT_ROOT, "data/processed/stsc")
    test_x = os.path.join(stsc_dir, "test_stsc.npy")
    test_p = os.path.join(stsc_dir, "test_params.csv")
    if not os.path.exists(test_x):
        print("Generating tiny STSC set...")
        from scripts.generate_synthetic_stsc import main as gen
        sys.argv = ["generate_synthetic_stsc.py", "--tiny"]
        gen()

    X = np.load(test_x)
    P = pd.read_csv(test_p)
    if tiny:
        X, P = X[:200], P.iloc[:200]

    x_mean = np.load(os.path.join(stsc_dir, "x_mean.npy"))
    x_scale = np.load(os.path.join(stsc_dir, "x_scale.npy"))
    flat = X.reshape(len(X), -1)
    x_std = torch.tensor((flat - x_mean) / x_scale, dtype=torch.float32)

    cfg_path = os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_STSC_A.json")
    # find latest ckpt if any
    ckpt = None
    saved = os.path.join(PROJECT_ROOT, "saved/stsc/models/PILA_STSC_A")
    if os.path.isdir(saved):
        runs = sorted(os.listdir(saved))
        if runs:
            cand = os.path.join(saved, runs[-1], "model_best.pth")
            if os.path.exists(cand):
                ckpt = cand

    model, cfg = load_pila(cfg_path, ckpt)
    physics = model.physics_model
    pred = recover_params_from_encoder(model, x_std, physics)

    rows = []
    for k in FRAC_KEYS_STSC:
        if k not in P.columns:
            continue
        yt = P[k].values
        yp = pred[k].cpu().numpy()
        r2 = r2_score(yt, yp) if np.std(yt) > 1e-8 else float("nan")
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rows.append({"model": "STSC", "param": k, "R2": r2, "RMSE": rmse})
        print(f"STSC {k}: R2={r2:.3f} RMSE={rmse:.4f}")
    return rows, P, pred


def eval_evi_baseline_note():
    """EVI 版对比：若有 test_params 与已训模型则评估 rice/wheat fraction。"""
    rows = []
    evi_params = os.path.join(PROJECT_ROOT, "data/processed/dpm/test_params.csv")
    ckpt = os.path.join(PROJECT_ROOT, "saved/dpm/models/PILA_DPM_A/0412_154312/model_best.pth")
    cfg_path = os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_DPM_A.json")
    if not (os.path.exists(evi_params) and os.path.getsize(evi_params) > 0 and os.path.exists(ckpt)):
        print("EVI baseline skip: missing test_params or checkpoint")
        return rows

    from model.model_phys_smpl import PHYS_VAE_SMPL
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = PHYS_VAE_SMPL(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)

    evi = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/dpm/test_evi.csv"))
    P = pd.read_csv(evi_params)
    n = min(500, len(evi))
    x_mean = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_mean.npy"))
    x_scale = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_scale.npy"))
    x = evi.iloc[:n].values.astype(np.float32)
    x_std = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    pred = recover_params_from_encoder(model, x_std, model.physics_model)
    for k in ["wheat_fraction", "rice_fraction"]:
        yt = P[k].values[:n]
        yp = pred[k].cpu().numpy()
        r2 = r2_score(yt, yp)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rows.append({"model": "EVI", "param": k, "R2": r2, "RMSE": rmse})
        print(f"EVI {k}: R2={r2:.3f} RMSE={rmse:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "reports")
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    rows_stsc, P, pred = eval_stsc(tiny=args.tiny)
    rows_evi = eval_evi_baseline_note()
    df = pd.DataFrame(rows_stsc + rows_evi)
    out_csv = os.path.join(out_dir, "compare_stsc_vs_evi.csv")
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv)

    # 面积参数散点（STSC）
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, k in zip(axes.ravel(), FRAC_KEYS_STSC):
        if k not in P.columns:
            continue
        yt = P[k].values
        yp = pred[k].cpu().numpy()
        ax.scatter(yt, yp, s=8, alpha=0.4)
        lim = [0, 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(k)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    fig.suptitle("STSC parameter recovery (fractions)")
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, "compare_stsc_fraction_scatter.png")
    fig.savefig(fig_path, dpi=150)
    print("saved", fig_path)
    print("OK: compare_stsc_vs_evi done")


if __name__ == "__main__":
    main()
