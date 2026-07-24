"""
生成多波段 STSC 合成训练集（LHS + 噪声增强）。

输出：
  data/processed/stsc/train_stsc.npy  (n, 7, 46)
  data/processed/stsc/train_params.csv
  valid/test 同理
  x_mean.npy / x_scale.npy  — 展平后 322 维标准化统计
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dpm.dpm_stsc import DPM_STSC


ORDER_GROUPS = [
    ["wheat_sos", "wheat_mat", "wheat_sen", "wheat_eos"],
    ["summer_sos", "summer_mat"],
    ["rice_sen", "rice_eos"],
    ["maize_sen", "maize_eos"],
    ["soy_sen", "soy_eos"],
]


def enforce_order(params: dict):
    for keys in ORDER_GROUPS:
        if len(keys) < 2:
            continue
        vals = np.stack([params[k] for k in keys], axis=0)
        vals_sorted = np.sort(vals, axis=0)
        for i, k in enumerate(keys):
            params[k] = vals_sorted[i]
    # 面积和 <= 1
    fracs = np.stack([params["wheat_frac"], params["rice_frac"],
                      params["maize_frac"], params["soy_frac"]], axis=0)
    total = fracs.sum(axis=0)
    scale = np.maximum(total, 1.0)
    params["wheat_frac"] = fracs[0] / scale
    params["rice_frac"] = fracs[1] / scale
    params["maize_frac"] = fracs[2] / scale
    params["soy_frac"] = fracs[3] / scale
    return params


def generate(n_samples, paras_path, endmembers_path, noise_levels, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(paras_path) as f:
        ranges = json.load(f)
    names = list(ranges.keys())
    mins = np.array([ranges[k]["min"] for k in names])
    maxs = np.array([ranges[k]["max"] for k in names])

    sampler = LatinHypercube(d=len(names), seed=seed)
    raw = mins + sampler.random(n=n_samples) * (maxs - mins)
    params = {k: raw[:, i] for i, k in enumerate(names)}
    params = enforce_order(params)
    params_arr = np.stack([params[k] for k in names], axis=1)

    model = DPM_STSC(endmembers_path=endmembers_path if os.path.exists(endmembers_path) else None,
                     learnable_endmembers=False)
    tensors = {k: torch.tensor(params[k], dtype=torch.float32) for k in names}
    with torch.no_grad():
        clean = model.run(**tensors).numpy()  # (N, 7, 46)

    all_x = [clean]
    all_p = [params_arr]
    for sigma in noise_levels:
        noisy = np.clip(clean + np.random.randn(*clean.shape) * sigma, 0.0, 1.0)
        all_x.append(noisy)
        all_p.append(params_arr)

    X = np.concatenate(all_x, axis=0).astype(np.float32)
    P = np.concatenate(all_p, axis=0).astype(np.float32)
    idx = np.random.permutation(len(X))
    n_train = int(len(X) * 0.8)
    n_valid = int(len(X) * 0.1)
    splits = {
        "train": idx[:n_train],
        "valid": idx[n_train:n_train + n_valid],
        "test": idx[n_train + n_valid:],
    }
    return X, P, splits, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--noise_levels", nargs="+", type=float,
                        default=[0.005, 0.01, 0.02, 0.03])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paras_path", default=os.path.join(PROJECT_ROOT, "configs/dpm_stsc_paras.json"))
    parser.add_argument("--endmembers", default=os.path.join(PROJECT_ROOT, "data/processed/stsc/endmembers.json"))
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "data/processed/stsc"))
    parser.add_argument("--tiny", action="store_true", help="快速小样本（打通流水线）")
    args = parser.parse_args()

    if args.tiny:
        args.n_samples = 500
        args.noise_levels = [0.01, 0.02]

    if not os.path.exists(args.endmembers):
        print("endmembers.json missing — creating demo endmembers")
        from scripts.extract_endmembers import default_demo_endmembers
        os.makedirs(os.path.dirname(args.endmembers), exist_ok=True)
        with open(args.endmembers, "w") as f:
            json.dump(default_demo_endmembers(), f, indent=2)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.n_samples} x {1+len(args.noise_levels)} samples...")
    X, P, splits, names = generate(
        args.n_samples, args.paras_path, args.endmembers, args.noise_levels, args.seed
    )
    print("Total:", X.shape)

    for split, idx in splits.items():
        np.save(os.path.join(args.output_dir, f"{split}_stsc.npy"), X[idx])
        header = ",".join(names)
        np.savetxt(
            os.path.join(args.output_dir, f"{split}_params.csv"),
            P[idx], delimiter=",", header=header, comments="",
        )
        print(f"  {split}: {len(idx)}")

    # 标准化统计：展平 (7*46,)
    flat = X[splits["train"]].reshape(len(splits["train"]), -1)
    x_mean = flat.mean(axis=0)
    x_scale = flat.std(axis=0)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    np.save(os.path.join(args.output_dir, "x_mean.npy"), x_mean.astype(np.float32))
    np.save(os.path.join(args.output_dir, "x_scale.npy"), x_scale.astype(np.float32))
    print("Saved x_mean/x_scale shape", x_mean.shape)
    print("Done.")


if __name__ == "__main__":
    main()
