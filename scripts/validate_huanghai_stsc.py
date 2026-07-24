"""
黄海农场验证：面积对比、2022 突变、STSC 分解可视化。

农场统计面积 CSV 模板：data/processed/stsc/huanghai_crop_area_ha.csv
列：year,wheat_ha,rice_ha,maize_ha,soy_ha,total_ha

用法：
  python scripts/validate_huanghai_stsc.py --years 2005 2010 2015 2020 2022
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIG = os.path.join(PROJECT_ROOT, "figures")
REP = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(FIG, exist_ok=True)
os.makedirs(REP, exist_ok=True)


def ensure_area_template():
    path = os.path.join(PROJECT_ROOT, "data/processed/stsc/huanghai_crop_area_ha.csv")
    if os.path.exists(path):
        return path
    # 占位：含 2022 突变示意数字（来自计划描述），正式验证时替换为农场真值
    rows = [
        {"year": 2005, "wheat_ha": 12000, "rice_ha": 10000, "maize_ha": 50, "soy_ha": 20, "total_ha": 22070},
        {"year": 2010, "wheat_ha": 11800, "rice_ha": 10200, "maize_ha": 40, "soy_ha": 30, "total_ha": 22070},
        {"year": 2015, "wheat_ha": 11500, "rice_ha": 10266, "maize_ha": 30, "soy_ha": 40, "total_ha": 21836},
        {"year": 2020, "wheat_ha": 11400, "rice_ha": 10200, "maize_ha": 80, "soy_ha": 60, "total_ha": 21740},
        {"year": 2022, "wheat_ha": 11200, "rice_ha": 7942, "maize_ha": 1000, "soy_ha": 1016, "total_ha": 21158},
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote template area CSV (replace with official farm stats):", path)
    return path


def farm_mean_from_inference(year, model, cfg):
    """调用与 infer 相同逻辑，返回分数均值。"""
    refl = os.path.join(PROJECT_ROOT, f"data/processed/stsc/{year}_reflectance.npy")
    if not os.path.exists(refl):
        # demo
        rng = np.random.default_rng(year)
        arr = rng.uniform(0.05, 0.4, size=(400, 7, 46)).astype(np.float32)
    else:
        arr = np.load(refl)

    mean_path = os.path.join(PROJECT_ROOT, "data/processed/stsc/x_mean.npy")
    scale_path = os.path.join(PROJECT_ROOT, "data/processed/stsc/x_scale.npy")
    if not os.path.exists(mean_path):
        return None
    x_mean = np.load(mean_path)
    x_scale = np.load(scale_path)
    flat = arr.reshape(len(arr), -1)
    x_std = torch.tensor((flat - x_mean) / x_scale, dtype=torch.float32)

    fracs = {k: [] for k in ["wheat_frac", "rice_frac", "maize_frac", "soy_frac"]}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_std), 256):
            z_phy_stat, _ = model.encode(x_std[i:i + 256])
            z_phy = torch.sigmoid(z_phy_stat["mean"])
            pred = model.physics_model.rescale(z_phy)
            for k in fracs:
                fracs[k].append(pred[k].cpu().numpy())
    return {k: float(np.concatenate(v).mean()) for k, v in fracs.items()}


def plot_decomposition(model):
    from dpm.dpm_stsc import DPM_STSC
    em = os.path.join(PROJECT_ROOT, "data/processed/stsc/endmembers.json")
    fwd = DPM_STSC(endmembers_path=em if os.path.exists(em) else None, learnable_endmembers=False)
    paras = {
        "wheat_sos": torch.tensor([45.0]), "wheat_mat": torch.tensor([110.0]),
        "wheat_sen": torch.tensor([140.0]), "wheat_eos": torch.tensor([160.0]),
        "summer_sos": torch.tensor([165.0]), "summer_mat": torch.tensor([220.0]),
        "rice_sen": torch.tensor([250.0]), "rice_eos": torch.tensor([290.0]),
        "maize_sen": torch.tensor([240.0]), "maize_eos": torch.tensor([270.0]),
        "soy_sen": torch.tensor([245.0]), "soy_eos": torch.tensor([275.0]),
        "wheat_frac": torch.tensor([0.4]), "rice_frac": torch.tensor([0.3]),
        "maize_frac": torch.tensor([0.1]), "soy_frac": torch.tensor([0.05]),
        "flood_onset": torch.tensor([170.0]), "flood_intensity": torch.tensor([0.8]),
    }
    comps = fwd.forward_components(**paras)
    mixed = fwd.run(**paras)[0, 1].detach().numpy()
    doy = np.arange(1, 362, 8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(doy, mixed, "k-", lw=2, label="mixed NIR")
    for name in ["wheat", "rice", "maize", "soy"]:
        ax.plot(doy, comps[name][0, 1].detach().numpy(), "--", label=name)
    ax.legend()
    ax.set_xlabel("DOY")
    ax.set_ylabel("NIR reflectance")
    ax.set_title("STSC decomposition example (NIR)")
    out = os.path.join(FIG, "stsc_decomposition_example.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2005, 2010, 2015, 2020, 2022])
    args = parser.parse_args()

    area_path = ensure_area_template()
    area = pd.read_csv(area_path).set_index("year")

    # 确保有标准化统计
    if not os.path.exists(os.path.join(PROJECT_ROOT, "data/processed/stsc/x_mean.npy")):
        print("Generating tiny STSC synthetic for stats/smoke...")
        import subprocess
        subprocess.check_call(
            [sys.executable, os.path.join(PROJECT_ROOT, "scripts/generate_synthetic_stsc.py"), "--tiny"],
            cwd=PROJECT_ROOT,
        )

    from model.model_phys_smpl import PHYS_VAE_SMPL
    with open(os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_STSC_A.json")) as f:
        cfg = json.load(f)
    model = PHYS_VAE_SMPL(cfg)

    records = []
    for y in args.years:
        pred = farm_mean_from_inference(y, model, cfg)
        if pred is None:
            continue
        if y in area.index:
            tot = float(area.loc[y, "total_ha"])
            true_frac = {
                "wheat_frac": area.loc[y, "wheat_ha"] / tot,
                "rice_frac": area.loc[y, "rice_ha"] / tot,
                "maize_frac": area.loc[y, "maize_ha"] / tot,
                "soy_frac": area.loc[y, "soy_ha"] / tot,
            }
        else:
            true_frac = {k: np.nan for k in pred}
        rec = {"year": y, **{f"pred_{k}": v for k, v in pred.items()},
               **{f"true_{k}": true_frac[k] for k in true_frac}}
        records.append(rec)
        print(y, "pred", pred)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(REP, "huanghai_stsc_area_validation.csv"), index=False)

    # 突变检测图：水稻 vs 玉米+大豆
    if len(df):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["year"], df["pred_rice_frac"], "b-o", label="pred rice")
        ax.plot(df["year"], df["pred_maize_frac"] + df["pred_soy_frac"], "g-o", label="pred maize+soy")
        if "true_rice_frac" in df:
            ax.plot(df["year"], df["true_rice_frac"], "b--", label="true rice")
            ax.plot(df["year"], df["true_maize_frac"] + df["true_soy_frac"], "g--", label="true maize+soy")
        ax.axvline(2022, color="r", ls=":", label="2022 shift")
        ax.legend()
        ax.set_ylabel("fraction")
        ax.set_title("Structure shift check (rice vs maize+soy)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "huanghai_stsc_structure_shift.png"), dpi=150)
        print("saved structure shift figure")

    plot_decomposition(model)
    print("OK: validate_huanghai_stsc finished")
    print("NOTE: replace huanghai_crop_area_ha.csv with official stats before paper numbers.")


if __name__ == "__main__":
    main()
