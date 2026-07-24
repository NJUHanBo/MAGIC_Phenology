"""
STSC 可视化：单作物曲面、混合像元、与 endmembers 参考曲面对比。
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dpm.dpm_stsc import DPM_STSC, BAND_NAMES

FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_paras(**overrides):
    base = {
        "wheat_sos": 40.0, "wheat_mat": 110.0, "wheat_sen": 140.0, "wheat_eos": 160.0,
        "summer_sos": 160.0, "summer_mat": 220.0,
        "rice_sen": 250.0, "rice_eos": 290.0,
        "maize_sen": 240.0, "maize_eos": 270.0,
        "soy_sen": 245.0, "soy_eos": 275.0,
        "wheat_frac": 0.0, "rice_frac": 0.0, "maize_frac": 0.0, "soy_frac": 0.0,
        "flood_onset": 170.0, "flood_intensity": 0.8,
    }
    base.update(overrides)
    return {k: torch.tensor([v], dtype=torch.float32) for k, v in base.items()}


def plot_surface(rho_7x46, title, path):
    """rho: (7, 46) numpy"""
    doy = np.arange(1, 362, 8)
    bands = np.arange(7)
    T, B = np.meshgrid(doy, bands)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, B, rho_7x46, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("DOY")
    ax.set_ylabel("Band index")
    ax.set_zlabel("Reflectance")
    ax.set_yticks(range(7))
    ax.set_yticklabels(BAND_NAMES, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    em_path = os.path.join(PROJECT_ROOT, "data/processed/stsc/endmembers.json")
    if not os.path.exists(em_path):
        # 生成 demo 端元
        from scripts.extract_endmembers import default_demo_endmembers
        os.makedirs(os.path.dirname(em_path), exist_ok=True)
        with open(em_path, "w") as f:
            json.dump(default_demo_endmembers(), f, indent=2)

    m = DPM_STSC(endmembers_path=em_path, learnable_endmembers=False)

    # 纯作物
    for crop, kw in [
        ("wheat", {"wheat_frac": 1.0}),
        ("rice", {"rice_frac": 1.0}),
        ("maize", {"maize_frac": 1.0}),
        ("soy", {"soy_frac": 1.0}),
    ]:
        out = m.run(**make_paras(**kw))[0].detach().numpy()
        plot_surface(out, f"Synthetic pure {crop} STSC", os.path.join(FIG_DIR, f"stsc_pure_{crop}.png"))

    # 混合：50% 小麦 + 30% 水稻 + 20% 背景
    mixed = m.run(**make_paras(wheat_frac=0.5, rice_frac=0.3))[0].detach().numpy()
    plot_surface(mixed, "Mixed 50% wheat + 30% rice + 20% bg", os.path.join(FIG_DIR, "stsc_mixed_example.png"))

    # 与 endmember 参考曲面对比（NIR 时间序列）
    with open(em_path) as f:
        em = json.load(f)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    doy = np.arange(1, 362, 8)
    for ax, crop in zip(axes.ravel(), ["wheat", "rice", "maize", "soy"]):
        ref = np.array(em[crop]["surface"])[1]
        syn = m.run(**make_paras(**{f"{crop}_frac": 1.0}))[0, 1].detach().numpy()
        ax.plot(doy, ref, "k--", label="endmember ref")
        ax.plot(doy, syn, "r-", label="DPM_STSC")
        ax.set_title(crop)
        ax.set_ylabel("NIR")
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("DOY")
    axes[-1, 1].set_xlabel("DOY")
    fig.suptitle("NIR: synthetic vs endmember reference surface")
    fig.tight_layout()
    outp = os.path.join(FIG_DIR, "stsc_vs_endmember_nir.png")
    fig.savefig(outp, dpi=150)
    print("saved", outp)
    print("OK: visualize_stsc done — check figures/stsc_*.png")


if __name__ == "__main__":
    main()
