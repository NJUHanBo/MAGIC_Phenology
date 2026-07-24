"""
从黄海农场纯像元（或 demo 合成）提取端元，写出 endmembers.json。

作物：wheat, rice, maize, soy + soil + water
每个端元：
  - peak_spectrum: 7 维（生长峰值附近时间窗均值）
  - surface: 7 x 46 参考曲面

用法：
  # demo（无真实标签时）
  python scripts/extract_endmembers.py --demo

  # 真实：提供像元索引列表 JSON
  python scripts/extract_endmembers.py \
    --reflectance data/processed/stsc/2015_reflectance.npy \
    --pure-pixels data/processed/stsc/pure_pixel_ids.json
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAND_NAMES = ["Red", "NIR", "Blue", "Green", "SWIR1230", "SWIR1628", "SWIR2105"]
CLASSES = ["wheat", "rice", "maize", "soy", "soil", "water"]


def default_demo_endmembers(rng=None):
    """物理合理的默认端元（用于打通 pipeline）。"""
    rng = rng or np.random.default_rng(0)
    # 典型植被：NIR 高、Red 低；土壤：红边附近较平；水体：全波段低、NIR 更低
    templates = {
        "wheat": np.array([0.05, 0.45, 0.03, 0.08, 0.25, 0.18, 0.10]),
        "rice":  np.array([0.04, 0.42, 0.03, 0.07, 0.22, 0.15, 0.09]),
        "maize": np.array([0.06, 0.48, 0.04, 0.09, 0.28, 0.20, 0.12]),
        "soy":   np.array([0.05, 0.40, 0.03, 0.08, 0.24, 0.17, 0.11]),
        "soil":  np.array([0.18, 0.22, 0.10, 0.15, 0.30, 0.35, 0.28]),
        "water": np.array([0.04, 0.02, 0.05, 0.04, 0.01, 0.01, 0.01]),
    }
    out = {}
    t = np.arange(46)
    # 生长曲线近似：夏峰
    g_summer = 1 / (1 + np.exp(-(t - 20) / 2)) * (1 / (1 + np.exp((t - 34) / 2)))
    g_winter = 1 / (1 + np.exp(-(t - 8) / 2)) * (1 / (1 + np.exp((t - 18) / 2)))
    for name, peak in templates.items():
        peak = peak + 0.01 * rng.standard_normal(7)
        peak = np.clip(peak, 0.01, 0.8)
        soil = templates["soil"]
        if name == "soil":
            surf = np.tile(peak[:, None], (1, 46))
        elif name == "water":
            surf = np.tile(peak[:, None], (1, 46))
        elif name == "wheat":
            g = g_winter
            surf = soil[:, None] + g[None, :] * (peak - soil)[:, None]
        else:
            g = g_summer
            surf = soil[:, None] + g[None, :] * (peak - soil)[:, None]
            if name == "rice":
                # 淹水：早期 NIR 压低
                flood = 1 / (1 + np.exp(-(t - 16) / 1.5)) * (1 / (1 + np.exp((t - 22) / 1.5)))
                water = templates["water"]
                surf = (1 - flood)[None, :] * surf + flood[None, :] * (
                    water[:, None] + g[None, :] * (peak - water)[:, None]
                )
        out[name] = {
            "peak_spectrum": peak.tolist(),
            "surface": surf.astype(float).tolist(),
            "band_names": BAND_NAMES,
        }
    return out


def extract_from_pure(reflectance: np.ndarray, pure: dict) -> dict:
    """
    reflectance: (n, 7, 46)
    pure: {class_name: [pixel_id, ...]}
    """
    out = {}
    for name in CLASSES:
        ids = pure.get(name, [])
        if not ids:
            raise ValueError(f"pure_pixel_ids missing class: {name}")
        subset = reflectance[np.array(ids)]
        surf = subset.mean(axis=0)  # (7, 46)
        # 峰值时间：NIR 最大
        peak_t = int(np.argmax(surf[1]))
        win = slice(max(0, peak_t - 1), min(46, peak_t + 2))
        peak = surf[:, win].mean(axis=1)
        out[name] = {
            "peak_spectrum": peak.tolist(),
            "surface": surf.astype(float).tolist(),
            "band_names": BAND_NAMES,
            "n_pixels": len(ids),
            "peak_t_index": peak_t,
        }
    return out


def plot_endmembers(endmembers: dict, out_png: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(BAND_NAMES))
    for name, payload in endmembers.items():
        ax.plot(x, payload["peak_spectrum"], marker="o", label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, rotation=30)
    ax.set_ylabel("Reflectance")
    ax.set_title("Endmember peak spectra")
    ax.legend()
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("saved", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--reflectance", type=str, default=None)
    parser.add_argument("--pure-pixels", type=str, default=None)
    parser.add_argument("--out", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/processed/stsc/endmembers.json"))
    parser.add_argument("--fig", type=str,
                        default=os.path.join(PROJECT_ROOT, "figures/endmember_spectra.png"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)

    if args.demo or not args.reflectance:
        endmembers = default_demo_endmembers()
        print("Using demo endmembers (physically plausible defaults)")
    else:
        refl = np.load(args.reflectance)
        with open(args.pure_pixels) as f:
            pure = json.load(f)
        endmembers = extract_from_pure(refl, pure)

    with open(args.out, "w") as f:
        json.dump(endmembers, f, indent=2)
    print("wrote", args.out)
    plot_endmembers(endmembers, args.fig)

    # 粗检：植被 NIR > Red
    for crop in ["wheat", "rice", "maize", "soy"]:
        s = endmembers[crop]["peak_spectrum"]
        assert s[1] > s[0], f"{crop}: NIR should exceed Red"
    print("OK: endmember physical sanity checks passed")


if __name__ == "__main__":
    main()
