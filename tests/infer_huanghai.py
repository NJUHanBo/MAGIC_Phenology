"""
Part 2: 黄海农场 2015 真实数据推理
- 加载 model_best.pth，对 huanghai_2015_evi.csv 做推理
- 提取作物面积比例 → 空间分布热力图
- 选样像元展示 EVI 分解
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dpm.dpm import DPM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_PATH = "saved/dpm/models/PILA_DPM_A/0412_154312/model_best.pth"
CONFIG_PATH = "configs/phys_smpl/PILA_DPM_A.json"
HUANGHAI_CSV = "data/processed/dpm/huanghai/huanghai_2015_evi.csv"
X_MEAN_PATH = "data/processed/dpm/x_mean.npy"
X_SCALE_PATH = "data/processed/dpm/x_scale.npy"


def load_model():
    from model.model_phys_smpl import PHYS_VAE_SMPL
    with open(os.path.join(PROJECT_ROOT, CONFIG_PATH)) as f:
        config = json.load(f)
    model = PHYS_VAE_SMPL(config)
    ckpt = torch.load(os.path.join(PROJECT_ROOT, MODEL_PATH), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, config


def load_huanghai_data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, HUANGHAI_CSV))
    rows = df['row'].values
    cols = df['col'].values
    evi_cols = [c for c in df.columns if c.startswith('EVI')]
    evi_raw = df[evi_cols].values.astype(np.float32)
    x_mean = np.load(os.path.join(PROJECT_ROOT, X_MEAN_PATH))
    x_scale = np.load(os.path.join(PROJECT_ROOT, X_SCALE_PATH))
    return evi_raw, rows, cols, x_mean, x_scale


def infer(model, evi_raw, x_mean, x_scale, paras_ranges):
    evi_std = (evi_raw - x_mean) / x_scale
    x_tensor = torch.tensor(evi_std, dtype=torch.float32)

    param_names = list(paras_ranges.keys())
    mins = np.array([paras_ranges[k]['min'] for k in param_names])
    maxs = np.array([paras_ranges[k]['max'] for k in param_names])

    with torch.no_grad():
        z_phy_stat, z_aux_stat = model.encode(x_tensor)
        z_phy = torch.sigmoid(z_phy_stat['mean'])
        params_phys = z_phy.numpy() * (maxs - mins) + mins

        z_aux = z_aux_stat['mean']
        x_recon_std = model.decode(z_phy, z_aux, epoch=200, epochs_pretrain=30, full=False)
        x_recon = x_recon_std.numpy() * x_scale + x_mean

    params_dict = {name: params_phys[:, i] for i, name in enumerate(param_names)}
    return params_dict, x_recon


def compute_crop_fractions(params_dict):
    A1 = params_dict['wheat_fraction']
    A2 = params_dict['rice_mix_maize_fraction']
    A3 = params_dict['maize_in_mix_fraction']

    wheat_frac = A1
    rice_frac = A2 * (1 - A3)
    maize_frac = A2 * A3

    return wheat_frac, rice_frac, maize_frac


def plot_spatial_maps(rows, cols, wheat_frac, rice_frac, maize_frac, save_path):
    row_range = (rows.min(), rows.max())
    col_range = (cols.min(), cols.max())
    n_rows = row_range[1] - row_range[0] + 1
    n_cols = col_range[1] - col_range[0] + 1

    crops = [
        ('Wheat Fraction', wheat_frac),
        ('Rice Fraction', rice_frac),
        ('Maize Fraction', maize_frac),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, frac) in zip(axes, crops):
        grid = np.full((n_rows, n_cols), np.nan)
        for r, c, v in zip(rows, cols, frac):
            grid[r - row_range[0], c - col_range[0]] = v

        im = ax.imshow(grid, cmap='YlGn', vmin=0, vmax=1, aspect='auto',
                       interpolation='nearest')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Huanghai Farm 2015 — Sub-pixel Crop Area Fractions (PILA_DPM_A)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Spatial maps saved: {save_path}")


def plot_evi_decomposition(evi_raw, x_recon, params_dict, save_path, n_samples=4):
    np.random.seed(123)
    idx = np.random.choice(len(evi_raw), n_samples, replace=False)
    doy = np.arange(1, 362, 8)

    dpm = DPM()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    param_names = list(params_dict.keys())

    for ax_i, si in enumerate(idx):
        ax = axes[ax_i]
        single_params = {k: torch.tensor([params_dict[k][si]], dtype=torch.float32)
                         for k in param_names}

        with torch.no_grad():
            wheat_evi = dpm.phenology_model(
                dpm.time_points,
                single_params['wheat_M'], single_params['wheat_m'],
                single_params['wheat_sos'], single_params['wheat_mat'],
                single_params['wheat_sen'], single_params['wheat_eos']
            ).numpy().flatten()

            rice_evi = dpm.phenology_model(
                dpm.time_points,
                single_params['rice_M'], single_params['rice_m'],
                single_params['rice_sos'], single_params['rice_mat'],
                single_params['rice_sen'], single_params['rice_eos']
            ).numpy().flatten()

            maize_evi = dpm.phenology_model(
                dpm.time_points,
                single_params['maize_M'], single_params['maize_m'],
                single_params['maize_sos'], single_params['maize_mat'],
                single_params['maize_sen'], single_params['maize_eos']
            ).numpy().flatten()

        A1 = params_dict['wheat_fraction'][si]
        A2 = params_dict['rice_mix_maize_fraction'][si]
        A3 = params_dict['maize_in_mix_fraction'][si]

        ax.plot(doy, evi_raw[si], 'ko-', ms=4, lw=1, label='Observed EVI', zorder=5)
        ax.plot(doy, x_recon[si], 'r--', lw=1.5, label='Reconstructed EVI', zorder=4)

        ax.fill_between(doy, 0, A1 * wheat_evi, alpha=0.3, color='gold',
                         label=f'Wheat ({A1:.0%})')
        ax.fill_between(doy, 0, A2*(1-A3) * rice_evi, alpha=0.3, color='green',
                         label=f'Rice ({A2*(1-A3):.0%})')
        ax.fill_between(doy, 0, A2*A3 * maize_evi, alpha=0.3, color='orange',
                         label=f'Maize ({A2*A3:.0%})')

        ax.set_xlabel('DOY', fontsize=11)
        ax.set_ylabel('EVI', fontsize=11)
        ax.set_title(f'Pixel #{si}', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(-0.05, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Huanghai Farm 2015 — EVI Decomposition by Crop', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  EVI decomposition saved: {save_path}")


def main():
    print("=" * 60)
    print("Part 2: Huanghai Farm 2015 Real Data Inference")
    print("=" * 60)

    print("\n[1] Loading model and data...")
    model, config = load_model()
    evi_raw, rows, cols, x_mean, x_scale = load_huanghai_data()
    paras_ranges = json.load(open(os.path.join(PROJECT_ROOT, "configs/dpm_paras.json")))
    print(f"  Pixels: {len(evi_raw)}")

    print("\n[2] Running inference...")
    params_dict, x_recon = infer(model, evi_raw, x_mean, x_scale, paras_ranges)

    print("\n[3] Computing crop fractions...")
    wheat_frac, rice_frac, maize_frac = compute_crop_fractions(params_dict)

    print(f"\n  Farm-wide statistics:")
    print(f"    Wheat fraction:  mean={wheat_frac.mean():.3f}, std={wheat_frac.std():.3f}")
    print(f"    Rice fraction:   mean={rice_frac.mean():.3f}, std={rice_frac.std():.3f}")
    print(f"    Maize fraction:  mean={maize_frac.mean():.3f}, std={maize_frac.std():.3f}")
    print(f"    Sum (wheat+rice+maize avg): {wheat_frac.mean()+rice_frac.mean()+maize_frac.mean():.3f}")

    print("\n[4] Generating spatial maps...")
    plot_spatial_maps(rows, cols, wheat_frac, rice_frac, maize_frac,
                      os.path.join(FIG_DIR, "huanghai_2015_crop_fractions.png"))

    print("\n[5] Generating EVI decomposition plots...")
    plot_evi_decomposition(evi_raw, x_recon, params_dict,
                           os.path.join(FIG_DIR, "huanghai_2015_evi_decomposition.png"))

    rec_mse = np.mean((evi_raw - x_recon) ** 2)
    rec_mae = np.mean(np.abs(evi_raw - x_recon))
    print(f"\n[6] Reconstruction quality on real data:")
    print(f"  MSE: {rec_mse:.6f}")
    print(f"  MAE: {rec_mae:.6f}")

    print("\n" + "=" * 60)
    print("Part 2 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
