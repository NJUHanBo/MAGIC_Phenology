"""
Part 1: 合成 test set 参数反演验证
- 加载 model_best.pth，对 test_evi 推理
- 与 test_params 真值逐参数比较
- 输出 R²/RMSE 汇总表、散点图、重建 EVI 曲线
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_PATH = "saved/dpm/models/PILA_DPM_A/0409_131847/model_best.pth"
CONFIG_PATH = "configs/phys_smpl/PILA_DPM_A.json"
TEST_EVI = "data/processed/dpm/test_evi.csv"
TEST_PARAMS = "data/processed/dpm/test_params.csv"
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


def load_test_data():
    import pandas as pd
    evi_df = pd.read_csv(os.path.join(PROJECT_ROOT, TEST_EVI))
    params_df = pd.read_csv(os.path.join(PROJECT_ROOT, TEST_PARAMS))
    x_mean = np.load(os.path.join(PROJECT_ROOT, X_MEAN_PATH))
    x_scale = np.load(os.path.join(PROJECT_ROOT, X_SCALE_PATH))
    return evi_df.values.astype(np.float32), params_df, x_mean, x_scale


def infer_parameters(model, evi_raw, x_mean, x_scale, paras_ranges):
    evi_std = (evi_raw - x_mean) / x_scale
    x_tensor = torch.tensor(evi_std, dtype=torch.float32)

    param_names = list(paras_ranges.keys())
    mins = np.array([paras_ranges[k]['min'] for k in param_names])
    maxs = np.array([paras_ranges[k]['max'] for k in param_names])

    all_params = []
    all_recon = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i:i+batch_size]
            z_phy_stat, z_aux_stat = model.encode(batch)
            z_phy = torch.sigmoid(z_phy_stat['mean'])
            params_01 = z_phy.numpy()
            params_phys = params_01 * (maxs - mins) + mins
            all_params.append(params_phys)

            z_aux = z_aux_stat['mean']
            x_recon = model.decode(z_phy, z_aux, epoch=200, epochs_pretrain=30, full=False)
            all_recon.append(x_recon.numpy())

    inferred = np.concatenate(all_params, axis=0)
    recon_std = np.concatenate(all_recon, axis=0)
    recon_raw = recon_std * x_scale + x_mean
    return inferred, recon_raw


def plot_scatter(true_vals, pred_vals, param_names, save_path):
    focus_params = [
        'wheat_fraction', 'rice_mix_maize_fraction', 'maize_in_mix_fraction',
        'rice_flood_depth', 'rice_flood_onset',
        'wheat_sos', 'rice_sos', 'rice_eos', 'maize_sos'
    ]
    focus_idx = [param_names.index(p) for p in focus_params]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, (ax, idx) in enumerate(zip(axes, focus_idx)):
        t = true_vals[:, idx]
        p = pred_vals[:, idx]
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))

        ax.scatter(t, p, alpha=0.15, s=8, c='steelblue', edgecolors='none')
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='1:1 line')
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel('True', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{focus_params[i]}\nR²={r2:.4f}  RMSE={rmse:.4f}', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scatter plot saved: {save_path}")


def plot_recon_curves(evi_raw, recon_raw, save_path, n_samples=4):
    np.random.seed(42)
    idx = np.random.choice(len(evi_raw), n_samples, replace=False)
    doy = np.arange(1, 362, 8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, (ax, si) in enumerate(zip(axes, idx)):
        ax.plot(doy, evi_raw[si], 'ko-', ms=4, lw=1, label='Original EVI')
        ax.plot(doy, recon_raw[si], 'r.-', ms=4, lw=1.5, label='Reconstructed EVI')
        ax.set_xlabel('DOY', fontsize=11)
        ax.set_ylabel('EVI', fontsize=11)
        ax.set_title(f'Sample #{si}', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle('EVI Reconstruction: Original vs Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Reconstruction curves saved: {save_path}")


def main():
    print("=" * 60)
    print("Part 1: Synthetic Test Set Verification")
    print("=" * 60)

    print("\n[1] Loading model and data...")
    model, config = load_model()
    evi_raw, params_df, x_mean, x_scale = load_test_data()
    paras_ranges = json.load(open(os.path.join(PROJECT_ROOT, "configs/dpm_paras.json")))
    param_names = list(paras_ranges.keys())
    true_params = params_df.values.astype(np.float32)
    print(f"  Test samples: {len(evi_raw)}")
    print(f"  Parameters: {len(param_names)}")

    print("\n[2] Running inference...")
    pred_params, recon_raw = infer_parameters(model, evi_raw, x_mean, x_scale, paras_ranges)

    print("\n[3] Parameter-wise R² and RMSE:")
    print(f"  {'Parameter':30s} {'R²':>8s} {'RMSE':>10s}")
    print(f"  {'-'*50}")
    for i, name in enumerate(param_names):
        r2 = r2_score(true_params[:, i], pred_params[:, i])
        rmse = np.sqrt(mean_squared_error(true_params[:, i], pred_params[:, i]))
        marker = " ***" if name in ['wheat_fraction', 'rice_mix_maize_fraction', 'maize_in_mix_fraction'] else ""
        print(f"  {name:30s} {r2:8.4f} {rmse:10.4f}{marker}")

    print("\n[4] Generating scatter plots...")
    plot_scatter(true_params, pred_params, param_names,
                 os.path.join(FIG_DIR, "synth_param_scatter.png"))

    print("\n[5] Generating reconstruction curves...")
    plot_recon_curves(evi_raw, recon_raw,
                      os.path.join(FIG_DIR, "synth_recon_curves.png"))

    rec_mse = np.mean((evi_raw - recon_raw) ** 2)
    rec_mae = np.mean(np.abs(evi_raw - recon_raw))
    print(f"\n[6] Overall reconstruction quality:")
    print(f"  MSE: {rec_mse:.6f}")
    print(f"  MAE: {rec_mae:.6f}")
    print(f"  Per-timestep avg error: {rec_mae:.4f} EVI units")

    print("\n" + "=" * 60)
    print("Part 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
