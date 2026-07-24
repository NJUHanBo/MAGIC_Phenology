"""
Generate synthetic DPM training data via Latin Hypercube Sampling + Gaussian noise.

Workflow:
  1. LHS sample N parameter vectors from dpm_paras.json ranges
  2. Run DPM forward model to get clean EVI curves
  3. Add Gaussian noise (multiple noise levels per sample) for data augmentation
  4. Save train / valid / test splits as CSV
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dpm.dpm import DPM


def enforce_order(params: dict, ranges: dict):
    """Enforce monotonicity: sos < mat < sen < eos for wheat and summer crop."""
    for crop in ['wheat']:
        keys = [f'{crop}_sos', f'{crop}_mat', f'{crop}_sen', f'{crop}_eos']
        vals = [params[k] for k in keys]
        vals_sorted = np.sort(vals, axis=0)
        for k, v in zip(keys, vals_sorted):
            params[k] = v
        params[f'{crop}_m'] = np.minimum(params[f'{crop}_m'], params[f'{crop}_M'] - 0.05)

    summer_keys = ['summer_sos', 'summer_mat', 'summer_sen', 'summer_eos_base']
    vals = [params[k] for k in summer_keys]
    vals_sorted = np.sort(vals, axis=0)
    for k, v in zip(summer_keys, vals_sorted):
        params[k] = v
    params['summer_m'] = np.minimum(params['summer_m'], params['summer_M'] - 0.05)

    params['delta_eos'] = np.clip(params['delta_eos'], 5.0, 60.0)
    params['wheat_fraction'] = np.clip(params['wheat_fraction'], 0.05, 0.95)
    params['rice_fraction'] = np.clip(params['rice_fraction'], 0.01, 0.99)
    return params


def generate(n_samples: int, paras_path: str, noise_levels: list,
             seed: int = 42, augment_factor: int = 3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(paras_path, 'r') as f:
        ranges = json.load(f)

    param_names = list(ranges.keys())
    n_params = len(param_names)
    mins = np.array([ranges[k]['min'] for k in param_names])
    maxs = np.array([ranges[k]['max'] for k in param_names])

    sampler = LatinHypercube(d=n_params, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    params_raw = mins + lhs_samples * (maxs - mins)

    params_dict = {k: params_raw[:, i] for i, k in enumerate(param_names)}
    params_dict = enforce_order(params_dict, ranges)

    params_array = np.stack([params_dict[k] for k in param_names], axis=1)

    dpm = DPM()
    params_tensor = {k: torch.tensor(params_dict[k], dtype=torch.float32) for k in param_names}
    with torch.no_grad():
        evi_clean = dpm.run(**params_tensor).numpy()

    n_time = evi_clean.shape[1]

    all_evi = [evi_clean]
    all_params = [params_array]

    for _ in range(augment_factor - 1):
        noise_std = np.random.choice(noise_levels)
        noise = np.random.randn(n_samples, n_time) * noise_std
        evi_noisy = np.clip(evi_clean + noise, 0.0, 1.0)
        all_evi.append(evi_noisy)
        all_params.append(params_array)

    all_evi = np.concatenate(all_evi, axis=0)
    all_params = np.concatenate(all_params, axis=0)

    total = all_evi.shape[0]
    idx = np.random.permutation(total)
    n_train = int(total * 0.8)
    n_valid = int(total * 0.1)

    splits = {
        'train': idx[:n_train],
        'valid': idx[n_train:n_train + n_valid],
        'test': idx[n_train + n_valid:]
    }

    evi_cols = [f'EVI_{i+1}' for i in range(n_time)]
    header_evi = ','.join(evi_cols)
    header_params = ','.join(param_names)

    return all_evi, all_params, splits, header_evi, header_params


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic DPM data')
    parser.add_argument('--n_samples', type=int, default=20000)
    parser.add_argument('--augment_factor', type=int, default=3,
                        help='Total copies (1=clean only, 3=clean + 2 noisy)')
    parser.add_argument('--noise_levels', nargs='+', type=float,
                        default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paras_path', type=str,
                        default=os.path.join(PROJECT_ROOT, 'configs', 'dpm_paras.json'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'processed', 'dpm'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.n_samples} LHS samples x {args.augment_factor} augmentation...")
    all_evi, all_params, splits, header_evi, header_params = generate(
        args.n_samples, args.paras_path, args.noise_levels,
        args.seed, args.augment_factor)

    total = all_evi.shape[0]
    print(f"Total samples: {total}")
    print(f"Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")

    for split_name, idx in splits.items():
        evi_path = os.path.join(args.output_dir, f'{split_name}_evi.csv')
        params_path = os.path.join(args.output_dir, f'{split_name}_params.csv')

        np.savetxt(evi_path, all_evi[idx], delimiter=',', header=header_evi, comments='')
        np.savetxt(params_path, all_params[idx], delimiter=',', header=header_params, comments='')
        print(f"  {split_name}: {len(idx)} samples -> {evi_path}")

    x_mean = all_evi[splits['train']].mean(axis=0)
    x_scale = all_evi[splits['train']].std(axis=0)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)

    np.save(os.path.join(args.output_dir, 'x_mean.npy'), x_mean)
    np.save(os.path.join(args.output_dir, 'x_scale.npy'), x_scale)
    print(f"Saved x_mean.npy and x_scale.npy (shape: {x_mean.shape})")

    print("Done.")


if __name__ == '__main__':
    main()
