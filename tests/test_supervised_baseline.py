"""
Supervised baseline: directly train MLP to predict DPM params from WEVI.
This gives the upper bound of what's achievable — if supervised learning
can't recover params, the forward model lacks identifiability.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    import pandas as pd
    x_mean = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_mean.npy"))
    x_scale = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_scale.npy"))

    train_evi = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/dpm/train_evi.csv")).values
    train_params = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/dpm/train_params.csv")).values
    test_evi = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/dpm/test_evi.csv")).values
    test_params = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/dpm/test_params.csv")).values

    paras = json.load(open(os.path.join(PROJECT_ROOT, "configs/dpm_paras.json")))
    param_names = list(paras.keys())
    mins = np.array([paras[k]['min'] for k in param_names])
    maxs = np.array([paras[k]['max'] for k in param_names])

    train_x = ((train_evi - x_mean) / x_scale).astype(np.float32)
    test_x = ((test_evi - x_mean) / x_scale).astype(np.float32)
    train_y = ((train_params - mins) / (maxs - mins)).astype(np.float32)
    test_y = ((test_params - mins) / (maxs - mins)).astype(np.float32)

    return train_x, train_y, test_x, test_y, param_names, mins, maxs


class InverseNet(nn.Module):
    def __init__(self, in_dim=46, out_dim=23, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def main():
    train_x, train_y, test_x, test_y, param_names, mins, maxs = load_data()
    print(f"Train: {train_x.shape}, Test: {test_x.shape}, Params: {len(param_names)}")

    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

    model = InverseNet(in_dim=train_x.shape[1], out_dim=len(param_names), hidden=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.6f}")

    model.eval()
    with torch.no_grad():
        pred_01 = model(torch.tensor(test_x)).numpy()
    pred_phys = pred_01 * (maxs - mins) + mins
    true_phys = test_y * (maxs - mins) + mins

    print(f"\n{'Parameter':30s} {'R²':>8s} {'RMSE':>10s}")
    print(f"{'-'*50}")
    for i, name in enumerate(param_names):
        r2 = r2_score(true_phys[:, i], pred_phys[:, i])
        rmse = np.sqrt(np.mean((true_phys[:, i] - pred_phys[:, i])**2))
        marker = " ***" if 'fraction' in name or 'flood' in name else ""
        print(f"{name:30s} {r2:8.4f} {rmse:10.4f}{marker}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    focus = ['wheat_fraction', 'rice_fraction', 'delta_eos',
             'wheat_M', 'summer_M', 'summer_m',
             'summer_sos', 'summer_sen', 'summer_eos_base']
    for ax, name in zip(axes.ravel(), focus):
        i = param_names.index(name)
        t, p = true_phys[:, i], pred_phys[:, i]
        r2 = r2_score(t, p)
        ax.scatter(t, p, alpha=0.15, s=8, c='steelblue', edgecolors='none')
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
        ax.set_title(f'{name}\nR²={r2:.4f}', fontsize=11)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures/supervised_baseline_scatter.png"), dpi=150)
    plt.close()
    print(f"\nSaved: figures/supervised_baseline_scatter.png")


if __name__ == '__main__':
    main()
