"""04 — 验收：MLP 拟合 y = sin(x) + noise，画出拟合曲线。"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.linspace(-np.pi, np.pi, 400).astype(np.float32)
    y = np.sin(x) + 0.05 * np.random.randn(len(x)).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(x[:, None]),
        torch.from_numpy(y[:, None]),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch+1:3d}  mse={total/len(ds):.5f}")

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x[:, None])
        y_hat = model(x_t).numpy().ravel()
        final_mse = float(np.mean((y_hat - np.sin(x)) ** 2))

    print(f"final MSE vs clean sin(x): {final_mse:.5f}")
    assert final_mse < 0.05, f"验收失败：MSE={final_mse}"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(x[::4], y[::4], s=8, alpha=0.4, label="noisy data")
    ax.plot(x, np.sin(x), "k--", lw=1.5, label="true sin(x)")
    ax.plot(x, y_hat, "r-", lw=2, label="MLP fit")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PyTorch MLP: sin(x) regression")
    out = os.path.join(OUT_DIR, "sin_fit.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved:", out)
    print("OK: 04_sin_regression passed")


if __name__ == "__main__":
    main()
