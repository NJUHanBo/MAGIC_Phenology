"""03 — nn.Module：两层 MLP，理解 forward / parameters。"""
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = TinyMLP()
print(model)
n_params = sum(p.numel() for p in model.parameters())
print("num parameters:", n_params)

x = torch.randn(8, 1)
y = model(x)
print("output shape:", tuple(y.shape))
assert y.shape == (8, 1)

print("OK: 03_mlp_module passed")
