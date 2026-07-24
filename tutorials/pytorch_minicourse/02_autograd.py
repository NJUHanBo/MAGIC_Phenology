"""02 — 自动求导：requires_grad / backward / grad。"""
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # dy/dx = 2x + 2 = 8 when x=3
y.backward()
print("x.grad (expect ~8):", x.grad.item())
assert abs(x.grad.item() - 8.0) < 1e-5

w = torch.randn(4, requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
print("w.grad:", w.grad)
assert w.grad is not None

print("OK: 02_autograd passed")
