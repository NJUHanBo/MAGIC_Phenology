"""01 — Tensor 基本操作：创建、reshape、索引、设备。"""
import torch

print("torch version:", torch.__version__)

x = torch.arange(12, dtype=torch.float32)
print("x:", x)
print("reshape (3,4):", x.reshape(3, 4))
print("slice x[2:5]:", x[2:5])

a = torch.randn(2, 3)
b = torch.ones_like(a)
print("a + b:\n", a + b)
print("matmul (2,3)@(3,2):\n", a @ torch.randn(3, 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
y = torch.zeros(2, 2, device=device)
print("y on", y.device, ":\n", y)

print("OK: 01_tensor_basics passed")
