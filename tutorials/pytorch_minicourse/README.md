# PyTorch 最小入门（约 3–5 天）

按顺序跑通四个脚本即可。不需要系统学完整教程。

```bash
conda activate magic
cd /path/to/MAGIC_Phenology

python tutorials/pytorch_minicourse/01_tensor_basics.py
python tutorials/pytorch_minicourse/02_autograd.py
python tutorials/pytorch_minicourse/03_mlp_module.py
python tutorials/pytorch_minicourse/04_sin_regression.py
```

**验收标准**：`04_sin_regression.py` 跑完后在 `tutorials/pytorch_minicourse/figures/sin_fit.png` 看到拟合曲线贴近 `sin(x)`，终端打印的最终 MSE < 0.05。
