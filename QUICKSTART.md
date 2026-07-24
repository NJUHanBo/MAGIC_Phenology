# QUICKSTART — MAGIC_Phenology / PILA-STSC

一句话目标：从 MODIS 混合像元的多波段时间序列，反演出亚像元作物面积比例和物候参数（小麦 / 水稻 / 玉米 / 大豆）。

当前基线版本用 EVI 一维曲线；升级方向是 STSC（Spectro-Temporal Surface of Crop）多波段曲面。细节见 `docs/method.md` 与 clawd wiki `pila-crop-unmixing.md`。

---

## 环境安装

推荐 Python 3.11：

```bash
conda create -n magic python=3.11 -y
conda activate magic

# CPU（本机调试）
pip install torch torchvision

# 或 GPU（有 CUDA 时）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__)"
```

Windows 额外说明见 `WINDOWS_SETUP.md`。

---

## 目录结构（与本项目相关）

| 路径 | 作用 |
|------|------|
| `dpm/dpm.py` | 动态物候前向模型（EVI 版，15 参数） |
| `dpm/dpm_stsc.py` | STSC 多波段前向模型（升级中） |
| `train_pila.py` | PILA 训练入口 |
| `configs/phys_smpl/` | PILA 训练配置（A/B/C） |
| `configs/dpm_paras.json` | 物候参数取值范围 |
| `scripts/` | 合成数据生成、WEVI 计算等 |
| `tests/` | 流水线与推理测试 |
| `tutorials/` | PyTorch 入门与验收脚本 |
| `data/processed/dpm/` | 已处理好的 EVI 训练/验证数据 |
| `saved/dpm/models/` | 训练好的 checkpoint |
| `docs/method.md` | 方法说明（中文） |

---

## 跑通现有 EVI 基线（约 1–2 小时 CPU）

工作目录必须是仓库根目录：

```bash
cd /path/to/MAGIC_Phenology
conda activate magic

# 1) 训练（可缩短 epochs 做冒烟测试）
python train_pila.py -c configs/phys_smpl/PILA_DPM_A.json

# 2) 合成数据参数恢复验证
python tests/verify_synthetic.py

# 3) 黄海农场真实数据推理（使用最新 checkpoint）
python tests/infer_huanghai.py
```

最新完整训练结果目录：`saved/dpm/models/PILA_DPM_A/0412_154312/`（200 epoch，`model_best.pth`）。

冒烟测试（只跑几个 batch，确认代码能跑）：见 `scripts/smoke_test_train.py`。

---

## 模型在干什么（给科研助理）

1. **前向模型（DPM）**：给定物候参数 + 面积比例 → 生成一条（或多波段）时间序列曲线。
2. **编码器（PILA）**：给定观测到的时间序列 → 反推出物候参数和面积比例。
3. **训练**：在合成数据上练「曲线 → 参数」；loss 主要看重建误差（重建曲线是否贴近输入）和参数循环一致性。
4. **推理**：把真实 MODIS 像元曲线喂给编码器，得到每个像元的作物面积比例。

输入（基线）：46 个时间点的 EVI（一年 8 天合成）。  
输出（基线）：15 个物理参数（小麦/夏季物候 + 面积比例等）。

---

## STSC 升级路线（执行清单）

按阶段推进，详见仓库内 `docs/STSC_WORKPLAN.md`：

0. PyTorch 入门 → 跑通现有代码  
1. GEE 导出 MODIS 7 波段数据 + QC + 端元提取  
2. 实现 `dpm_stsc.py` 并通过单元测试  
3. 生成多波段合成训练集  
4. 适配 PILA 编码器并训练  
5. 黄海农场真实数据验证  

---

## 常见问题

- **找不到数据文件**：确认 `data/processed/dpm/` 下有 `train_evi.csv` 等；若缺失，先跑 `scripts/generate_synthetic_dpm.py`。
- **CUDA 报错**：本机无 GPU 时配置里会自动用 CPU，属正常。
- **推理脚本指向旧模型**：`tests/infer_huanghai.py` 中 `MODEL_PATH` 应为 `0412_154312`。
