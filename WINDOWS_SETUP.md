# Windows 服务器运行环境配置指南

## 系统要求

- **操作系统**: Windows Server 2016 或更高版本
- **Python版本**: Python 3.10.x（推荐 3.10.0）
- **内存**: 建议至少 8GB RAM
- **存储**: 根据数据集大小，建议至少 50GB 可用空间

## 环境配置步骤

### 1. 安装 Python

1. 从 [Python官网](https://www.python.org/downloads/) 下载 Python 3.10.x
2. 安装时**务必勾选** "Add Python to PATH"
3. 验证安装：
   ```cmd
   python --version
   ```
   应显示 `Python 3.10.x`

### 2. 安装依赖包

#### 方法一：使用 requirements.txt（推荐）

```cmd
cd /d [项目路径]\MAGIC_Phenology
pip install -r requirements.txt
```

#### 方法二：手动安装核心依赖

如果遇到某些包安装问题，可以分步安装：

```cmd
# 核心深度学习框架
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 科学计算库
pip install numpy scipy==1.10.1 pandas==1.5.3 scikit-learn

# 数据处理
pip install h5py openpyxl==3.1.5 rasterio==1.4.3

# 可视化
pip install matplotlib seaborn plotly

# 其他工具
pip install tqdm wandb torchdiffeq
```

**注意**: 
- `rasterio` 在 Windows 上可能需要额外安装 GDAL 依赖，如果安装失败，可以尝试：
  ```cmd
  pip install rasterio==1.4.3 --no-binary rasterio
  ```
  或者从 [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) 下载预编译的 wheel 文件

### 3. 设置环境变量

在 Windows 上设置 `PYTHONPATH` 环境变量：

#### 方法一：临时设置（当前会话有效）

```cmd
set PYTHONPATH=%CD%
```

#### 方法二：永久设置（推荐）

1. 右键"此电脑" → "属性" → "高级系统设置"
2. 点击"环境变量"
3. 在"用户变量"或"系统变量"中：
   - 变量名：`PYTHONPATH`
   - 变量值：`[项目完整路径]\MAGIC_Phenology`
   - 例如：`C:\Users\YourName\Documents\MAGIC_Phenology`

#### 方法三：在脚本中设置（推荐用于批处理文件）

创建 `run_train.bat` 文件：
```batch
@echo off
set PYTHONPATH=%CD%
python train.py --config configs\AE_RTM_A.json
```

## 运行项目

### 训练模型

**Linux/Mac 原始命令**:
```bash
python3 train.py --config configs/AE_RTM_A.json
```

**Windows 对应命令**:
```cmd
python train.py --config configs\AE_RTM_A.json
```

**注意**: Windows 使用反斜杠 `\` 作为路径分隔符，Linux/Mac 使用正斜杠 `/`

### 评估模型

**Linux/Mac 原始命令**:
```bash
python3 test_AE_RTM.py --config pretrained/AE_RTM_A/config.json --resume pretrained/AE_RTM_A/model_best.pth
```

**Windows 对应命令**:
```cmd
python test_AE_RTM.py --config pretrained\AE_RTM_A\config.json --resume pretrained\AE_RTM_A\model_best.pth
```

### 创建 Windows 批处理脚本

可以将 `run_train.sh` 转换为 `run_train.bat`:

```batch
@echo off
set PYTHONPATH=%CD%

REM Train AE_RTM_A (classical autoencoder)
python train.py --config configs\AE_RTM_A.json

REM Train AE_DPM_A (classical autoencoder)
python train.py --config configs\AE_DPM_A.json
```

## 常见问题排查

### 1. 模块导入错误

**问题**: `ModuleNotFoundError: No module named 'xxx'`

**解决**:
- 确认已安装所有依赖：`pip install -r requirements.txt`
- 确认 `PYTHONPATH` 环境变量已正确设置
- 尝试在项目根目录运行：`python -m train` 而不是 `python train.py`

### 2. Rasterio 安装失败

**问题**: `rasterio` 包安装失败

**解决**:
- 安装 GDAL：从 [OSGeo4W](https://trac.osgeo.org/osgeo4w/) 或使用 conda：
  ```cmd
  conda install -c conda-forge gdal
  pip install rasterio==1.4.3
  ```

### 3. 路径相关问题

**问题**: 配置文件中的路径无法找到

**解决**:
- 检查配置文件（如 `configs/AE_RTM_A.json`）中的路径
- 确保使用 Windows 路径格式（反斜杠 `\` 或双反斜杠 `\\`）
- 或者使用原始字符串（Python 会自动处理）

### 4. PyTorch CPU 版本

**注意**: 本项目使用 CPU 版本的 PyTorch，如果服务器有 GPU 并想使用 GPU 加速：

```cmd
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

然后修改代码中的设备设置（通常在 `train.py` 或配置文件中）。

## 验证安装

运行以下命令验证环境是否正确配置：

```cmd
python -c "import torch; import numpy; import pandas; import rasterio; print('所有核心依赖已正确安装')"
```

## 注意事项

1. **路径分隔符**: Windows 使用 `\`，Linux/Mac 使用 `/`。在 Python 代码中可以使用 `os.path.join()` 或 `pathlib.Path` 来避免跨平台问题。

2. **文件权限**: 确保对项目目录和数据目录有读写权限。

3. **长路径支持**: Windows 10/Server 2016+ 默认支持长路径，如果遇到路径过长问题，可以启用长路径支持。

4. **编码问题**: 如果遇到中文路径或文件名问题，确保使用 UTF-8 编码。

## 推荐工具

- **终端**: PowerShell 或 Windows Terminal（比传统 CMD 更好用）
- **Python 环境管理**: 考虑使用 `conda` 或 `venv` 创建独立环境
- **代码编辑器**: VS Code 或 PyCharm（便于调试）

## 使用 Conda 环境（可选）

如果服务器已安装 Anaconda/Miniconda，可以使用项目提供的 `environment.yml`:

```cmd
conda env create -f environment.yml
conda activate mres
```

注意：`environment.yml` 中的某些包是 Linux 特定的，可能需要手动调整。




