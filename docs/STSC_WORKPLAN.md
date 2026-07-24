# PILA-STSC 执行清单（给孙夕又）

完整计划见 Cursor plan：`pila-stsc_unmixing_restart`。本文件是仓库内可执行索引。

## 阶段零
1. 读 `QUICKSTART.md`，装 `magic` 环境
2. 完成 `tutorials/pytorch_minicourse/` 四个脚本；验收：`python tutorials/pytorch_minicourse/04_sin_regression.py`
3. 冒烟：`python scripts/smoke_test_train.py`
4. 完整训练（可选）：`python train_pila.py -c configs/phys_smpl/PILA_DPM_A.json`
5. 验证：`python tests/verify_synthetic.py`；推理：`python tests/infer_huanghai.py`

## 阶段一
1. GEE：`scripts/gee_export_modis_stsc.py`（需 Earth Engine 认证）
2. QC：`scripts/qc_modis_stsc.py`
3. 端元：`scripts/extract_endmembers.py` → `data/processed/stsc/endmembers.json`

## 阶段二
1. 读 `dpm/dpm.py`（EVI 版）与 `dpm/dpm_stsc.py`（STSC 版）
2. 跑单元测试：`python -m pytest tests/test_dpm_stsc.py -v`（或 `python tests/test_dpm_stsc.py`）
3. 可视化：`python scripts/visualize_stsc.py`

## 阶段三
1. 参数范围：`configs/dpm_stsc_paras.json`
2. 生成：`python scripts/generate_synthetic_stsc.py`
3. DataLoader：`datasets/timeseriesSTSC.py`

## 阶段四
1. 配置：`configs/phys_smpl/PILA_STSC_A.json`
2. 训练：`python train_pila.py -c configs/phys_smpl/PILA_STSC_A.json`
3. 对比：`python scripts/compare_stsc_vs_evi.py`

## 阶段五
1. 推理：`python tests/infer_huanghai_stsc.py`
2. 面积/突变/分解：`scripts/validate_huanghai_stsc.py`
