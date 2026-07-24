"""冒烟测试：用现有 PILA_DPM_A 配置跑 1 个 epoch，确认训练链路可启动。"""
import os
import sys
import json
import copy
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def ensure_data():
    train_csv = os.path.join(PROJECT_ROOT, "data/processed/dpm/train_evi.csv")
    need = (not os.path.exists(train_csv)) or os.path.getsize(train_csv) == 0
    if not need:
        return
    print("WARNING: train_evi.csv missing/empty — regenerating small synthetic set...")
    from scripts.generate_synthetic_dpm import generate
    out_dir = os.path.join(PROJECT_ROOT, "data/processed/dpm")
    os.makedirs(out_dir, exist_ok=True)
    all_evi, all_params, splits, header_evi, header_params = generate(
        n_samples=200,
        paras_path=os.path.join(PROJECT_ROOT, "configs/dpm_paras.json"),
        noise_levels=[0.01, 0.02],
        seed=0,
        augment_factor=2,
    )
    for split_name, idx in splits.items():
        np_path = os.path.join(out_dir, f"{split_name}_evi.csv")
        pp_path = os.path.join(out_dir, f"{split_name}_params.csv")
        import numpy as np
        np.savetxt(np_path, all_evi[idx], delimiter=",", header=header_evi, comments="")
        np.savetxt(pp_path, all_params[idx], delimiter=",", header=header_params, comments="")
    import numpy as np
    x_mean = all_evi[splits["train"]].mean(axis=0)
    x_scale = all_evi[splits["train"]].std(axis=0)
    x_scale = np.where(x_scale < 1e-6, 1.0, x_scale)
    np.save(os.path.join(out_dir, "x_mean.npy"), x_mean)
    np.save(os.path.join(out_dir, "x_scale.npy"), x_scale)
    print("Regenerated small synthetic EVI set.")


def main():
    ensure_data()
    cfg_path = os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_DPM_A.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = copy.deepcopy(cfg)
    cfg["trainer"]["epochs"] = 1
    cfg["trainer"]["save_period"] = 1
    cfg["trainer"]["early_stop"] = 1
    cfg["trainer"]["phys_vae"]["epochs_pretrain"] = 0
    cfg["data_loader"]["args"]["batch_size"] = 32
    cfg["name"] = "PILA_DPM_SMOKE"
    smoke_cfg = os.path.join(PROJECT_ROOT, "configs/phys_smpl/_smoke_PILA_DPM_A.json")
    with open(smoke_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    print("Running 1-epoch smoke train with", smoke_cfg)
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "train_pila.py"), "-c", smoke_cfg]
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    print("OK: smoke_test_train finished")


if __name__ == "__main__":
    main()
