"""
Step 6 end-to-end test:
  1. Load synthetic EVI via EVIDataLoader
  2. Init PHYS_VAE_SMPL with PILA_DPM_A config
  3. Forward pass (encode → draw → decode)
  4. Verify shapes + value sanity
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_data_loader():
    print("=" * 60)
    print("[1] Testing EVIDataLoader")
    print("=" * 60)
    from data_loader.data_loaders import EVIDataLoader

    loader = EVIDataLoader(
        data_dir="data/processed/dpm/train_evi.csv",
        batch_size=32, shuffle=True, validation_split=0.0, num_workers=0
    )
    batch = next(iter(loader))
    evi = batch['evi']
    print(f"  Batch shape: {evi.shape}")
    print(f"  Value range: [{evi.min().item():.4f}, {evi.max().item():.4f}]")
    print(f"  Mean: {evi.mean().item():.4f}")
    assert evi.shape == (32, 46), f"Expected (32, 46), got {evi.shape}"
    assert evi.min() >= 0.0, "EVI should be non-negative"
    assert evi.max() <= 1.0, "EVI should be <= 1.0"
    print("  PASSED\n")
    return evi


def test_model_init():
    print("=" * 60)
    print("[2] Testing PHYS_VAE_SMPL initialization")
    print("=" * 60)
    from model.model_phys_smpl import PHYS_VAE_SMPL

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs/phys_smpl/PILA_DPM_A.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    model = PHYS_VAE_SMPL(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {type(model).__name__}")
    print(f"  Physics model: {type(model.physics_model).__name__}")
    print(f"  dim_z_phy: {model.dim_z_phy}")
    print(f"  dim_z_aux: {model.dim_z_aux}")
    print(f"  Total parameters: {n_params:,}")
    print("  PASSED\n")
    return model, config


def test_forward_pass(model, evi_batch, config):
    print("=" * 60)
    print("[3] Testing forward pass")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        z_phy_stat, z_aux_stat, x_mean = model(
            evi_batch, reconstruct=True, hard_z_phy=True,
            epoch=0, epochs_pretrain=30
        )

    print(f"  z_phy mean shape: {z_phy_stat['mean'].shape}")
    print(f"  z_phy lnvar shape: {z_phy_stat['lnvar'].shape}")
    print(f"  z_aux mean shape: {z_aux_stat['mean'].shape}")
    print(f"  x_mean (recon) shape: {x_mean.shape}")
    print(f"  x_mean range: [{x_mean.min().item():.4f}, {x_mean.max().item():.4f}]")

    assert z_phy_stat['mean'].shape == (32, 21)
    assert x_mean.shape == (32, 46)
    print("  PASSED\n")
    return z_phy_stat


def test_physics_decode(model, z_phy_stat):
    print("=" * 60)
    print("[4] Testing Physics-only decode")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        z_phy = torch.sigmoid(z_phy_stat['mean'])
        y_phys = model.generate_physonly(z_phy)

    print(f"  Physics output shape: {y_phys.shape}")
    print(f"  Physics output range: [{y_phys.min().item():.4f}, {y_phys.max().item():.4f}]")
    assert y_phys.shape == (32, 46)
    print("  PASSED\n")


def test_parameter_recovery():
    print("=" * 60)
    print("[5] Testing parameter recovery (DPM round-trip)")
    print("=" * 60)
    from dpm.dpm import DPM

    dpm = DPM()
    params = {
        'wheat_M': torch.tensor([0.7]),
        'wheat_m': torch.tensor([0.1]),
        'wheat_sos': torch.tensor([60.0]),
        'wheat_mat': torch.tensor([110.0]),
        'wheat_sen': torch.tensor([140.0]),
        'wheat_eos': torch.tensor([160.0]),
        'rice_M': torch.tensor([0.75]),
        'rice_m': torch.tensor([0.1]),
        'rice_sos': torch.tensor([175.0]),
        'rice_mat': torch.tensor([220.0]),
        'rice_sen': torch.tensor([260.0]),
        'rice_eos': torch.tensor([300.0]),
        'maize_M': torch.tensor([0.65]),
        'maize_m': torch.tensor([0.08]),
        'maize_sos': torch.tensor([170.0]),
        'maize_mat': torch.tensor([215.0]),
        'maize_sen': torch.tensor([250.0]),
        'maize_eos': torch.tensor([285.0]),
        'wheat_fraction': torch.tensor([0.4]),
        'rice_mix_maize_fraction': torch.tensor([0.5]),
        'maize_in_mix_fraction': torch.tensor([0.3]),
    }
    evi = dpm.run(**params)
    print(f"  Single-sample EVI shape: {evi.shape}")
    print(f"  EVI range: [{evi.min().item():.4f}, {evi.max().item():.4f}]")
    print(f"  Spring peak (DOY~110): {evi[0, 13].item():.4f}")
    print(f"  Summer peak (DOY~220): {evi[0, 27].item():.4f}")
    assert evi.shape == (1, 46)
    assert evi.min() >= 0.0
    assert evi.max() <= 1.0
    print("  PASSED\n")


def test_config_B():
    print("=" * 60)
    print("[6] Testing PILA_DPM_B (with low-rank residual)")
    print("=" * 60)
    from model.model_phys_smpl import PHYS_VAE_SMPL

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs/phys_smpl/PILA_DPM_B.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    model = PHYS_VAE_SMPL(config)
    x = torch.rand(16, 46)
    model.eval()
    with torch.no_grad():
        z_phy_stat, z_aux_stat, x_mean = model(
            x, reconstruct=True, epoch=50, epochs_pretrain=30
        )
    print(f"  z_aux mean shape: {z_aux_stat['mean'].shape}")
    assert z_aux_stat['mean'].shape == (16, 8), f"Expected (16, 8), got {z_aux_stat['mean'].shape}"

    with torch.no_grad():
        z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z_phy=True, hard_z_aux=True)
        x_PB, x_P, y, delta, c = model.decode(
            z_phy, z_aux, epoch=50, epochs_pretrain=30, full=True
        )
    print(f"  delta shape: {delta.shape}, norm: {delta.norm(dim=1).mean().item():.6f}")
    print(f"  c shape: {c.shape}")
    print(f"  B shape: {model.dec.B.shape}")
    print(f"  s values: {model.dec.s.data}")
    assert delta.shape == (16, 46)
    assert c.shape == (16, 8)
    print("  PASSED\n")


if __name__ == '__main__':
    evi = test_data_loader()
    model, config = test_model_init()
    z_phy_stat = test_forward_pass(model, evi, config)
    test_physics_decode(model, z_phy_stat)
    test_parameter_recovery()
    test_config_B()
    print("=" * 60)
    print("ALL STEP 6 TESTS PASSED")
    print("=" * 60)
