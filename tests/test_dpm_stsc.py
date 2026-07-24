"""
单元测试：DPM_STSC 光谱层 / 淹水 / 混合层。

运行：
  python tests/test_dpm_stsc.py
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dpm.dpm_stsc import DPM_STSC


def _base_paras(B=4, device="cpu"):
    """一组合法默认参数。"""
    def c(v):
        return torch.full((B,), float(v), device=device)

    return {
        "wheat_sos": c(40), "wheat_mat": c(110), "wheat_sen": c(140), "wheat_eos": c(160),
        "summer_sos": c(160), "summer_mat": c(220),
        "rice_sen": c(250), "rice_eos": c(290),
        "maize_sen": c(240), "maize_eos": c(270),
        "soy_sen": c(245), "soy_eos": c(275),
        "wheat_frac": c(0.25), "rice_frac": c(0.25), "maize_frac": c(0.25), "soy_frac": c(0.15),
        "flood_onset": c(170), "flood_intensity": c(0.8),
    }


def test_spectral_layer_g_all_one():
    m = DPM_STSC(learnable_endmembers=False)
    B, T = 3, 46
    g = torch.ones(B, T)
    out = m.spectral_layer(g, m.rho_wheat, m.rho_soil)
    # g=1 → 应等于 veg
    assert torch.allclose(out, m.rho_wheat.view(1, 7, 1).expand(B, 7, T), atol=1e-5)


def test_spectral_layer_g_all_zero():
    m = DPM_STSC(learnable_endmembers=False)
    B, T = 3, 46
    g = torch.zeros(B, T)
    out = m.spectral_layer(g, m.rho_wheat, m.rho_soil)
    assert torch.allclose(out, m.rho_soil.view(1, 7, 1).expand(B, 7, T), atol=1e-5)


def test_pure_wheat_mix():
    m = DPM_STSC(learnable_endmembers=False)
    p = _base_paras(B=2)
    p["wheat_frac"] = torch.ones(2)
    p["rice_frac"] = torch.zeros(2)
    p["maize_frac"] = torch.zeros(2)
    p["soy_frac"] = torch.zeros(2)
    comps = m.forward_components(**p)
    mixed = m.run(**p)
    assert torch.allclose(mixed, comps["wheat"], atol=1e-5)


def test_flood_lowers_nir():
    m = DPM_STSC(learnable_endmembers=False)
    p = _base_paras(B=1)
    p["wheat_frac"] = torch.zeros(1)
    p["maize_frac"] = torch.zeros(1)
    p["soy_frac"] = torch.zeros(1)
    p["rice_frac"] = torch.ones(1)

    p_flood = {k: v.clone() for k, v in p.items()}
    p_noflood = {k: v.clone() for k, v in p.items()}
    p_noflood["flood_intensity"] = torch.zeros(1)

    nir_flood = m.run(**p_flood)[0, 1]  # NIR
    nir_noflood = m.run(**p_noflood)[0, 1]
    # 淹水窗口内 NIR 应更低
    t = m.time_points
    onset = p_flood["flood_onset"][0].item()
    win = (t > onset) & (t < onset + 40)
    assert (nir_flood[win].mean() < nir_noflood[win].mean()).item()


def test_output_shape():
    m = DPM_STSC(learnable_endmembers=False)
    p = _base_paras(B=5)
    out = m.run(**p)
    assert out.shape == (5, 7, 46)


def main():
    test_spectral_layer_g_all_one()
    print("PASS spectral g=1")
    test_spectral_layer_g_all_zero()
    print("PASS spectral g=0")
    test_pure_wheat_mix()
    print("PASS pure wheat mix")
    test_flood_lowers_nir()
    print("PASS flood lowers NIR")
    test_output_shape()
    print("PASS output shape")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
