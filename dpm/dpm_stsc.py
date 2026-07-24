"""
DPM_STSC: 三层前向模型 — 物候层 → 光谱层 → 混合层。

输出混合反射率曲面 (batch, 7, 46)。

可学习端元：5 类植被/背景 peak 光谱（wheat/rice/maize/soy/soil）+ water，
初始值从 endmembers.json 加载。
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn


BAND_NAMES = ["Red", "NIR", "Blue", "Green", "SWIR1230", "SWIR1628", "SWIR2105"]
N_BANDS = 7
N_TIME = 46


class DPM_STSC(nn.Module):
    """
    三层前向模型：
    Layer 1 - 物候层：双 logistic 归一化生长曲线 g(t) ∈ [0,1]
    Layer 2 - 光谱层：rho = rho_bg + g * (rho_veg - rho_bg)
    Layer 3 - 混合层：反射率空间线性叠加

    输入物理参数（编码器反演，约 22 个，见 configs/dpm_stsc_paras.json）：
      wheat: M,m,sos,mat,sen,eos (6) — M/m 在 STSC 中用于构造 g 的幅度，
             实际反射率幅度主要由端元决定；这里用 (S1-S2) 归一化到 [0,1] 的 g。
      summer shared: sos, mat (2) + 各作物独立 sen/eos:
             rice_sen, rice_eos, maize_sen, maize_eos, soy_sen, soy_eos (6)
      summer shared amplitude helpers: summer_M, summer_m (2) — 用于 g 形状稳定性
      fractions: wheat_frac, rice_frac, maize_frac (3)；soy_frac = max(0, 1 - sum - bg)
        为简化，用 4 个面积：wheat/rice/maize/soy，约束非负且和 <= 1，背景 = 1-sum
      flood: flood_onset, flood_intensity (2)

    端元（模型参数，非编码器输出）：
      rho_wheat/rice/maize/soy/soil/water: 各 (7,)
    """

    def __init__(
        self,
        time_points: Optional[torch.Tensor] = None,
        endmembers_path: Optional[str] = None,
        learnable_endmembers: bool = True,
    ):
        super().__init__()
        if time_points is None:
            self.register_buffer("time_points", torch.arange(1, 362, 8).float())
        else:
            self.register_buffer("time_points", time_points.float())

        init = self._default_endmembers()
        if endmembers_path and os.path.exists(endmembers_path):
            with open(endmembers_path) as f:
                em = json.load(f)
            for k in ["wheat", "rice", "maize", "soy", "soil", "water"]:
                init[k] = torch.tensor(em[k]["peak_spectrum"], dtype=torch.float32)

        self.learnable_endmembers = learnable_endmembers
        for name, vec in init.items():
            if learnable_endmembers:
                setattr(self, f"rho_{name}", nn.Parameter(vec.clone()))
            else:
                self.register_buffer(f"rho_{name}", vec.clone())

    @staticmethod
    def _default_endmembers() -> Dict[str, torch.Tensor]:
        return {
            "wheat": torch.tensor([0.05, 0.45, 0.03, 0.08, 0.25, 0.18, 0.10]),
            "rice":  torch.tensor([0.04, 0.42, 0.03, 0.07, 0.22, 0.15, 0.09]),
            "maize": torch.tensor([0.06, 0.48, 0.04, 0.09, 0.28, 0.20, 0.12]),
            "soy":   torch.tensor([0.05, 0.40, 0.03, 0.08, 0.24, 0.17, 0.11]),
            "soil":  torch.tensor([0.18, 0.22, 0.10, 0.15, 0.30, 0.35, 0.28]),
            "water": torch.tensor([0.04, 0.02, 0.05, 0.04, 0.01, 0.01, 0.01]),
        }

    def phenology_g(self, t, sos, mat, sen, eos):
        """
        归一化双 logistic 生长曲线 g(t) ∈ 约 [0,1]。
        忽略绝对 M/m，只保留形状（便于光谱层用端元控制幅度）。
        Returns: (batch, n_time)
        """
        t = t.unsqueeze(0)
        sos = sos.unsqueeze(1)
        mat = mat.unsqueeze(1)
        sen = sen.unsqueeze(1)
        eos = eos.unsqueeze(1)

        exp_arg1 = 2 * (sos + mat - 2 * t) / (mat - sos + 1e-6)
        exp_arg1 = torch.clamp(exp_arg1, -500, 500)
        s1 = 1 / (1 + torch.exp(exp_arg1))

        exp_arg2 = 2 * (sen + eos - 2 * t) / (eos - sen + 1e-6)
        exp_arg2 = torch.clamp(exp_arg2, -500, 500)
        s2 = 1 / (1 + torch.exp(exp_arg2))

        g = (s1 - s2).clamp(0.0, 1.0)
        return g

    def spectral_layer(self, g: torch.Tensor, rho_veg: torch.Tensor, rho_bg: torch.Tensor):
        """
        g: (B, T)
        rho_veg / rho_bg: (7,) or (B, 7)
        returns: (B, 7, T)
        """
        if rho_veg.dim() == 1:
            rho_veg = rho_veg.unsqueeze(0)
        if rho_bg.dim() == 1:
            rho_bg = rho_bg.unsqueeze(0)
        # (B, 7, 1) + (B, 1, T) * (B, 7, 1)
        g = g.unsqueeze(1)  # (B, 1, T)
        veg = rho_veg.unsqueeze(-1)  # (B, 7, 1)
        bg = rho_bg.unsqueeze(-1)
        return bg + g * (veg - bg)

    def flood_window(self, t: torch.Tensor, flood_onset: torch.Tensor, flood_intensity: torch.Tensor):
        """
        W(t) ∈ [0,1]：淹水权重。onset 为 DOY；intensity 控制窗口宽度/强度。
        flood_onset, flood_intensity: (B,)
        returns: (B, T)
        """
        t = t.unsqueeze(0)  # (1, T)
        onset = flood_onset.unsqueeze(1)
        # intensity 映射为窗口半宽（天），范围约 5–40
        half = (5.0 + 35.0 * flood_intensity.clamp(0, 1)).unsqueeze(1)
        # 平滑窗：进入淹水后一段时间内 W≈intensity
        enter = torch.sigmoid((t - onset) / 3.0)
        leave = torch.sigmoid((onset + 2 * half - t) / 3.0)
        w = enter * leave * flood_intensity.unsqueeze(1).clamp(0, 1)
        return w

    def rice_background(self, t, flood_onset, flood_intensity):
        """时间变化的水稻背景端元：土壤 ↔ 水体。returns (B, 7, T)"""
        w = self.flood_window(t, flood_onset, flood_intensity)  # (B, T)
        soil = self.rho_soil  # (7,)
        water = self.rho_water
        # (B, 7, T)
        w = w.unsqueeze(1)
        return w * water.view(1, -1, 1) + (1 - w) * soil.view(1, -1, 1)

    def forward_components(self, **paras):
        """返回各作物 STSC 分量，便于可视化分解。"""
        device = paras["wheat_sos"].device
        t = self.time_points.to(device)
        B = paras["wheat_sos"].shape[0]

        g_wheat = self.phenology_g(
            t, paras["wheat_sos"], paras["wheat_mat"], paras["wheat_sen"], paras["wheat_eos"]
        )
        # 夏季共用 sos/mat，各作物独立 sen/eos
        g_rice = self.phenology_g(
            t, paras["summer_sos"], paras["summer_mat"], paras["rice_sen"], paras["rice_eos"]
        )
        g_maize = self.phenology_g(
            t, paras["summer_sos"], paras["summer_mat"], paras["maize_sen"], paras["maize_eos"]
        )
        g_soy = self.phenology_g(
            t, paras["summer_sos"], paras["summer_mat"], paras["soy_sen"], paras["soy_eos"]
        )

        soil = self.rho_soil.unsqueeze(0).expand(B, -1)
        rho_wheat = self.spectral_layer(g_wheat, self.rho_wheat.unsqueeze(0).expand(B, -1), soil)

        rho_bg_rice = self.rice_background(t, paras["flood_onset"], paras["flood_intensity"])
        # 水稻：用随时间变化的背景
        g_r = g_rice.unsqueeze(1)
        veg_r = self.rho_rice.view(1, -1, 1).expand(B, -1, t.numel())
        rho_rice = rho_bg_rice + g_r * (veg_r - rho_bg_rice)

        rho_maize = self.spectral_layer(g_maize, self.rho_maize.unsqueeze(0).expand(B, -1), soil)
        rho_soy = self.spectral_layer(g_soy, self.rho_soy.unsqueeze(0).expand(B, -1), soil)
        rho_bg = soil.unsqueeze(-1).expand(B, N_BANDS, t.numel())

        return {
            "wheat": rho_wheat,
            "rice": rho_rice,
            "maize": rho_maize,
            "soy": rho_soy,
            "background": rho_bg,
            "g_wheat": g_wheat,
            "g_rice": g_rice,
            "g_maize": g_maize,
            "g_soy": g_soy,
        }

    def mix(self, comps: dict, wheat_frac, rice_frac, maize_frac, soy_frac):
        """线性混合。fractions: (B,)"""
        aw = wheat_frac.view(-1, 1, 1)
        ar = rice_frac.view(-1, 1, 1)
        am = maize_frac.view(-1, 1, 1)
        as_ = soy_frac.view(-1, 1, 1)
        ab = (1.0 - wheat_frac - rice_frac - maize_frac - soy_frac).clamp(min=0.0).view(-1, 1, 1)
        mixed = (
            aw * comps["wheat"]
            + ar * comps["rice"]
            + am * comps["maize"]
            + as_ * comps["soy"]
            + ab * comps["background"]
        )
        return mixed.clamp(0.0, 1.0)

    def run(self, **paras):
        """
        Returns mixed reflectance (B, 7, T).
        """
        # 面积约束
        for k in ["wheat_frac", "rice_frac", "maize_frac", "soy_frac"]:
            paras[k] = paras[k].clamp(0.0, 1.0)
        total = paras["wheat_frac"] + paras["rice_frac"] + paras["maize_frac"] + paras["soy_frac"]
        scale = torch.clamp(total, min=1.0)
        for k in ["wheat_frac", "rice_frac", "maize_frac", "soy_frac"]:
            paras[k] = paras[k] / scale

        comps = self.forward_components(**paras)
        return self.mix(
            comps,
            paras["wheat_frac"],
            paras["rice_frac"],
            paras["maize_frac"],
            paras["soy_frac"],
        )

    # 兼容旧接口命名
    def forward(self, **paras):
        return self.run(**paras)
