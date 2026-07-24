"""
Compute WEVI (Water-Enhanced Vegetation Index) from EVI and Band2(NIR).

WEVI = EVI - lambda * max(NIR - EVI + delta, 0)

For normal vegetation (EVI > NIR): WEVI = EVI (unchanged)
For flooded rice paddies (NIR > EVI): WEVI dips below EVI, encoding flood signal
"""
import os, sys, glob, re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIF_DIR_EVI = "/Users/hanbo/Downloads/黄海农场数据/EVI"
TIF_DIR_NIR = "/Users/hanbo/Downloads/黄海农场数据/Band2"
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dpm", "huanghai")

STANDARD_DOY = list(range(1, 362, 8))
DELTA = 0.05
LAMBDA = 1.5


def load_year_data(tif_dir, year, scale_factor=1.0):
    """Load all TIFs for a year, return list of (doy, 2d_array)."""
    import rasterio
    pattern = os.path.join(tif_dir, f"*{year}*.tif")
    files = sorted(glob.glob(pattern))
    result = []
    for fp in files:
        match = re.search(r'(\d{8})\.tif$', os.path.basename(fp))
        if not match:
            continue
        dt = datetime.strptime(match.group(1), '%Y%m%d')
        doy = dt.timetuple().tm_yday
        with rasterio.open(fp) as src:
            data = src.read(1).astype(np.float32) * scale_factor
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata * scale_factor] = np.nan
        result.append((doy, data))
    result.sort(key=lambda x: x[0])
    return result


def compute_wevi_arrays(evi_data, nir_data, delta=DELTA, lam=LAMBDA):
    """
    Compute WEVI from paired EVI and NIR data.
    Both inputs: list of (doy, 2d_array).
    Returns: list of (doy, 2d_wevi_array) aligned to common DOYs.
    """
    evi_dict = {d: a for d, a in evi_data}
    nir_dict = {d: a for d, a in nir_data}
    common_doys = sorted(set(evi_dict.keys()) & set(nir_dict.keys()))

    result = []
    for doy in common_doys:
        evi = evi_dict[doy]
        nir = nir_dict[doy]
        flood_indicator = np.maximum(nir - evi + delta, 0)
        wevi = evi - lam * flood_indicator
        result.append((doy, wevi))
    return result


def plot_evi_vs_wevi(year=2015, pixels=None):
    """Plot EVI vs WEVI for sample pixels to visualize flood dip."""
    if pixels is None:
        pixels = [(20, 15), (25, 30), (30, 20), (22, 40)]

    evi_data = load_year_data(TIF_DIR_EVI, year, scale_factor=1.0)
    nir_data = load_year_data(TIF_DIR_NIR, year, scale_factor=0.0001)
    wevi_data = compute_wevi_arrays(evi_data, nir_data)

    evi_dict = {d: a for d, a in evi_data}
    wevi_dict = {d: a for d, a in wevi_data}
    nir_dict = {d: a for d, a in nir_data}
    common_doys = sorted(wevi_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for ax_i, (r, c) in enumerate(pixels):
        ax = axes[ax_i]
        doys_evi, vals_evi = [], []
        doys_wevi, vals_wevi = [], []
        doys_nir, vals_nir = [], []

        for doy in sorted(evi_dict.keys()):
            v = evi_dict[doy][r, c]
            if not np.isnan(v) and v > -0.5:
                doys_evi.append(doy)
                vals_evi.append(v)

        for doy in common_doys:
            v = wevi_dict[doy][r, c]
            if not np.isnan(v) and abs(v) < 2.0:
                doys_wevi.append(doy)
                vals_wevi.append(v)

        for doy in sorted(nir_dict.keys()):
            v = nir_dict[doy][r, c]
            if not np.isnan(v) and v > 0:
                doys_nir.append(doy)
                vals_nir.append(v)

        ax.plot(doys_evi, vals_evi, 'ko-', ms=4, lw=1, label='EVI')
        ax.plot(doys_wevi, vals_wevi, 'b^-', ms=4, lw=1.5, label=f'WEVI (λ={LAMBDA}, δ={DELTA})')
        ax.plot(doys_nir, vals_nir, 'g.--', ms=3, lw=0.8, alpha=0.6, label='NIR (Band2)')

        ax.axvspan(145, 185, alpha=0.1, color='cyan', label='Flood window')
        ax.axhline(y=0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('DOY', fontsize=11)
        ax.set_ylabel('Index Value', fontsize=11)
        ax.set_title(f'Pixel ({r},{c})', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(-0.3, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Huanghai Farm {year} — EVI vs WEVI (flood dip detection)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, f"huanghai_{year}_evi_vs_wevi.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_evi_vs_wevi(2015)
    plot_evi_vs_wevi(2010)
