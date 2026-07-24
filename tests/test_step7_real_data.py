"""
Step 7 test: Huanghai Farm TIF → WEVI CSV pipeline + model inference on real data.

Workflow:
  1. Scan TIF files (EVI + NIR), parse dates, group by year
  2. For each year, align to standard 46-period DOY grid
  3. Compute WEVI = EVI - λ·max(NIR - EVI + δ, 0)
  4. Extract pixel-level WEVI time series → CSV
  5. Standardize and run through trained PHYS_VAE_SMPL
"""
import sys, os, json, glob, re
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIF_DIR_EVI = "/Users/hanbo/Downloads/黄海农场数据/EVI"
TIF_DIR_NIR = "/Users/hanbo/Downloads/黄海农场数据/Band2"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dpm", "huanghai")

STANDARD_DOY = list(range(1, 362, 8))  # 46 periods: 1,9,17,...,361
WEVI_DELTA = 0.05
WEVI_LAMBDA = 1.5


def parse_tif_files(tif_dir):
    """Scan TIF files, extract dates, group by year."""
    pattern = os.path.join(tif_dir, "*.tif")
    files = sorted(glob.glob(pattern))

    year_files = {}
    for fp in files:
        fname = os.path.basename(fp)
        match = re.search(r'(\d{8})\.tif$', fname)
        if not match:
            continue
        date_str = match.group(1)
        dt = datetime.strptime(date_str, '%Y%m%d')
        year = dt.year
        doy = dt.timetuple().tm_yday
        if year not in year_files:
            year_files[year] = []
        year_files[year].append((doy, fp))

    for y in year_files:
        year_files[y].sort(key=lambda x: x[0])

    return year_files


def compute_wevi_cube(evi_cube, nir_cube, delta=WEVI_DELTA, lam=WEVI_LAMBDA):
    """
    Compute WEVI from aligned EVI and NIR cubes (46, rows, cols).
    WEVI = EVI - lambda * max(NIR - EVI + delta, 0)
    """
    flood_indicator = np.maximum(nir_cube - evi_cube + delta, 0)
    wevi_cube = evi_cube - lam * flood_indicator
    return wevi_cube


def read_tif(filepath):
    """Read a single TIF file, return 2D array."""
    import rasterio
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def align_to_standard_doy(year_data):
    """
    Align observed DOY-indexed data to standard 46-period grid.
    Uses nearest-neighbor interpolation for missing periods.

    year_data: list of (doy, 2d_array) sorted by doy
    Returns: (46, rows, cols) array
    """
    observed_doys = [d for d, _ in year_data]
    arrays = [a for _, a in year_data]
    rows, cols = arrays[0].shape

    result = np.full((46, rows, cols), np.nan, dtype=np.float32)

    for i, target_doy in enumerate(STANDARD_DOY):
        distances = [abs(target_doy - od) for od in observed_doys]
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] <= 8:
            result[i] = arrays[nearest_idx]

    return result


def extract_pixel_timeseries(cube, min_valid_ratio=0.7):
    """
    Extract valid pixel time series from (46, rows, cols) cube.
    Returns: (n_pixels, 46) array, pixel_coords list
    """
    n_time, rows, cols = cube.shape
    all_ts = []
    coords = []

    for r in range(rows):
        for c in range(cols):
            ts = cube[:, r, c]
            valid_ratio = np.sum(~np.isnan(ts)) / n_time
            if valid_ratio >= min_valid_ratio:
                ts_filled = ts.copy()
                nan_mask = np.isnan(ts_filled)
                if nan_mask.any() and not nan_mask.all():
                    valid_idx = np.where(~nan_mask)[0]
                    nan_idx = np.where(nan_mask)[0]
                    ts_filled[nan_idx] = np.interp(nan_idx, valid_idx, ts_filled[valid_idx])
                all_ts.append(ts_filled)
                coords.append((r, c))

    return np.array(all_ts, dtype=np.float32), coords


def test_tif_parsing():
    print("=" * 60)
    print("[1] Parsing TIF files (EVI + NIR)")
    print("=" * 60)
    evi_files = parse_tif_files(TIF_DIR_EVI)
    nir_files = parse_tif_files(TIF_DIR_NIR)
    print(f"  EVI years: {sorted(evi_files.keys())}")
    print(f"  NIR years: {sorted(nir_files.keys())}")
    for y in [2001, 2010, 2020]:
        if y in evi_files:
            print(f"  {y} EVI: {len(evi_files[y])} files, NIR: {len(nir_files.get(y, []))} files")
    assert len(evi_files) >= 15, "Expected at least 15 years"
    print("  PASSED\n")
    return evi_files, nir_files


def test_single_year_pipeline(evi_files, nir_files, test_year=2015):
    print("=" * 60)
    print(f"[2] Processing year {test_year} → WEVI")
    print("=" * 60)
    evi_entries = evi_files[test_year]
    nir_entries = nir_files.get(test_year, [])
    print(f"  EVI files: {len(evi_entries)}, NIR files: {len(nir_entries)}")

    print("  Reading EVI TIFs...")
    evi_year_data = [(doy, read_tif(fp)) for doy, fp in evi_entries]
    print(f"  Reading NIR TIFs...")
    nir_year_data = [(doy, read_tif(fp) * 0.0001) for doy, fp in nir_entries]
    print(f"  Array shape: {evi_year_data[0][1].shape}")

    print("  Aligning EVI to 46-period grid...")
    evi_cube = align_to_standard_doy(evi_year_data)
    print("  Aligning NIR to 46-period grid...")
    nir_cube = align_to_standard_doy(nir_year_data)

    print(f"  Computing WEVI (λ={WEVI_LAMBDA}, δ={WEVI_DELTA})...")
    wevi_cube = compute_wevi_cube(evi_cube, nir_cube)
    valid_pct = np.sum(~np.isnan(wevi_cube)) / wevi_cube.size * 100
    print(f"  WEVI cube shape: {wevi_cube.shape}, valid: {valid_pct:.1f}%")

    print("  Extracting pixel WEVI time series...")
    ts_array, coords = extract_pixel_timeseries(wevi_cube)
    print(f"  Valid pixels: {len(coords)}, shape: {ts_array.shape}")
    print(f"  WEVI range: [{np.nanmin(ts_array):.4f}, {np.nanmax(ts_array):.4f}]")
    print(f"  WEVI mean: {np.nanmean(ts_array):.4f}")
    assert ts_array.shape[1] == 46
    print("  PASSED\n")
    return ts_array, coords


def test_model_inference(ts_array):
    print("=" * 60)
    print("[3] Model inference on real WEVI data")
    print("=" * 60)
    from model.model_phys_smpl import PHYS_VAE_SMPL

    x_mean = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_mean.npy"))
    x_scale = np.load(os.path.join(PROJECT_ROOT, "data/processed/dpm/x_scale.npy"))

    ts_std = (ts_array - x_mean) / x_scale
    print(f"  Standardized range: [{ts_std.min():.4f}, {ts_std.max():.4f}]")

    config_path = os.path.join(PROJECT_ROOT, "configs/phys_smpl/PILA_DPM_B.json")
    with open(config_path) as f:
        config = json.load(f)

    model = PHYS_VAE_SMPL(config)
    model.eval()

    x_tensor = torch.tensor(ts_std[:64], dtype=torch.float32)
    with torch.no_grad():
        z_phy_stat, z_aux_stat, x_recon = model(
            x_tensor, reconstruct=True, hard_z_phy=True, hard_z_aux=True,
            epoch=100, epochs_pretrain=30
        )

    z_phy = torch.sigmoid(z_phy_stat['mean'])
    print(f"  Input shape: {x_tensor.shape}")
    print(f"  z_phy shape: {z_phy.shape}")
    print(f"  z_phy range: [{z_phy.min().item():.4f}, {z_phy.max().item():.4f}]")

    paras_ranges = json.load(open(os.path.join(PROJECT_ROOT, "configs/dpm_paras.json")))
    param_names = list(paras_ranges.keys())
    print(f"\n  Sample pixel 0 inferred parameters (untrained model):")
    for i, name in enumerate(param_names):
        minv = paras_ranges[name]['min']
        maxv = paras_ranges[name]['max']
        val = z_phy[0, i].item() * (maxv - minv) + minv
        print(f"    {name:30s}: {val:.4f}  (range [{minv}, {maxv}])")
    print("  PASSED\n")


def test_save_csv(ts_array, coords, year=2015):
    print("=" * 60)
    print(f"[4] Saving WEVI CSV for year {year}")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wevi_cols = [f'EVI_{i+1}' for i in range(46)]
    header = ','.join(['row', 'col'] + wevi_cols)

    rows = []
    for (r, c), ts in zip(coords, ts_array):
        row_data = [str(r), str(c)] + [f'{v:.6f}' for v in ts]
        rows.append(','.join(row_data))

    csv_path = os.path.join(OUTPUT_DIR, f"huanghai_{year}_wevi.csv")
    with open(csv_path, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(row + '\n')

    print(f"  Saved: {csv_path}")
    print(f"  Rows: {len(rows)}, Cols: {len(wevi_cols) + 2}")
    file_size = os.path.getsize(csv_path) / 1024
    print(f"  File size: {file_size:.1f} KB")
    print("  PASSED\n")
    return csv_path


if __name__ == '__main__':
    evi_files, nir_files = test_tif_parsing()
    ts_array, coords = test_single_year_pipeline(evi_files, nir_files, test_year=2015)
    test_model_inference(ts_array)
    csv_path = test_save_csv(ts_array, coords, year=2015)

    print("=" * 60)
    print("ALL STEP 7 TESTS PASSED (WEVI)")
    print("=" * 60)
