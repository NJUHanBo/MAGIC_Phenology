"""
质量控制：云掩膜后的 MODIS GeoTIFF → 时间插值 → numpy (n_pixels, 7, 46)。

输入目录结构（GEE 导出下载后）：
  data/raw/modis_stsc/mod09a1_stsc_huanghai_YYYY.tif
  或 data/raw/modis_stsc/YYYY/*.tif

输出：
  data/processed/stsc/YYYY_reflectance.npy   # (n, 7, 46)
  data/processed/stsc/YYYY_rowcol.npy        # (n, 2) row, col
  data/processed/stsc/qa_report.csv          # 每年有效观测比例
"""
from __future__ import annotations

import argparse
import os
import glob
import csv
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_BANDS = 7
N_TIME = 46


def temporal_linear_interpolate(arr: np.ndarray) -> np.ndarray:
    """
    arr: (n_pix, n_bands, n_time)，NaN 为缺失。
    对每个像元×波段沿时间线性插值；两端外推用最近有效值。
    """
    out = arr.copy()
    n_pix, n_bands, n_t = out.shape
    t = np.arange(n_t)
    for i in range(n_pix):
        for b in range(n_bands):
            y = out[i, b, :]
            good = np.isfinite(y)
            if good.sum() == 0:
                out[i, b, :] = 0.0
                continue
            if good.all():
                continue
            out[i, b, :] = np.interp(t, t[good], y[good])
    return out


def load_year_geotiff(path: str):
    try:
        import rasterio
    except ImportError as e:
        raise SystemExit("需要 rasterio: pip install rasterio") from e

    with rasterio.open(path) as src:
        data = src.read()  # (bands_stacked, H, W)
        profile = src.profile
    # 期望 bands = 7 * n_scenes；若不足 46 景，右侧 pad NaN
    n_layers, H, W = data.shape
    if n_layers % N_BANDS != 0:
        raise ValueError(f"{path}: band count {n_layers} not divisible by {N_BANDS}")
    n_scenes = n_layers // N_BANDS
    data = data.astype(np.float32)
    # GEE 导出可能已乘 0.0001；若数值 > 1.5 视为未缩放
    if np.nanmax(data) > 1.5:
        data = data * 0.0001
    data[data < -0.01] = np.nan
    data[data > 1.0] = np.nan

    # reshape -> (7, n_scenes, H, W) then pad/crop to 46
    cube = data.reshape(N_BANDS, n_scenes, H, W)
    if n_scenes < N_TIME:
        pad = np.full((N_BANDS, N_TIME - n_scenes, H, W), np.nan, dtype=np.float32)
        cube = np.concatenate([cube, pad], axis=1)
    elif n_scenes > N_TIME:
        cube = cube[:, :N_TIME]

    # (H*W, 7, 46)
    arr = np.transpose(cube, (2, 3, 0, 1)).reshape(-1, N_BANDS, N_TIME)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rowcol = np.stack([rows.ravel(), cols.ravel()], axis=1)
    return arr, rowcol, profile, H, W


def process_year(tif_path: str, out_dir: str, year: int):
    arr, rowcol, profile, H, W = load_year_geotiff(tif_path)
    valid_ratio = float(np.isfinite(arr).mean())
    arr_filled = temporal_linear_interpolate(arr)

    # 去掉全年全零/全无效像元（可选：保留全部）
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{year}_reflectance.npy"), arr_filled.astype(np.float32))
    np.save(os.path.join(out_dir, f"{year}_rowcol.npy"), rowcol.astype(np.int32))
    meta = {"year": year, "H": H, "W": W, "valid_ratio_before_interp": valid_ratio,
            "n_pixels": int(arr.shape[0]), "source": tif_path}
    with open(os.path.join(out_dir, f"{year}_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"{year}: shape={arr_filled.shape}, valid_ratio={valid_ratio:.3f}")
    return meta


def find_tif(raw_dir: str, year: int):
    patterns = [
        os.path.join(raw_dir, f"mod09a1_stsc_huanghai_{year}.tif"),
        os.path.join(raw_dir, f"{year}", "*.tif"),
        os.path.join(raw_dir, f"*{year}*.tif"),
    ]
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return sorted(hits)[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(PROJECT_ROOT, "data/raw/modis_stsc"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data/processed/stsc"))
    parser.add_argument("--year-start", type=int, default=2001)
    parser.add_argument("--year-end", type=int, default=2020)
    parser.add_argument("--demo", action="store_true",
                        help="无真实 GeoTIFF 时生成假数据，打通后续流水线")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_rows = []

    if args.demo:
        rng = np.random.default_rng(0)
        for year in range(args.year_start, min(args.year_end, args.year_start) + 1):
            H, W = 20, 20
            arr = rng.uniform(0.05, 0.45, size=(H * W, N_BANDS, N_TIME)).astype(np.float32)
            # 随机挖洞再插值
            mask = rng.random(arr.shape) < 0.1
            arr[mask] = np.nan
            valid_ratio = float(np.isfinite(arr).mean())
            arr = temporal_linear_interpolate(arr)
            rowcol = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
            np.save(os.path.join(args.out_dir, f"{year}_reflectance.npy"), arr)
            np.save(os.path.join(args.out_dir, f"{year}_rowcol.npy"), rowcol.astype(np.int32))
            report_rows.append({"year": year, "valid_ratio_before_interp": valid_ratio, "n_pixels": H * W, "demo": True})
            print(f"demo {year}: wrote synthetic array")
    else:
        for year in range(args.year_start, args.year_end + 1):
            tif = find_tif(args.raw_dir, year)
            if tif is None:
                print(f"SKIP {year}: no GeoTIFF under {args.raw_dir}")
                continue
            meta = process_year(tif, args.out_dir, year)
            report_rows.append(meta)

    report_path = os.path.join(args.out_dir, "qa_report.csv")
    if report_rows:
        keys = sorted({k for r in report_rows for k in r.keys()})
        with open(report_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(report_rows)
        print("QA report:", report_path)
    else:
        print("No years processed. Use --demo or place GeoTIFFs in", args.raw_dir)


if __name__ == "__main__":
    main()
