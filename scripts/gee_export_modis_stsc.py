"""
GEE: 导出黄海农场及周边 MODIS MOD09A1 7 波段 x 46 期/年 地表反射率。

用法（需先 ee.Authenticate / ee.Initialize）：
  python scripts/gee_export_modis_stsc.py --year 2015 --to-drive
  python scripts/gee_export_modis_stsc.py --year-start 2001 --year-end 2020 --to-drive

输出到 Google Drive 文件夹 MAGIC_STSC（可改 --drive-folder）。
下载后放到 data/raw/modis_stsc/YYYY/ 再用 qc_modis_stsc.py 处理。
"""
import argparse

try:
    import ee
except ImportError as e:
    raise SystemExit("需要 earthengine-api: pip install earthengine-api") from e

# 黄海农场中心附近约 50km x 50km（可按需微调）
HUANGHAI_ROI = {
    "west": 119.40,
    "south": 34.55,
    "east": 119.95,
    "north": 35.05,
}

BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04",
         "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
QA_BAND = "StateQA"  # MOD09A1 uses StateQA; some catalogs label state_1km


def get_roi():
    return ee.Geometry.Rectangle([
        HUANGHAI_ROI["west"], HUANGHAI_ROI["south"],
        HUANGHAI_ROI["east"], HUANGHAI_ROI["north"],
    ])


def mask_clouds(img):
    """用 StateQA 掩膜云/雪/阴影（简化：取 bit 0-2 cloud state == 0 clear）。"""
    qa = img.select(QA_BAND)
    # Bits 0-1: cloud state (00 = clear)
    cloud_state = qa.bitwiseAnd(3)
    clear = cloud_state.eq(0)
    # Bit 2: cloud shadow
    shadow = qa.bitwiseAnd(1 << 2).eq(0)
    # Bits 12-13: snow/ice (00 = none) — optional
    snow = qa.bitwiseAnd(3 << 12).eq(0)
    mask = clear.And(shadow).And(snow)
    return img.select(BANDS).updateMask(mask).multiply(0.0001).copyProperties(img, ["system:time_start"])


def year_stack(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    col = (ee.ImageCollection("MODIS/061/MOD09A1")
           .filterBounds(get_roi())
           .filterDate(start, end)
           .map(mask_clouds))

    # 期望约 46 景；按时间排序后 toBands
    col = col.sort("system:time_start")
    stacked = col.toBands()
    return stacked.clip(get_roi()).set("year", year)


def export_year(year, drive_folder, scale=500):
    img = year_stack(year)
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=f"MOD09A1_STSC_huanghai_{year}",
        folder=drive_folder,
        fileNamePrefix=f"mod09a1_stsc_huanghai_{year}",
        region=get_roi(),
        scale=scale,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Started export task for {year}: {task.id}")
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=None,
                        help="GEE Cloud project id，如 ee-xxxx")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--year-start", type=int, default=2001)
    parser.add_argument("--year-end", type=int, default=2020)
    parser.add_argument("--drive-folder", type=str, default="MAGIC_STSC")
    parser.add_argument("--to-drive", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印 ROI/波段，不提交任务")
    args = parser.parse_args()

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    years = [args.year] if args.year else list(range(args.year_start, args.year_end + 1))
    print("ROI:", HUANGHAI_ROI)
    print("Years:", years)
    print("Bands:", BANDS)

    if args.dry_run:
        print("dry-run: skip export")
        return

    if not args.to_drive:
        print("未指定 --to-drive，退出。确认后加上 --to-drive 提交 Drive 导出。")
        return

    for y in years:
        export_year(y, args.drive_folder)
    print("全部任务已提交。到 GEE Tasks 面板查看进度，下载到 data/raw/modis_stsc/")


if __name__ == "__main__":
    main()
