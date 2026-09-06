from pathlib import Path
import sys
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def require_ee():
    try:
        import ee  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "earthengine-api is required. Install with: pip install earthengine-api"
        ) from exc
    return ee


def add_indices(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = image.normalizedDifference(["B8", "B5"]).rename("NDRE")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2"),
        },
    ).rename("EVI")
    return image.addBands([ndvi, ndre, evi])


def image_to_feature(ee, image, geometry):
    date = image.date().format("YYYY-MM-dd")
    cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE")
    reduced = image.select(["NDVI", "NDRE", "EVI"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        bestEffort=True,
    )
    return ee.Feature(
        None,
        {
            "date": date,
            "ndvi": reduced.get("NDVI"),
            "ndre": reduced.get("NDRE"),
            "evi": reduced.get("EVI"),
            "cloud_pct": cloud_pct,
        },
    )


def _load_registry(interim_dir: Path) -> pd.DataFrame:
    path = interim_dir / "coordinate_registry.csv"
    if not path.exists():
        try:
            from src.data.build_coordinate_registry import main as build_registry

            build_registry()
        except Exception as exc:
            logger.warning("Failed to build coordinate registry: %s", exc)
    if not path.exists():
        raise FileNotFoundError("Missing coordinate_registry.csv. Run build_coordinate_registry.py first.")
    return pd.read_csv(path)


def _load_cases(interim_dir: Path) -> pd.DataFrame:
    path = interim_dir / "cases.csv"
    if not path.exists():
        raise FileNotFoundError("Missing cases.csv. Run preprocess_all.py first.")
    cases = pd.read_csv(path)
    for col in ["sowing_date", "harvest_date"]:
        if col in cases.columns:
            cases[col] = pd.to_datetime(cases[col], errors="coerce")
    return cases


def _write_manual_report(report_path: Path, merged: pd.DataFrame, buffer_m: int) -> None:
    lines = [
        "# GEE Export Tasks",
        "",
        "Use these coordinates for manual Sentinel-2 exports.",
        f"Buffer: {buffer_m} m",
        "",
        "## Fields",
    ]
    if merged.empty:
        lines.append("- No rows in coordinate registry.")
    else:
        for _, row in merged.iterrows():
            case_id = row.get("case_id")
            field_id = row.get("field_id")
            district = row.get("district")
            lat = row.get("latitude")
            lon = row.get("longitude")
            sowing = row.get("sowing_date")
            harvest = row.get("harvest_date")
            season = f"{row.get('season_year_start')}-{row.get('season_year_end')}"
            lines.append(
                f"- case_id={case_id}, field_id={field_id}, district={district}, "
                f"lat={lat}, lon={lon}, sowing={sowing}, harvest={harvest}, season={season}"
            )

    lines.extend(
        [
            "",
            "## Export instructions",
            "1. Filter Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) by date range.",
            "2. Apply cloud filter and compute NDVI/NDRE/EVI.",
            "3. Reduce over a 500 m buffer around the field point.",
            "4. Export columns: case_id, field_id, district, date, ndvi_raw, ndre, evi, cloud_pct, source.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sentinel-2 indices via GEE")
    parser.add_argument("--cloud-max", type=float, default=80.0, help="Max cloud %")
    parser.add_argument("--scale", type=int, default=10, help="Pixel scale (m)")
    parser.add_argument("--project", type=str, default=None, help="GEE project ID")
    parser.add_argument("--buffer-m", type=int, default=500, help="Point buffer in meters")
    args = parser.parse_args()

    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    out_dir = Path(config["paths"]["satellite_dir"])
    ensure_dir(out_dir)
    ensure_dir(reports_dir)

    registry = _load_registry(interim_dir)
    cases = _load_cases(interim_dir)
    if registry.empty:
        logger.warning("Coordinate registry is empty. Fill it before exporting Sentinel-2.")
        return

    merged = registry.merge(
        cases[["case_id", "district", "sowing_date", "harvest_date", "season_year_start", "season_year_end"]],
        on="case_id",
        how="left",
    )

    report_path = reports_dir / "gee_export_tasks.md"
    _write_manual_report(report_path, merged, args.buffer_m)
    logger.info("Wrote %s", report_path)

    ee = require_ee()
    try:
        ee.Initialize(project=args.project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=args.project)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", args.cloud_max))
        .map(add_indices)
    )

    for _, row in merged.iterrows():
        case_id = row.get("case_id")
        field_id = row.get("field_id")
        district = row.get("district")
        lat = row.get("latitude")
        lon = row.get("longitude")
        sowing = row.get("sowing_date")
        harvest = row.get("harvest_date")

        if pd.isna(case_id) or pd.isna(field_id) or pd.isna(lat) or pd.isna(lon):
            logger.warning("Missing case_id/field_id/lat/lon. Skipping.")
            continue
        if pd.isna(sowing) or pd.isna(harvest):
            logger.warning("Missing sowing/harvest dates for %s. Skipping.", case_id)
            continue

        point = ee.Geometry.Point([float(lon), float(lat)])
        geometry = point.buffer(args.buffer_m)
        start_date = str(pd.to_datetime(sowing).date())
        end_date = str(pd.to_datetime(harvest).date())

        filtered = collection.filterDate(start_date, end_date).filterBounds(geometry)

        features = filtered.map(lambda img: image_to_feature(ee, img, geometry))
        try:
            data = features.getInfo()["features"]
        except Exception as exc:
            logger.error("GEE export failed for %s: %s", case_id, exc)
            continue

        rows = []
        for feat in data:
            props = feat.get("properties", {})
            rows.append(
                {
                    "case_id": case_id,
                    "field_id": field_id,
                    "district": district,
                    "date": props.get("date"),
                    "ndvi_raw": props.get("ndvi"),
                    "ndre": props.get("ndre"),
                    "evi": props.get("evi"),
                    "cloud_pct": props.get("cloud_pct"),
                    "source": "GEE_S2_SR_HARMONIZED",
                }
            )

        if not rows:
            logger.warning("No Sentinel-2 rows for %s", case_id)
            continue

        field_tag = str(field_id).replace(" ", "_")
        out_path = out_dir / f"sentinel2_{case_id}_{field_tag}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
