from pathlib import Path
import sys
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)

YEAR_RE = re.compile(r"\b(20\d{2}-\d{2})\b")


def _clean_num(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "NA", "N/A"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _row_has_token(row, token: str) -> bool:
    return row.astype(str).str.contains(token, case=False, na=False).any()


def _find_header_row(df: pd.DataFrame, start_idx: int) -> int | None:
    for i in range(start_idx, min(start_idx + 80, len(df))):
        row = df.iloc[i]
        if _row_has_token(row, "DISTRICT") and _row_has_token(row, "SR"):
            return i
    return None


def _extract_year_columns(header_row: pd.Series) -> list[tuple[str, int]]:
    cols = []
    for idx, val in header_row.items():
        match = YEAR_RE.search(str(val))
        if match:
            cols.append((match.group(1), idx))
    return cols


def _extract_metric_cols(subheader_row: pd.Series, start_col: int, end_col: int) -> dict[str, int]:
    area_col = prod_col = yield_col = None
    for idx in range(start_col, end_col):
        val = str(subheader_row.get(idx, ""))
        if re.search(r"AREA", val, re.IGNORECASE):
            area_col = idx
        if re.search(r"PROD", val, re.IGNORECASE):
            prod_col = idx
        if re.search(r"YIELD", val, re.IGNORECASE):
            yield_col = idx
    return {"area": area_col, "prod": prod_col, "yield": yield_col}


def _parse_sheet(df: pd.DataFrame, source_file: str, source_sheet: str) -> list[dict]:
    rows = []
    total_wheat_rows = df.apply(lambda r: _row_has_token(r, "TOTAL WHEAT"), axis=1)
    total_indices = list(df.index[total_wheat_rows])
    if not total_indices:
        return rows

    for start_idx in total_indices:
        header_idx = _find_header_row(df, start_idx)
        if header_idx is None:
            continue

        header_row = df.iloc[header_idx]
        year_cols = _extract_year_columns(header_row)
        if not year_cols:
            continue

        # Identify district column
        district_col = None
        for idx, val in header_row.items():
            if re.search(r"DISTRICT", str(val), re.IGNORECASE):
                district_col = idx
                break
        if district_col is None:
            continue

        subheader_row = df.iloc[header_idx + 1]
        year_cols_sorted = sorted(year_cols, key=lambda x: x[1])

        metric_cols = {}
        for i, (year, col_idx) in enumerate(year_cols_sorted):
            end_col = year_cols_sorted[i + 1][1] if i + 1 < len(year_cols_sorted) else col_idx + 30
            metric_cols[year] = _extract_metric_cols(subheader_row, col_idx, end_col)

        # Data starts after subheader
        for i in range(header_idx + 2, min(header_idx + 200, len(df))):
            row = df.iloc[i]
            district = row.get(district_col)
            if pd.isna(district):
                break
            district_str = str(district).strip()
            if not district_str:
                break
            if re.search(r"TOTAL|STATE|GUJARAT", district_str, re.IGNORECASE):
                continue

            for year, cols in metric_cols.items():
                area = row.get(cols["area"]) if cols["area"] is not None else None
                prod = row.get(cols["prod"]) if cols["prod"] is not None else None
                yld = row.get(cols["yield"]) if cols["yield"] is not None else None

                if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in [area, prod, yld]):
                    continue

                rows.append(
                    {
                        "district": district_str,
                        "state": "Gujarat",
                        "crop": "wheat",
                        "season": "rabi",
                        "agri_year": year,
                        "season_year_start": year.split("-")[0],
                        "season_year_end": "20" + year.split("-")[1],
                        "area_000_ha": _clean_num(area),
                        "production_000_mt": _clean_num(prod),
                        "yield_kg_ha": _clean_num(yld),
                        "source_file": source_file,
                        "source_sheet": source_sheet,
                        "notes": "",
                    }
                )

    return rows


def main() -> None:
    config = load_config()
    apy_dir = Path(config["paths"]["apy_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    files = [
        apy_dir / "20-23.xlsx",
        apy_dir / "23-24.xlsx",
        apy_dir / "24-25.xlsx",
    ]

    for f in files[:2]:
        if not f.exists():
            raise FileNotFoundError(f"Missing APY file: {f.name}")

    rows = []
    for fpath in files:
        if not fpath.exists():
            continue
        xl = pd.ExcelFile(fpath)
        for sheet in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name=sheet, header=None)
            rows.extend(_parse_sheet(df, fpath.name, sheet))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No APY rows extracted from XLSX files.")

    # Prefer latest values for overlapping years
    df = df.sort_values(["district", "agri_year", "source_file"])
    prefer_files = {
        "2020-21": "20-23.xlsx",
        "2021-22": "23-24.xlsx",
        "2022-23": "23-24.xlsx",
        "2023-24": "23-24.xlsx",
    }
    df = df[
        df.apply(
            lambda r: prefer_files.get(r["agri_year"], r["source_file"]) == r["source_file"],
            axis=1,
        )
    ]

    # Remove duplicates
    df = df.drop_duplicates(subset=["district", "agri_year"], keep="last")

    # State benchmark from 24-25 only if present
    state_mask = df["district"].str.contains("GUJARAT|STATE", case=False, na=False)
    state_df = df[state_mask].copy()
    df = df[~state_mask].copy()

    out_all = interim_dir / "apy_yield_all_districts.csv"
    df.to_csv(out_all, index=False)

    if not state_df.empty:
        state_df.to_csv(interim_dir / "apy_state_benchmark.csv", index=False)

    # Report
    report_path = reports_dir / "apy_ingestion_report.md"
    years = sorted(df["agri_year"].dropna().unique().tolist())
    districts = sorted(df["district"].dropna().unique().tolist())
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# APY Ingestion Report\n\n")
        f.write("## XLSX files used\n")
        for fpath in files:
            f.write(f"- {fpath.name if fpath.exists() else fpath.name + ' (missing)'}\n")
        f.write("\n## Years extracted\n")
        f.write(", ".join(years) + "\n\n")
        f.write(f"## District-year rows: {len(df)}\n")
        f.write(f"## Number of districts: {len(districts)}\n\n")
        f.write("## Duplicate handling\n")
        f.write("- Preferred latest values (23-24.xlsx) for 2021-22 to 2023-24.\n")
        f.write("- Used 20-23.xlsx for 2020-21.\n\n")
        f.write("## 24-25 handling\n")
        f.write("- State-level rows saved as benchmark only if present.\n\n")
        f.write("## Notes\n")
        f.write("- Verify district counts and years after parsing.\n")

    logger.info("Wrote %s", out_all)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
