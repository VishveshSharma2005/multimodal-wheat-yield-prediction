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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_tables_from_pdf(pdf_path: Path, preferred_years: set[str]) -> list[dict]:
    import pdfplumber  # type: ignore

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                }
            )
            if not tables:
                continue

            for table_idx, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue

                header = [_normalize_text(h) for h in table[0]]
                if not header or "Area" not in " ".join(header):
                    continue

                for raw in table[1:]:
                    if not raw or len(raw) < 5:
                        continue

                    row = [_normalize_text(v) for v in raw]
                    district = row[0]
                    crop_table = row[1] if len(row) > 1 else ""
                    agri_year = row[2] if len(row) > 2 else ""
                    area = row[3] if len(row) > 3 else None
                    production = row[4] if len(row) > 4 else None
                    yield_val = row[5] if len(row) > 5 else None

                    if not district or "TOTAL" in district.upper() and district.upper() != "TOTAL WHEAT":
                        continue

                    crop_table_norm = crop_table.upper()
                    if "TOTAL WHEAT" not in crop_table_norm:
                        continue

                    if agri_year and preferred_years and agri_year not in preferred_years:
                        continue

                    rows.append(
                        {
                            "district": district,
                            "state": "Gujarat",
                            "crop": "wheat",
                            "season": "rabi",
                            "agri_year": agri_year,
                            "season_year_start": agri_year.split("-")[0] if "-" in agri_year else "",
                            "season_year_end": "20" + agri_year.split("-")[1] if "-" in agri_year else "",
                            "area_000_ha": _clean_num(area),
                            "production_000_mt": _clean_num(production),
                            "yield_kg_ha": _clean_num(yield_val),
                            "source_file": pdf_path.name,
                            "source_page": page_idx,
                            "source_table": f"table_{table_idx}",
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

    pdf_20_23 = apy_dir / "20-23.pdf"
    pdf_23_24 = apy_dir / "23-24.pdf"
    pdf_24_25 = apy_dir / "24-25.pdf"

    if not pdf_20_23.exists() or not pdf_23_24.exists():
        raise FileNotFoundError("Required APY PDFs missing in data/raw/apy")

    preferred_20_23 = {"2020-21"}
    preferred_23_24 = {"2021-22", "2022-23", "2023-24"}

    rows = []
    rows.extend(_parse_tables_from_pdf(pdf_20_23, preferred_20_23))
    rows.extend(_parse_tables_from_pdf(pdf_23_24, preferred_23_24))

    df = pd.DataFrame(rows)
    if df.empty:
        out_all = interim_dir / "apy_yield_all_districts.csv"
        df.to_csv(out_all, index=False)
        report_path = reports_dir / "apy_extraction_report.md"
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# APY Extraction Report\n\n")
            f.write("## PDFs used\n")
            f.write(f"- {pdf_20_23.name}\n")
            f.write(f"- {pdf_23_24.name}\n")
            f.write(f"- {pdf_24_25.name if pdf_24_25.exists() else 'not found'}\n\n")
            f.write("## Extraction status\n")
            f.write("- No machine-readable tables found. PDFs appear scanned.\n\n")
            f.write("## Next steps\n")
            f.write("- Provide machine-readable CSVs from data.gov.in or OCR the PDFs.\n")
        logger.warning("No APY rows extracted. PDFs may be scanned images.")
        return

    # Separate state-level rows
    is_state = df["district"].str.contains("GUJARAT|STATE", case=False, na=False)
    state_df = df[is_state].copy()
    df = df[~is_state].copy()

    df = df.drop_duplicates(
        subset=["district", "agri_year"], keep="last"
    ).reset_index(drop=True)

    # State benchmark extraction if present
    benchmark_rows = []
    if pdf_24_25.exists():
        bench = _parse_tables_from_pdf(pdf_24_25, {"2024-25"})
        benchmark_rows = [
            r for r in bench if "STATE" in r["district"].upper() or "GUJARAT" in r["district"].upper()
        ]

    out_all = interim_dir / "apy_yield_all_districts.csv"
    df.to_csv(out_all, index=False)

    if not state_df.empty or benchmark_rows:
        state_all = pd.concat([state_df, pd.DataFrame(benchmark_rows)], ignore_index=True)
        state_all.to_csv(interim_dir / "apy_state_benchmark.csv", index=False)

    # Report
    report_path = reports_dir / "apy_extraction_report.md"
    years = sorted(df["agri_year"].dropna().unique().tolist())
    districts = sorted(df["district"].dropna().unique().tolist())
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# APY Extraction Report\n\n")
        f.write("## PDFs used\n")
        f.write(f"- {pdf_20_23.name}\n")
        f.write(f"- {pdf_23_24.name}\n")
        f.write(f"- {pdf_24_25.name if pdf_24_25.exists() else 'not found'}\n\n")
        f.write("## Years extracted\n")
        f.write(", ".join(years) + "\n\n")
        f.write(f"## District-year rows: {len(df)}\n")
        f.write(f"## Number of districts: {len(districts)}\n\n")
        f.write("## Duplicate handling\n")
        f.write("- Kept latest PDF values for duplicate years (23-24 preferred).\n\n")
        f.write("## 24-25 handling\n")
        f.write("- Stored only as state benchmark if present.\n\n")
        f.write("## Warnings/limitations\n")
        f.write("- PDF table layouts can vary. Verify outputs visually.\n")

    logger.info("Wrote %s", out_all)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
