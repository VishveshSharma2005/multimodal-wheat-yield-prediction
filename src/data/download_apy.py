from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.paths import ensure_dir
from src.utils.logging_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    apy_dir = Path(config["paths"]["apy_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(reports_dir)

    csv_files = sorted(apy_dir.glob("*.csv"))
    pdf_files = sorted(apy_dir.glob("*.pdf"))
    if pdf_files:
        logger.warning("PDF files found in %s. Convert to CSV and re-run.", apy_dir)
    if not csv_files:
        logger.warning("No APY CSVs found in %s", apy_dir)
        logger.info("Manual download required. See reports/data_sources_report.md")
        return

    frames = []
    for path in csv_files:
        if "trace" in path.stem or "field_registry" in path.stem:
            logger.info("Skipping non-model CSV: %s", path.name)
            continue
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)

    if not frames:
        logger.warning("No APY data loaded.")
        return

    out_df = pd.concat(frames, ignore_index=True)
    out_path = interim_dir / "apy_yield.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
