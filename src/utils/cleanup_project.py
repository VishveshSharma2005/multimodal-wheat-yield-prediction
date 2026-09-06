from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import shutil


ROOT = Path(__file__).resolve().parents[2]

PROTECTED_PATHS = {
    ROOT / "data" / "raw" / "apy",
    ROOT / "data" / "interim" / "apy_yield_all_districts.csv",
    ROOT / "data" / "interim" / "cases.csv",
    ROOT / "data" / "interim" / "weather_daily.csv",
    ROOT / "data" / "interim" / "satellite_observations.csv",
    ROOT / "data" / "interim" / "image_metadata.csv",
    ROOT / "data" / "processed" / "stage_features.csv",
    ROOT / "data" / "processed" / "stage_features_with_split.csv",
    ROOT / "data" / "processed" / "targets.csv",
    ROOT / "src",
    ROOT / "configs",
    ROOT / "reports" / "apy_ingestion_report.md",
    ROOT / "reports" / "training_data_verification.md",
    ROOT / "README.md",
}


def _is_protected(path: Path) -> bool:
    for protected in PROTECTED_PATHS:
        try:
            if protected in path.parents or path == protected:
                return True
        except Exception:
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe cleanup utility (dry-run by default)")
    parser.add_argument("--apply", action="store_true", help="Move files into archive")
    args = parser.parse_args()

    candidates: list[tuple[Path, str]] = []

    # Empty APY extraction outputs
    empty_candidates = [
        ROOT / "data" / "interim" / "apy_yield.csv",
        ROOT / "data" / "interim" / "apy_state_benchmark.csv",
    ]
    for path in empty_candidates:
        if path.exists() and path.stat().st_size == 0 and not _is_protected(path):
            candidates.append((path, "Empty APY extraction output"))

    # Superseded APY single-district CSVs (outside protected raw/apy)
    for path in (ROOT / "data" / "raw" / "apy").glob("*.csv"):
        if _is_protected(path):
            continue
        if "mehsana" in path.name or "trace" in path.name or "field_registry" in path.name:
            candidates.append((path, "Superseded by all-district APY extraction"))

    # Old extraction report
    old_report = ROOT / "reports" / "apy_extraction_report.md"
    if old_report.exists() and not _is_protected(old_report):
        candidates.append((old_report, "Superseded by apy_ingestion_report.md"))

    if not candidates:
        print("No cleanup candidates found.")
        return

    print("Cleanup candidates:")
    for path, reason in candidates:
        print(f"- {path.relative_to(ROOT)} :: {reason}")

    if not args.apply:
        print("Dry run only. Use --apply to move files into archive.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ROOT / "archive" / f"unused_files_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path, reason in candidates:
        if not path.exists() or _is_protected(path):
            continue
        target = archive_dir / path.name
        print(f"Moving {path} -> {target} ({reason})")
        shutil.move(str(path), str(target))


if __name__ == "__main__":
    main()
