from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def _venv_python() -> Path | None:
    """Return the interpreter of a project-local .venv, if one is usable."""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def _run_script(script_path: Path) -> int:
    venv_python = _venv_python()
    python_exe = venv_python if venv_python is not None else Path(sys.executable)
    print(f"Running: {python_exe} {script_path}")
    result = subprocess.run([str(python_exe), str(script_path)], cwd=str(ROOT))
    return result.returncode


def main() -> None:
    scripts = [
        ROOT / "src" / "data" / "create_prediction_2025_cases.py",
        ROOT / "src" / "data" / "build_coordinate_registry.py",
        ROOT / "src" / "data" / "download_weather.py",
        ROOT / "src" / "data" / "check_satellite_files.py",
        ROOT / "src" / "data" / "ingest_satellite.py",
        ROOT / "src" / "data" / "ingest_apy_2025_ground_truth.py",
        ROOT / "src" / "features" / "extract_image_features.py",
        ROOT / "src" / "data" / "build_image_stage_features.py",
        ROOT / "src" / "data" / "build_stage_table.py",
        ROOT / "src" / "data" / "verify_training_data.py",
        ROOT / "src" / "data" / "diagnose_features.py",
        ROOT / "src" / "models" / "train_stagewise_regression.py",
        ROOT / "src" / "models" / "train_quintile_classifier.py",
        ROOT / "src" / "models" / "train_lstm_sequence_model.py",
        ROOT / "src" / "models" / "predict_stagewise.py",
        ROOT / "src" / "visualization" / "plot_results.py",
    ]

    exit_codes = []
    for script in scripts:
        if not script.exists():
            print(f"Warning: missing script {script}")
            continue
        exit_codes.append(_run_script(script))

    if any(code != 0 for code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
