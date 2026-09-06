from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def _run(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script}")
    print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script.relative_to(ROOT)} failed with exit code {result.returncode}")


def main() -> None:
    scripts = [
        ROOT / "src" / "deep_learning" / "build_cnn_stage_features.py",
        ROOT / "src" / "data" / "build_stage_table.py",
        ROOT / "src" / "data" / "verify_training_data.py",
        ROOT / "src" / "data" / "diagnose_features.py",
        ROOT / "src" / "models" / "train_stagewise_regression.py",
        ROOT / "src" / "models" / "train_quintile_classifier.py",
        ROOT / "src" / "models" / "train_lstm_sequence_model.py",
        ROOT / "src" / "models" / "predict_stagewise.py",
        ROOT / "src" / "models" / "run_cnn_ablation_study.py",
        ROOT / "src" / "models" / "generate_cnn_improvement_report.py",
    ]

    for script in scripts:
        _run(script)

    print("\nCNN-integrated training pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
