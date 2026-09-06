from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def _run_script(script_path: Path) -> int:
    python_exe = Path(sys.executable)
    print(f"Running: {python_exe} {script_path}")
    result = subprocess.run([str(python_exe), str(script_path)], cwd=str(ROOT))
    return result.returncode


def main() -> None:
    scripts = [
        ROOT / "src" / "deep_learning" / "train_densenet121.py",
        ROOT / "src" / "deep_learning" / "train_mobilenetv2.py",
        ROOT / "src" / "deep_learning" / "train_inceptionv3.py",
        ROOT / "src" / "deep_learning" / "train_vgg16.py",
        ROOT / "src" / "deep_learning" / "train_xception.py",
        ROOT / "src" / "deep_learning" / "extract_cnn_features.py",
        ROOT / "src" / "deep_learning" / "evaluate_cnn_models.py",
        ROOT / "src" / "deep_learning" / "compare_cnn_models.py",
        ROOT / "src" / "deep_learning" / "build_cnn_stage_features.py",
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
