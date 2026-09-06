from pathlib import Path
import yaml

from .paths import config_path
from .paths import resolve_project_path


def load_config() -> dict:
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str(resolve_project_path(value))

    return config


def resolve_path(config: dict, *parts: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root.joinpath(*parts)
