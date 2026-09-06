from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return project_root() / "configs" / "config.yaml"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve project-relative paths against the current repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def to_project_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()
