'''
为整个工程提供统一的绝对路径
'''

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> str:
    return str(_PROJECT_ROOT)


def _resolve_relative_path(relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("relative_path must be a relative path")

    resolved = (_PROJECT_ROOT / candidate).resolve(strict=False)
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"relative_path escapes project root: {relative_path}") from exc

    return resolved


def is_path_within_project(path: str | Path) -> bool:
    candidate = Path(path).resolve(strict=False)
    try:
        candidate.relative_to(_PROJECT_ROOT)
        return True
    except ValueError:
        return False


def get_abs_path(relative_path: str) -> str:
    return str(_resolve_relative_path(relative_path))


if __name__ == "__main__":
    print(get_abs_path("data/processed_data.csv"))